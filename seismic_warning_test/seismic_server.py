__author__ = 'Kyle Benson'

# TODO: update with RideD description
NEW_SCRIPT_DESCRIPTION = '''Simple aggregating server that receives 'picks' from clients
indicating a possible seismic event using a simple JSON format over UDP.
It aggregates together all of the readings over a short period of time and
then forwards the combined data to interested client devices. The forwarding is done
using the RideD resilient multicast middleware.'''

# @author: Kyle Benson
# (c) Kyle Benson 2017

import logging as log
import sys
import argparse
import time
import asyncore
import socket
import json
import signal
import os
from threading import Timer

import ride

# Buffer size for receiving packets
BUFF_SIZE = 4096
PUBLICATION_TOPIC = 'seismic_alert'


def parse_args(args):
    ##################################################################################
    # ################      ARGUMENTS       ###########################################
    # ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    # action is one of: store[_const,_true,_false], append[_const], count
    # nargs is one of: N, ?(defaults to const when no args), *, +, argparse.REMAINDER
    # help supports %(var)s: help='default value is %(default)s'
    ##################################################################################

    parser = argparse.ArgumentParser(description=NEW_SCRIPT_DESCRIPTION,
                                     parents=(ride.ride_d.RideD.get_arg_parser(),),
                                     #formatter_class=argparse.RawTextHelpFormatter,
                                     #epilog='Text to display at the end of the help print',
                                     )

    # parameters used for simulation/testing
    parser.add_argument('--delay', type=float, default=5,
                        help='''time period (in secs) during which sensor readings
                         are aggregated before sending the data to interested parties''')
    parser.add_argument('--quit_time', '-q', type=float, default=30,
                        help='''delay (in secs) before quitting''')
    parser.add_argument('--no-ride', action='store_false', dest='with_ride',
                        help='''disable use of RIDE middleware, in which case aggregated data
                         is sent to subscribers' unicast IP addresses (specified in --subs).''')
    parser.add_argument('--debug', '-d', type=str, default='info', nargs='?', const='debug',
                        help='''set debug level for logging facility (default=%(default)s,
                         %(const)s when specified with no arg)''')

    # networking configuration
    parser.add_argument('--recv_port', type=int, default=9999,
                        help='''UDP port number from which data should be received''')
    parser.add_argument('--send_port', type=int, default=9998,
                        help='''UDP port number to which data should be sent''')

    # data exchange configuration
    parser.add_argument('--pubs', nargs='+', default=[],
                        help='''list of publisher DPIDs (not used if configured without RIDE)''')
    parser.add_argument('--subs', nargs='+', default=[],
                        help='''list of subscriber DPIDs (if configured with RIDE) or their IP addresses
                         (if configured without RIDE).''')

    return parser.parse_args(args)


class SeismicServer(asyncore.dispatcher):
    EXIT_CODE_NO_SUBSCRIBERS = 10

    def __init__(self, config):
        asyncore.dispatcher.__init__(self)

        log_level = log.getLevelName(config.debug.upper())
        log.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log_level)

        # store configuration options and validate them
        self.config = config

        # Stores received events indexed by their 'id'
        self.events_rcvd = dict()

        #### BEGIN RIDE-D CONFIGURATION

        # Set up RideD resilient multicast middleware if enabled. If disabled, we still need
        # to build a RideD object in order to set up static unicast routes for subscribers in order
        # to avoid controller recovering failed paths quickly due to Mininet's zero latency
        # control plane network.  Either way, we set static routes for publishers in
        # order to ensure we know what route their publications took.
        self.rided = ride.ride_d.RideD.build_from_args(config, pre_parsed=True)

        for sub in config.subs:
            # HACK: similar to the try statement below, we should only register subscribers
            # that are reachable in our topology view or we'll cause errors later...
            try:
                route = self.rided.topology_manager.get_path(config.dpid, sub, weight='latency')
                self.rided.add_subscriber(sub, topic_id=PUBLICATION_TOPIC)

                # Set static unicast routes to subscribers for non-ride config.
                try:
                    # NOTE: because we only set static routes when not using RideD, this shouldn't
                    # interfere with other routes set for multicast usage.
                    if not config.with_ride:
                        log.debug("Installing static route for subscriber: %s" % route)
                        flow_rules = self.rided.topology_manager.build_flow_rules_from_path(route)
                        for r in flow_rules:
                            self.rided.topology_manager.install_flow_rule(r)
                except Exception as e:
                    log.error("Error installing flow rules for static subscriber routes: %s" % e)
            except Exception as e:
                log.warning("Route between subscriber %s and server %s not found: skipping... Error: %s" % (sub, config.dpid, e))

        for pub in config.pubs:
            # HACK: we get the shortest path (as per networkx) and set that as a static route
            # to prevent the controller from changing the path later since we don't dynamically
            # update the routes currently.
            try:
                route = self.rided.topology_manager.get_path(pub, config.dpid, weight='latency')
                log.debug("Installing static route publisher subscriber: %s" % route)
                flow_rules = self.rided.topology_manager.build_flow_rules_from_path(route)
                for r in flow_rules:
                    self.rided.topology_manager.install_flow_rule(r)
                self.rided.set_publisher_route(pub, route)
            except Exception as e:
                log.warning("Route between publisher %s and server %s not found: skipping... Error: %s" % (pub, config.dpid, e))

        # BUGFIX: if all subscribers are unreachable in the topology due to failure updates
        # propagating to the controller, we won't have registered any subs for the topic.
        try:
            self.rided.get_subscribers_for_topic(PUBLICATION_TOPIC)
        except KeyError:
            log.error("No subscribers reachable by server!  Aborting...")
            exit(self.EXIT_CODE_NO_SUBSCRIBERS)

        # Set up multicast trees!
        if config.with_ride:
            self.addresses = self.config.addresses
            mdmts = self.rided.build_mdmts()[PUBLICATION_TOPIC]
            self.rided.install_mdmts(mdmts)
        # Because we potentially use different IDs to identify hosts than IP address,
        # we need to do a conversion here if we're not using RideD.
        else:
            self.addresses = [self.rided.topology_manager.get_ip_address(s) for s in self.config.subs]

        #### END RIDE-D CONFIGURATION

        # queue seismic event aggregation and forwarding
        # need to store references to cancel them when finish() is called
        self.next_timer = Timer(self.config.delay, self.send_events).start()

        # setup UDP network socket to listen for events on and multicast aggregated events on
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
        self.bind(('', self.config.recv_port))

        # we assume that the quit_time is far enough in the future that
        # no interesting sensor data will arrive around then, hence not
        # worrying about flushing the buffer
        Timer(self.config.quit_time, self.finish).start()

        # Sending SIGINT results in the process hanging in the asyncore loop for a bit
        def __sigint_handler(sig, frame):
            self.exit_now()
        signal.signal(signal.SIGINT, __sigint_handler)


    def send_events(self):
        log.debug("Checking for aggregated events...")
        if len(self.events_rcvd) > 0:
            agg_events = dict()
            # aggregated events are expected as an array
            agg_events['events'] = [v for v in self.events_rcvd.values()]
            agg_events['id'] = 'aggregator'
            data = json.dumps(agg_events)

            # Gather subscriber IP addresses if using unicast or just use single multicast
            # one if configured with RIDE.
            if self.config.with_ride:
                addresses = [self.rided.get_best_multicast_address(PUBLICATION_TOPIC)]
                # ENHANCE: add some unicast addresses for those we think are unreachable via multicast
            else:
                addresses = self.addresses

            for address in addresses:
                # TODO: test and determine whether or not we need to lock the data structures
                self.sendto(data, (address, self.config.send_port))
                log.info("Aggregated events sent to %s" % address)

        # don't forget to schedule the next time we send aggregated events
        self.next_timer = Timer(self.config.delay, self.send_events)
        self.next_timer.start()

    def handle_read(self):
        """
        Receive an event and record the time we received it.
        If it's already been received, we simply record the fact that
        we've received a duplicate.
        """

        # Need to get the IP address of the sending publisher
        # data = self.recv(BUFF_SIZE)
        data, (ip, port) = self.socket.recvfrom(BUFF_SIZE)

        # ENHANCE: handle packets too large to fit in this buffer
        try:
            event = json.loads(data)
            log.info("received event %s" % event)

            # Aggregate events together in an array
            if event['id'] not in self.events_rcvd:
                event['time_aggd'] = time.time()
                self.events_rcvd[event['id']] = event

        except ValueError:
            log.error("Error parsing JSON from %s" % data)
        except IndexError as e:
            log.error("Malformed event dict: %s" % e)

        # Need to notify RideD that we received a successful publication.
        # TODO: may need to wrap this with mutex
        if self.config.with_ride:
            publisher = ip
            self.rided.notify_publication(publisher, id_type='ip')

    def run(self):
        try:
            asyncore.loop()
        except:
            # seems as though this just crashes sometimes when told to quit
            log.error("Error in SeismicServer.run() can't recover...")
            self.finish()

    def finish(self):
        # need to cancel the next timer or the loop could keep going
        self.next_timer.cancel()
        self.close()

        # TODO: clear out all the routes we added without deleting ALL flows.
        # This will require storing some metadata about the flows/groups we added.

    def exit_now(self, error_code=1):
        # HACK: kill the whole process immediately and include the error code (regular exit only kills thread)
        os._exit(error_code)

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    client = SeismicServer(args)
    client.run()
