#! /usr/bin/python

# @author: Kyle Benson
# (c) Kyle Benson 2017
import os
import traceback

import errno

from smart_campus_experiment import SmartCampusExperiment

import logging as log
import json
import argparse
import time
import signal
import ipaddress

from mininet.net import Mininet
from mininet.node import RemoteController, Host, OVSKernelSwitch
from mininet.node import Switch, Link, Node  # these just used for types in docstrings
from mininet.cli import CLI
from mininet.link import TCLink, Intf

from topology_manager.networkx_sdn_topology import NetworkxSdnTopology
from topology_manager.test_sdn_topology import mac_for_host  # used for manual MAC assignment

EXPERIMENT_DURATION = 35  # in seconds
SEISMIC_EVENT_DELAY = 25  # seconds before the 'earthquake happens', i.e. sensors start sending data
IPERF_BASE_PORT = 5000  # background traffic generators open several iperf connections starting at this port number
OPENFLOW_CONTROLLER_PORT = 6653  # we assume the controller will always be at the default port
IP_SUBNET = '10.0.0.0/24'  # subnet for all hosts
WITH_LOGS = True  # output seismic client/server stdout to a log file
# HACK: rather than some ugly hacking at Mininet's lack of API for allocating the next IP address,
# we just put the NAT/server interfaces in a hard-coded subnet.
NAT_SERVER_IP_ADDRESS = '11.0.0.%d/24'
# TODO: use a different address base...
MULTICAST_ADDRESS_BASE = u'224.0.0.1'  # must be unicode!
# When True, runs host processes with -00 command for optimized python code
OPTIMISED_PYTHON = False

class MininetSmartCampusExperiment(SmartCampusExperiment):
    """
    Version of SmartCampusExperiment that runs the experiment in Mininet emulation.
    This includes some background traffic and....
    """

    def __init__(self, controller_ip='127.0.0.1', controller_port=8181,
                 tree_choosing_heuristic='importance', topology_adapter='onos',  # need to save these to pass to RideD
                 n_traffic_generators=5, traffic_generator_bandwidth=10,
                 *args, **kwargs):
        # TODO: add additional params before committing
        """
        Mininet and the SdnTopology adapter will be started by this constructor.
        NOTE: you must start the remote SDN controller before constructing/running the experiment!

        :param tree_choosing_heuristic: explicit in this version since we are running an
         actual emulation and so cannot check all the heuristics at once
        :param args: see args of superclass
        :param kwargs: see kwargs of superclass
        """
        super(MininetSmartCampusExperiment, self).__init__(*args, **kwargs)
        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.tree_choosing_heuristic = tree_choosing_heuristic
        self.topology_adapter = topology_adapter
        self.n_traffic_generators = n_traffic_generators
        self.traffic_generator_bandwidth = traffic_generator_bandwidth

        # These will all be filled in by calling setup_mininet()
        #TODO: do we actually need all these???
        self.hosts = []
        self.switches = []
        self.links = []
        self.server = None
        self.net = None
        self.controller = None
        self.nat = None

        # HACK: Mininet doesn't exit properly for some reason so we can't do >1 run...
        if self.nruns > 1:
            log.warning("nruns > 1 not currently supported for Mininet experiment.  Only doing 1 run...")
            self.nruns = 1

        # HACK: We just manually allocate IP addresses rather than adding a controller API to request them.
        base_addr = ipaddress.IPv4Address(MULTICAST_ADDRESS_BASE)
        self.mcast_address_pool = [str(base_addr + i) for i in range(kwargs['ntrees'])]

    # argument parser that can be combined with others when this class is used in a script
    # need to not add help options to use that feature, though
    arg_parser = argparse.ArgumentParser(parents=[SmartCampusExperiment.get_arg_parser()])  #add_help=False ???
    # experimental treatment parameters: all taken from parents
    # background traffic generation
    arg_parser.add_argument('--ngenerators', '-g', default=5, dest='n_traffic_generators', type=int,
                            help='''number of hosts that generate random traffic to cause congestion (default=%(default)s)''')
    arg_parser.add_argument('--generator-bandwidth', '-bw', default=10, dest='traffic_generator_bandwidth', type=float,
                            help='''bandwidth (in Mbps) of iperf for congestion traffic generating hosts (default=%(default)s)''')

    def record_result(self, result):
        """Result is a dict that includes the percentage of subscribers
        reachable as well as metadata such as run #"""
        self.results['results'].append(result)

    def output_results(self):
        """Outputs the results to a file"""
        print self.output_filename
        log.info("Results: %s" % json.dumps(self.results, sort_keys=True, indent=2))
        with open(self.output_filename, "w") as f:
            json.dump(self.results, f, sort_keys=True, indent=2)

    def set_interrupt_signal(self):
        # ignore it so we can terminate Mininet commands without killing Mininet
        # TODO: something else?
        return

    def setup_topology(self):
        """
        Builds the Mininet network, including all hosts, servers, switches, links, and NATs.
        This relies on reading the topology file using a NetworkxSdnTopology helper.
        :param str topology_file: file name of topology to import
        """
        self.net = Mininet(topo=None,
                           build=False,
                           ipBase=IP_SUBNET,
                           autoSetMacs=True,
                           # autoStaticArp=True
                           )

        log.info('*** Adding controller')
        self.controller = self.net.addController(name='c0',
                                         controller=RemoteController,
                                         ip=self.controller_ip,
                                         port=OPENFLOW_CONTROLLER_PORT,
                                         )

        # import the switches, hosts, and server(s) from our specified file
        topo = NetworkxSdnTopology(self.topology_filename)
        self.topo = topo

        def __get_mac_for_switch(switch):
            # BUGFIX: need to manually specify the mac to set DPID properly or Mininet
            # will just use the number at the end of the name, causing overlaps.
            # HACK: slice off the single letter at start of name, which we assume it has;
            # then convert the number to a MAC.
            mac = mac_for_host(int(switch[1:]))
            # Disambiguate one switch type from another by setting the first letter
            # to be a unique one corresponding to switch type and add in the other 0's.
            first_letter = switch[0]
            if first_letter == 'm':
                first_letter = 'a'
            # rest fit in hex except for rack switches
            mac = first_letter + '0:00:00' + mac[3:]
            return str(mac)

        for switch in topo.get_switches():
            mac = __get_mac_for_switch(switch)
            s = self.net.addSwitch(switch, dpid=mac, cls=OVSKernelSwitch)
            log.debug("adding switch %s at DPID %s" % (switch, s.dpid))
            self.switches.append(s)

        for host in topo.get_hosts():
            h = self.net.addHost(host)
            self.hosts.append(h)

        for server in topo.get_servers():
            # HACK: we actually add a switch in case the server is multi-homed since it's very
            # difficult to work with multiple interfaces on a host (e.g. ONOS can only handle
            # a single MAC address per host).
            server_switch_name = server.replace('s', 'e')
            server_switch_dpid = __get_mac_for_switch(server_switch_name)
            # Keep server name for switch so that the proper links will be added later.
            self.server_switch = self.net.addSwitch(server, dpid=server_switch_dpid, cls=OVSKernelSwitch)
            s = self.net.addHost('h' + server)
            self.server = s
            self.net.addLink(self.server_switch, self.server)

        for link in topo.get_links():
            from_link = link[0]
            to_link = link[1]
            log.debug("adding link from %s to %s" % (from_link, to_link))

            # Get link attributes for configuring realistic traffic control settings
            # For configuration options, see mininet.link.TCIntf.config()
            attributes = link[2]
            _bw = attributes.get('bw', 10)  # in Mbps
            _delay = '%fms' % attributes.get('latency', 10)
            _jitter = '1ms'
            _loss = self.publication_error_rate

            l = self.net.addLink(self.net.get(from_link), self.net.get(to_link),
                                 cls=TCLink, bw=_bw, delay=_delay, jitter=_jitter, loss=_loss
                                 )
            self.links.append(l)

        # add NAT so the server can communicate with SDN controller's REST API
        # NOTE: because we didn't add it to the actual SdnTopology, we don't need
        # to worry about it getting failed.  However, we do need to ensure it
        # connects directly to the server to avoid failures disconnecting it.
        # HACK: directly connect NAT to the server, set a route for it, and
        # handle this hacky IP address configuration
        nat_ip = NAT_SERVER_IP_ADDRESS % 2
        srv_ip = NAT_SERVER_IP_ADDRESS % 3
        self.nat = self.net.addNAT(connect=self.server)
        self.nat.configDefault(ip=nat_ip)

        # Now we set the IP address for the server's new interface.
        # NOTE: we have to set the default route after starting Mininet it seems...
        srv_iface = sorted(self.server.intfNames())[-1]
        self.server.intf(srv_iface).setIP(srv_ip)

    # HACK: because self.topo stores nodes by just their string name, we need to
    # convert them into actual Mininet hosts for use by this experiment.

    def _get_mininet_nodes(self, nodes):
        """
        Choose the actual Mininet Hosts (rather than just strings) that will
        be subscribers.
        :param List[str] nodes:
        :return List[Node] mininet_nodes:
        """
        return [self.net.get(n) for n in nodes]

    def choose_publishers(self):
        """
        Choose the actual Mininet Hosts (rather than just strings) that will
        be publishers.
        :return List[Host] publishers:
        """
        return self._get_mininet_nodes(super(MininetSmartCampusExperiment, self).choose_publishers())

    def choose_subscribers(self):
        """
        Choose the actual Mininet Hosts (rather than just strings) that will
        be subscribers.
        :return List[Host] subscribers:
        """
        return self._get_mininet_nodes(super(MininetSmartCampusExperiment, self).choose_subscribers())

    def choose_server(self):
        """
        Choose the actual Mininet Host (rather than just strings) that will
        be the server.
        :return Host server:
        """
        return self.server

    def get_failed_nodes_links(self):
        fnodes, flinks = super(MininetSmartCampusExperiment, self).get_failed_nodes_links()
        # NOTE: we can just pass the links as strings
        return self._get_mininet_nodes(fnodes), flinks

    def run_experiment(self, failed_nodes, failed_links, server, publishers, subscribers):
        """
        Configures all appropriate settings, runs the experiment, and
        finally tears it down before returning the results.
        (Assumes Mininet has already been started).

        :param List[Node] failed_nodes:
        :param List[str] failed_links:
        :param Host server:
        :param List[Host] publishers:
        :param List[Host] subscribers:
        :rtype dict:
        """

        log.info('*** Starting network')
        self.net.build()
        self.net.start()
        self.net.waitConnected()  # ensure switches connect

        # give controller time to converge topology so pingall works
        time.sleep(5)

        # setting the server's default route for controller access needs to be
        # done after the network starts up
        nat_ip = self.nat.IP()
        srv_iface = self.server.intfNames()[-1]
        self.server.setDefaultRoute('via %s dev %s' % (nat_ip, srv_iface))

        # We also have to manually configure the routes for the multicast addresses
        # the server will use.
        for a in self.mcast_address_pool:
            self.server.setHostRoute(a, self.server.intf().name)

        # this needs to come after starting network or no interfaces/IP addresses will be present
        log.debug("\n".join("added host %s at IP %s" % (host.name, host.IP()) for host in self.net.hosts))
        log.debug('links: %s' % [(l.intf1.name, l.intf2.name) for l in self.net.links])

        log.info('*** Pinging hosts so controller can gather IP addresses...')
        # don't want the NAT involved as hosts won't get a route to it
        loss = self.net.ping(hosts=[h for h in self.net.hosts if h != self.nat], timeout=2)
        if loss > 0:
            log.warning("ping had a loss of %f" % loss)

        # This needs to occur AFTER pingAll as the exchange of ARP messages
        # is used by the controller (ONOS) to learn hosts' IP addresses
        self.net.staticArp()

        log.info('*** Network set up!\n*** Configuring experiment...')

        self.setup_traffic_generators()
        # NOTE: it takes a second or two for the clients to actually start up!
        # log.debug('*** Starting clients at time %s' % time.time())
        self.setup_seismic_test(publishers, subscribers, server)
        # log.debug('*** Done starting clients at time %s' % time.time())

        # Apply actual failure model: we schedule these to fail when the earthquake hits
        # so there isn't time for the topology to update on the controller,
        # which would skew the results incorrectly. Since it may take a few cycles
        # to fail a lot of nodes/links, we schedule the failures for a second before.
        # ENCHANCE: instead of just 1 sec before, should try to figure out how long
        # it'll take for different machines/configurations and time it better...
        log.info('*** Configuration done!  Waiting for earthquake to start...')
        time.sleep(SEISMIC_EVENT_DELAY - 1)
        log.info('*** Earthquake at %s!  Applying failure model...' % time.time())
        for link in failed_links:
            self.net.configLinkStatus(link[0], link[1], 'down')
        for node in failed_nodes:
            node.stop(deleteIntfs=False)

        # log.debug('*** Failure model finished applying at %s' % time.time())

        log.info("*** Waiting for experiment to complete...")

        time.sleep(EXPERIMENT_DURATION - SEISMIC_EVENT_DELAY)

        # TODO: some sort of meaningful results?  maybe save filenames of the seismic hosts?
        return {}

    def setup_traffic_generators(self):
        """Each traffic generating host starts an iperf process aimed at
         (one of) the server(s) in order to generate random traffic and create
         congestion in the experiment.  Traffic is a mix of UDP and TCP."""

        generators = self._get_mininet_nodes(self._choose_random_hosts(self.n_traffic_generators))

        # TODO: include the cloud_server as a possible traffic generation/reception
        # point here?  could also use other hosts as destinations...
        srv = self.server

        log.info("*** Starting background traffic generators")
        # We enumerate the generators to fill the range of ports so that the server
        # can listen for each iperf client.
        for n, g in enumerate(generators):
            log.info("iperf from %s to %s" % (g, srv))
            g.popen('iperf -p %d -t %d -u -b %dM -c %s &' % (IPERF_BASE_PORT + n, EXPERIMENT_DURATION,
                                                             self.traffic_generator_bandwidth, srv.IP()))
            srv.popen('iperf -p %d -t %d -u -s &' % (IPERF_BASE_PORT + n, EXPERIMENT_DURATION))

            # can't do this as there's no option to put it in the background
            # self.net.iperf([g,s])

    def setup_seismic_test(self, sensors, subscribers, server):
        """
        Sets up the seismic sensing test scenario in which each sensor reports
        a sensor reading to the server, which will aggregate them together and
        multicast the result back out to each subscriber.  The server uses RIDE-D:
        a reliable multicast method in which several maximally-disjoint multicast
        trees (MDMTs) are installed in the SDN topology and intelligently
        choosen from at alert-time based on various heuristics.
        :param List[Host] sensors:
        :param List[Host] subscribers:
        :param Host server:
        """

        # Save Popen objects to later ensure procs terminate
        self.popens = []

        delay = SEISMIC_EVENT_DELAY  # seconds before sensors start picking
        quit_time = EXPERIMENT_DURATION

        # SETUP SERVER

        log.info("Seismic server on host %s" % server.name)

        cmd = "python %s seismic_warning_test/seismic_server.py -a %s --quit_time %d" % \
              ("-O" if OPTIMISED_PYTHON else "", ' '.join(self.mcast_address_pool), quit_time)
        # HACK: we pass the static lists of publishers/subscribers via cmd line so as to avoid having to build an
        # API server for RideD to pull this info from.  ENHANCE: integrate a pub-sub broker agent API on controller.
        cmd += " --subs %s --pubs %s" % (' '.join(self.get_host_dpid(h) for h in subscribers),
                                         ' '.join(self.get_host_dpid(h) for h in sensors))
        # Add RideD arguments to the server command.
        cmd += " --ntrees %d --mcast-construction-algorithm %s --choosing-heuristic %s --dpid %s --ip %s --port %d"\
               % (self.ntrees, ' '.join(self.tree_construction_algorithm), self.tree_choosing_heuristic,
                  self.get_host_dpid(self.server), self.controller_ip, self.controller_port)
        if WITH_LOGS:
            cmd += " > logs/srv 2>&1"

        log.debug(cmd)
        # HACK: Need to set PYTHONPATH since we don't install our Python modules directly and running Mininet
        # as root strips this variable from our environment.
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
        p = server.popen(cmd, shell=True, env=env)
        self.popens.append(p)

        #  SETUP CLIENTS

        sensors = set(sensors)
        subscribers = set(subscribers)

        log.info("Running seismic test client on %d subscribers and %d sensors" % (len(subscribers), len(sensors)))
        server_ip = server.IP()
        assert server_ip != '127.0.0.1', "ERROR: server.IP() returns localhost!"
        for client in sensors.union(subscribers):
            client_id = client.name
            cmd = "python %s seismic_warning_test/seismic_client.py --id %s --delay %d --quit_time %d" % \
                  ("-O" if OPTIMISED_PYTHON else "", client_id, delay, quit_time)
            if client in sensors:
                cmd += ' -a %s' % server_ip
            if client in subscribers:
                cmd += ' -l'
            if WITH_LOGS:
                cmd += " > logs/%s 2>&1" % client_id

            # the node.sendCmd option in mininet only allows a single
            # outstanding command at a time and cancels any current
            # ones when net.CLI is called.  Hence, we need popen.
            log.debug(cmd)
            p = client.popen(cmd, shell=True, env=env)
            self.popens.append(p)

    def teardown_experiment(self):
        log.info("*** Experiment complete!\n")

        # need to check if the programs have finished before we exit mininet!
        # First, we check the server to see if it even ran properly.
        ret = self.popens[0].wait()
        if ret != 0:
            from seismic_warning_test.seismic_server import SeismicServer
            if ret == SeismicServer.EXIT_CODE_NO_SUBSCRIBERS:
                log.error("Server proc exited due to no reachable subscribers: this experiment is a wash!")
                # TODO: handle this error appropriately: mark results as junk?
            else:
                log.error("Server proc exited with code %d" % self.popens[0].returncode)
        for p in self.popens[1:]:
            ret = p.wait()
            if ret is not None and ret != 0:
                if ret == errno.ENETUNREACH:
                    # TODO: handle this error appropriately: record failed clients in results?
                    log.error("Client proc failed due to unreachable network!")
                else:
                    log.error("Client proc exited with code %d" % p.returncode)

        # TODO: make this optional (maybe accessible via ctrl-c?)
        CLI(self.net)

        # TODO: figure out why this gives a 'OSError: File name too long'
        try:
            self.net.stop()
        except OSError as e:
            print traceback.format_exc()

    def get_host_dpid(self, host):
        """
        Returns the data plane ID for the given host that is recognized by the
        particular SDN controller currently in use.
        :param Host host:
        :return:
        """
        if self.topology_adapter == 'onos':
            # TODO: verify this vibes with ONOS properly; might need VLAN??
            dpid = host.defaultIntf().MAC().upper() + '/None'
        elif self.topology_adapter == 'floodlight':
            dpid = host.IP()
        else:
            raise ValueError("Unrecognized topology adapter type %s" % self.topology_adapter)
        return dpid

if __name__ == "__main__":
    import sys
    exp = MininetSmartCampusExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()

