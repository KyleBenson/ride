__author__ = 'kebenson'

NEW_SCRIPT_DESCRIPTION = '''Simple client that sends a 'pick' indicating a possible seismic event to a server using
a simple JSON format over UDP.'''

# @author: Kyle Benson
# (c) Kyle Benson 2016

import logging as log
log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

import sys
import argparse
import time
import asyncore
import socket
import json
import os
from threading import Timer


# Buffer size for receiving packets
BUFF_SIZE = 4096


def parse_args(args):
    ##################################################################################
    # ################      ARGUMENTS       ###########################################
    # ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    # action is one of: store[_const,_true,_false], append[_const], count
    # nargs is one of: N, ?(defaults to const when no args), *, +, argparse.REMAINDER
    # help supports %(var)s: help='default value is %(default)s'
    ##################################################################################

    parser = argparse.ArgumentParser(description=NEW_SCRIPT_DESCRIPTION,
                                     #formatter_class=argparse.RawTextHelpFormatter,
                                     #epilog='Text to display at the end of the help print',
                                     )


    parser.add_argument('--listen', '-l', action="store_true",
                        help='''client will act as a subscriber and listen for incoming messages
                        as well as output statistics at the end.''')
    parser.add_argument('--file', '-f', type=str, default="output_events_rcvd",
                        help='''file to write statistics on when picks were sent/recvd to
                        (Note that it appends a '_' and the id to this string)''')
    parser.add_argument('--id', type=str, default=None,
                        help='''unique identifier of this client (used for
                         naming output files and including in event message;
                         default=process ID)''')

    parser.add_argument('--delay', '-d', type=float, default=1,
                        help='''delay (in secs) before sending the event
                        (when the simulated earthquake occurs)''')
    parser.add_argument('--retransmit', '-r', type=float, default=2,
                        help='''delay (in secs) before resending the event for reliability''')
    parser.add_argument('--quit_time', '-q', type=float, default=10,
                        help='''delay (in secs) before quitting and recording statistics''')

    parser.add_argument('--recv_port', type=int, default=9998,
                        help='''UDP port number from which data should be received''')
    parser.add_argument('--send_port', type=int, default=9999,
                        help='''UDP port number to which data should be sent''')
    parser.add_argument('--address', '-a', type=str, default=None,
                        help='''If specified, enables sending sensor data to the given IP address''')

    return parser.parse_args(args)


class SeismicClient(asyncore.dispatcher):

    def __init__(self, config):
        # super(SeismicClient, self).__init__()
        asyncore.dispatcher.__init__(self)

        # store configuration options and validate them
        self.config = config
        if self.config.id is None:
            self.config.id = str(os.getpid())

        # self.my_ip = socket.gethostbyname(socket.gethostname())

        # Stores received UNIQUE events indexed by their 'id'
        # Includes the time they were received at
        self.events_rcvd = dict()

        # TODO: need to record time we started the quake somehow?

        # queue seismic event reporting
        if self.config.address is not None:
            self.next_timer = Timer(self.config.delay, self.send_event)
            self.next_timer.start()

        # setup UDP network socket to listen for events on and send them via
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.bind(('', self.config.recv_port))

        # record statistics after experiment finishes then clean up
        Timer(self.config.quit_time, self.finish).start()

    def send_event(self):
        event = dict()
        event['time_sent'] = time.time()
        event['id'] = self.config.id

        self.sendto(json.dumps(event), (self.config.address, self.config.send_port))

        # don't forget to schedule the next time we send aggregated events
        self.next_timer = Timer(self.config.retransmit, self.send_event)
        self.next_timer.start()

    def process_event(self, event):
        """Helper function for handle_read()"""
        if event['id'] not in self.events_rcvd:
            event['time_rcvd'] = time.time()
            event['copies_rcvd'] = 1
            self.events_rcvd[event['id']] = event
        else:
            self.events_rcvd[event['id']]['copies_rcvd'] += 1

    def handle_read(self):
        """
        Receive an event and record the time we received it.
        If it's already been received, we simply record the fact that
        we've received a duplicate.
        """

        if not self.config.listen:
            return

        data = self.recv(BUFF_SIZE)
        # ENHANCE: handle packets too large to fit in this buffer
        try:
            event = json.loads(data)
            log.info("received event %s" % event)

            # Aggregated events will have all the events in an array
            if event['id'].startswith("aggregator"):
                for e in event['events']:
                    self.process_event(e)

            else:
                self.process_event(event)

        except ValueError:
            log.error("Error parsing JSON from %s" % data)

    def run(self):
        try:
            asyncore.loop()
        except:
            # seems as though this just crashes sometimes when told to quit
            log.error("Error in SeismicClient.run() can't recover...")
            return

    def finish(self):
        try:
            self.next_timer.cancel()
        except AttributeError:
            # This just means we don't send data
            pass

        if self.config.listen:
            self.record_stats()
        self.close()

    def record_stats(self):
        """Records the received picks for consumption by another script
        that will analyze the resulting performance."""

        fname = "_".join([self.config.file, self.config.id])
        with open(fname, "w") as f:
            f.write(json.dumps(self.events_rcvd))


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    client = SeismicClient(args)
    client.run()
