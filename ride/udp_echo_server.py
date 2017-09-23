#! /usr/bin/env python

"""This file includes an EchoServer class used for testing with a DataPathMonitor."""

# Reference: https://docs.python.org/2/library/asyncore.html
import sys
from socket import *
import time

import asyncore
import socket
import logging as log
from threading import Timer
import argparse
import random

def parse_args(args):


    parser = argparse.ArgumentParser()

    # parameters used for simulation/testing

    parser.add_argument('--quit_time', '-q', type=float, default=0,
                        help='''delay (in secs) before quitting (default=0, which means never quit)''')
    parser.add_argument('--port', '-p', type=int, default=9999,
                        help='''UDP Echo Port''')
    parser.add_argument('--loss_rate', '-l', type=float, default=0.0,
                        help='''Simulate loss rate by having the EchoServer simply not respond to some % of packets.''')
    parser.add_argument('--response_delay', '-d', type=float, default=0.0,
                        help='''Simulate congestion-induced delay by having the EchoServer simply wait some time before
                         responding to probes.  By default we don't delay; the units are in seconds.''')


    return parser.parse_args(args)

class EchoServer(asyncore.dispatcher):

    def __init__(self, config):
        # Old-style class 'super' call...
        asyncore.dispatcher.__init__(self)

        self.config = config
        self.loss_rate = config.loss_rate
        self.response_delay = config.response_delay

        # setup socket
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.set_reuse_addr()
        self.bind(('', config.port))

        if self.config.quit_time:
            Timer(self.config.quit_time, self.finish).start()

    def handle_read(self):
        """Receive the probe, optionally ignore it (loss_rate), and then optionally delay the echo response."""
        data, addr = self.recvfrom(2048)
        log.debug("EchoServer read data: %s" % data)
        if self.loss_rate == 0 or random.random() > self.loss_rate:
            if self.response_delay:
                log.debug("EchoServer delaying response by %fs" % self.response_delay)
                Timer(self.response_delay, self.sendto, args=[data, addr]).start()
                # time.sleep(self.response_delay)
            else:
                self.sendto(data, addr)

    def writable(self):
        return False

    def finish(self):
        self.close()

    def run(self):
        try:
            asyncore.loop()
        except Exception as e:
            log.error("Error in EchoServer run() can't recover...")
            self.finish()

if __name__ == "__main__":
    log.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log.DEBUG)
    args = parse_args(sys.argv[1:])
    server = EchoServer(args)
    server.run()