#! /usr/bin/env python

# Reference: https://docs.python.org/2/library/asyncore.html
import sys
from socket import *
import time


import asyncore
import socket
import logging as log
from threading import Timer
import argparse
import json

def parse_args(args):


    parser = argparse.ArgumentParser()

    # parameters used for simulation/testing

    parser.add_argument('--quit_time', '-q', type=float, default=30,
                        help='''delay (in secs) before quitting''')
    parser.add_argument('--port', '-p', type=int, default=38693,
                        help='''UDP Echo Port''')
    parser.add_argument('--host', '-s', type=str, default="127.0.0.1",
                        help='''Server Addr''')
    parser.add_argument('--result_file', '-f', type=str, default="./test_detector.log",
                        help='''log file''')

    return parser.parse_args(args)

PROBE_TIMEOUT = 3
BUFSIZE = 4096

class TestDetector(object):

    def __init__(self, config):
        self.config = config
        log.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log.DEBUG)
        self.echo_server = self.config.host, self.config.port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(PROBE_TIMEOUT)

        self._file = open(self.config.result_file, 'w',1)
        self._seq = 0

        Timer(self.config.quit_time, self.finish).start()
        self._END = False
        #self.listen(5)

    def finish(self):
        self._file.close()
        self._END = True

    def run(self):
        while True:
            if not self._END:
                data = {}
                data['seq'] = self._seq
                current_time_milis = int(time.time() * 1000)
                data['time_sent'] = current_time_milis

                self._socket.sendto(json.dumps(data), self.echo_server)
                self._file.write("Sent Probe (seq:%d)\n" % (self._seq))
                try:
                    recv_data_str, fromaddr = self._socket.recvfrom(BUFSIZE)
                except socket.timeout:
                    line = "Timeout Probe (seq:%d)" % (self._seq)
                else:
                    receive_time_milis = int(time.time() * 1000)
                    recv_data = json.loads(recv_data_str)
                    sent_time_milis = recv_data['time_sent']
                    delay = receive_time_milis - sent_time_milis
                    receive_seq = recv_data['seq']

                    line = "Received Probe (seq:%d), delay = %d" % (receive_seq, delay)
                self._file.write(line)
                self._file.write('\n')
                self._seq += 1
            else:
                return

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    server = TestDetector(args)
    server.run()


