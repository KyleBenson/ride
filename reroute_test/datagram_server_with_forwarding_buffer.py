import logging as log

import sys
import argparse
import time
import socket
import json
import os
import select

from collections import deque

# Buffer size for receiving packets
BUFF_SIZE = 4096
DEFAULT_OUTPUT_FILE_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "events")

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', '-f', type=str, default=DEFAULT_OUTPUT_FILE_BASE)
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--port', type=int, default=4002)
    parser.add_argument('--bufsize', type=int, default=100)
    parser.add_argument('--address', '-a', type=str, default=None)
    parser.add_argument('--control_port', type=int, default=4002)

    return parser.parse_args(args)


class DatagramServer(object):

    def __init__(self, args):
        self.config = args
        if self.config.id is None:
            self.config.id = str(os.getpid())

        try:
            self._fd = open(self.config.file,'w')
        except Exception as e:
            log.error("Can't open server log file")

        self._cloud_ip = self.config.address
        self._port = self.config.port
        self._id = self.config.id

        self._seq = 0

        self._addr = self._cloud_ip, self._port
        self._local_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._local_socket.bind(('', self._port))
        self._local_socket.setblocking(0)

        self._control_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._control_socket.bind(('', self.config.control_port))
        self._control_socket.setblocking(0)

        self._buf = deque(maxlen=self.config.bufsize)


    def record_data(self,data):
        event = json.loads(data)
        line = event['id'] + ',' + event['seq'] + ',' + event['send_time'] + ',' +str(time.time())
        self._fd.write(line)
        self._fd.write("\n")

    def dump_buffer_to_cloud(self):
        buf_list = list(self._buf)
        for data in buf_list:
            try:
                self._local_socket.sendto(data, self._addr)
            except socket.error as e:
                log.error(e)


    def run(self):
        while True:
            readable, writable, exceptional = select.select([self._control_socket,self._local_socket], [], [], 1)
            for s in readable:
                data, fromaddr = s.recvfrom(BUFF_SIZE)
                if s.getsocketname()[1] == self.config.control_port:
                    self.dump_buffer_to_cloud()
                elif s.getsocketname()[1] == self._port:
                    self.record_data(data)
                    self._buf.append(data)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    client = DatagramServer(args)
    log.debug("Server started at time %s" %  time.time())
    client.run()
