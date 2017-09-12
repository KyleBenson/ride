import logging as log

import sys
import argparse
import time
import socket
import json
import os


# Buffer size for receiving packets
BUFF_SIZE = 4096

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--file', '-f', type=str, default="./datagram_server.log")
    parser.add_argument('--id', type=str, default=None)
    parser.add_argument('--port', type=int, default=4002)

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

        self._port = self.config.port
        self._id = self.config.id

        self._seq = 0

        self._local_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._local_socket.bind(('', self._port))


    def record_data(self,data):
        event = json.loads(data)
        line = str(event['id']) + ',' + str(event['seq']) + ',' + str(event['send_time']) + ',' +str(time.time())
        self._fd.write(line)
        self._fd.write("\n")


    def run(self):
        while True:
            data, fromaddr = self._local_socket.recvfrom(BUFF_SIZE)
            self.record_data(data)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    client = DatagramServer(args)
    log.debug("Server started at time %s" %  time.time())
    client.run()
