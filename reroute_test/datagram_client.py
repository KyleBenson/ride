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

    parser.add_argument('--file', '-f', type=str, default="./datagram_client.log")
    parser.add_argument('--id', type=str, default='test')
    parser.add_argument('--port', type=int, default=4002)
    parser.add_argument('--address', '-a', type=str, default="localhost")
    parser.add_argument('--interval', '-i', type=float, default=0.1)

    return parser.parse_args(args)


class DatagramClient(object):

    def __init__(self, args):
        self.config = args
        if self.config.id is None:
            self.config.id = str(os.getpid())

        try:
            self._fd = open(self.config.file,'w')
        except Exception as e:
            log.error("Can't open client log file")

        self._cloud_ip = self.config.address
        self._port = self.config.port
        self._id = self.config.id
        self._interval = self.config.interval

        self._seq = 0

        self._addr = self._cloud_ip, self._port
        self._local_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send_event(self):
        curr_time = time.time()
        log.info("Sending event at time %s" % curr_time)

        event = dict()
        event['seq'] = self._seq
        event['id'] = self.config.id

        event['send_time'] = str(time.time())

        try:
            self._local_socket.sendto(json.dumps(event), self._addr)
        except socket.error as e:
            log.error(e)
        else:
            self._fd.write(str(self._seq))
            self._fd.write('\n')
            self._seq+=1

    def run(self):
        while True:
            self.send_event()
            time.sleep(self._interval)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    client = DatagramClient(args)
    log.debug("Client started at time %s" %  time.time())
    client.run()
