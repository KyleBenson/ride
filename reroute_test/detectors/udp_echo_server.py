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

def parse_args(args):


    parser = argparse.ArgumentParser()

    # parameters used for simulation/testing

    parser.add_argument('--quit_time', '-q', type=float, default=10000,
                        help='''delay (in secs) before quitting''')
    parser.add_argument('--port', '-p', type=int, default=38693,
                        help='''UDP Echo Port''')


    return parser.parse_args(args)

class EchoServer(asyncore.dispatcher):

    def __init__(self, config):
        self.config = config
        asyncore.dispatcher.__init__(self)
        log.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log.DEBUG)
        self.create_socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.set_reuse_addr()
        self.bind(('', config.port))
        Timer(self.config.quit_time, self.finish).start()
        #self.listen(5)

    def handle_read(self):
        data, addr = self.recvfrom(2048)
        print(data)
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
    args = parse_args(sys.argv[1:])
    server = EchoServer(args)
    server.run()


# DEFAULT_PORT = 4444
# BUFSIZE = 1024
#
#
# def server():
#     if len(sys.argv) > 1:
#         port = eval(sys.argv[2])
#     else:
#         port = DEFAULT_PORT
#     s = socket(AF_INET, SOCK_DGRAM)
#     s.bind(('', port))
#     print 'udp echo server ready'
#     while 1:
#         data, addr = s.recvfrom(BUFSIZE)
#         print 'server received %r from %r' % (data, addr)
#         s.sendto(data, addr)
#
# if __name__ == "__main__":
#     server()