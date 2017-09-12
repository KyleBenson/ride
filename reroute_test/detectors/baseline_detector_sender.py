#! /usr/bin/env python


import sys
import socket
import time
import threading
import json
import os
import errno
from onos_sdn_topology import OnosSdnTopology as SdnTopology

DEFAULT_PORT = 4444
BUFSIZE = 1024

STATE = 0 # 0-T, 1-T->S, 2-S
INTERVAL = 1
FAST_PROBE_TIMES = 4
TIMEOUT = 0.6 #in seconds
SEND_LOCK = False
SENT_SIGNAL = False


class ProbeSender(threading.Thread):
    def __init__(self, probe_socket, target_addr):
        threading.Thread.__init__(self)
        self.local_socket = probe_socket
        self.target_addr = target_addr
        #self.interval_one_loss = 0.2
        self.thread_stop = False
        self.fast_probe_times = 4
        self.threshold = 100
        self.loss_penalty = 5

        self.current_out_seq = 0

    def send_next_probe(self):
        data = {}
        self.current_out_seq = self.current_out_seq + 1
        data['seq'] = self.current_out_seq
        current_time_milis = int(time.time() * 1000)
        data['time_sent'] = current_time_milis
        self.local_socket.sendto(json.dumps(data), self.target_addr)

    def run(self):
        global SEND_LOCK
        global STATE
        global SENT_SIGNAL
        while not self.thread_stop: #and not SEND_LOCK:
            print "sender: Current STate: "  + str(STATE)
            if STATE == 0:#T
                self.send_next_probe()
                SEND_LOCK = True
                SENT_SIGNAL = True
                time.sleep(INTERVAL)
                continue

            if STATE == 1:#T-S
                print "Begin to send fast probes"

                i = FAST_PROBE_TIMES
                while i > 0:
                    self.send_next_probe()
                    print "fast probe sent #" + str(i)
                    time.sleep(TIMEOUT) # in base line method, the interval of fast probe is the timeout time
                    if STATE != 1:
                        break
                    i = i - 1
                SEND_LOCK = True
                SENT_SIGNAL = True
                time.sleep(INTERVAL)
                continue

            if STATE == 2:#T
                self.send_next_probe()
                SENT_SIGNAL = True
                time.sleep(INTERVAL)
                continue



    def stop(self):
        self.thread_stop = True


class AckReceiver(threading.Thread):
    def __init__(self, ack_socket):
        threading.Thread.__init__(self)
        self.local_socket = ack_socket
        self.thread_stop = False
        self.pipe_location = "./pipe"

        try:
            print os.getcwd()
            self.pipe_fd = os.open(self.pipe_location, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            print e
            print "ERROR: Can\'t open pipe file"
            if e.errno == errno.ENXIO:
                return
        else:
            self._second_pipe_writable = True


    def print_message(self, data):
        data_loaded = json.loads(data)
        seq = data_loaded['seq']
        sent_time = data_loaded['time_sent']
        current_time_milis = int(time.time() * 1000)
        print "SEQ: " + str(seq) + "| RTT =" + str(current_time_milis - sent_time) + "ms"

    def run(self):
        global STATE
        global SEND_LOCK
        global TIMEOUT
        while not self.thread_stop:
            SEND_LOCK = False
            print "Recevier: current state: " + str(STATE)
            if STATE == 0:
                self.local_socket.settimeout(TIMEOUT)
                try:
                    print "try get recv, current timeout: " + str(self.local_socket.gettimeout())
                    data, fromaddr = self.local_socket.recvfrom(BUFSIZE)
                except socket.error:
                    STATE = 1
                    self.local_socket.settimeout(TIMEOUT * FAST_PROBE_TIMES)
                    print "Normal probe time out! Prepare to send fast probes"
                else:
                    self.print_message(data)
                finally:
                    SEND_LOCK = False
                continue

            if STATE == 1:
                self.local_socket.settimeout(TIMEOUT * FAST_PROBE_TIMES)
                try:
                    print "try get recv, current timeout: " + str(self.local_socket.gettimeout())
                    data, fromaddr = self.local_socket.recvfrom(BUFSIZE) #flush?
                except socket.error:
                    STATE = 2
                    if self._second_pipe_writable is True:
                        try:
                            # with os.fdopen(os.open(self._second_pipe_location, os.O_WRONLY|os.O_NONBLOCK)) as second_pipe:
                            # self._second_pipe.write(data_line)
                            os.write(self.pipe_fd, "2\n")
                        except OSError as e:
                            print
                            "ERROR: Can\'t Write line to second pipe file"
                            print e
                            if e.errno == errno.EPIPE:
                                self._second_pipe_writable = False
                                os.close(self.pipe_fd)
                    print "all fast probes time out! GO INTO SUSPECT STATE"
                else:
                    STATE = 0
                    self.local_socket.settimeout(TIMEOUT)
                    print "Got reply for fast probe, go back to normal"
                finally:
                    SEND_LOCK = False
                continue

            if STATE == 2:
                self.local_socket.settimeout(None)
                SEND_LOCK = False
                print "try get recv, current timeout: " + str(self.local_socket.gettimeout())
                data, fromaddr = self.local_socket.recvfrom(BUFSIZE) #flush?
                STATE = 0
                if self._second_pipe_writable is True:
                    try:
                        # with os.fdopen(os.open(self._second_pipe_location, os.O_WRONLY|os.O_NONBLOCK)) as second_pipe:
                        # self._second_pipe.write(data_line)
                        os.write(self.pipe_fd, "0\n")
                    except OSError as e:
                        print
                        "ERROR: Can\'t Write line to second pipe file"
                        print
                        e
                        if e.errno == errno.EPIPE:
                            self._second_pipe_writable = False
                            os.close(self.pipe_fd)
                self.local_socket.settimeout(TIMEOUT)
                SEND_LOCK = False
                continue

    def stop(self):
        self.thread_stop = True


def usage():
    sys.stdout = sys.stderr
    print 'or:    udp_echo_client host [port] (client)'
    sys.exit(2)

def main():
    if len(sys.argv) < 2:
        usage()
    host = sys.argv[1]
    if len(sys.argv) > 2:
        port = eval(sys.argv[3])
    else:
        port = DEFAULT_PORT

    #st = SdnTopology()

    addr = host, port
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(TIMEOUT)
    s.bind(('', 0))
    print 'udp echo client ready'

    probe_sender_th = ProbeSender(s, addr)
    ack_receiver_th = AckReceiver(s)
    probe_sender_th.daemon = True
    ack_receiver_th.daemon = True

    ack_receiver_th.start()
    probe_sender_th.start()

    #time.sleep(10)

    #probe_sender_th.interval_normal = 0.2

    while(1):
        time.sleep(1)


if __name__ == "__main__":
    main()



