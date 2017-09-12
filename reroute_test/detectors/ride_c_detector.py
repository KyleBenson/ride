from detector import Detector
import argparse
import sys
import time
import socket
import json
import math

BUFSIZE = 4096

def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', '-i', type=str,
                        help='''detector id''')
    parser.add_argument('--quit_time', '-q', type=float, default=10000,
                        help='''delay (in secs) before quitting''')
    parser.add_argument('--port', '-p', type=int, default=38693,
                        help='''UDP Echo Port''')
    parser.add_argument('--host', '-s', type=str, default="10.0.0.2",
                        help='''UDP Echo Server Addr''')

    parser.add_argument('--init_window', '-a', type=int, default=100,
                        help='''UDP Echo Server Addr''')
    parser.add_argument('--max_detection_time', '-m', type=int, default=3000,
                        help='''UDP Echo Server Addr''')
    parser.add_argument('--max_false_positive', '-x', type=float, default=0.01,
                        help='''UDP Echo Server Addr''')


    parser.add_argument('--ride_c_addr', '-r', type=str, default="11.0.0.2",
                        help='''UDP Echo Server Addr''')
    parser.add_argument('--ride_c_port', '-t', type=int, default=36463,
                        help='''UDP Echo Server Addr''')
    parser.add_argument('--result_file', '-f', type=str, default="./ride_c_detector.log",
                        help='''log file''')

    return parser.parse_args(args)


class RideCDetector(Detector):
    def __init__(self, config):
        super(RideCDetector,self).__init__(id=config.id,ride_c_addr=config.ride_c_addr,ride_c_port=config.ride_c_port,
                          server_addr=config.host,server_port=config.port,quit_time=config.quit_time,file_name=config.result_file)
        self.config = config
        # total number of sent packet in normal status(connected status)
        self._seq =0
        self._total_sent = 0
        self._total_received = 0
        self._link_loss = 0
        self._rtt_a = None
        self._detection_window_size = 0
        self._sending_interval = 0

        self._timeout = self.config.max_detection_time

    def send_probe(self,count=False):
        self._detecting_socket.settimeout(self._timeout/1000)
        data = {}
        data['seq'] = self._seq
        current_time_milis = int(time.time() * 1000)
        data['time_sent'] = current_time_milis

        self._detecting_socket.sendto(json.dumps(data), self.echo_server)
        self._file.write("Sent Probe (seq:%d)\n" % (self._seq))
        self._seq += 1

        if count:
            self._total_sent+=1

    def on_probe_receive(self, recv_data_str, count=False):
        receive_time_milis = int(time.time() * 1000)
        if count:
            self._total_received += 1
        recv_data = json.loads(recv_data_str)
        sent_time_milis = recv_data['time_sent']
        delay = receive_time_milis - sent_time_milis
        receive_seq = recv_data['seq']
        line = "Received Probe (seq:%d), delay = %d" % (receive_seq, delay)
        self._file.write(line+'\n')

        return delay

    def link_init_detection(self):
        self._file.write("Initial Detection\n")
        for i in range(self.config.init_window):
            self.send_probe(count=True)
            try:
                recv_data_str, fromaddr = self._detecting_socket.recvfrom(BUFSIZE)
            except socket.timeout:
                self._file.write("Timeout Probe (seq:%d)\n" % (self._seq))
            else:
                delay = self.on_probe_receive(recv_data_str, count=True)
                if self._rtt_a is None:
                    self._rtt_a = delay
                else:
                    self._rtt_a = 0.8 * self._rtt_a + 0.2 * delay
        self._file.write("Initial Detection Finished\n")

        self._link_loss = 1.0 - float(self._total_received)/self._total_sent
        self._timeout = 2 * self._rtt_a
        self._detection_window_size  = math.ceil(math.log(self.config.max_false_positive, self._link_loss))
        self._sending_interval = self.config.max_detection_time / self._detection_window_size

        self._file.write("links status: link_loss:%f, rtt_a:%d, Nb:%d, interval:%f" % (self._link_loss, self._rtt_a,self._detection_window_size,self._sending_interval))


    def link_recover_detection(self):
        self._file.write("Recover Detection")
        self._timeout = self.config.max_detection_time
        successive_count = 0
        while self._link_status == 0:
            self.send_probe(count=False)
            try:
                recv_data_str, fromaddr = self._detecting_socket.recvfrom(BUFSIZE)
            except socket.timeout:
                line = "Timeout Probe (seq:%d)" % (self._seq)
                successive_count = 0
            else:
                successive_count +=1
                delay = self.on_probe_receive(recv_data_str,count=False)

            if successive_count > 3:
                self.update_link_status(1)
                return


    def adapt_detector(self):
        self._link_loss = 1.0 - float(self._total_received) / self._total_sent
        self._detection_window_size = math.ceil(math.log(self.config.max_false_positive, self._link_loss))
        self._timeout = self._rtt_a * 2
        self._sending_interval = self.config.max_detection_time / self._detection_window_size

    def failure_detection(self, successive_fails):
        if successive_fails > self._detection_window_size or self._rtt_a > self._sending_interval:
            return 0 #fail

    def run(self):
        #Initial Phase:
        self.link_init_detection()
        successive_fails = 0
        while True:
            if not self._END:
                if self._link_status == 1:
                    self
                    self.send_probe(count=True)
                    try:
                        recv_data_str, fromaddr = self._detecting_socket.recvfrom(BUFSIZE)
                    except socket.timeout:
                        line = "Timeout Probe (seq:%d)" % (self._seq)
                        self._file.write(line+'\n')
                        successive_fails += 1
                    else:
                        successive_fails = 0
                        delay = self.on_probe_receive(recv_data_str,count=True)
                        self._rtt_a = 0.8 * self._rtt_a + 0.2 * delay
                    state = self.failure_detection(successive_fails)
                    if state == 0:
                        self.update_link_status(0)
                    else:
                        self.adapt_detector()
                        time.sleep(self._sending_interval/1000)
                else:
                    self.link_recover_detection()
                    successive_fails = 0
            else:
                return

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    detector = RideCDetector(args)
    detector.run()