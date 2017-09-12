import time
import socket
from threading import Timer
import logging as log
import json

class Detector(object):
    def __init__(self, id, ride_c_addr="11.0.0.2", ride_c_port=36463, server_addr="10.0.0.2", server_port=82649, quit_time=60, file_name=None):
        self.id = id
        self._link_status = 1
        self._ride_c_addr = ride_c_addr
        self._ride_c_port = ride_c_port
        self._server_addr = server_addr
        self._server_port = server_port


        if file_name is not None:
            self.file_name = file_name
            self._file = open(file_name, 'w', 1)

        self._ride_c_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._ride_c_socket.connect((self._ride_c_addr, self._ride_c_port))
        except Exception as e:
            log.error("Detector %s Cannot Connect to RIDEC, Terminate" % (self.id))
            return
        else:
            self.set_keepalive_linux(sock=self._ride_c_socket)

        self._detecting_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.echo_server = self._server_addr, self._server_port

        Timer(quit_time, self.finish).start()
        self._END = False

    def send_link_status_to_ride_c(self,status,detection_time):
        data = {}
        data["id"] = self.id
        data["status"] = status
        data["time"] = detection_time
        self._ride_c_socket.send(json.dumps(data))

    def update_link_status(self, status):
        detection_time = int(time.time()*1000)
        self._link_status = status
        self.send_link_status_to_ride_c(status,detection_time)

    def set_keepalive_linux(self, sock, after_idle_sec=1, interval_sec=3, max_fails=5):
        #Reference :https://stackoverflow.com/questions/12248132/how-to-change-tcp-keepalive-timer-using-python-script
        """Set TCP keepalive on an open socket.

        It activates after 1 second (after_idle_sec) of idleness,
        then sends a keepalive ping once every 3 seconds (interval_sec),
        and closes the connection after 5 failed ping (max_fails), or 15 seconds
        """
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, after_idle_sec)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, interval_sec)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, max_fails)

    def finish(self):
        self._file.close()
        self._END = True

    def run(self):
        raise NotImplementedError