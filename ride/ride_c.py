#! /usr/bin/env python

# Reference: https://docs.python.org/2/library/asyncore.html
import sys
from socket import *
import time
import threading


import asyncore
import socket
import logging as log
from threading import Timer
import argparse
import json
import random

from smart_campus_experiment import SmartCampusExperiment, DISTANCE_METRIC

PORT = 38693

class RideC(threading.Thread, asyncore.dispatcher):

    def __init__(self, topo, topology_adapter, gateways, hosts, server, cloud, port=PORT ):
        asyncore.dispatcher.__init__(self)
        threading.Thread.__init__(self)
        log.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log.DEBUG)
        self.topo = topo
        self.topology_adapter = topology_adapter
        self.gateways = gateways
        self.hosts=hosts
        self.server = server
        self.server_ip = server.IP()
        self.cloud = cloud
        self.cloud_ip = cloud.IP()
        self.gw_to_hosts_map = {}
        self.gw_avalability = {}
        self.total_available_gateways = 0
        for gw in gateways:
            self.gw_avalability[gw.name] = 1
            self.total_available_gateways += 1
            self.gw_to_hosts_map[gw.name] = []
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)
        self.set_reuse_addr()
        self.bind(('', port))
        self.listen(5)

    def handle_read(self):
        data_str, addr = self.recvfrom(2048)
        recv_data = json.loads(data_str)
        id = recv_data["id"]
        status = recv_data["status"]
        self.on_link_status_change(id, status)


    def on_link_status_change(self, id, status):
        self.gw_avalability['g'+id] = status
        if self.gw_avalability['g'+id] == 0:
            self.total_available_gateways -= 1
            if self.total_available_gateways > 0:
                self.set_alterpath_for_hosts_on_gateway('g'+id)
            else:
                self.set_all_to_edge()

    def set_alterpath_for_hosts_on_gateway(self, gateway_name):
        impact_hosts = self.gw_to_hosts_map[gateway_name]
        for gw,availability in self.gw_avalability.items():
            if availability == 1:
                for h in impact_hosts:
                    try:
                        route = self.topo.get_path(h.name, gw, weight=DISTANCE_METRIC)
                        route = self._get_mininet_nodes(route)
                        flow_rules = self.build_flow_rules_from_path_to_gateway(route, self.cloud_ip)
                        for r in flow_rules:
                            self.topology_adapter.install_flow_rule(r)
                    except Exception as e:
                        log.error("Error installing flow rules for routes: %s" % e)
                        raise e

    def set_all_to_edge(self):
        for h in self.hosts:
            try:
                original_server_name = self.topo.get_servers()[0]
                route = self.topo.get_path(h.name, original_server_name, weight=DISTANCE_METRIC)
                route = self._get_mininet_nodes(route)
                route.insert(len(route), self.get_host_dpid(self.server))
                flow_rules = self.topology_adapter.build_flow_rules_edge_reroute(route,self.cloud_ip)
                for r in flow_rules:
                    self.topology_adapter.install_flow_rule(r)
            except Exception as e:
                log.error("Error installing flow rules for routes: %s" % e)
                raise e

    def writable(self):
        return False

    def finish(self):
        self.close()

    def setup_init_flows(self):
        for h in self.hosts:
            gw = random.sample(self.gateways, 1)
            self.gw_to_hosts_map[gw.name].append(h)
            try:
                route = self.topo.get_path(h.name, gw.name, weight=DISTANCE_METRIC)
                route = self._get_mininet_nodes(route)
                flow_rules = self.build_flow_rules_from_path_to_gateway(route, self.cloud_ip)
                for r in flow_rules:
                    self.topology_adapter.install_flow_rule(r)
            except Exception as e:
                log.error("Error installing flow rules for routes: %s" % e)
                raise e

    def run(self):
        self.setup_init_flows()
        asyncore.loop()


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