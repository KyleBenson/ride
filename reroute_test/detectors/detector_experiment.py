#!/usr/bin/python

import argparse
import time

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.cli import CLI
from mininet.log import setLogLevel, info

import os

from sys import argv

NAT_SERVER_IP_ADDRESS = '11.0.0.%d/24'

# Experiment Setting
DETECTION_ALGO = "ride_c_detector"
EXPERIMENT_TIME = 100
LINK_DOWN_TIME = 60
LINK_BACK_TIME = 80
ECHO_SERVER_PORT = 38693

# Network Setting
LINK_BANDWIDTH = 10 # Mbps
LINK_DELAY = '50ms'
LINK_LOSS_RATE = 5 #%

class SingleSwitchTopo( Topo ):
    "Single switch connected to 2 hosts."
    def build( self, loss=10, delay='5ms', bandwidth=10 ):
        switch = self.addSwitch('s1')
        h1 = self.addHost('h1')
        self.addLink(h1, switch, use_htb=True)
        h2 = self.addHost('h2')
        self.addLink(h2, switch,
                     bw=bandwidth, delay=delay, loss=loss, use_htb=True)



def perfTest( loss=10, delay='5ms', bandwidth=10 ):
    "Create network and run simple performance test for our detectors"
    topo = SingleSwitchTopo( loss=loss, delay=delay, bandwidth=bandwidth)
    net = Mininet( topo=topo,
                   link=TCLink,
                   autoStaticArp=True)
    h1, h2 = net.getNodeByName('h1', 'h2')
    nat_ip = NAT_SERVER_IP_ADDRESS % 2
    detector_ip = NAT_SERVER_IP_ADDRESS % 3
    server_ip =NAT_SERVER_IP_ADDRESS % 4
    nat_switch = net.addSwitch('s3')
    nat = net.addNAT(connect=nat_switch)
    nat.configDefault(ip=nat_ip)

    net.addLink(h1, nat_switch, use_htb=True)
    net.addLink(h2, nat_switch, use_htb=True)

    h1_iface = sorted(h1.intfNames())[-1]
    h1.intf(h1_iface).setIP(detector_ip)
    h2_iface = sorted(h2.intfNames())[-1]
    h2.intf(h2_iface).setIP(server_ip)

    nat_switch.start([])

    net.start()
    info( "Dumping host connections\n" )
    dumpNodeConnections(net.hosts)
    info( "Experiment Start\n" )

    h1, h2 = net.getNodeByName('h1', 'h2')

    # Set up file env
    env = os.environ.copy()
    if 'PYTHONPATH' not in env:
        env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

    # Start echo server
    echo_server_ip = h2.IP()
    # h2.popen("python udp_echo_server.py --port %d --quit_time %d" % (ECHO_SERVER_PORT, EXPERIMENT_TIME),
    #          shell=True, env=env)
    #
    # # Start detector
    # h1.popen("python %s.py --host %s --port %d --quit_time %d" % (DETECTION_ALGO, echo_server_ip, ECHO_SERVER_PORT, EXPERIMENT_TIME),
    #          shell=True, env=env)

    # time.sleep(LINK_DOWN_TIME)
    # print("Link Down Now")
    # net.configLinkStatus('s1','h2','down')
    # print("Link Down Happened @ %d" % ( int(time.time() * 1000)) )
    #
    # time.sleep(LINK_BACK_TIME - LINK_DOWN_TIME)
    # print("Link UP Now")
    # net.configLinkStatus('s1', 'h2', 'up')
    # print("Link Up Happened @ %d" % (int(time.time() * 1000)))

    #time.sleep(EXPERIMENT_TIME - LINK_BACK_TIME)
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    perfTest( loss=LINK_LOSS_RATE, delay=LINK_DELAY, bandwidth=LINK_BANDWIDTH )