#!/usr/bin/python

import argparse

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import CPULimitedHost
from mininet.link import TCLink
from mininet.util import dumpNodeConnections
from mininet.cli import CLI
from mininet.log import setLogLevel, info

from sys import argv

# Experiment Setting
DETECTION_ALGO = "ridec_detector"
EXPERIMENT_TIME = 60
LINK_DOWN_TIME = 20
LINK_BACK_TIME = 40

# Network Setting
LINK_BANDWIDTH = 10 # Mbps
LINK_DELAY = '100ms'
LINK_LOSS_RATE = 10 #%

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
    topo = SingleSwitchTopo( loss=10, delay='5ms', bandwidth=10)
    net = Mininet( topo=topo,
                   link=TCLink,
                   autoStaticArp=True)
    net.start()
    info( "Dumping host connections\n" )
    dumpNodeConnections(net.hosts)
    info( "Testing bandwidth between h1 and h2\n" )

    h1, h2 = net.getNodeByName('h1', 'h2')
    # Start echo server
    h2.popen("python udp_echo_server")


    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel( 'info' )
    perfTest( loss=LINK_LOSS_RATE, delay=LINK_DELAY, bandwidth=LINK_BANDWIDTH )