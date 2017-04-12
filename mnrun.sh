#!/bin/bash

# start everything fresh
tools/onos_reset.sh
sudo mn -c > /dev/null
mkdir logs 2> /dev/null
rm logs/* 2> /dev/null

# use first non-loopback interface as IP address of SDN Controller (we run it locally)
CONTROLLER_IP=`hostname -I | awk '{print $1}'`
echo "SDN controller at $CONTROLLER_IP"
sudo ./mininet_smart_campus_experiment.py --ip "$CONTROLLER_IP" $@

sudo mn -c > /dev/null
