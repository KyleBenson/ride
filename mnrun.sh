#!/bin/bash

echo "Resetting ONOS, Mininet, and log files to start fresh..." 
tools/onos_reset.sh > /dev/null 2>&1
sudo mn -c > /dev/null 2>&1

# use first non-loopback interface as IP address of SDN Controller (we run it locally)
CONTROLLER_IP=`hostname -I | awk '{print $1}'`
echo "SDN controller at $CONTROLLER_IP"
sudo ./mininet_smart_campus_experiment.py --ip "$CONTROLLER_IP" $@
