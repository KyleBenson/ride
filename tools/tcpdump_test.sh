#!/bin/bash
# tcpdump -D returns a list of interface indices; the mininet hosts seem to all have their interface as index 3
sudo tcpdump -vv -l -K -i 3 udp port 9999
