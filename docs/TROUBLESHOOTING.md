# Troubleshooting

Getting all of the pieces working can be a bit tricky and may require some hacking for an untested system.
We used Ubuntu 14-16 so expect potential issues if you don't follow all of this setup exactly.

## Running manual tests / experimental setup in Mininet

First, get your OF controller running and get OVS setup.  Run Mininet with:

`sudo mn --controller=remote --topo=linear,4,2`

Open an xterm to several of the hosts and configure them for multicast by adding an IP route e.g.:

`ip route add 224.0.0.0/4 dev eth0`

Now you should be able to run the IoT client (i.e. scale_client) and have it publish events to this multicast address.

If you want to inspect the OpenFlow how a packet will be processed by the flow table rules on a particular switch in Mininet, you can do:

`s1 ovs-appctl ofproto/trace 's1' [flow-expression]`
Where s1 is the switch being inspected (only the second occurrence matters) and [flow-expression] could look like: `'in_port=1,ipv4,nw_dst=224.0.0.1,nw_src=10.0.0.1'`

NOTE about ARP: in order to deliver messages to a destination (especially via multicast) the MAC address will have to be proper.  This may require ARP resolution (flooding a request and then receiving a response to tell the MAC address for the destination IP), or it may require setting the destination MAC to be a multi/broadcast address e.g. ff:ff:ff:ff:ff:ff

## Getting ONOS running properly

If you're using ONOS as your controller, which is our suggested default for the SdnTopology, make sure that your ONOS instance is up and running properly.  If it's not using the default port (8181), you'll need to set that manually in your configurations.

We recommend installing ONOS from the tarball rather than from scratch (building from source), which the troubleshooting steps below relate to.

NOTE: just use version 1.11.1 as it seems far more stable than the ones that caused the issues below!

Sometimes ONOS just stops working properly... it might start, but the web server is not accessible and just keeps responding with NOT_FOUND.
I'm unsure why, but reinstalling ONOS seems to help.  Do `onos-kill; onos-uninstall; mvn clean; onos-build; onos-install` to reinstall it one step at a time.
This also assume that your *cell* is set properly: make sure that `onos-cell` refers to the right user/password on your systems for it to run as and that it doesn't refer to *localhost* in the `$OC??` variables (use 127.0.0.1 if you're running it locally).

Most of the development/testing was done on ONOS version 1.9.0.

