# sdn_pubsub_mcast
Installs SDN (OpenFlow) rules for accomplishing pub-sub via multicast.

## Getting Started

**Networkx:** Note that you may need a 'bleeding edge' version of networkx for the default multicast tree generation (Steiner tree) algorithm.
  As of 1/2017 it is not included in the officially released version, so we just checked out
   the third-party fork that the pull request is based on in an external directory and
   created a symbolic link in this directory to make the import work.
   
### Mininet-based Experiments

To run the Mininet-based experiments, we'll obviously need to install Mininet and its Python API (e.g. via pip).  We just cloned the source repo and symlinked it to inside this repo's main directory.
We also seem to have made a change to a file in mininet for multiple experiment runs with switches we create:
```
diff --git mininet/net.py mininet/net.py
index a7c159e..b518958 100755
--- mininet/net.py
+++ mininet/net.py
@@ -520,9 +520,6 @@ def stop( self ):
         for swclass, switches in groupby(
                 sorted( self.switches, key=type ), type ):
             switches = tuple( switches )
-            if hasattr( swclass, 'batchShutdown' ):
-                success = swclass.batchShutdown( switches )
-                stopped.update( { s: s for s in success } )
         for switch in self.switches:
             info( switch.name + ' ' )
             if switch not in stopped:
```

You'll also need to run an *SDN controller* locally (or remotely, though we never tried that).
Currently we only fully support ONOS.  We install it using the directions on their website and configure it to run as a service with the following options file (in `/opt/onos/options`):
```
ONOS_USER=sdn
ONOS_APPS=openflow-base,drivers,openflow,fwd,proxyarp
JAVA_HOME=/usr/lib/jvm/java-8-oracle
```
If you're running on a physical machine rather than a VM, you should probably change the user/password for the karaf GUI in `/opt/onos/apache-karaf-$KARAF_VERSION/etc/users.properties` to something more secure.
Note that because ONOS uses a clustering service, you may run into problems with dynamic IP addresses (i.e. the master node is no longer available because your machine IS the master node but its IP address changed).


Make sure to get **Open vSwitch** set up properly too!  You can install it with Mininet's install script.

You may need to tweak some settings inside mininet_smart_campus_experiment.py e.g. the IP addresses of the Mininet hosts might conflict with your OS's networking configuration esp. if you're running inside a virtual machine.

This repository assumes that the scale_client and seismic_warning_test repos are on your PYTHONPATH.  Really, they'll need to be in *root*'s PYTHONPATH!  I typically use symbolic links to just put them in the same directory; make sure you use an absolute (NOT relative) path for this or it won't work right!
   
**Running the SCALE Client from within Mininet**: to properly run this, we opt to specify a different user that will have a virtual environment they can run the scale client in.  You can change the `SCALE_USER` at the top of the scale_config.py file. Make sure this user has the `virtualenvwrapper.sh` script available (you can edit the `SCALE_CLIENT_BASE_COMMAND` as well if you use virtualenv differently or want to forego it entirely), create a virtual environment with the name `ride_scale_client`, and install the dependencies (both for scale_client and for ride) in that environment with `pip install -r requirements.txt`


## Troubleshooting

### Running manual tests / experimental setup in Mininet

First, get your OF controller running and get OVS setup.  Run Mininet with:

`sudo mn --controller=remote --topo=linear,4,2`

Open an xterm to several of the hosts and configure them for multicast by adding an IP route e.g.:

`ip route add 224.0.0.0/4 dev eth0`

Now you should be able to run the IoT client (i.e. scale_client) and have it publish events to this multicast address.

If you want to inspect the OpenFlow how a packet will be processed by the flow table rules on a particular switch in Mininet, you can do:

`s1 ovs-appctl ofproto/trace 's1' [flow-expression]`
Where s1 is the switch being inspected (only the second occurrence matters) and [flow-expression] could look like: `'in_port=1,ipv4,nw_dst=224.0.0.1,nw_src=10.0.0.1'`

NOTE about ARP: in order to deliver messages to a destination (especially via multicast) the MAC address will have to be proper.  This may require ARP resolution (flooding a request and then receiving a response to tell the MAC address for the destination IP), or it may require setting the destination MAC to be a multi/broadcast address e.g. ff:ff:ff:ff:ff:ff

### Getting ONOS running properly

If you're using ONOS as your controller, which is our suggested default for the SdnTopology, make sure that your ONOS instance is up and running properly.  If it's not using the default port (8181), you'll need to set that manually in your configurations.

Sometimes ONOS just stops working properly... it might start, but the web server is not accessible and just keeps responding with NOT_FOUND.
I'm unsure why, but reinstalling ONOS seems to help.  Do `onos-kill; onos-uninstall; mvn clean; onos-build; onos-install` to reinstall it one step at a time.
This also assume that your *cell* is set properly: make sure that `onos-cell` refers to the right user/password on your systems for it to run as and that it doesn't refer to *localhost* in the `$OC??` variables (use 127.0.0.1 if you're running it locally).

Most of the development/testing was done on ONOS version 1.9.0.


## Architecture and Workflow


### SCALE Client integration

To demonstrate the RIDE approach, we integrated it with the SCALE Client as a RIDE-D module.  The general idea is that the IoT device clients (publishers) forward seismic-related `SensedEvent`s to the server for aggregation and multicast alerting to subscribers.

#### Overall flow

1) Subscribers contact server to subscribe to alerts (as described below, they create the CoAP resource end-point at this time).
2) RIDE-D module on the server collects network topology info including the RIDE-C paths taken from publishers to the server.
3) It establishes MDMTs for the alert subscribers and installs these flow rules via SDN REST API.
4) At earthquake time, the publishers send their picks to the seismic server.
5) The server aggregates these picks together for a couple seconds (representing the earthquake-detection algorithm's running time) and updates its current STT view based on these recently-received picks.
6) It then sends the aggregated picks (mostly just the publisher IDs and timestamps) out in an alert message.
 This CoAP PUT message traverses the MDMT(s) chosen by the RIDE-D module based on the STT.
The intermediate SDN switches route it and copy it appropriately, eventually translating the multicast destination IP address to the end-host before delivery to it.


#### Challenges to overcome

* Will 1 CoAP packet fit the entire alert message with hundreds of publishers?  May not need to worry about this since the Mininet-based experiments will be of limited size anyway... Future work would explore something like a Bloom filter.
* Handling subscriptions at server: how often should we re-analyze and re-construct the MDMTs?  Certainly (almost) immediately if a new subscriber arrives!
* Server should always send an alert even if no publications received as the subscribers may still be reachable (falsely results in 0.0 reachability).
 

#### Publisher configuration

The publisher IoT clients simply send a seismic `SensedEvent` to the server (via its well-known IP address) at a predetermined time (i.e. the time of the simulated earthquake).
This report time includes a small amount of random delay to account for the fact that the *picking algorithm* will actually detect the seismic wave at slightly different times in different devices due to 1) the difference in wave arrival time and 2) subtle differences between devices e.g. OS thread scheduling, amount of normal background shaking, etc.

#### Subscriber configuration

The subscriber clients will receive these multicast alerts via CoAP PUT.
By using our SDN-based IP multicast-based resilient pub-sub, these clients are essentially unmodified.
They only require UDP-based CoAP functionality; specifically must accept non-CONfirmable messages as alerts.
They must also **subscribe** to the alerts by contacting the server in advance.

For testing purposes, the subscribers are responsible for **gathering statistics** about the alerts.
They look at the aggregate event's metadata to see which publishers' picks are included in the aggregate one and record when they received these for later analysis.

NOTE: because the subscribers are receiving these alerts from a non-CON multicast message, we need to create the resource end-point at subscribe time.  Otherwise, the PUT would be rejected and a POST would be needed, but the server won't know to do this.


#### Seismic Server Configuration

The seismic server runs the main RIDE-D module.

multiple multicast dst addresses?  dst_ip translated to actual host before delivery so just need to make it non-CON msg:
* could configure sink as multicast, but coapthon impl forces ALL_COAP as the dst_ip when doing this...
* how to explicitly set non-CON?
* socket-level ride-d impl: intercept multicast msg at socket level (ride-d socket that we passed to coap client constructor), determine which route would be best, and change dst_ip to that one.  Client will respond (if it does at all) with unicast msg anyway...


event-reporter plug-in (or ride-d-sink) needs to choose from multiple addresses based on STT
* how to handle OVERLAPPING mdmts????

TODO 
====

## Documentation

discuss rest api, sdn topo mgr, and cli interfaces

talk about config files?

document the mess that is network/sdn/netx_topology, esp. the fact that a host can be client or server; everything else is a switch, which can be a gateway