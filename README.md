# sdn_pubsub_mcast
Installs SDN (OpenFlow) rules for accomplishing pub-sub via multicast.

## Getting Started

First, install the Python [requirements.txt](requirements.txt).

**Networkx:** Note that you may need a 'bleeding edge' version of networkx for the default multicast tree generation (Steiner tree) algorithm.
  As of 1/2017 it is not included in the officially released version, so we just checked out
   the third-party fork that the pull request is based on in an external directory and
   created a symbolic link in this directory to make the import work.
You can just clone [my networkx repo](https://github.com/KyleBenson/networkx/tree/digraph_fix) and use the branch 'digraph-fix' that will have steinertree in it...
   
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
Note that the config.py file contains some reset commands used to clean out and restart OVS between experiment runs: you may wish to change these to e.g. `service ovs-vswitch restart`

You may need to tweak some settings inside config.py e.g. the IP addresses of the Mininet hosts might conflict with your OS's networking configuration esp. if you're running inside a virtual machine.

This repository assumes that the [scale_client](https://github.com/KyleBenson/scale_client) and [seismic_warning_test](https://github.com/KyleBenson/seismic_warning_test) repos are on your PYTHONPATH.  Really, they'll need to be in *root*'s PYTHONPATH!  I typically use symbolic links to just put them in the same directory; make sure you use an absolute (NOT relative) path for this or it won't work right!
   
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

We recommend installing ONOS from the tarball rather than from scratch (building from source), which the troubleshooting steps below relate to.

Sometimes ONOS just stops working properly... it might start, but the web server is not accessible and just keeps responding with NOT_FOUND.
I'm unsure why, but reinstalling ONOS seems to help.  Do `onos-kill; onos-uninstall; mvn clean; onos-build; onos-install` to reinstall it one step at a time.
This also assume that your *cell* is set properly: make sure that `onos-cell` refers to the right user/password on your systems for it to run as and that it doesn't refer to *localhost* in the `$OC??` variables (use 127.0.0.1 if you're running it locally).

Most of the development/testing was done on ONOS version 1.9.0.


## Architecture and Workflow

Here we briefly explain the workflow and how we implemented the architecture.

### REST API Adapter for SDN Controller Interaction

To connect the RIDE middleware running at the edge with the SDN controller, we wrote an adapter for connecting to the controller's REST APIs.
This adapter enables network topology/state collection and flow rule maintenance using a *topology manager* layer implemented as an inheritance hierarchy.
See the [REST API Adapter documentation](topology_manager/README.md) for more details.

### Overall flow

1) Subscribers contact server to subscribe to alerts (as described below, they create the CoAP resource end-point at this time).
2) RIDE-D module on the server connects to SDN controller via its REST API (see `topology_manager` folder) for using network topology info to establish MDMTs (and install associated flow rules) for the alert subscribers.
3) As publications arrive, RIDE builds up the STT; RIDE-C simultaneously monitors cloud connection DataPaths and adapts to failures/congestion by re-routing to another DataPath or the edge. 
4) At earthquake time, the publishers send their picks to the seismic server.
5) The server aggregates these picks together for a couple seconds (representing the earthquake-detection algorithm's running time) and updates its current STT view based on these recently-received picks.
6) It then sends the aggregated picks (mostly just the publisher IDs and timestamps) out in an alert message.
 This CoAP PUT message traverses the MDMT(s) chosen by the RIDE-D module based on the STT.
The intermediate SDN switches route it and copy it appropriately, eventually translating the multicast destination IP address to the end-host before delivery to it.


### SCALE Client integration

To demonstrate the RIDE approach, we integrated it with the SCALE Client as a seismic alerting scenario.
The general idea is that the IoT device clients (publishers) forward seismic-related `SensedEvent`s to the server(s) for aggregation multicast alerting to subscribers.
Both of these client types are basic unmodified SCALE Clients, although subscribers have a special seismic alerting application for recording the experiment results.

The cloud server sends alerts via unicast to all subscribers, but the edge server runs RIDE-enabled services: `RideCApplication` and `RideDEventSink`.
The former sets up publisher routes and configures `DataPathMonitor`s that internally publish events to update the status of each registered DataPath when it changes.
In response to these changes, it uses the REST API to configure re-direction flow routes to seamlessly forward seismic publications to the edge server.
It can also clear this redirection and return to the cloud-based operation.
When operating in edge mode, the `SeismicAlertServer` running on the edge server host will use RIDE-D-based resilient multicast alerting.
The MDMTs were pre-configured based on the network topology constructed from interaction with the SDN controller via its REST API.
It installs flow rules to realize these MDMTs with each having an assigned IPv4 address.

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

The seismic server runs the `SeismicAlertServer` Application as well as the main RIDE modules.
It runs a `CoapServer` to receive seismic `SensedEvent`s from the *publishers* and quickly store them in a queue for processing.
Periodically (2 secs. by default) the server aggregates together all of the seismic events as a list of unique event IDs containing the publishing client address and earthquake sequence number.
It internally publishes a `seismic_alert` event containing this aggregation; the SCALE client `EventReporter` forwards this event to the `RideDEventSink`.
When this event sink is configured for multicast, it selects the best MDMT using the configured policy and does a CoAP PUT to the URI containing 1) the MDMT's multicast address; 2) the application-specified path for receiving these alerts, which is configured to expose an updateable CoAP resource  at subscription time.

NOTE: because CoAP multicast is best-effort (non-CONfirmable), we currently do not make use of ACKs in multicast alerting mode (the responses are unrecognized by the CoAP protocol layer).

NOTE: the current implementation handles *STT* construction/estimation directly in the RideD middleware rather than RideC publishing updates for it.

### Challenges overcome

* To ensure a single CoAP packet fits the entire alert message with hundreds of publishers, we implemented a method for extracting only the unique seismic event ID (publisher hostname and sequence #) to include in the packet and throw out older (by seq #) picks until the aggregated alert fits in a single packet.
* Handling subscriptions at server: how often should we re-analyze and re-construct the MDMTs?  Currently, we simply have the Ride-D module run its `update()` method once after a delay that essentially guarantees all subscriptions have arrived.  Otherwise, we had problems with the topology being updated and routing around any failures automatically (or causing errors due to now-missing components).
 

TODO 
====

## Documentation

* Details about configuration files for customizing a RIDE installation (for now see in-line comments in the `config.py` and `ride/config.py` files)
* Architecture diagram to explain the inheritance model
* Explain output results format and the `statistics.py` parsing/analysis modules.

## A More Robust Implementation

In the RIDE paper, we presented a particular architecture and separation of concerns.
However, this current implementation uses the SDN controller's REST API and a Python-based adaptation layer to facilitate RIDE's SDN interactions.
A more robust implementation as an actual SDN controller application (i.e. a Network Operating System service) runs RIDE's main logic on the controller with thin clients on the edge/cloud servers.
While that approach could demonstrate better performance by leveraging the SDN controller's carefully-engineered services (e.g. distributed topology synchronization), we considered it an engineering exercise and focused on rapidly prototyping RIDE to study its algorithm's performance.

This decision allowed us to:
1) Easily migrate between SDN controller platforms and isolate RIDE from performance issues related to their implementations. Note that we started with Floodlight and moved to ONOS upon encountering some bugs.
2) Test the system and its algorithms as stand-alone software.  See the `ride` package with the self-contained RIDE C and D algorithms.
3) Iteratively refine the pieces to work on a real network stack (emulated in Mininet) and deploy them in our real system testbed.
4) Later run large-scale simulations over various parameters. See the [NetworkX-based simulated experiment version](networkx_smart_campus_experiment.py).

Furthermore, this prototype exhibits some slight architectural differences from the one presented in the paper:
1) Publishers and subscribers register directly with the RIDE-C/D modules instead of seamlessly contacting the data exchange service with registration requests.
The latter implementation could be done in three ways: 
    1. Running a RIDE-enabled data exchange instance that forwards these requests to the RIDE service as part of processing them.  This does not meet our goal of using unmodified IoT data exchange services.
    2. The SDN data plane forwards requests to both RIDE and the unmodified service.  A problem with this is how to identify new requests by network-layer information e.g. is this packet a subscription request or perhaps just a publication for a previously-advertised topic?
    3. A RIDE SDN controller application intercepts new requests (similar problem to above) to parse and process them directly before forwarding the originals to an unmodified data exchange instance.
2) RIDE-C currently publishes the assigned publisher routes to RIDE-D, and the latter maintains the *STT* instead of the former.
3) The experiment directly configures the cloud echo server rather than RIDE-C programmatically spawning the instance.


## Possible Enhancements

* How to handle overlapping MDMTs? Could combine flow rules (e.g. match 224.0.0.[1-4]) to save flow table space.  Could also consider incremental adjustments to MDMTs to minimize possible down-time as well as overhead.
* Extend CoAPthon to handle non-CON responses and treat them like ACKs for multicast alerting?