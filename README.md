# Resilient IoT Data Exchange (Ride)

Resilient IoT Data Exchange (Ride) leveraging SDN and edge computing.
It extends existing publish-subscribe-oriented IoT data exchanges without client modifications to facilitate resilient real-time event-collection and emergency alert dissemination despite prevalent network failures and congestion (e.g. due to a severe earthquake).
The first phase, Ride-C, leverages programmable SDN-enabled infrastructure for gathering and utilizing network-awareness to improve IoT data collection.
It monitors cloud data paths (overlay links from the local smart campus network to the cloud IoT service) for failures/degraded quality and rapidly switches over to another available one (if available) or to the backup edge service.
The second phase, Ride-D, uses this information to disseminate time critical alerts via an intelligent network and application-aware multicast mechanism.
It pre-computes multiple Maximally-Disjoint Multicast Trees (MDMTs) for the registered subscribers and alert topic, installs them to the SDN data plane, and intelligently selects the best for alert dissemination based on current network state.

<img src="docs/approach.png" height=500>

This repository includes the algorithms, prototype implementation, SDN controller REST API adapter, and Mininet-based experimental framework we used to evaluate its performance.
The [docs](docs/) folder contains further documentation, including [implementation decision notes](docs/IMPLEMENTATION_NOTES.md) and a list of [TODOs/future work](docs/TODO.md).

Refer to [this link for the formal research paper](http://www.ics.uci.edu/~dsm/papers/2018/ride-iotdi-2018.pdf) that presents Ride and please cite it as:

Kyle E. Benson, Guoxi Wang, Young-Jin Kim, and Nalini Venkatasubramanian. "Ride: A Resilient IoT Data Exchange Middleware Leveraging SDN and Edge Cloud Resources". *in Proceedings of 2018 ACM/IEEE Third International Conference on Internet-of-Things Design and Implementation (IoTDI)*, Orlando, Florida, USA, 2018.

## Getting Started

Follow the [installation instructions](docs/INSTALL.md) to install Ride.


## Architecture and Workflow

Here we briefly explain the workflow and how we implemented the architecture.
Ride's theoretical architecture is as follows:

<img src="docs/ride_architecture.png" height=600>

### Overall flow

Refer to the following figure and workflow sequence for an overview of the whole workflow:

<img src="docs/ride_workflow.png" height=400>

1) The server creates a CoAP resource end-point to receive publications and subscription requests. Subscribers PUT alert subscriptions on this server end-point and then create a CoAP resource end-point to receive the alerts.
2) Ride-C/D modules on the server connect to SDN controller via its REST API (see [topology_manager docs](topology_manager/README.md)) to collect network topology information.  This information is used to establish network routes (and then install associated flow rules) for publications, cloud data path (CDP) probes, and the MDMT routes that deliver alerts to subscribers and their responses back to the server.  These routes use flow rules that match based on both IPv4 address and port # in order to distinguish multiple data flows (i.e. topics) between two end-points.  This is especially important for Ride-D's reliable retransmission mechanism (i.e. unicast alert responses must flow along the multicast tree used to get them there) and the CDP probes.
3) Ride-C monitors CDPs and adapts to failures/congestion by re-routing to another CDP or the edge using redirection flow rules that translate the destination IP address.
3) At earthquake time, the publishers send their picks to the currently-active seismic server (be it cloud or edge).
4) As publications arrive, Ride stores them in a temporary buffer for aggregation into an alert every 2 seconds (representing the earthquake-detection algorithm's running time).  As it processes them, it updates its current STT view.
6) It then sends the aggregated picks (mostly just the publisher IDs and timestamps) out in an alert message.
 This CoAP PUT message traverses the MDMT(s) chosen by the Ride-D module based on the STT.
The intermediate SDN switches route it and copy it (using OpenFlow groups tables) appropriately.  They eventually translate the multicast destination IP address to the end-host before delivery to avoid multicast group configurations on the IoT end-devices.
7) Subscribers respond to alerts with acknowledgements that are routed back along the same MDMT used to deliver the alert.
8) Ride-D marks which subscribers were reached, further updates the STT with the routes taken by these responses, and retransmits the alert after some timeout period if some subscribers remain unreached.  These retransmissions only consider the unreached subscribers (i.e. it trims them and unnecessary links from the MDMTs) in order to prefer MDMTs that help reach them specifically.

### REST API Adapter for SDN Controller Interaction

To connect the Ride middleware running at the edge with the SDN controller, we wrote an adapter for connecting to the controller's REST APIs.
This adapter enables network topology/state collection and flow rule maintenance using a *topology manager* layer implemented as an inheritance hierarchy.
See the [REST API Adapter documentation](topology_manager/README.md) for more details.

### Example application: Seismic alerting service

To show an example scenario and study Ride's effectiveness in a motivating realistic scenario, we implemented a seismic alerting service that gathers ground shaking data from IoT sensors, detects possible earthquakes, and issues alerts to interested subscribers (i.e. human users or actuating devices).
This application, based on the [Community Seismic Network (CSN) concept](http://csn.caltech.edu/), is built around our multi-purpose IoT system called SCALE.
See the [SCALE Client integration notes](docs/SCALE_CLIENT_INTEGRATION.md) for details.

