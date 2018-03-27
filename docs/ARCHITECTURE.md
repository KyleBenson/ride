# Architecture and Workflow

See [Architecture and Workflow](docs/ARCHITECTURE.md) for details about how Ride works.

Here we briefly explain the workflow and how we implemented the architecture.
Ride's theoretical architecture is as follows:

<img src="docs/ride_architecture.png" height=600>

## Overall flow

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

## REST API Adapter for SDN Controller Interaction

To connect the Ride middleware running at the edge with the SDN controller, we wrote an adapter for connecting to the controller's REST APIs.
This adapter enables network topology/state collection and flow rule maintenance using a *topology manager* layer implemented as an inheritance hierarchy.
See the [REST API Adapter documentation](topology_manager/README.md) for more details.
