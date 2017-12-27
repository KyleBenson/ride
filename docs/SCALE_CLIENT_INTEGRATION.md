# SCALE Client integration

To demonstrate the Ride approach, we integrated it with the SCALE Client as a seismic alerting scenario.
The general idea is that the IoT device clients (publishers) forward seismic-related `SensedEvent`s to the server(s) for aggregation multicast alerting to subscribers.
Both of these client types are basic unmodified SCALE Clients, although subscribers have a special seismic alerting application for recording the experiment results.

The cloud server sends alerts via unicast to all subscribers, but the edge server runs Ride-enabled services: `RideCApplication` and `RideDEventSink`.
The former sets up publisher routes and configures `DataPathMonitor`s that internally publish events to update the status of each registered DataPath when it changes.
In response to these changes, it uses the REST API to configure re-direction flow routes to seamlessly forward seismic publications to the edge server.
It can also clear this redirection and return to the cloud-based operation.
When operating in edge mode, the `SeismicAlertServer` running on the edge server host will use Ride-D-based resilient multicast alerting.
The MDMTs were pre-configured based on the network topology constructed from interaction with the SDN controller via its REST API.
It installs flow rules to realize these MDMTs with each having an assigned IPv4 address.

## Publisher configuration

The publisher IoT clients simply send a seismic `SensedEvent` to the server (via its well-known IP address) at a predetermined time (i.e. the time of the simulated earthquake).
This report time includes a small amount of random delay to account for the fact that the *picking algorithm* will actually detect the seismic wave at slightly different times in different devices due to 1) the difference in wave arrival time and 2) subtle differences between devices e.g. OS thread scheduling, amount of normal background shaking, etc.

## Subscriber configuration

The subscriber clients will receive these multicast alerts via CoAP PUT.
By using our SDN-based IP multicast-based resilient pub-sub, these clients are essentially unmodified.
They only require UDP-based CoAP functionality; specifically must accept non-CONfirmable messages as alerts.
They must also **subscribe** to the alerts by contacting the server in advance.

For testing purposes, the subscribers are responsible for **gathering statistics** about the alerts.
They look at the aggregate event's metadata to see which publishers' picks are included in the aggregate one and record when they received these for later analysis.

NOTE: because the subscribers are receiving these alerts from a non-CON multicast message, we need to create the resource end-point at subscribe time.  Otherwise, the PUT would be rejected and a POST would be needed, but the server won't know to do this.


## Seismic Server Configuration

The seismic server runs the `SeismicAlertServer` Application as well as the main Ride modules.
It runs a `CoapServer` to receive seismic `SensedEvent`s from the *publishers* and quickly store them in a queue for processing.
Periodically (2 secs. by default) the server aggregates together all of the seismic events as a list of unique event IDs containing the publishing client address and earthquake sequence number.
It internally publishes a `seismic_alert` event containing this aggregation; the SCALE client `EventReporter` forwards this event to the `RideDEventSink`.
When this event sink is configured for multicast, it selects the best MDMT using the configured policy and does a CoAP PUT to the URI containing 1) the MDMT's multicast address; 2) the application-specified path for receiving these alerts, which is configured to expose an updateable CoAP resource  at subscription time.

NOTE: because CoAP multicast is best-effort (non-CONfirmable), we currently do not make use of ACKs in multicast alerting mode (the responses are unrecognized by the CoAP protocol layer).

NOTE: the current implementation handles *STT* construction/estimation directly in the RideD middleware rather than RideC publishing updates for it.

