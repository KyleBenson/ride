# Challenges overcome

* To ensure a single CoAP packet fits the entire alert message with hundreds of publishers, we implemented a method for extracting only the unique seismic event ID (publisher hostname and sequence #) to include in the packet and throw out older (by seq #) picks until the aggregated alert fits in a single packet.
* Handling subscriptions at server: how often should we re-analyze and re-construct the MDMTs?  Currently, we simply have the Ride-D module run its `update()` method once after a delay that essentially guarantees all subscriptions have arrived.  Otherwise, we had problems with the topology being updated and routing around any failures automatically (or causing errors due to now-missing components).
* Extended CoAPthon to handle non-CON multicast responses and treat them like ACKs for multicast alerting
