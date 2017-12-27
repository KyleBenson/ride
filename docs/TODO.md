TODO 
====

## Documentation

* Details about configuration files for customizing a Ride installation (for now see in-line comments in the `config.py` and `Ride/config.py` files)
* Architecture diagram to explain the inheritance model
* Explain output results format and the `statistics.py` parsing/analysis modules.

## A More Robust Implementation

In the Ride paper, we presented a particular architecture and separation of concerns.
However, this current implementation uses the SDN controller's REST API and a Python-based adaptation layer to facilitate Ride's SDN interactions.
A more robust implementation as an actual SDN controller application (i.e. a Network Operating System service) runs Ride's main logic on the controller with thin clients on the edge/cloud servers.
While that approach could demonstrate better performance by leveraging the SDN controller's carefully-engineered services (e.g. distributed topology synchronization), we considered it an engineering exercise and focused on rapidly prototyping Ride to study its algorithm's performance.

This decision allowed us to:
1) Easily migrate between SDN controller platforms and isolate Ride from performance issues related to their implementations. Note that we started with Floodlight and moved to ONOS upon encountering some bugs.
2) Test the system and its algorithms as stand-alone software.  See the `Ride` package with the self-contained Ride C and D algorithms.
3) Iteratively refine the pieces to work on a real network stack (emulated in Mininet) and deploy them in our real system testbed.
4) Later run large-scale simulations over various parameters. See the [NetworkX-based simulated experiment version](networkx_smart_campus_experiment.py).

Furthermore, this prototype exhibits some slight architectural differences from the one presented in the paper:
1) Publishers and subscribers register directly with the Ride-C/D modules instead of seamlessly contacting the data exchange service with registration requests.
The latter implementation could be done in three ways: 
    1. Running a Ride-enabled data exchange instance that forwards these requests to the Ride service as part of processing them.  This does not meet our goal of using unmodified IoT data exchange services.
    2. The SDN data plane forwards requests to both Ride and the unmodified service.  A problem with this is how to identify new requests by network-layer information e.g. is this packet a subscription request or perhaps just a publication for a previously-advertised topic?
    3. A Ride SDN controller application intercepts new requests (similar problem to above) to parse and process them directly before forwarding the originals to an unmodified data exchange instance.
2) Ride-C currently publishes the assigned publisher routes to Ride-D, and the latter maintains the *STT* instead of the former.
3) The experiment directly configures the cloud echo server rather than Ride-C programmatically spawning the instance.


## Possible Enhancements

* How to handle overlapping MDMTs? Could combine flow rules (e.g. match 224.0.0.[1-4]) to save flow table space.  Could also consider incremental adjustments to MDMTs to minimize possible down-time as well as overhead.
