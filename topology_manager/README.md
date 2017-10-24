Topology Manager & REST API
===========================

The topology management subsystem provides a mechanism for interacting with the SDN controller through its REST APIs.
This essentially makes an adapter for developing controller-agnostic SDN applications in Python to facilitate the SDN capabilities of RIDE.

Inheritance Model
-----------------

This subsystem (ab)uses inheritance to separate concerns and algorithms into different classes as well as provide for an adaptation layer to integrate different SDN controllers:

* `NetworkTopology` uses networkx graph data structures and algorithms to enable abstract network topology management in terms of: hosts (client or server devices), switches (can be a router or gateway too), and links connecting these components
* `SdnTopology` extends this to provide flow rule and group construction/installation/deletion
* `[Onos|Floodlight]SdnTopology` uses the respective SDN Controller's REST API adapter to integrate that 
* `NetworkxSdnTopology` implements the same interface as above but is designed for working with a networkx-based 

Command-line Interface Adapter
------------------------------

The `sdn_cli.py` file provides an incomplete CLI for using the above SDN topology management classes.
It provides some basic features such as listing available hosts/switches (in the `SdnTopology` format rather than the raw REST API format), installing unicast/multicast static flow paths, etc.
We used this for some basic testing of these SDN mechanism implementations.
For more advanced features, you'll probably want to just write a script and modify the specific parameters directly before running it.