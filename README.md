# sdn_pubsub_mcast
Installs SDN (OpenFlow) rules for accomplishing pub-sub via multicast.

Note that you may need a 'bleeding edge' version of networkx for the default multicast tree generation (Steiner tree) algorithm.
  As of 1/2017 it is not included in the officially released version, so we just checked out
   the third-party fork that the pull request is based on in an external directory and
   created a symbolic link in this directory to make the import work. 