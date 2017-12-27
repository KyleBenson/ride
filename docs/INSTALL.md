# Installing Ride

If you have issues getting everything working after following these directions, refer to the [troubleshooting tips](TROUBLESHOOTING.md).

We recommend using a Python `virtualenv` to both avoid polluting your system-wide Python distribution and also to support a slight hack used for the Mininet version (root runs a command that will be executed by non-root user so they'll both need this virtualenv).

First, install the Python [requirements.txt](../requirements.txt).

**Networkx:** Note that you may need a 'bleeding edge' version of networkx for the default multicast tree generation (Steiner tree) algorithm.
  As of 1/2017 it is not included in the officially released version, so we just checked out
   the third-party fork that the pull request is based on in an external directory and
   created a symbolic link in this directory to make the import work.
You can just clone [my networkx repo](https://github.com/KyleBenson/networkx/tree/digraph_fix) and use the branch 'digraph-fix' that will have steinertree in it...

## Networkx-based large-scale Ride-D experiments

To scale up larger than Mininet can handle (i.e. hundreds of nodes), we included a NetworkX-only version of the experimental framework.
You can run this through the `networkx_smart_campus_experiment.py` script (see its help info for details by adding the `--help` command).

## Mininet-based Experiments

Summary of requirements:
* Mininet and Open vSwitch
* [ONOS](https://wiki.onosproject.org/display/ONOS/Installing+and+running+ONOS) SDN controller
* [SCALE client](https://github.com/KyleBenson/scale_client)
* [seismic_warning_test](https://github.com/KyleBenson/seismic_warning_test), which contains the adapter code to integrate Ride into our test seismic warning app as a SCALE client app
* [networkx with digraph fix and steinertree](https://github.com/KyleBenson/networkx/tree/digraph_fix)
* [coapthon](https://github.com/KyleBenson/CoAPthon.git) (our modified fork).

WARNING: Mininet needs to run as root!  Therefore, you're going to have to set up your environment so that root shares the same Python `virtualenv` as your regular user!  You could alternatively run everything as root in which case you'll need to tweak the `config.py` some...

To run the Mininet-based experiments, we'll obviously need to install Mininet and its Python API (e.g. via pip).  We just cloned the source repo and symlinked it to inside this repo's main directory so that Python can import classes for the Mininet API.
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
Currently we only fully support ONOS (v1.11 seems to work best).  We install it from tarball using the directions on their website and configure it to run as a service with the following options file (in `/opt/onos/options`):
```
ONOS_USER=sdn
ONOS_APPS=openflow-base,drivers,openflow,fwd,proxyarp
JAVA_HOME=/usr/lib/jvm/java-8-oracle
```
If you're running on a physical machine rather than a VM, you should probably change the user/password for the karaf GUI in `/opt/onos/apache-karaf-$KARAF_VERSION/etc/users.properties` to something more secure.
Note that because ONOS uses a clustering service, you may run into problems with dynamic IP addresses (i.e. the master node is no longer available because your machine IS the master node but its IP address changed).

NOTE: you might want to use `/opt/onos/bin/onos-user-key karaf ~/.ssh/id_rsa.pub` so you can run `/opt/onos/bin/onos` without typing in the password.  This is used to reset ONOS between each run so you'll probably NEED to!


Make sure to get **Open vSwitch** set up properly too!  You can install it with Mininet's install script.
Note that the config.py file contains some reset commands used to clean out and restart OVS between experiment runs: you may wish to change these to e.g. `service ovs-vswitch restart` or `systemctl restart openvswitch-switch`

You may need to tweak some settings inside config.py e.g. the IP addresses of the Mininet hosts might conflict with your OS's networking configuration esp. if you're running inside a virtual machine that uses a `10.0.*.*` address.

This repository assumes that the python packages (i.e. folder containing `__init__.py`) within the following repositories are on your PYTHONPATH:
* [scale_client](https://github.com/KyleBenson/scale_client)
* [seismic_warning_test](https://github.com/KyleBenson/seismic_warning_test)
* [coapthon](https://github.com/KyleBenson/CoAPthon.git) (our modified fork).
  
Really, they'll need to be in the *root* user's PYTHONPATH since that's how you'll be executing it!
I typically use symbolic links to just put them in the same directory; note that you may need to use an absolute (NOT relative) path for this, although I think this issue may have been related to crossing filesystem boundaries (my repository was on my Mac OSX system while we actually ran Mininet inside a Linux VM).
   
**Running the SCALE Client from within Mininet**: to properly run this, we opt to specify a different user that will have a virtual environment they can run the scale client in.  You can change the `SCALE_USER` at the top of the scale_config.py file. Make sure this user has the `virtualenvwrapper.sh` script available (you can edit the `SCALE_CLIENT_BASE_COMMAND` as well if you use virtualenv differently or want to forego it entirely), create a virtual environment with the name `ride_scale_client`, and install the dependencies (both for scale_client and for ride) in that environment with `pip install -r requirements.txt`

Also note that you'll need to point the Scale Client to the proper SDN controller IP address.  Set the appropriate address of the machine you're running it on in `config.py`.

