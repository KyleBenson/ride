####      CONFIGURATIONS
# Configurations that you might need to change for a specific installation.
# TODO: this file can be sourced in bash to collect these variables too!  BUT we need to do string formatting differently...

# When True, runs host processes in mininet with -OO command for optimized python code
OPTIMISED_PYTHON = False

# can just change this and not the others if everything is running under same user account...
DEFAULT_USER='vagrant'

# This will control a lot of delays and debugging
TESTING = True
WITH_LOGS = TESTING  # output seismic client/server stdout to a log file

##########         ONOS     CONFIG      ###########

CONTROLLER_IP="10.0.2.15"
ONOS_ADMIN_USER=DEFAULT_USER  # user that can run ONOS commands, NOT who is running the ONOS service!
ONOS_USER=ONOS_ADMIN_USER     # the user actually running the ONOS service
ONOS_ADMIN_PORT=8101  # SSH port, NOT REST API port!
ONOS_HOSTNAME='localhost'

# For accessing REST API
ONOS_API_USER='karaf'
ONOS_API_PASSWORD='karaf'

CONTROLLER_SERVICE_RESTART_CMD='service onos restart'

# Since Mininet runs as root, we need a way of invoking ONOS commands as the ONOS user
# to reset the controller in between executions.  This is a HACK as we couldn't get the
# shell to execute with ONOS_ADMIN_USER's proper env variables that let us use the 'onos' command,
# which would let us just call 'tools/onos_reset.sh'
CONTROLLER_RESET_CMD="su -c 'ssh -p %d %s@%s wipe-out please' %s" % (ONOS_ADMIN_PORT, ONOS_USER, ONOS_HOSTNAME, ONOS_ADMIN_USER)


##########    SCALE CLIENT CONFIG       ###########

### Configurations for actually running SCALE as the test client applications
# Make sure you setup a virtual environment called 'scale_client' for this user!
SCALE_USER=DEFAULT_USER
# XXX: HACK: since Mininet runs as root and we use virtual environments, we have to run the client
# within the venv but at the right location, under the right user, with the right PYTHONPATH,
# all as a large complicated command passed as a string to 'su'...
SCALE_EXTRA_ARGS=\
    "--disable-log-module topology_manager.sdn_topology urllib3.connectionpool " \
    "--raise-errors " \
    # "--format-logging '%%(levelname)-6s : %%(name)-55s (%%(asctime)2s) : %%(message)s'"  # TODO: this doesn't work right now due to coapthon logging bug.... add timestamps; make sure to use '%%' to keep it from doing the formatting yet!
    # " --enable-log-module coapthon " \
# Change this command to match your environment's configuration as necessary
VIRTUAL_ENV_CMD="export WORKON_HOME=~/.venvs; source ~/.local/bin/virtualenvwrapper.sh; workon ride_scale_client;"
# WARNING: this took a long time to get actually working as the quotes are quite finicky... careful modifying!
SCALE_CLIENT_BASE_COMMAND='su -c "pushd .; %s popd; python %s -m scale_client %s %%s" ' % (VIRTUAL_ENV_CMD, "-OO" if OPTIMISED_PYTHON else "", SCALE_EXTRA_ARGS) + SCALE_USER


##### Misc.
IGNORE_OUTPUT = ' > /dev/null 2>&1'

### Open vSwitch
# Basically these are aliases for restarting OVS between runs, which seems to help solve some issues with larger topos...

# Change these paths depending on your system's installation
OVS_PREFIX_DIR='/usr/local'
OVS_SCHEMA='/home/%s/repos/ovs/vswitchd/vswitch.ovsschema;' % DEFAULT_USER
OVS_KERNEL_FILE='/home/%s/repos/ovs/datapath/linux/openvswitch.ko' % DEFAULT_USER

# These two collections of commands are assumed to be run as root!
RESET_OVS='''sudo pkill ovsdb-server;
sudo pkill ovs-vswitchd;
sudo rm -f %s/var/run/openvswitch/*;
sudo rm -f %s/etc/openvswitch/*;
ovsdb-tool create %s/etc/openvswitch/conf.db %s
sudo modprobe libcrc32c;
sudo modprobe gre;
sudo modprobe nf_conntrack;
sudo modprobe nf_nat_ipv6;
sudo modprobe nf_nat_ipv4;
sudo modprobe nf_nat;
sudo modprobe nf_defrag_ipv4;
sudo modprobe nf_defrag_ipv6;
sudo insmod %s
''' % (OVS_PREFIX_DIR, OVS_PREFIX_DIR, OVS_PREFIX_DIR, OVS_SCHEMA, OVS_KERNEL_FILE)

RUN_OVS='''sudo ovsdb-server --remote=punix:%s/var/run/openvswitch/db.sock \
    --private-key=db:Open_vSwitch,SSL,private_key \
    --certificate=db:Open_vSwitch,SSL,certificate \
    --bootstrap-ca-cert=db:Open_vSwitch,SSL,ca_cert \
    --pidfile --detach --log-file;
ovs-vsctl --db=unix:%s/var/run/openvswitch/db.sock --no-wait init;
sudo ovs-vswitchd --pidfile --detach --log-file''' % (OVS_PREFIX_DIR, OVS_PREFIX_DIR)


##########    NETWORKING CONFIG       ###########

IPERF_BASE_PORT = 5000  # background traffic generators open several iperf connections starting at this port number
PROBE_BASE_SRC_PORT = 9900  # ensure this doesn't collide with any other apps/protocols you're using!
ECHO_SERVER_PORT = 9999
COAP_CLIENT_SRC_PORT = 7777  # for RemoteCoapEventSink; needed to properly identify publisher traffic
OPENFLOW_CONTROLLER_PORT = 6653  # we assume the controller will always be at the default port
HOST_IP_N_MASK_BITS = 9
# subnet for all hosts (if you change this, update the __get_ip_for_host() function!)
# NOTE: we do /9 so as to avoid problems with addressing e.g. the controller on the local machine
# (vagrant uses 10.0.2.* for VM's IP address).
IP_SUBNET = '10.128.0.0/%d' % HOST_IP_N_MASK_BITS
# HACK: rather than some ugly hacking at Mininet's lack of API for allocating the next IP address,
# we just put the NAT/server interfaces in a hard-coded subnet.
NAT_SERVER_IP_ADDRESS = '11.0.0.%d/24'
MULTICAST_ADDRESS_BASE = u'224.0.0.1'  # must be unicode!


#########       EXPERIMENT    CONFIG    ##########

## Delays (in seconds) control when events happen, how long experiment lasts, and waiting for convergence
SEISMIC_EVENT_DELAY = 60 if not TESTING else 35  # seconds before the 'earthquake happens', i.e. sensors start sending data
TIME_BETWEEN_SEISMIC_EVENTS = 20 if not TESTING else 10  # for multiple earthquakes / aftershock events
SLEEP_TIME_BETWEEN_RUNS = 15 if not TESTING else 5 # give Mininet/OVS/ONOS a chance to reconverge after cleanup
# make sure this leaves enough time for all data_path failures, recoveries, and publishing additional
# seismic events so we can see the recovery!  Maybe just configure this dynamically in the future?
EXPERIMENT_DURATION = SEISMIC_EVENT_DELAY + 10 + TIME_BETWEEN_SEISMIC_EVENTS * (3 if TESTING else 4)
# whether to set static ARP and ping between all pairs or just cloud/edge servers
# set this to True if the controller topology doesn't seem to be including all expected hosts
ALL_PAIRS = False
# Since we're using the scale client for background traffic, we need to specify an interval between SensedEvents
# ENHANCE: base this on the traffic_generator_bandwidth parameter
IOT_CONGESTION_INTERVAL = 0.05
DEFAULT_TOPOLOGY_ADAPTER = 'onos'