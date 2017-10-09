####      CONFIGURATIONS
# Configurations that you might need to change for a specific installation.

# can just change this and not the others if everything is running under same user account...
DEFAULT_USER='vagrant'

# This will control a lot of delays and debugging
TESTING = False
WITH_LOGS = True  # output seismic client/server stdout to a log file

# When True, runs host processes in mininet with -OO command for optimized python code
OPTIMISED_PYTHON = not TESTING

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
CLEANUP_SCALE_CLIENTS="ps aux | grep '\-m scale_client' | grep -v 'grep' | awk '{print $2;}' | xargs -n 1 kill -9"

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


import re
from topology_manager.test_sdn_topology import mac_for_host


def get_ip_mac_for_host(host):
    # See note in mininet_smart_campus_experiment.setup_topology about host format
    # XXX: differentiate between regular hosts and server hosts
    if '-' in host:
        host_num, building_type, building_num = re.match('h(\d+)-([mb])(\d+)', host).groups()
    else:  # must be a server
        building_type, host_num = re.match('h?([xs])(\d+)', host).groups()
        building_num = 0

    # Assign a number according to the type of router this host is attached to
    if building_type == 'b':
        router_code = 131
        router_mac_code = 'bb'
    elif building_type == 'm':
        router_code = 144
        router_mac_code = 'aa'
    # cloud
    elif building_type == 'x':
        router_code = 199
        router_mac_code = 'cc'
    # edge server
    elif building_type == 's':
        router_code = 255
        router_mac_code = '55'
    else:
        raise ValueError("unrecognized building type '%s' so cannot assign host IP address!" % building_type)
    _ip = "10.%d.%s.%s/%d" % (router_code, building_num, host_num, HOST_IP_N_MASK_BITS)
    _mac = mac_for_host(int(host_num))
    _mac = "00:%s:%s%s" % (router_mac_code, str(building_num).rjust(2, '0'), _mac[8:])
    # XXX: onos expects upper case mac addresses
    return _ip, _mac.upper()


def get_mac_for_switch(switch, is_cloud=False, is_server=False):
    # BUGFIX: need to manually specify the mac to set DPID properly or Mininet
    # will just use the number at the end of the name, causing overlaps.
    # HACK: slice off the single letter at start of name, which we assume it has;
    # then convert the number to a MAC.
    mac = mac_for_host(int(switch[1:]))
    # Disambiguate one switch type from another by setting the first letter
    # to be a unique one corresponding to switch type and add in the other 0's.
    # XXX: if the first letter is outside those available in hexadecimal, assign one that is
    first_letter = switch[0]
    if first_letter == 'm':
        first_letter = 'a'
    elif first_letter == 'g':
        first_letter = 'e'
    # We'll just label rack/floor switches the same way; we don't actually even use them currently...
    elif first_letter == 'r':
        first_letter = 'f'

    # XXX: we're out of letters! need to assign a second letter for the cloud/server switches...
    second_letter = '0'
    if is_cloud:
        second_letter = 'c'
    elif is_server:
        second_letter = 'e'

    mac = first_letter + second_letter + ':00:00:' + mac[3:]
    return str(mac).lower()

#########       EXPERIMENT    CONFIG    ##########

## Delays (in seconds) control when events happen, how long experiment lasts, and waiting for convergence
SEISMIC_EVENT_DELAY = 60 if not TESTING else 40  # seconds before the 'earthquake happens', i.e. sensors start sending data
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
IOT_CONGESTION_INTERVAL = 0.1
DEFAULT_TOPOLOGY_ADAPTER = 'onos'
