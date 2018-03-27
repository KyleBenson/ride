IPERF_BASE_PORT = 5000  # background traffic generators open several iperf connections starting at this port number
PROBE_BASE_SRC_PORT = 9900  # ensure this doesn't collide with any other apps/protocols you're using!
ECHO_SERVER_PORT = 9999
COAP_CLIENT_BASE_SRC_PORT = 7777  # for RemoteCoapEventSink; needed to properly identify publisher traffic
OPENFLOW_CONTROLLER_PORT = 6653  # we assume the controller will always be at the default port
HOST_IP_N_MASK_BITS = 9
# subnet for all hosts (if you change this, update the __get_ip_for_host() function!)
# NOTE: we do /9 so as to avoid problems with addressing e.g. the controller on the local machine
# (vagrant uses 10.0.2.* for VM's IP address).
IP_SUBNET = '10.128.0.0/%d' % HOST_IP_N_MASK_BITS
# HACK: rather than some ugly hacking at Mininet's lack of API for allocating the next IP address,
# we just put the NAT/server interfaces in a hard-coded subnet.
NAT_SERVER_IP_ADDRESS = '11.0.0.%d/24'
## multiple MDMTs (multicast alerting trees) are distinguished by unique IP address/UDP src port network addresses
## NOTE: the UDP port is necessary so that replies can be routed along the particular MDMT used to disseminate the alert
MULTICAST_ADDRESS_BASE = u'224.0.0.1'  # must be unicode!
MULTICAST_ALERT_BASE_SRC_PORT = 4000

DEFAULT_ERROR_RATE = 0.0  # a rate in range [0 .. 1.0]
DEFAULT_LATENCY = 10.0  # in ms
DEFAULT_JITTER = 1.0    # in ms
DEFAULT_BANDWIDTH = 1  # in Mbps
