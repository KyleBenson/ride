import re
from net import HOST_IP_N_MASK_BITS
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
