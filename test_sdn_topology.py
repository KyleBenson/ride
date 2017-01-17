# Test suite for SdnTopology classes

# helps determine various helper functions
import json

API_TYPE = 'floodlight'

# Helper functions for Mininet-based tests

def mac_for_host(host_num):
    """Assuming you started mininet with --mac option, this returns a
    mac address for host h<host_num> e.g. for h1 do mac_for_host(1) --> 00:00:00:00:00:01"""

    # format int as hex with proper number of octets, then add :'s using some pymagic
    num = format(host_num, 'x').rjust(12, '0')
    num = ':'.join(s.encode('hex') for s in num.decode('hex'))
    return num


def id_for_host(host_num):
    id = mac_for_host(host_num)
    if API_TYPE == 'onos':
        id += '/None'
    return id


def dpid_for_switch(switch_num):
    """This returns a DPID for switch s<switch_num>
    e.g. for s1 do dpid_for_switch(1) --> 00:00:00:00:00:00:00:01 in Floodlight
    and of:0000000000000001 in ONOS"""

    if API_TYPE == 'floodlight':
        # Assume we won't have enough switches to ever break this...
        return "00:00:%s" % mac_for_host(switch_num)
    elif API_TYPE == 'onos':
        return "of:0000%s" % mac_for_host(switch_num).replace(':', '')


#### Tests


def test_path_flow(st):
    """Test simple static flow entries for a basic path between h1 and h16"""

    # path = st.get_path("10.0.0.1", "10.0.0.16")
    path = st.get_path(id_for_host(1), id_for_host(16))
    # print "Path:", path
    rules = st.get_flow_rules_from_path(path)
    rules.extend(st.get_flow_rules_from_path(list(reversed(path))))
    print "Rules: %s" % rules
    for rule in rules:
        assert st.install_flow_rule(rule)


def test_group_flows(st):
    """Test the functions used to construct and install groups"""

    switch = dpid_for_switch(2)
    matches = st.get_matches(ipv4_src='10.0.0.1', eth_type='0x0800')
    actions = st.get_actions(("group", 1))

    flow = st.get_flow_rule(switch, matches, actions, priority=500)
    buckets = [st.get_bucket(st.get_actions(("output", 3)))]
    gflow = st.get_group_flow_rule(switch, buckets)
    print "Flow rules for groups:", flow
    print "Groups:", gflow
    assert st.install_group(gflow)
    assert st.install_flow_rule(flow)


def test_mcast_flows(st):
    mcast_tree = st.get_multicast_tree(id_for_host(1), [id_for_host(2), id_for_host(9), id_for_host(15)])
    # print list(mcast_tree.nodes())
    matches = st.get_matches(ipv4_src="10.0.0.1", ipv4_dst="224.0.0.1", eth_type='0x0800')
    flows = st.get_flow_rules_from_multicast_tree(mcast_tree, id_for_host(1), matches)
    print "Mcast flows: %s" % flows
    for flow in flows:
        assert st.install_flow_rule(flow)


def test_utils(st):
    """Test the various smaller helper functions that
    essentially adapt to the specific controller's REST API."""
    assert st.get_ip_address(id_for_host(1)) == "10.0.0.1"

    assert st.get_ports_for_nodes(id_for_host(1), dpid_for_switch(2)) == (0,1)
    assert st.get_ports_for_nodes(dpid_for_switch(1), dpid_for_switch(2)) == (1,5)

    print 'test_utils passed!'


def test_flow_helpers(st):
    """Test the helper functions used to construct flow rules"""
    if API_TYPE == "floodlight":
        # actions
        a1 = st.get_actions(('output', 2))
        assert a1 == "output=2"
        a2 = st.get_actions('strip_vlan', ('output', 2))
        assert a2 == "strip_vlan,output=2"
        a3 = st.get_actions(('strip_vlan'), ('set_ipv4_dst', '10.0.0.1'), ('output', 2))
        assert a3 == "strip_vlan,set_field=ipv4_dst->10.0.0.1,output=2"

        # matches
        m1 = st.get_matches(ipv4_src="10.0.0.3", in_port=3)
        assert m1 == {"in_port": "3", "ipv4_src": "10.0.0.3"}

        # TODO: whole flows
    elif API_TYPE == "onos":
        # actions
        a1 = st.get_actions(('output', 2))
        assert a1 == [{"type": "OUTPUT", "port": "2"}]
        a2 = st.get_actions(('set_eth_dst', "00:11:22:33:44:55"), ('output', 2))
        assert a2 == [{"type": "L2MODIFICATION", "subtype": "ETH_DST", "mac": "00:11:22:33:44:55"}, {"type": "OUTPUT", "port": "2"}]

        # matches
        # TODO: double-check that ONOS accepts ints here
        m1 = st.get_matches(ipv4_src="10.0.0.3", in_port=3)
        # NOTE: need to sort since kwargs aren't ordered
        assert sorted(m1) == sorted([{"type": "IN_PORT", "port": 3}, {"type": "IPV4_SRC", "ip": "10.0.0.3/32"}])

        # whole flows
        f1 = st.get_flow_rule("of:000000000021", m1, a1, priority=100)
        assert f1 == {"priority": 100, "isPermanent": True, "deviceId": "of:000000000021",
                      "treatment": {"instructions": a1}, "selector": {"criteria": m1}}
        try:
            f1 = st.get_flow_rule("of:000000000021", m1, a1)
            assert False == "not specifying priority should throw an assertion"
        except ValueError:
            pass

    print "test_flow_helpers() passed!"


def test_group_helpers(st):
    """Test the helper functions used to construct groups"""
    if API_TYPE == 'floodlight':
        # TODO: add some unit tests here instead of relying on test_group_flows()
        pass
    elif API_TYPE == 'onos':
        a1 = st.get_actions(('output', 2))
        a2 = st.get_actions(('set_eth_dst', "00:11:22:33:44:55"), ('output', 2))
        b1 = st.get_bucket(a1, weight=0.5)
        assert b1 == {"treatment": {"instructions": a1}, "weight": 0.5}
        b2 = st.get_bucket(a2, watch_group=5)
        assert b2 == {"treatment": {"instructions": a2}, "watchGroup": "5"}

        switch = dpid_for_switch(1)
        g = st.get_group_flow_rule(switch, [b1, b2])
        assert g == {
            "type": "ALL",
            "deviceId": switch,
            "appCookie": "SdnTopology",
            "groupId": "1",
            "buckets": [b1, b2]
        }


def run_tests(st):
    # These tests assume running mininet locally in the following way
    # AFTER starting Floodlight:
    # sudo mn --controller remote --mac --topo=tree,2,4
    #
    # THEN run 'pingall 10' in Mininet to populate the
    # hosts' IP addresses in Floodlight

    test_utils(st)
    test_flow_helpers(st)
    test_path_flow(st)

    test_group_helpers(st)
    test_group_flows(st)
    test_mcast_flows(st)

if __name__ == '__main__':
    import sys
    valid_args = ['floodlight', 'onos']
    if len(sys.argv) < 2 or sys.argv[1] not in valid_args:
        print "Usage: argument must be one of %s" % valid_args
        print "Defaulting to %s tests" % API_TYPE
    else:
        global API_TYPE
        API_TYPE = sys.argv[1]

    if API_TYPE == 'floodlight':
        from floodlight_sdn_topology import FloodlightSdnTopology as SdnTopology
    elif API_TYPE == 'onos':
        from onos_sdn_topology import OnosSdnTopology as SdnTopology
    st = SdnTopology()

    run_tests(st)