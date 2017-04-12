# Test suite for SdnTopology classes

import time

from ipaddress import IPv4Address
HOST_ADDRESS_BASE = IPv4Address(u'10.0.0.1')
H1_ADDRESS = str(HOST_ADDRESS_BASE)
MULTICAST_ADDRESS_BASE = IPv4Address(u"224.0.0.1")
MULTICAST_ADDRESS = str(MULTICAST_ADDRESS_BASE)

global API_TYPE
API_TYPE = 'onos'

# Helper functions for Mininet-based tests

def mac_for_host(host_num):
    """Assuming you started mininet with --mac option, this returns a
    mac address for host h<host_num> e.g. for h1 do mac_for_host(1) --> 00:00:00:00:00:01"""

    # format int as hex with proper number of octets, then add :'s using some pymagic
    num = format(host_num, 'x').rjust(12, '0')
    num = ':'.join(s.encode('hex') for s in num.decode('hex'))
    if API_TYPE == 'onos':
        num = num.upper()
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

def clear_tables_and_wait(switch=None):
    """
    Clears all flow rules and groups from all switches or optionally just the
    one specified switch.  Returns after the changes have been committed.
    :param switch:
    :return:
    """
    st.remove_all_flow_rules()
    st.remove_all_groups()
    ngroups = len(st.get_groups(switch))
    print "waiting for all old groups to clear..."
    while ngroups != 0:
        time.sleep(1)
        ngroups = len(st.get_groups(switch))

#### Tests

def test_basic_flow_installation(st):
    """Test simple flow entries for some edge cases in the APIs"""

    # get # flow rules, check after adding, after deleting, then after adding again
    switch = dpid_for_switch(1)
    assert st.remove_all_flow_rules()  # clear out previous flow rules just in case
    # HACK: ONOS-SPECIFIC: we define this special helper function to only get flows that we have
    # complete control over in this context (static flows for this given switch).
    def __get_relevant_flows():
        flows = st.get_flow_rules(switch)
        # print "%d flows before filter:" % len(flows), flows
        flows = [f for f in flows if f['appId'] == 'org.onosproject.rest']
        # print "%d flows after filter:" % len(flows), flows
        return flows

    nflows = len(__get_relevant_flows())

    matches = st.build_matches(ipv4_src=H1_ADDRESS, eth_type='0x0800')
    actions = st.build_actions(("output", 1))
    rule = st.build_flow_rule(switch, matches, actions)
    assert st.install_flow_rule(rule)

    new_nflows = len(__get_relevant_flows())
    assert new_nflows == nflows+1, "should have one more flow rule (old %d != new %d) after adding one!" % (nflows+1, new_nflows)

    assert st.remove_all_flow_rules()
    new_nflows = len(__get_relevant_flows())
    assert new_nflows == nflows, "should have one less flow rule (old %d != new %d) after deleting one!" % (nflows, new_nflows)

    matches = st.build_matches(ipv4_src=H1_ADDRESS)
    rule = st.build_flow_rule(switch, matches, actions)
    assert st.install_flow_rule(rule), "should be able to install flow rule without specifying eth_type match"
    new_nflows = len(__get_relevant_flows())
    assert new_nflows == nflows+1, "should have one more flow rule (old %d != new %d) after adding one!" % (nflows+1, new_nflows)

    print 'test_basic_flow_installation passed!'


def test_path_flow(st):
    """Test simple static flow entries for a basic path between h1 and h16"""

    # path = st.get_path("10.0.0.1", "10.0.0.16")
    path = st.get_path(id_for_host(1), id_for_host(16))
    # print "Path:", path
    rules = st.build_flow_rules_from_path(path)
    rules.extend(st.build_flow_rules_from_path(list(reversed(path))))
    # print "Rules: %s" % rules
    for rule in rules:
        assert st.install_flow_rule(rule)

    print 'test_path_flow passed!'


def test_group_flows(st):
    """Test the functions used to construct and install groups, including
    adding multiple groups to verify they don't overwrite each other or
    something of the sort."""

    switch = dpid_for_switch(2)

    print "test_group_flows entered: this involves some sleep statements so be patient..."

    # We'll verify the groups have been added by counting the # groups present.
    clear_tables_and_wait(switch)

    buckets = [st.build_bucket(st.build_actions(("output", 3)))]
    gflow = st.build_group(switch, buckets, group_id=3)

    matches = st.build_matches(ipv4_src=H1_ADDRESS)
    actions = st.build_actions(("group", 3))
    flow = st.build_flow_rule(switch, matches, actions, priority=500)
    # print "Flow rules for groups:", flow
    # print "Groups:", gflow
    assert st.install_group(gflow)
    assert st.install_flow_rule(flow)

    print "waiting for added group to show up (something is likely wrong if this lasts >5-10 seconds!)"
    ngroups = -1
    while ngroups != 1:
        time.sleep(1)
        groups = st.get_groups(switch)
        ngroups = len(groups)
        # print st.rest_api.pretty_format_parsed_response(groups)

    # install a second group flow rule and verify that it's reflected in the flow tables
    buckets = [st.build_bucket(st.build_actions(("output", 1)))]
    gflow = st.build_group(switch, buckets, group_id=4)

    matches = st.build_matches(ipv4_src='10.0.0.2')
    actions = st.build_actions(("group", 4))
    flow = st.build_flow_rule(switch, matches, actions, priority=500)
    # print "Flow rules for groups:", flow
    # print "Groups:", gflow
    assert st.install_group(gflow)
    assert st.install_flow_rule(flow)

    print "waiting for added group to show up (something is likely wrong if this lasts >5-10 seconds!)"
    ngroups = -1
    while ngroups != 2:
        time.sleep(1)
        groups = st.get_groups(switch)
        ngroups = len(groups)
        # print st.rest_api.pretty_format_parsed_response(groups)

    # Now we want to add a group with the same group ID to a different switch
    # and verify that this works as expected.
    switch = dpid_for_switch(1)
    gflow = st.build_group(switch, buckets, group_id=4)

    matches = st.build_matches(ipv4_src='10.0.0.2')
    actions = st.build_actions(("group", 4))
    flow = st.build_flow_rule(switch, matches, actions, priority=500)
    assert st.install_group(gflow)
    assert st.install_flow_rule(flow)

    print "waiting for added group to show up (something is likely wrong if this lasts >5-10 seconds!)"
    ngroups = -1
    while ngroups != 3:
        time.sleep(1)
        groups = st.get_groups()
        ngroups = len(groups)
        # print st.rest_api.pretty_format_parsed_response(groups)

    # CLEAN UP
    st.remove_all_groups()
    ngroups = len(st.get_groups())
    print "CLEANUP: waiting for all old groups to clear..."
    while ngroups != 0:
        time.sleep(1)
        ngroups = len(st.get_groups())

    print 'test_group_flows passed!'


def test_mcast_flows(st):
    mcast_tree = st.get_multicast_tree(id_for_host(1), [id_for_host(2), id_for_host(9), id_for_host(15)])
    # print list(mcast_tree.nodes())
    matches = st.build_matches(ipv4_src=H1_ADDRESS, ipv4_dst=MULTICAST_ADDRESS, eth_type='0x0800')
    gflows, flows = st.build_flow_rules_from_multicast_tree(mcast_tree, id_for_host(1), matches)
    # print "Mcast flows: %s" % json.dumps(flows)
    for gf in gflows:
        # print json.dumps(gf)
        assert st.install_group(gf)
    for flow in flows:
        # print json.dumps(flow)
        assert st.install_flow_rule(flow), "problem installing flow: %s" % flow

    # TODO: programmatically verify the expected rules are now present on the switch's flow tables...

    # Also verify that we can request a multicast tree containing an unknown host without
    # crashing the whole process.
    mcast_tree = st.get_multicast_tree(id_for_host(1), [id_for_host(2), id_for_host(9), id_for_host(15), id_for_host(222)])
    # print list(mcast_tree.nodes())
    matches = st.build_matches(ipv4_src=H1_ADDRESS, ipv4_dst=MULTICAST_ADDRESS, eth_type='0x0800')
    gflows, flows = st.build_flow_rules_from_multicast_tree(mcast_tree, id_for_host(1), matches, group_id=5)
    # print "Mcast flows: %s" % json.dumps(flows)
    for gf in gflows:
        # print json.dumps(gf)
        assert st.install_group(gf)
    for flow in flows:
        # print json.dumps(flow)
        assert st.install_flow_rule(flow), "problem installing flow: %s" % flow

    print 'test_mcast_flows passed!'

def test_multiple_mcast_trees(st):
    """Builds and installs 2 multicast trees to verify the disjoint mcast functionality works"""

    clear_tables_and_wait()

    ntrees = 2
    destinations = [id_for_host(2), id_for_host(9), id_for_host(15)]
    root = id_for_host(1)
    trees = st.get_redundant_multicast_trees(root, destinations, k=ntrees)
    for i, t in enumerate(trees):
        matches = st.build_matches(ipv4_dst=str(MULTICAST_ADDRESS_BASE+i))
        groups, flow_rules = st.build_flow_rules_from_multicast_tree(t, root, matches, group_id=i+10)
        for g in groups:
            assert st.install_group(g)
        for fr in flow_rules:
            assert st.install_flow_rule(fr)

    print 'multiple mcast trees configured: verify them in Mininet by using netcat on each host in', destinations
    print 'test_multiple_mcast_trees passed!'


def test_utils(st):
    """Test the various smaller helper functions that
    essentially adapt to the specific controller's REST API."""
    assert st.get_ip_address(id_for_host(1)) == H1_ADDRESS

    assert st.get_ports_for_nodes(id_for_host(1), dpid_for_switch(2)) == (0,1)
    assert st.get_ports_for_nodes(dpid_for_switch(1), dpid_for_switch(2)) == (1,5)

    print 'test_utils passed!'


def test_flow_helpers(st):
    """Test the helper functions used to construct flow rules"""
    if API_TYPE == "floodlight":
        # actions
        a1 = st.build_actions(('output', 2))
        assert a1 == "output=2"
        a2 = st.build_actions('strip_vlan', ('output', 2))
        assert a2 == "strip_vlan,output=2"
        a3 = st.build_actions(('strip_vlan'), ('set_ipv4_dst', '10.0.0.1'), ('output', 2))
        assert a3 == "strip_vlan,set_field=ipv4_dst->10.0.0.1,output=2"

        # matches
        m1 = st.build_matches(ipv4_src="10.0.0.3", in_port=3)
        assert m1 == {"in_port": "3", "ipv4_src": "10.0.0.3"}

        # TODO: whole flows
    elif API_TYPE == "onos":
        # actions
        a1 = st.build_actions(('output', 2))
        assert a1 == [{"type": "OUTPUT", "port": "2"}]
        a2 = st.build_actions(('set_eth_dst', "00:11:22:33:44:55"), ('output', 2))
        assert a2 == [{"type": "L2MODIFICATION", "subtype": "ETH_DST", "mac": "00:11:22:33:44:55"}, {"type": "OUTPUT", "port": "2"}]

        # matches
        m1 = st.build_matches(ipv4_src="10.0.0.3", in_port=3)
        # NOTE: need to sort since kwargs aren't ordered
        assert sorted(m1) == sorted([{"type": "IN_PORT", "port": 3}, {"type": "IPV4_SRC", "ip": "10.0.0.3/32"},
                                     {"type": "ETH_TYPE", "ethType": "0x0800"}])

        # whole flows
        f1 = st.build_flow_rule("of:000000000021", m1, a1, priority=100)
        assert f1 == {"priority": 100, "isPermanent": True, "deviceId": "of:000000000021",
                      "treatment": {"instructions": a1}, "selector": {"criteria": m1}}
        try:
            f1 = st.build_flow_rule("of:000000000021", m1, a1)
            assert 'priority' in f1, "flow rules MUST include a priority!"
        except ValueError:
            pass

    print "test_flow_helpers() passed!"


def test_group_helpers(st):
    """Test the helper functions used to construct groups"""
    if API_TYPE == 'floodlight':
        # TODO: add some unit tests here instead of relying on test_group_flows()
        pass
    elif API_TYPE == 'onos':
        a1 = st.build_actions(('output', 2))
        a2 = st.build_actions(('set_eth_dst', "00:11:22:33:44:55"), ('output', 2))
        b1 = st.build_bucket(a1, weight=0.5)
        assert b1 == {"treatment": {"instructions": a1}, "weight": 0.5}
        b2 = st.build_bucket(a2, watch_group=5)
        assert b2 == {"treatment": {"instructions": a2}, "watchGroup": "5"}

        switch = dpid_for_switch(1)
        g = st.build_group(switch, [b1, b2])
        assert g == {
            "type": "ALL",
            "deviceId": switch,
            # HACK: we probably shouldn't be replicating this logic to test it here but...
            "appCookie": "0x%s%s%s" % (switch[3:], st.MAGIC_DELIMETER_appCookie, "1"),
            "groupId": "1",
            "buckets": [b1, b2]
        }, g

    print 'test_group_helpers passed!'


def run_tests(st):
    # These tests assume running mininet locally in the following way
    # AFTER starting Floodlight:
    # sudo mn --controller remote --mac --topo=tree,2,4
    #
    # THEN run 'pingall 10' in Mininet to populate the
    # hosts' IP addresses in Floodlight

    test_utils(st)
    test_flow_helpers(st)
    test_basic_flow_installation(st)
    test_path_flow(st)

    test_group_helpers(st)
    test_group_flows(st)
    test_mcast_flows(st)
    test_multiple_mcast_trees(st)

    print "ALL TESTS PASSED!!"

if __name__ == '__main__':
    # Uncomment to enable other modules' logging facilities
    # import logging as log
    # log.basicConfig(format='%(levelname)s:%(message)s', level=log.DEBUG)

    import sys
    valid_args = ['floodlight', 'onos']
    if len(sys.argv) < 2 or sys.argv[1] not in valid_args:
        print "Usage: argument must be one of %s" % valid_args
        print "Defaulting to %s tests" % API_TYPE
    else:
        API_TYPE = sys.argv[1]

    if API_TYPE == 'floodlight':
        from floodlight_sdn_topology import FloodlightSdnTopology as SdnTopology
    elif API_TYPE == 'onos':
        from onos_sdn_topology import OnosSdnTopology as SdnTopology
    st = SdnTopology()

    run_tests(st)