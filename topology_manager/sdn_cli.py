#! /usr/bin/python
from __future__ import print_function

from onos_sdn_topology import OnosSdnTopology

SDN_CLI_DESCRIPTION = '''CLI-based interface to the various sdn_topology implementations.
Useful for configuring an SDN controller to install flows without having to manually type them out
and properly format them.'''

# @author: Kyle Benson
# (c) Kyle Benson 2017

import argparse
import logging
#from os.path import isdir
#from os import listdir
#from getpass import getpass
#password = getpass('Enter password: ')

def parse_args(args):
##################################################################################
#################      ARGUMENTS       ###########################################
# ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
# action is one of: store[_const,_true,_false], append[_const], count
# nargs is one of: N, ?(defaults to const when no args), *, +, argparse.REMAINDER
# help supports %(var)s: help='default value is %(default)s'
# Mutually exclusive arguments:
# group = parser.add_mutually_exclusive_group()
# group.add_argument(...)
##################################################################################

    parser = argparse.ArgumentParser(description=SDN_CLI_DESCRIPTION,
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     #epilog='Text to display at the end of the help print',
                                     #parents=[parent1,...], # add parser args from these ArgumentParsers
                                     # NOTE: for multiple levels of arg
                                     # scripts, will have to use add_help=False
                                     # or may consider using parse_known_args()
                                     )

    parser.add_argument('--debug', '-d', action='store_true',
                        help='''enable debug-level logging for all modules (has global effect)''')

    # Configuring the manager itself
    parser.add_argument('--type', type=str, default='onos',
                        help='''type of SDN topology manager to run operations on (default=%(default)s)''')
    parser.add_argument('--ip', type=str, default='localhost',
                        help='''SDN controller's host address (default=%(default)s)''')
    parser.add_argument('--port', '-p', type=int, default=8181,
                        help='''SDN controller's REST API port (default=%(default)s)''')

    # Displaying info
    parser.add_argument('command', default=['hosts'], nargs='*',
                        help='''command to execute can be one of (default=%(default)s):
    hosts [include_attributes]          - print the available hosts (with attributes if optional argument is yes/true; NO *ARGS/**KWARGS!)
    switches|devices                    - print the available switches (alias: devices)
    path src dst                        - build and install a path between src and dst using flow rules
    m[ulti]cast addr src dst1 [dst2...] - build and install a multicast tree for IP address 'addr' from src to all of dst1,2... using flow rules (NO *ARGS/**KWARGS!)
    mdmts ntrees alg <mcast_args>       - same as mcast, except it builds 'ntrees' multiple maximally-disjoint multicast trees using the algorithm 'alg'
    redirect src old_dst new_dst        - redirect packets from src to old_dst by installing flow rules that convert ipv4_dst to that of new_dst
    del-flow switch_id flow_id          - delete the requested flow rule
    del-flows                           - deletes all flow rules
    del-groups                          - deletes all groups

    NOTE: the arguments for most commands with optional parameters will be passed as
     *args positional arguments to the relevant function call,
      so please see their method signatures for more details.
    NOTE: you can also specify **kwargs (for those commands that don't say otherwise) by doing e.g.:
    redirect source=<src_ip> old_dest=<old_dst_ip> new_dest=<new_dst_ip>
    The **kwargs may be passed directly to the relevant function or they may be passed to build_flow_rules[...]()
    e.g. specifying the 'priority' does the latter
    ''')


    # joins logging facility with argparse
    # parser.add_argument('--debug', '-d', type=str, default='info', nargs='?', const='debug',
    #                     help=''set debug level for logging facility (default=%(default)s, %(const)s when specified with no arg)'')


    return parser.parse_args(args)

if __name__ == "__main__":
    import sys
    args = parse_args(sys.argv[1:])

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)

    if args.type == 'onos':
        topo = OnosSdnTopology(ip=args.ip, port=args.port)
    else:
        raise ValueError("unrecognized / unsupported SdnTopology implementation of type: %s" % args.type)

    # Extract the relevant command and any positional/keyword arguments to its corresponding function call.
    cmd = args.command[0]
    cmd_args = []
    cmd_kwargs = {}
    for a in args.command[1:]:
        if '=' in a:
            first, second = a.split('=')
            cmd_kwargs[first] = second
        else:
            cmd_args.append(a)
    nargs = len(cmd_args) + len(cmd_kwargs)

    if cmd == 'hosts':
        include_attrs = True if (cmd_args and cmd_args[0].lower() in ('y', 'yes', 't', 'true')) else False
        if include_attrs:
            print("Hosts:\n%s" % '\n'.join(str(h) for h in topo.get_hosts(attributes=True)))
        else:
            print("Hosts:\n%s" % '\n'.join(topo.get_hosts()))

    elif cmd == 'switches' or cmd == 'devices':
        include_attrs = True if (cmd_args and cmd_args[0].lower() in ('y', 'yes', 't', 'true')) else False
        if include_attrs:
            print("Hosts:\n%s" % '\n'.join(str(h) for h in topo.get_switches(attributes=True)))
        else:
            print("Hosts:\n%s" % '\n'.join(topo.get_hosts()))

    elif cmd == 'path':
        assert nargs >= 2, "path command must at least have the 2 hosts specified!"
        if 'weight' in cmd_kwargs:
            path = topo.get_path(*cmd_args, weight=cmd_kwargs.pop('weight'))
        else:
            path = topo.get_path(*cmd_args)
        print("Installing Path:", path)
        rules = topo.build_flow_rules_from_path(path, **cmd_kwargs)
        rules.extend(topo.build_flow_rules_from_path(list(reversed(path)), **cmd_kwargs))
        for rule in rules:
            assert topo.install_flow_rule(rule), "error installing rule: %s" % rule

    elif cmd == 'mcast' or cmd == 'multicast':
        assert nargs >= 3, "must specify an IP address and at least 2 host IDs to build a multicast tree!"
        # TODO: validate that first arg is an IP address...
        src_host = cmd_args[1]
        mcast_tree = topo.get_multicast_tree(src_host, cmd_args[2:])
        address = cmd_args[0]

        print("Installing multicast tree:", list(mcast_tree.nodes()))
        matches = topo.build_matches(ipv4_src=topo.get_ip_address(src_host), ipv4_dst=address, eth_type='0x0800')
        gflows, flows = topo.build_flow_rules_from_multicast_tree(mcast_tree, src_host, matches)

        print("installing groups:", gflows)
        for gf in gflows:
            assert topo.install_group(gf)

        print("installing flows:", flows)
        for flow in flows:
            assert topo.install_flow_rule(flow), "problem installing flow: %s" % flow

    elif cmd == 'mdmts':
        assert nargs >= 5, "must specify #MDMTs, an MDMT-construction algorithm, an IP address," \
                           " and at least 2 host IDs to build a disjoint multicast trees!"

        ntrees = int(cmd_args[0])
        algorithm = cmd_args[1]
        address = cmd_args[2]
        src_host = cmd_args[3]
        dests = cmd_args[4:]

        mdmts = topo.get_redundant_multicast_trees(src_host, destinations=dests, k=ntrees, algorithm=algorithm)
        for i, mcast_tree in enumerate(mdmts):
            print("Installing multicast tree:", list(mcast_tree.nodes()))
            matches = topo.build_matches(ipv4_src=topo.get_ip_address(src_host), ipv4_dst=address, eth_type='0x0800')
            gflows, flows = topo.build_flow_rules_from_multicast_tree(mcast_tree, src_host, matches, group_id=i+10)

            print("installing groups:", gflows)
            for gf in gflows:
                assert topo.install_group(gf)

            print("installing flows:", flows)
            for flow in flows:
                assert topo.install_flow_rule(flow), "problem installing flow: %s" % flow

    elif cmd == 'redirect':
        assert nargs >= 3, "redirect command must at least have the source, old_destination, and new_destination hosts specified!"
        if len(cmd_args) >= 3:
            print("Redirecting packets from %s originally bound for %s to instead go to %s" % (cmd_args[0], cmd_args[1], cmd_args[2]))

        flow_rules = topo.build_redirection_flow_rules(*cmd_args, **cmd_kwargs)
        for f in flow_rules:
            assert topo.install_flow_rule(f), "problem installing flow: %s" % f

    elif cmd == 'del-flow':
        assert nargs >= 2, "delete flow command must at least have the switchId and flowId!"
        topo.remove_flow_rule(*cmd_args, **cmd_kwargs)

    elif cmd == 'del-flows':
        topo.remove_all_flow_rules()
    elif cmd == 'del-groups':
        topo.remove_all_groups()

    else:
        print("ERROR: unrecognized command: %s\nrest of args were: %s" % (cmd, args.command[1:]))

    # enables logging for all classes
    # log_level = log.getLevelName(args.debug.upper())
    # log.basicConfig(format='%(levelname)s:%(message)s', level=log_level)