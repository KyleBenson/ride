#!/usr/bin/python

# NOTE: to view documentation for the ONOS REST API, navigate to http://localhost:8181/onos/v1/docs/
# (assuming you're running ONOS locally on port 8181 or at least have port forwarding to your VM).

# Based on cli.py from Floodlight's GitHub repo @ https://github.com/floodlight/floodlight/blob/master/example/cli.py

import sys
import argparse
import json

from base_rest_api import BaseRestApi

usage_desc = """
Command descriptions:

    hosts [id]
    links
    devices [device_id]
    switches (alias of devices)
    ports <device_id>
    flows [device_id]
    post_flow <device_id> <json_flow_rule>
    del_flows
    groups [device_id [group_key]]
    post_group <device_id> <json_group>
    del_group <device_id> <group_key>
    paths <src_element_id> <dst_element_id>
    intents [app_id intent_id]
    post_intent <json_intent>
    apps
    statistics (not yet implemented)
"""


class OnosRestApi(BaseRestApi):
    """REST API helper object for the ONOS controller."""

    def __init__(self, server, port, username='karaf', password='karaf'):
        super(OnosRestApi, self).__init__(server, port, username, password)
        self.base_path = "/onos/v1"

    def get_links(self, link_id=None):
        """Get all links.  Specifying a specific one is undefined for current ONOS API."""
        if link_id is not None:
            raise NotImplementedError
        path = self.base_path + '/links'
        return self.get(path)['links']

    # TODO: verify this ovveriding works properly
    def get_hosts(self, host_id=None):
        """Get all hosts or a specific one."""
        path = self.base_path + '/hosts'
        if host_id is not None:
            path += '/%s' % host_id
        return self.get(path)['hosts']

    def get_switches(self, switch_id=None):
        """Get all switches (a.k.a. devices) or a specific one if specified."""
        path = self.base_path + '/devices'
        if switch_id is not None:
            path += '/%s' % switch_id
        return self.get(path)['devices']

    def get_ports(self, switch_id):
        """Get all ports for the specified switch device."""
        path = '%s/devices/%s/ports' % (self.base_path, switch_id)
        return self.get(path)

    def push_flow_rule(self, rule, switch_id=None):
        """Push the specified flow rule to the controller for the specified switch.
        see the following links for documentation about flow rules:
        https://wiki.onosproject.org/display/ONOS/Flow+Rule+Instructions
        https://wiki.onosproject.org/display/ONOS/Flow+Rule+Criteria
        empty 'treatment' drops packets
        priority and isPermanent are only truly required fields"""

        # First, verify we set switch_id and/or the deviceId field in the rule correctly
        # We at least need to build the URL path correctly,
        # but don't NEED to include the deviceId in the JSON
        if switch_id is None:
            if 'deviceId' not in rule:
                raise ValueError('Must specify switch id for fow rule %s' % rule)
            switch_id = rule['deviceId']

        path = '%s/flows/%s' % (self.base_path, switch_id)
        return self.set(path, rule)

    def get_flow_rules(self, switch_id=None):
        """Get all flow rules or a specific switch's if specified.
        NOTE: check the flow rule's 'state' field if you previously
        deleted it and don't expect to see it as sometimes it will
        be 'PENDING_REMOVE'"""
        path = self.base_path + '/flows'
        if switch_id is not None:
            path += '/%s' % switch_id
        flows = self.get(path)['flows']
        # Need to filter out any flows that haven't been fully deleted yet to avoid confusing users.
        flows = [f for f in flows if f['state'] != 'PENDING_REMOVE']
        return flows

    def remove_flow_rule(self, switch_id, flow_id):
        """Removes the requested flow rule (installed using this REST API) from the specified switch."""
        # Return value of flow ID from get_flow_rules is in hex, but removal requires int so we need to do a
        # conversion if the user specified it as a hex string
        if isinstance(flow_id, basestring):
            flow_id = int(flow_id, 16 if flow_id.startswith('0x') else 10)

        path = self.base_path + '/flows/%s/%s' % (switch_id, flow_id)
        return self.remove(path)

    def remove_all_flow_rules(self):
        # Can delete all flows by just specifying to delete all rules from the rest api app
        path = self.base_path + '/flows/application/org.onosproject.rest'
        return self.remove(path)

    def push_group(self, group, switch_id=None):
        """Push the specified group to the controller for the specified switch."""
        # First, verify we set switch_id and/or the deviceId field in the rule correctly
        # We at least need to build the URL path correctly,
        # but don't NEED to include the deviceId in the JSON
        if switch_id is None:
            if 'deviceId' not in group:
                raise ValueError('Must specify switch id for fow rule %s' % group)
            switch_id = group['deviceId']

        path = '%s/groups/%s' % (self.base_path, switch_id)
        return self.set(path, group)

    def get_groups(self, switch_id=None):
        """
        Get all groups or a specific switch's if specified.
        WARNING: do not try to add a group with an ID (appCookie?) matching another group
         that is currently of 'state' 'PENDING_DELETE' as otherwise your new group will
         be deleted in a few seconds!
        :return List[dict] groups:
        """
        path = self.base_path + '/groups'
        if switch_id is not None:
            path += '/%s' % switch_id
        groups = self.get(path)['groups']
        # ENHANCE: could filter out any groups that haven't been fully deleted yet to avoid confusing users.
        # The problem is that we cannot add a group with the same ID as one pending deletion as otherwise
        # the group will just be deleted in a few seconds.
        # groups = [g for g in groups if g['state'] != 'PENDING_DELETE']
        return groups

    def get_group_key(self, group):
        """
        Gets the unique group key from the given group.
        :type dict group:
        :return:
        """
        return group['appCookie']

    def remove_all_groups(self, switch_id=None):
        """Remove all groups or optionally all groups from the specified switch."""
        # we have to do this by iteratively removing each individual group unfortunately
        # TODO: verify this will delete any flow rules associated with the group!

        if switch_id is None:
            switches_to_clear = [s['id'] for s in self.get_switches()]
        else:
            switches_to_clear = [switch_id]

        ret = True
        for s in switches_to_clear:
            groups = self.get_groups(s)
            group_keys = [self.get_group_key(g) for g in groups]
            for k in group_keys:
                r = self.remove_group(s, k)
                ret = r and ret

        return ret

    def remove_group(self, switch_id, group_key):
        """Remove the specified group.
        :param switch_id: switch's DPID
        :param group_key: controller-specific ID for groups (NOT necessarily the groupId!)
        """
        path = '%s/groups/%s/%s' % (self.base_path, switch_id, group_key)
        return self.remove(path)

    # ONOS-specific methods

    def get_paths(self, src_device_id, dst_device_id):
        """'Gets set of pre-computed shortest paths between the
        specified source and destination network elements.'
        NOTE: hosts are not network elements!"""
        path = '%s/paths/%s/%s' % (self.base_path, src_device_id, dst_device_id)
        return self.get(path)

    def push_intent(self, intent):
        """Push the specified intent to the controller."""
        path = self.base_path + '/intents'
        return self.set(path, intent)

    def get_intents(self, app_id=None, intent_id=None):
        """Get all intents or a specific one if specified."""
        path = self.base_path + '/intents'
        if app_id is not None and intent_id is not None:
            path += '/%s/%s' % (app_id, intent_id)
        return self.get(path)

    def get_apps(self):
        """Get all installed applications."""
        path = self.base_path + '/applications'
        return self.get(path)

    def get_statistics(self):
        """Get port statistics of all devices."""
        # TODO: support rest of ONOS REST API for statistics (need to feed args to this function)
        path = self.base_path + '/statistics/ports'
        return self.get(path)

    # ENHANCE: could add flow objectives, meters, component configuration

    def run_command(self, cmd, other_args):
        if cmd == 'hosts':
            return self.get_hosts(*other_args)
        elif cmd == 'links':
            return self.get_links(*other_args)
        elif cmd == 'switches' or cmd == 'devices':
            return self.get_switches(*other_args)
        elif cmd == 'ports':
            return self.get_ports(*other_args)
        elif cmd == 'flows':
            return self.get_flow_rules(*other_args)
        elif cmd == 'post_flow':
            return self.push_flow_rule(*other_args)
        elif cmd == 'del_flows':
            return self.remove_all_flow_rules()
        elif cmd == 'groups':
            return self.get_groups(*other_args)
        elif cmd == 'post_group':
            return self.push_group(other_args[1], other_args[0])
        elif cmd == 'del_group':
            return self.remove_group(*other_args)
        elif cmd == 'del_groups':
            return self.remove_all_groups()
        elif cmd == 'paths':
            return self.get_paths(*other_args)
        elif cmd == 'intents':
            return self.get_intents(*other_args)
        elif cmd == 'post_intent':
            return self.push_intent(*other_args)
        elif cmd == 'apps':
            return self.get_apps()
        elif cmd == 'statistics':
            return self.get_statistics()
        else:
            print usage_desc
            exit(0)


def main(argv):
    parser = argparse.ArgumentParser(description='process args', usage=usage_desc)
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', default=8181)
    parser.add_argument('--username', '-u', default='karaf')
    parser.add_argument('--password', '-p', default='karaf')
    parser.add_argument('cmd')
    parser.add_argument('otherargs', nargs='*')
    args = parser.parse_args(argv)

    rest = OnosRestApi(args.ip, args.port, username=args.username, password=args.password)
    return rest.run_command(args.cmd, args.otherargs)

if __name__ == '__main__':
    other_args = sys.argv[1:]
    if len(other_args) <= 0:
        other_args.append("switches")
    out = main(other_args)

    # TODO: jsonify sooner?
    print BaseRestApi.pretty_format_parsed_response(out)
    # print "Number of items: " + str(len(out))
