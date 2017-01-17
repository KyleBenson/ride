#!/usr/bin/python

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
    groups [device_id [group_key]]
    post_group <device_id> <json_group>
    paths <src_element_id> <dst_element_id>
    intents [app_id intent_id]
    post_intent <json_intent>
    apps
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
        """Get all flow rules or a specific switch's if specified."""
        path = self.base_path + '/flows'
        if switch_id is not None:
            path += '/%s' % switch_id
        # ENHANCE: optionally add specific flow rule?  Maybe that gives statistics?
        return self.get(path)

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
        """Get all groups or a specific switch's if specified."""
        path = self.base_path + '/groups'
        if switch_id is not None:
            path += '/%s' % switch_id
        # ENHANCE: add specific group ID?  Maybe needed for more detail?
        return self.get(path)

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
        elif cmd == 'groups':
            return self.get_groups(*other_args)
        elif cmd == 'post_group':
            return self.push_group(*other_args)
        elif cmd == 'paths':
            return self.get_paths(*other_args)
        elif cmd == 'intents':
            return self.get_intents(*other_args)
        elif cmd == 'post_intent':
            return self.push_intent(*other_args)
        elif cmd == 'apps':
            return self.get_apps(*other_args)
        else:
            print usage_desc
            exit(0)


def main(argv):
    parser = argparse.ArgumentParser(description='process args', usage=usage_desc)
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', default=8181)
    parser.add_argument('cmd')
    parser.add_argument('otherargs', nargs='*')
    args = parser.parse_args(argv)

    rest = OnosRestApi(args.ip, args.port)
    return rest.run_command(args.cmd, args.otherargs)

if __name__ == '__main__':
    other_args = sys.argv[1:]
    if len(other_args) <= 0:
        other_args.append("switches")
    out = main(other_args)

    # TODO: jsonify sooner?
    print BaseRestApi.pretty_format_parsed_response(out)
    # print "Number of items: " + str(len(out))
