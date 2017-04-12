#!/usr/bin/python

# Based on cli.py from Floodlight's GitHub repo @ https://github.com/floodlight/floodlight/blob/master/example/cli.py

from base_rest_api import BaseRestApi
import sys
import argparse

usage_desc = """
Command descriptions:

    hosts [debug]
    links [tunnellinks]
    port <blocked | broadcast>
    memory
    switches
    switchclusters
    counter [DPID] <name>
    switch_stats [DPID] <port | queue | flow | aggregate | desc | table | features | host>
"""


class FloodlightRestApi(BaseRestApi):
    """REST API helper object for the Floodlight controller."""

    def __init__(self, server, port):
        super(FloodlightRestApi, self).__init__(server, port)

    def get_links(self, link_id=[]):
        """Get all links or a specific one if specified."""
        return self.run_command('links', link_id)

    def get_hosts(self):
        """Get all hosts. Note that we cut off the extraneous 'devices'
        top-most element that Floodlight includes for some reason."""
        path = '/wm/device/'
        return self.get(path)['devices']

    def get_switches(self, switch_id=None):
        """Get all switches (a.k.a. devices)."""
        if switch_id is not None:
            raise NotImplementedError
        return self.run_command('switches')

    # TODO: implement this after figuring out what to do with arg
    # def get_ports(self, ????):
    #     return self.run_command('ports')

    def push_flow_rule(self, rule, switch_id=None):
        """Push the specified static flow rule to the controller for the specified switch."""
        path = '/wm/staticentrypusher/json'
        # Floodlight reads the switch_id from the JSON object so verify it's present
        if 'switch' not in rule:
            if switch_id is None:
                raise ValueError('Must specify switch id for fow rule %s' % rule)
            rule['switch'] = switch_id
        return self.set(path, rule)

    def get_flow_rules(self, switch_id):
        """Get all static flow rules or a specific switch's if specified."""
        path = '/wm/staticflowpusher/list/%s/json' % switch_id
        flows = self.get(path)
        assert isinstance(flows, list), "return value should be a list of dicts for get_flow_rules: need to fix the API!"
        return flows

    def remove_all_flow_rules(self):
        # TODO:
        raise NotImplementedError

    def push_group(self, group, switch_id=None):
        """Push the specified group to the controller for the specified switch."""
        # Floodlight REST API treats groups as a special case of flow rules
        return self.push_flow_rule(group, switch_id)

    def get_groups(self, switch_id):
        """Get all groups or a specific switch's if specified."""
        # Floodlight REST API treats groups as a special case of flow rules
        return self.get_flow_rules(switch_id)

    def run_command(self, cmd, other_args=[]):
        """Only supports GET commands"""
        path = self.lookup_path(cmd, other_args)
        return self.get(path)

    @staticmethod
    def lookup_path(cmd, other_args):
        # TODO: refactor this to defer to common method calls.
        path = ''

        numargs = len(other_args)

        if cmd == 'switch_stats':
            if numargs == 1:
                path = '/wm/core/switch/all/'+ other_args[0]+'/json'
            elif numargs == 2:
                path = '/wm/core/switch/'+ other_args[0]+'/'+ other_args[1]+'/json'

        elif cmd == 'switches':
            path = '/wm/core/controller/switches/json'

        elif cmd == 'counter':
            if numargs == 1:
                path = '/wm/core/counter/'+ other_args[0]+'/json'
            elif numargs == 2:
                path = '/wm/core/counter/'+ other_args[0]+'/'+ other_args[1]+'/json'

        elif cmd == 'memory':
            path = '/wm/core/memory/json'

        elif cmd == 'links':
            if numargs == 0:
                path = '/wm/topology/links/json'
            elif numargs == 1:
                path = '/wm/topology/'+ other_args[0]+'/json'

        elif cmd == 'port' and numargs == 1:
            if other_args[0] == "blocked":
                path = '/wm/topology/blockedports/json'
            elif other_args[0] == "broadcast":
                path = '/wm/topology/broadcastdomainports/json'

        elif cmd == 'switchclusters':
            path = '/wm/topology/switchclusters/json'

        elif cmd == 'hosts':
            path = '/wm/device/'
            if len(other_args) == 1 and other_args[0] == 'debug':
                path = '/wm/device/debug'
        else:
            print usage_desc
            raise NotImplementedError
        return path


def main(argv):
    parser = argparse.ArgumentParser(description='process args', usage=usage_desc)
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', default=8080)
    parser.add_argument('cmd')
    parser.add_argument('otherargs', nargs='*')
    args = parser.parse_args(argv)

    rest = FloodlightRestApi(args.ip, args.port)
    return rest.run_command(args.cmd, args.otherargs)

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) <= 0:
        args.append("switches")
    out = main(args)

    print BaseRestApi.pretty_format_parsed_response(out)
