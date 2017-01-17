#!/usr/bin/python

# Based on cli.py from Floodlight's GitHub repo @ https://github.com/floodlight/floodlight/blob/master/example/cli.py

import sys
import argparse
import json
import requests

usage_desc = """
Command descriptions:

    hosts [id]
    links
    switches
    ports <device_id>
    flows [device_id]
    post_flow <device_id> <json_flow_rule>
    groups [device_id [group_key]]
    post_group <device_id> <json_group>
"""

class BaseRestApi(object):
    """Base (abstract) REST API helper object for the various SDN controllers.
    The general idea is specify the server once and then use simple methods
    to interact with the REST API, including ones specific to things like
    flow rule insertion.  Should also support CLI interaction.

    NOTE that most controllers have quite different APIs and formats.
    Method signatures may vary between controllers (e.g. see get_ports()).
    Do not expect the return value to be the same, or even similar,
    between them.  You will get back the raw return value (assumed JSON) and so
    you may need some sort of adapters (e.g. classes) for meaningful
    interaction with your application (see sdn_topology.py)."""

    def __init__(self, server, port, username=None, password=None):
        super(BaseRestApi, self).__init__()
        self.server = server
        self.port = port
        self.username = username
        self.password = password

        # Using the session object allows connection pooling --> better performance?
        self.session = requests.Session()
        if self.username is not None and self.password is not None:
            self.session.auth = (self.username, self.password)

    # SDN controller-specific methods

    def get_links(self, link_id=None):
        """Get all links or a specific one if specified."""
        raise NotImplementedError
        path = 'path/to/links/here'
        return self.get(path)

    def get_hosts(self):
        """Get all hosts. Subclasses may optionally specify a specific host id."""
        raise NotImplementedError
        path = 'path/to/hosts/here'
        return self.get(path)

    def get_switches(self, switch_id=None):
        """Get all switches (a.k.a. devices) or a specific one if specified."""
        raise NotImplementedError
        path = 'path/to/switches/here'
        return self.get(path)

    # Recommended method, but controllers' implementations and API
    # vary too much to prescribe a signature
    # def get_ports(self, ):

    def push_flow_rule(self, rule, switch_id):
        """Push the specified flow rule to the controller for the specified switch."""
        raise NotImplementedError
        path = 'path/to/flow/insertion/here'
        return self.set(path, rule)

    def get_flow_rules(self, switch_id=None):
        """Get all flow rules or a specific switch's if specified."""
        raise NotImplementedError
        path = 'path/to/flow/request/here'
        return self.get(path)

    def push_group(self, group, switch_id):
        """Push the specified group to the controller for the specified switch."""
        raise NotImplementedError
        path = 'path/to/group/insertion/here'
        return self.set(path, rule)

    def get_groups(self, switch_id=None):
        """Get all groups or a specific switch's if specified."""
        raise NotImplementedError
        path = 'path/to/group/request/here'
        return self.get(path)

    def run_command(self, cmd, other_args):
        """Execute the requested command.  For use with CLI args.
        Should defer to other methods as much as possible."""
        raise NotImplementedError

    # General REST API methods to be used as helpers for the SDN APIs
    # Should not need to overwrite these

    def get(self, path):
        ret = self.rest_call(path, {}, 'GET')
        return ret.json()

    def set(self, path, data):
        ret = self.rest_call(path, data, 'POST')
        return 200 <= ret.status_code < 300

    def remove(self, path, data={}):
        ret = self.rest_call(path, data, 'DELETE')
        return 200 <= ret.status_code < 300

    def rest_call(self, path, data, action):
        # NOTE: we need to do json.dumps(data) as otherwise requests
        # puts the dicts in single-quoted strings, which Floodlight
        # cannot handle.
        req = requests.Request(action, 'http://%s:%s%s' % (self.server, self.port, path),
                               data=json.dumps(data), auth=self.session.auth)
        resp = self.session.send(req.prepare())
        resp.raise_for_status()  # only raises if error
        return resp

    @staticmethod
    def pretty_format_parsed_response(value):
        """Pretty format the given dict value for pretty printing.
        Used mainly for logging/testing."""
        return json.dumps(value, sort_keys=True, indent=2)


def main(argv):
    parser = argparse.ArgumentParser(description='process args', usage=usage_desc)
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', default=8080)
    parser.add_argument('cmd')
    parser.add_argument('otherargs', nargs='*')
    args = parser.parse_args(argv)

    rest = BaseRestApi(args.ip, args.port)
    return rest.run_command(args.cmd, args.otherargs)

if __name__ == '__main__':
    raise NotImplementedError

    args = sys.argv[1:]
    if len(args) <= 0:
        args.append("switches")
    out = main(args)

    # TODO: jsonify sooner?
    print BaseRestApi.pretty_format_parsed_response(out)
    # print "Number of items: " + str(len(out))