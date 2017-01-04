#!/usr/bin/python

# Based on cli.py from Floodlight's GitHub repo @ https://github.com/floodlight/floodlight/blob/master/example/cli.py

import sys
import argparse
import json
import httplib
# import urllib2

usage_desc = """
Command descriptions:

    host [debug]
    link [tunnellinks]
    port <blocked | broadcast>
    memory
    switch
    switchclusters
    counter [DPID] <name>
    switch_stats [DPID] <port | queue | flow | aggregate | desc | table | features | host>
"""


class RestApi(object):
    """REST API helper object for the Floodlight controller."""

    def __init__(self, server, port):
        self.server = server
        self.port = port

    def push_flow_rule(self, rule):
        path = '/wm/staticentrypusher/json'
        return self.set(path, rule)

    def get(self, path):
        ret = self.rest_call(path, {}, 'GET')
        return ret[2]
        # f = urllib2.urlopen('http://'+self.server+':'+str(self.port)+path)
        # ret = f.read()
        # return json.loads(ret)

    def set(self, path, data):
        ret = self.rest_call(path, data, 'POST')
        return ret[0] == 200

    def remove(self, objtype, data):
        ret = self.rest_call(data, 'DELETE')
        return ret[0] == 200

    def rest_call(self, path, data, action):
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json',
            }
        body = json.dumps(data)
        conn = httplib.HTTPConnection(self.server, self.port)
        conn.request(action, path, body, headers)
        response = conn.getresponse()
        ret = (response.status, response.reason, response.read())
        conn.close()
        # print str(ret[2])
        return ret

    @staticmethod
    def lookup_path(cmd, other_args):
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

        elif cmd == 'link':
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
            path = ''
            exit(0)
        return path


def main(argv):
    parser = argparse.ArgumentParser(description='process args', usage=usage_desc)
    parser.add_argument('--ip', default='localhost')
    parser.add_argument('--port', default=8080)
    parser.add_argument('cmd')
    parser.add_argument('otherargs', nargs='*')
    args = parser.parse_args(argv)

    rest = RestApi(args.ip, args.port)
    path = rest.lookup_path(args.cmd, args.otherargs)
    out = rest.get(path)
    return out

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) <= 0:
        args.append("switches")
    out = main(args)

    # TODO: jsonify sooner?
    print json.dumps(json.loads(out), sort_keys=True, indent=4)
    # print "Number of items: " + str(len(out))
