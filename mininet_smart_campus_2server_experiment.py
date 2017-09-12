#! /usr/bin/python

# @author: Kyle Benson
# (c) Kyle Benson 2017
import os

import errno
import re

from subprocess import Popen

from smart_campus_experiment import SmartCampusExperiment, DISTANCE_METRIC

import logging as log
import json
import argparse
import time
import signal
import ipaddress

from mininet.net import Mininet
from mininet.node import RemoteController, Host, OVSKernelSwitch
from mininet.node import Switch, Link, Node  # these just used for types in docstrings
from mininet.cli import CLI
from mininet.link import TCLink, Intf

from topology_manager.networkx_sdn_topology import NetworkxSdnTopology
from topology_manager.test_sdn_topology import mac_for_host  # used for manual MAC assignment
from ride.ride_c import RideC


EXPERIMENT_DURATION = 120  # in seconds
# EXPERIMENT_DURATION = 10  # for testing
CLOUD1_BREAK_EVENT_DELAY = 60  #
CLOUD2_BREAK_EVENT_DELAY = 90
OPENFLOW_CONTROLLER_PORT = 6653  # we assume the controller will always be at the default port
# subnet for all hosts (if you change this, update the __get_ip_for_host() function!)
# NOTE: we do /9 so as to avoid problems with addressing e.g. the controller on the local machine
# (vagrant uses 10.0.2.* for VM's IP address).
IP_SUBNET = '10.128.0.0/9'
NAT_SERVER_IP_ADDRESS = '11.0.0.%d/24'
# TODO: use a different address base...

SLEEP_TIME_BETWEEN_RUNS = 5  # give Mininet/OVS/ONOS a chance to reconverge after cleanup
DEFAULT_TOPOLOGY_ADAPTER = 'onos'

class MininetSmartCampus2ServerExperiment(SmartCampusExperiment):
    """
    Version of SmartCampusExperiment that runs the experiment in Mininet emulation.
    This includes some background traffic and....

    It outputs the following files (where * is a string representing a summary of experiment parameters):
      - results_*.json : the results file output by this experiment that contains all of the parameters
          and information about publishers/subscribers/the following output locations for each experimental run
      - outputs_*/client_events_{$HOST_ID}.json : contains events sent/recvd by seismic client
      - logs_*/{$HOST_ID} : log files storing seismic client/server's stdout/stderr
      NOTE: the folder hierarchy is important as the results_*.json file contains relative paths pointing
          to the other files from its containing directory.
    """

    def __init__(self, controller_ip='11.0.0.2', controller_port=8181,
                 topology_adapter=DEFAULT_TOPOLOGY_ADAPTER,
                 show_cli=False,
                 *args, **kwargs):
        """

        """

        super(MininetSmartCampus2ServerExperiment, self).__init__(*args, **kwargs)

        self.controller_ip = controller_ip
        self.controller_port = controller_port
        self.topology_adapter_type = topology_adapter
        # set later as it needs resetting between runs and must be created after the network starts up
        self.topology_adapter = None
        # This gets passed to seismic hosts
        self.debug_level = kwargs.get('debug', 'error')

        # These will all be filled in by calling setup_mininet()
        #TODO: do we actually need all these???
        self.hosts = []
        self.switches = []
        self.gateways = []
        self.links = []
        self.net = None
        self.controller = None
        self.nat = None

        self.server = None
        self.server_switch = None
        self.pingers = []
        # Save Popen objects to later ensure procs terminate before exiting Mininet
        # or we'll end up with hanging procs.
        self.popens = []
        # Need to save client/server iperf procs separately as we need to terminate the server ones directly.
        self.client_iperfs = []
        self.server_iperfs = []

        # We'll drop to a CLI after the experiment completes for
        # further poking around if we're only doing a single run.
        self.show_cli = self.nruns == 1 or show_cli

        # HACK: We just manually allocate IP addresses rather than adding a controller API to request them.


    @classmethod
    def get_arg_parser(cls, parents=(SmartCampusExperiment.get_arg_parser(),), add_help=True):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        # argument parser that can be combined with others when this class is used in a script
        # need to not add help options to use that feature, though
        # TODO: document some behavior that changes with the Mininet version:
        # -- pubs/subs are actual client processes
        arg_parser = argparse.ArgumentParser(parents=parents, add_help=add_help)
        # experimental treatment parameters: all taken from parents
        # background traffic generation
        arg_parser.add_argument('--ngenerators', '-g', default=0, dest='n_traffic_generators', type=int,
                                help='''number of hosts that generate random traffic to cause congestion (default=%(default)s)''')
        arg_parser.add_argument('--generator-bandwidth', '-bw', default=10, dest='traffic_generator_bandwidth', type=float,
                                help='''bandwidth (in Mbps) of iperf for congestion traffic generating hosts (default=%(default)s)''')
        arg_parser.add_argument('--cli', '-cli', dest='show_cli', action='store_true',
                                help='''force displaying the Mininet CLI after running the experiment. Normally it is
                                 only displayed iff nruns==1. This is useful for debugging problems as it prevents
                                the OVS/controller state from being wiped after the experiment.''')
        arg_parser.add_argument('--comparison', default=None,
                                help='''use the specified comparison strategy rather than RIDE-D.  Can be one of:
                                 unicast (send individual unicast packets to each subscriber),
                                 oracle (modifies experiment duration to allow server to retransmit aggregated
                                 packets enough times that the SDN controller should detect failures and recover paths).''')

        return arg_parser


    def setup_topology(self):
        """
        Builds the Mininet network, including all hosts, servers, switches, links, and NATs.
        This relies on reading the topology file using a NetworkxSdnTopology helper.

        NOTE: we assume that the topology file was generated by (or follows the same naming conventions as)
        the campus_topo_gen.py module.  In particular, the naming conventions is used to identify different
        types of hosts/switches as well as to assign MAC/IP addresses in a more legible manner.  i.e.
        Hosts are assigned IP addresses with the format "10.[131/200 for major/minor buildings respectively].building#.host#".
        Switch DPIDs (MAC addresses) are assigned with first letter being type (minor buildings are 'a' and
         the server switch is 'e') and the last digits being its #.
        :param str topology_file: file name of topology to import
        """
        self.net = Mininet(topo=None,
                           build=False,
                           ipBase=IP_SUBNET,
                           autoSetMacs=True,
                           # autoStaticArp=True
                           )

        log.info('*** Adding controller')
        self.controller = self.net.addController(name='c0',
                                         controller=RemoteController,
                                         ip=self.controller_ip,
                                         port=OPENFLOW_CONTROLLER_PORT,
                                         )

        # import the switches, hosts, and server(s) from our specified file
        self.topo = NetworkxSdnTopology(self.topology_filename)

        def __get_mac_for_switch(switch):
            # BUGFIX: need to manually specify the mac to set DPID properly or Mininet
            # will just use the number at the end of the name, causing overlaps.
            # HACK: slice off the single letter at start of name, which we assume it has;
            # then convert the number to a MAC.
            mac = mac_for_host(int(switch[1:]))
            # Disambiguate one switch type from another by setting the first letter
            # to be a unique one corresponding to switch type and add in the other 0's.
            first_letter = switch[0]
            if first_letter == 'm':
                first_letter = 'a'
            if first_letter == 'g':
                first_letter = 'e'
            # rest fit in hex except for rack switches
            mac = first_letter + '0:00:00' + mac[3:]
            return str(mac)

        for switch in self.topo.get_switches():
            mac = __get_mac_for_switch(switch)
            s = self.net.addSwitch(switch, dpid=mac, cls=OVSKernelSwitch)
            log.debug("adding switch %s at DPID %s" % (switch, s.dpid))
            self.switches.append(s)
            print switch
            if switch[0]=='g':
                print "found gateway"
                self.gateways.append(s)

        def __get_ip_for_host(host):
            # See note in docstring about host format
            host_num, building_type, building_num = re.match('h(\d+)-([mb])(\d+)', host).groups()
            return "10.%d.%s.%s" % (131 if building_type == 'b' else 200, building_num, host_num)

        for host in self.topo.get_hosts():
            h = self.net.addHost(host, ip=__get_ip_for_host(host))
            self.hosts.append(h)

        for server in self.topo.get_servers():
            # HACK: we actually add a switch in case the server is multi-homed since it's very
            # difficult to work with multiple interfaces on a host (e.g. ONOS can only handle
            # a single MAC address per host).
            server_switch_name = server.replace('s', 'e')
            server_switch_dpid = __get_mac_for_switch(server_switch_name)
            # Keep server name for switch so that the proper links will be added later.
            self.server_switch = self.net.addSwitch(server, dpid=server_switch_dpid, cls=OVSKernelSwitch)
            s = self.net.addHost('h' + server)
            self.server = s
            self.net.addLink(self.server_switch, self.server)

        for cloud in self.topo.get_clouds():
            # HACK: Same hack with adding local server
            cloud_switch_name = cloud.replace('x', 'f')
            cloud_switch_dpid = __get_mac_for_switch(cloud_switch_name)
            # Keep server name for switch so that the proper links will be added later.
            self.cloud_switch = self.net.addSwitch(cloud, dpid=cloud_switch_dpid, cls=OVSKernelSwitch)
            x = self.net.addHost('h' + cloud)
            self.cloud = x
            self.net.addLink(self.cloud_switch, self.cloud)

        for pinger in self.topo.get_pingers():
            p = self.net.addHost(pinger)
            self.pingers.append(p)

        for link in self.topo.get_links():
            from_link = link[0]
            to_link = link[1]
            log.debug("adding link from %s to %s" % (from_link, to_link))

            # Get link attributes for configuring realistic traffic control settings
            # For configuration options, see mininet.link.TCIntf.config()
            attributes = link[2]
            _bw = attributes.get('bw', 10)  # in Mbps
            _delay = '%fms' % attributes.get('latency', 10)
            _jitter = '1ms'
            _loss = self.error_rate

            l = self.net.addLink(self.net.get(from_link), self.net.get(to_link),
                                 cls=TCLink, bw=_bw, delay=_delay, jitter=_jitter, loss=_loss
                                 )
            self.links.append(l)

        # add NAT so the server can communicate with SDN controller's REST API
        # NOTE: because we didn't add it to the actual SdnTopology, we don't need
        # to worry about it getting failed.  However, we do need to ensure it
        # connects directly to the server to avoid failures disconnecting it.
        # HACK: directly connect NAT to the server, set a route for it, and
        # handle this hacky IP address configuration
        nat_switch = self.net.addSwitch('s9999')

        nat_ip = NAT_SERVER_IP_ADDRESS % 2
        srv_ip = NAT_SERVER_IP_ADDRESS % 3
        self.nat = self.net.addNAT(connect=nat_switch)
        self.nat.configDefault(ip=nat_ip)

        self.net.addLink(self.server,nat_switch)

        # Now we set the IP address for the server's new interface.
        # NOTE: we have to set the default route after starting Mininet it seems...
        srv_iface = sorted(self.server.intfNames())[-1]
        self.server.intf(srv_iface).setIP(srv_ip)

        pn = 1
        for p in self.pingers:
            self.net.addLink(p, nat_switch)
            p_iface = sorted(p.intfNames())[-1]
            pinger_nat_ip = NAT_SERVER_IP_ADDRESS % (3+pn)
            p.intf(p_iface).setIP(pinger_nat_ip)
            pn+=1

    # HACK: because self.topo stores nodes by just their string name, we need to
    # convert them into actual Mininet hosts for use by this experiment.

    def _get_mininet_nodes(self, nodes):
        """
        Choose the actual Mininet Hosts (rather than just strings) that will
        be subscribers.
        :param List[str] nodes:
        :return List[Node] mininet_nodes:
        """
        return [self.net.get(n) for n in nodes]


    def run_experiment(self):
        """

        """

        log.info('*** Starting network')
        self.net.build()
        self.net.start()
        self.net.waitConnected()  # ensure switches connect

        # give controller time to converge topology so pingall works
        time.sleep(5)

        # setting the server's default route for controller access needs to be
        # done after the network starts up
        nat_ip = self.nat.IP()
        srv_iface = self.server.intfNames()[-1]
        self.server.setDefaultRoute('via %s dev %s' % (nat_ip, srv_iface))

        # this needs to come after starting network or no interfaces/IP addresses will be present
        log.debug("\n".join("added host %s at IP %s" % (host.name, host.IP()) for host in self.net.hosts))
        log.debug('links: %s' % [(l.intf1.name, l.intf2.name) for l in self.net.links])

        log.info('*** Pinging hosts so controller can gather IP addresses...')
        # don't want the NAT involved as hosts won't get a route to it
        # TODO: could just ping the server from each host as we don't do any host-to-host
        # comms and the whole point of this is really just to establish the hosts in the
        # controller's topology.  ALSO: we need to either modify this or call ping manually
        # because having error_rate > 0 leads to ping loss, which could results in a host
        # not being known!
        loss = self.net.ping(hosts=[h for h in self.net.hosts if h != self.nat], timeout=2)
        if loss > 0:
            log.warning("ping had a loss of %f" % loss)

        # This needs to occur AFTER pingAll as the exchange of ARP messages
        # is used by the controller (ONOS) to learn hosts' IP addresses
        self.net.staticArp()

        self.setup_topology_manager()
        #CLI(self.net)

        log.info('*** Network set up!\n*** Configuring experiment...')
        self.setup_reroute_test()
        
        log.info('*** Configuration done!  Waiting for Link Break')
        time.sleep(CLOUD1_BREAK_EVENT_DELAY - 1)
        log.info('*** Link Break at %s!  ' % time.time())
        self.net.configLinkStatus('g0', 'x0', 'down')

        time.sleep(CLOUD2_BREAK_EVENT_DELAY - CLOUD1_BREAK_EVENT_DELAY)
        log.info('*** Link Break at %s!  ' % time.time())
        self.net.configLinkStatus('g1', 'x0', 'down')

        log.info("*** Waiting for experiment to complete...")
        time.sleep(EXPERIMENT_DURATION - CLOUD2_BREAK_EVENT_DELAY)

        return

    def setup_topology_manager(self):
        """
        Starts a SdnTopology for the given controller (topology_manager) type.  Used for setting
        routes, clearing flows, etc.
        :return:
        """
        SdnTopologyAdapter = None
        if self.topology_adapter_type == 'onos':
            from topology_manager.onos_sdn_topology import OnosSdnTopology as SdnTopologyAdapter
        elif self.topology_adapter_type == 'floodlight':
            from topology_manager.floodlight_sdn_topology import FloodlightSdnTopology as SdnTopologyAdapter
        else:
            log.error("Unrecognized topology_adapter_type type %s.  Can't reset controller between runs or manipulate flows properly!")
            exit(102)

        if SdnTopologyAdapter is not None:
            self.topology_adapter = SdnTopologyAdapter(ip=self.controller_ip, port=self.controller_port)



    def setup_reroute_test(self):
        """

       """
        quit_time = EXPERIMENT_DURATION


        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
        ####################
        ### SETUP Local Server
        ####################
        server_ip = self.server.IP()

        log.info("edge server on host %s" % self.server.name)
        # setup static route for hosts -> edge server
        for h in self.hosts:
            try:
                original_server_name = self.topo.get_servers()[0]
                route = self.topo.get_path(h.name, original_server_name, weight=DISTANCE_METRIC)
                # Next, convert the NetworkxTopology nodes to the proper ID
                route = self._get_mininet_nodes(route)
                ky = [self.get_node_dpid(n) for n in route]
                # Then we need to modify the route to account for the real Mininet server 'hs0'
                route.insert(len(route), self.get_host_dpid(self.server))
                log.debug("Installing static route for host %s: %s" % (h, route))

                flow_rules = self.topology_adapter.build_flow_rules_from_path(route)
                for r in flow_rules:
                    self.topology_adapter.install_flow_rule(r)
            except Exception as e:
                log.error("Error installing flow rules for host to edge routes: %s" % e)
                raise e

        cmd = "python reroute_test/datagram_server.py"
        p = self.server.popen(cmd, shell=True, env=env)
        self.popens.append(p)

        ####################
        ### SETUP Cloud Server
        ####################

        cloud_ip = self.cloud.IP()
        nat_ip = self.nat.IP()

        cloud_in_topo = self.topo.get_clouds()[0]
        for gw in self.topo.get_cloud_gateways():
            out_port, _ = self.topo.get_ports_for_nodes(gw, cloud_in_topo)
            actions = self.topo.build_actions(("output", out_port))
            matches = self.topo.build_matches(ipv4_dst=cloud_ip)
            rule = self.topology_adapter.build_flow_rule(gw, matches, actions)
            self.topology_adapter.install_flow_rule(rule)

        cmd = "python reroute_test/datagram_server.py"
        p_server = self.cloud.popen(cmd, shell=True, env=env)
        cmd = "python reroute_test/udp_echo_server.py"
        p_echo = self.cloud.popen(cmd, shell=True, env=env)
        self.popens.append(p_server)
        self.popens.append(p_echo)

        ####################
        ### SETUP RIDEC
        ####################
        ride_c = RideC(self.topo, self.topology_adapter, self.gateways, self.hosts, self.server, self.cloud)
        ride_c.setDaemon(True)
        ride_c.start()

        ####################
        ### SETUP Pingers
        ####################
        for pinger in self.pingers:
            try:
                original_cloud_name = self.topo.get_clouds()[0]
                route = self.topo.get_path(pinger.name, original_cloud_name, weight=DISTANCE_METRIC)
                # Next, convert the NetworkxTopology nodes to the proper ID
                route = self._get_mininet_nodes(route)
                # Then we need to modify the route to account for the real Mininet server 'hs0'
                route.insert(len(route), self.get_host_dpid(self.server))
                log.debug("Installing static route for pinger %s: %s" % (pinger, route))

                flow_rules = self.topology_adapter.build_flow_rules_from_path(route)
                for r in flow_rules:
                    self.topology_adapter.install_flow_rule(r)
            except Exception as e:
                log.error("Error installing flow rules for pinger routes: %s" % e)
                raise e

            cmd = "python reroute_test/detector/ride_c_detector.py --id %s --host %s --ride_c_addr %s" %\
                  (pinger.name, cloud_ip, nat_ip)
            p = pinger.popen(cmd, shell=True, env=env)
            self.popens.append(p)

        ####################
        ### SETUP Clients
        ####################
        for host in self.hosts:
            try:
                original_cloud_name = self.topo.get_clouds()[0]
                route = self.topo.get_path(host.name, original_cloud_name, weight=DISTANCE_METRIC)
                # Next, convert the NetworkxTopology nodes to the proper ID
                route = self._get_mininet_nodes(route)
                # Then we need to modify the route to account for the real Mininet server 'hs0'
                route.insert(len(route), self.get_host_dpid(self.server))
                log.debug("Installing static route for host %s: %s" % (host, route))

                flow_rules = self.topology_adapter.build_flow_rules_from_path(route)
                for r in flow_rules:
                    self.topology_adapter.install_flow_rule(r)
            except Exception as e:
                log.error("Error installing flow rules for static subscriber routes: %s" % e)
                raise e

            cmd = "python reroute_test/datagram_client.py --id %s --address %s --interval %f" %\
                  (host.name, cloud_ip, 0.1)
            p = host.popen(cmd, shell=True, env=env)
            self.popens.append(p)


        return

    def teardown_experiment(self):
        log.info("*** Experiment complete! Waiting for all host procs to exit...")

        # need to check if the programs have finished before we exit mininet!
        # First, we check the server to see if it even ran properly.
        ret = self.popens[0].wait()
        if ret != 0:
            from seismic_warning_test.seismic_server import SeismicServer
            if ret == SeismicServer.EXIT_CODE_NO_SUBSCRIBERS:
                log.error("Server proc exited due to no reachable subscribers: this experiment is a wash!")
                # TODO: handle this error appropriately: mark results as junk?
            else:
                log.error("Server proc exited with code %d" % self.popens[0].returncode)
        for p in self.popens[1:]:
            ret = p.wait()
            while ret is None:
                ret = p.wait()
            if ret != 0:
                if ret == errno.ENETUNREACH:
                    # TODO: handle this error appropriately: record failed clients in results?
                    log.error("Client proc failed due to unreachable network!")
                else:
                    log.error("Client proc exited with code %d" % p.returncode)

        log.debug("*** All processes exited!  Cleaning up Mininet...")

        if self.show_cli:
            CLI(self.net)

        # Clear out all the flows/groups from controller
        if self.topology_adapter is not None:
            log.debug("Removing groups and flows via REST API.  This could take a while while we wait for the transactions to commit...")
            self.topology_adapter.remove_all_flow_rules()

            # We loop over doing this because we want to make sure the groups have been fully removed
            # before continuing to the next run or we'll have serious problems.
            # NOTE: some flows will still be present so we'd have to check them after
            # filtering only those added by REST API, hence only looping over groups for now...
            ngroups = 1
            while ngroups > 0:
                self.topology_adapter.remove_all_groups()
                time.sleep(1)
                leftover_groups = self.topology_adapter.get_groups()
                ngroups = len(leftover_groups)
                # len(leftover_groups) == 0, "Not all groups were cleared after experiment! Still left: %s" % leftover_groups

        # BUG: This might error if a process (e.g. iperf) didn't finish exiting.
        try:
            self.net.stop()
        except OSError as e:
            log.error("Stopping Mininet failed, but we'll keep going.  Reason: %s" % e)

        # We seem to still have process leakage even after the previous call to stop Mininet,
        # so let's do an explicit clean between each run.
        p = Popen('sudo mn -c > /dev/null 2>&1', shell=True)
        p.wait()

        # Sleep for a bit so the controller/OVS can finish resetting
        time.sleep(SLEEP_TIME_BETWEEN_RUNS)

    def get_host_dpid(self, host):
        """
        Returns the data plane ID for the given host that is recognized by the
        particular SDN controller currently in use.
        :param Host host:
        :return:
        """
        if self.topology_adapter_type == 'onos':
            # TODO: verify this vibes with ONOS properly; might need VLAN??
            dpid = host.defaultIntf().MAC().upper() + '/None'
        elif self.topology_adapter_type == 'floodlight':
            dpid = host.IP()
        else:
            raise ValueError("Unrecognized topology adapter type %s" % self.topology_adapter_type)
        return dpid

    def get_switch_dpid(self, switch):
        """
        Returns the data plane ID for the given switch that is recognized by the
        particular SDN controller currently in use.
        :param Switch switch:
        :return:
        """
        if self.topology_adapter_type == 'onos':
            dpid = 'of:' + switch.dpid
        elif self.topology_adapter_type == 'floodlight':
            raise NotImplementedError()
        else:
            raise ValueError("Unrecognized topology adapter type %s" % self.topology_adapter_type)
        return dpid

    def get_node_dpid(self, node):
        """
        Returns the data plane ID for the given node by determining whether it's a
        Switch or Host first.
        :param node:
        :return:
        """
        if isinstance(node, Switch):
            return self.get_switch_dpid(node)
        elif isinstance(node, Host):
            return self.get_host_dpid(node)
        else:
            raise TypeError("Unrecognized node type for: %s" % node)



if __name__ == "__main__":
    import sys
    exp = MininetSmartCampus2ServerExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()

