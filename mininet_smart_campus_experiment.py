#! /usr/bin/python

# @author: Kyle Benson
# (c) Kyle Benson 2017

import os
import errno
import re
from subprocess import Popen

import topology_manager
from smart_campus_experiment import SmartCampusExperiment, DISTANCE_METRIC

import logging
log = logging.getLogger(__name__)
LOGGERS_TO_DISABLE = ('sdn_topology', 'topology_manager.sdn_topology', 'connectionpool', 'urllib3.connectionpool')

import json
import argparse
import time
import ipaddress

from mininet.net import Mininet
from mininet.node import RemoteController, Host, OVSKernelSwitch
from mininet.node import Switch, Link, Node  # these just used for types in docstrings
from mininet.cli import CLI
from mininet.link import TCLink, Intf

from topology_manager.networkx_sdn_topology import NetworkxSdnTopology
from topology_manager.test_sdn_topology import mac_for_host  # used for manual MAC assignment

from scale_config import *

EXPERIMENT_DURATION = 90  # in seconds
# EXPERIMENT_DURATION = 10  # for testing
SEISMIC_EVENT_DELAY = 60  # seconds before the 'earthquake happens', i.e. sensors start sending data
# SEISMIC_EVENT_DELAY = 5  # for testing
IPERF_BASE_PORT = 5000  # background traffic generators open several iperf connections starting at this port number
OPENFLOW_CONTROLLER_PORT = 6653  # we assume the controller will always be at the default port
# subnet for all hosts (if you change this, update the __get_ip_for_host() function!)
# NOTE: we do /9 so as to avoid problems with addressing e.g. the controller on the local machine
# (vagrant uses 10.0.2.* for VM's IP address).
HOST_IP_N_MASK_BITS = 9
IP_SUBNET = '10.128.0.0/%d' % HOST_IP_N_MASK_BITS
WITH_LOGS = True  # output seismic client/server stdout to a log file
# HACK: rather than some ugly hacking at Mininet's lack of API for allocating the next IP address,
# we just put the NAT/server interfaces in a hard-coded subnet.
NAT_SERVER_IP_ADDRESS = '11.0.0.%d/24'
# TODO: use a different address base...
MULTICAST_ADDRESS_BASE = u'224.0.0.1'  # must be unicode!
SLEEP_TIME_BETWEEN_RUNS = 5  # give Mininet/OVS/ONOS a chance to reconverge after cleanup

# Default values
DEFAULT_TREE_CHOOSING_HEURISTIC = 'importance'
DEFAULT_TOPOLOGY_ADAPTER = 'onos'

class MininetSmartCampusExperiment(SmartCampusExperiment):
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

    def __init__(self, controller_ip='127.0.0.1', controller_port=8181,
                 # need to save these two params to pass to RideD
                 tree_choosing_heuristic=DEFAULT_TREE_CHOOSING_HEURISTIC,
                 topology_adapter=DEFAULT_TOPOLOGY_ADAPTER,
                 n_traffic_generators=0, traffic_generator_bandwidth=10,
                 show_cli=False, comparison=None,
                 *args, **kwargs):
        """
        Mininet and the SdnTopology adapter will be started by this constructor.
        NOTE: you must start the remote SDN controller before constructing/running the experiment!
        :param controller_ip: IP address of SDN controller that we point RideD towards: it must be accessible by the server Mininet host!
        :param controller_port: REST API port of SDN controller
        :param tree_choosing_heuristic: explicit in this version since we are running an
         actual emulation and so cannot check all the heuristics at once
        :param topology_adapter: type of REST API topology adapter we use: one of 'onos', 'floodlight'
        :param n_traffic_generators: number of background traffic generators to run iperf on
        :param traffic_generator_bandwidth: bandwidth (in Mbps; using UDP) to set the iperf traffic generators to
        :param show_cli: display the Mininet CLI in between each run (useful for debugging)
        :param comparison: disable RIDE-D and use specified comparison strategy (unicast or oracle)
        :param args: see args of superclass
        :param kwargs: see kwargs of superclass
        """

        # We want this parameter overwritten in results file for the proper configuration.
        self.comparison = comparison
        if comparison is not None:
            assert comparison in ('oracle', 'unicast'), "Uncrecognized comparison method: %s" % comparison
            kwargs['tree_construction_algorithm'] = (comparison,)

        super(MininetSmartCampusExperiment, self).__init__(*args, **kwargs)
        # save any additional parameters the Mininet version adds
        self.results['params']['experiment_type'] = 'mininet'
        self.results['params']['tree_choosing_heuristic'] = self.tree_choosing_heuristic = tree_choosing_heuristic
        self.results['params']['n_traffic_generators'] = self.n_traffic_generators = n_traffic_generators
        self.results['params']['traffic_generator_bandwidth'] = self.traffic_generator_bandwidth = traffic_generator_bandwidth

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
        self.cloud_gateways = []
        # XXX: see note in setup_topology() about replacing cloud hosts with a switch to ease multi-homing
        self.cloud_switches = []
        self.links = []
        self.net = None
        self.controller = None
        self.nat = None

        self.server_switch = None
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
        base_addr = ipaddress.IPv4Address(MULTICAST_ADDRESS_BASE)
        self.mcast_address_pool = [str(base_addr + i) for i in range(kwargs['ntrees'])]

        # Disable some of the more verbose and unnecessary loggers
        for _logger_name in LOGGERS_TO_DISABLE:
            l = logging.getLogger(_logger_name)
            l.setLevel(logging.WARNING)

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

    @classmethod
    def build_default_results_file_name(cls, args, dirname='results'):
        """
        :param args: argparse object (or plain dict) with all args info (not specifying ALL args is okay)
        :param dirname: directory name to place the results files in
        :return: string representing the output_filename containing a parameter summary for easy identification
        """
        # HACK: we need to add the additional parameters this experiment version bring in
        # We add them at the end, though we'll replace the choosing_heuristic with the comparison metric if specified.
        output_filename = super(MininetSmartCampusExperiment, cls).build_default_results_file_name(args, dirname)
        if isinstance(args, argparse.Namespace):
            choosing_heuristic = args.tree_choosing_heuristic if args.comparison is None else args.comparison
        else:
            choosing_heuristic = args.get('tree_choosing_heuristic', DEFAULT_TREE_CHOOSING_HEURISTIC)\
                if args.get('comparison', None) is None else args['comparison']
        replacement = '_%s.json' % choosing_heuristic
        output_filename = output_filename.replace('.json', replacement)
        return output_filename

    def set_interrupt_signal(self):
        # ignore it so we can terminate Mininet commands without killing Mininet
        # TODO: something else?
        return

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

        def __get_mac_for_switch(switch, is_cloud=False, is_server=False):
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

            mac = first_letter + second_letter + ':00:00' + mac[3:]
            return str(mac)

        for switch in self.topo.get_switches():
            mac = __get_mac_for_switch(switch)
            s = self.net.addSwitch(switch, dpid=mac, cls=OVSKernelSwitch)
            log.debug("adding switch %s at DPID %s" % (switch, s.dpid))
            self.switches.append(s)
            if self.topo.is_cloud_gateway(switch):
                self.cloud_gateways.append(s)

        def __get_ip_for_host(host):
            # See note in docstring about host format
            # XXX: differentiate between regular hosts and server hosts
            if '-' in host:
                host_num, building_type, building_num = re.match('h(\d+)-([mb])(\d+)', host).groups()
            else:  # must be a server
                building_type, host_num = re.match('h?([xs])(\d+)', host).groups()
                building_num = 0

            # Assign a number according to the type of router this host is attached to
            if building_type == 'b':
                router_code = 131
            elif building_type == 'm':
                router_code = 144
            # cloud
            elif building_type == 'x':
                router_code = 199
            # edge server
            elif building_type == 's':
                router_code = 255
            else:
                raise ValueError("unrecognized building type '%s' so cannot assign host IP address!" % building_type)
            return "10.%d.%s.%s/%d" % (router_code, building_num, host_num, HOST_IP_N_MASK_BITS)

        for host in self.topo.get_hosts():
            h = self.net.addHost(host, ip=__get_ip_for_host(host))
            self.hosts.append(h)

        for server in self.topo.get_servers():
            # HACK: we actually add a switch in case the server is multi-homed since it's very
            # difficult to work with multiple interfaces on a host (e.g. ONOS can only handle
            # a single MAC address per host).
            server_switch_name = server.replace('s', 'e')
            server_switch_dpid = __get_mac_for_switch(server_switch_name, is_server=True)
            # Keep server name for switch so that the proper links will be added later.
            self.server_switch = self.net.addSwitch(server, dpid=server_switch_dpid, cls=OVSKernelSwitch)
            host = 'h' + server
            s = self.net.addHost(host, ip=__get_ip_for_host(host))
            # ENHANCE: handle multiple servers
            self.server = s
            self.net.addLink(self.server_switch, self.server)

        for cloud in self.topo.get_clouds():
            # Only consider the cloud special if we've enabled doing so
            if self.with_cloud:
                # HACK: Same hack with adding local server
                cloud_switch_name = cloud.replace('x', 'f')
                cloud_switch_dpid = __get_mac_for_switch(cloud_switch_name, is_cloud=True)
                # Keep server name for switch so that the proper links will be added later.
                cloud_switch = self.net.addSwitch(cloud, dpid=cloud_switch_dpid, cls=OVSKernelSwitch)
                self.cloud_switches.append(cloud_switch)
                # ENHANCE: handle multiple clouds
                host = 'h' + cloud
                self.cloud = self.net.addHost(host, ip=__get_ip_for_host(host))
                self.net.addLink(cloud_switch, self.cloud)
            # otherwise just add a host to prevent topology errors
            else:
                self.net.addHost(cloud)
                self.cloud = self.net.addHost(cloud)

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
        nat_ip = NAT_SERVER_IP_ADDRESS % 2
        srv_ip = NAT_SERVER_IP_ADDRESS % 3
        self.nat = self.net.addNAT(connect=self.server)
        self.nat.configDefault(ip=nat_ip)

        # Now we set the IP address for the server's new interface.
        # NOTE: we have to set the default route after starting Mininet it seems...
        srv_iface = sorted(self.server.intfNames())[-1]
        self.server.intf(srv_iface).setIP(srv_ip)

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

    def choose_publishers(self):
        """
        Choose the actual Mininet Hosts (rather than just strings) that will
        be publishers.
        :return List[Host] publishers:
        """
        return self._get_mininet_nodes(super(MininetSmartCampusExperiment, self).choose_publishers())

    def choose_subscribers(self):
        """
        Choose the actual Mininet Hosts (rather than just strings) that will
        be subscribers.
        :return List[Host] subscribers:
        """
        return self._get_mininet_nodes(super(MininetSmartCampusExperiment, self).choose_subscribers())

    def choose_server(self):
        """
        Choose the actual Mininet Host (rather than just strings) that will
        be the server.
        :return Host server:
        """
        # HACK: call the super version of this so that we increment the random number generator correctly
        super(MininetSmartCampusExperiment, self).choose_server()
        return self.server

    def get_failed_nodes_links(self):
        fnodes, flinks = super(MininetSmartCampusExperiment, self).get_failed_nodes_links()
        # NOTE: we can just pass the links as strings
        return self._get_mininet_nodes(fnodes), flinks

    def run_experiment(self):
        """
        Configures all appropriate settings, runs the experiment, and
        finally tears it down before returning the results.
        (Assumes Mininet has already been started).

        Returned results is a dict containing the 'logs_dir' and 'outputs_dir' for
        this run as well as lists of 'subscribers' and 'publishers' (their app IDs
        (Mininet node names), which will appear in the name of their output file).

        :rtype dict:
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

        # We also have to manually configure the routes for the multicast addresses
        # the server will use.
        for a in self.mcast_address_pool:
            self.server.setHostRoute(a, self.server.intf().name)

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

        log.info('*** Network set up!\n*** Configuring experiment...')

        self.setup_traffic_generators()
        # NOTE: it takes a second or two for the clients to actually start up!
        # log.debug('*** Starting clients at time %s' % time.time())
        logs_dir, outputs_dir = self.setup_seismic_test(self.publishers, self.subscribers, self.server)
        # log.debug('*** Done starting clients at time %s' % time.time())

        # Apply actual failure model: we schedule these to fail when the earthquake hits
        # so there isn't time for the topology to update on the controller,
        # which would skew the results incorrectly. Since it may take a few cycles
        # to fail a lot of nodes/links, we schedule the failures for a second before.
        # ENCHANCE: instead of just 1 sec before, should try to figure out how long
        # it'll take for different machines/configurations and time it better...
        log.info('*** Configuration done!  Waiting for earthquake to start...')
        time.sleep(SEISMIC_EVENT_DELAY - 1)
        log.info('*** Earthquake at %s!  Applying failure model...' % time.time())
        for link in self.failed_links:
            self.net.configLinkStatus(link[0], link[1], 'down')
        for node in self.failed_nodes:
            node.stop(deleteIntfs=False)

        # log.debug('*** Failure model finished applying at %s' % time.time())

        log.info("*** Waiting for experiment to complete...")

        time.sleep(EXPERIMENT_DURATION - SEISMIC_EVENT_DELAY)

        return {'outputs_dir': outputs_dir, 'logs_dir': logs_dir,
                'publishers': [p.name for p in self.publishers],
                'subscribers': [s.name for s in self.subscribers]}

    def setup_topology_manager(self):
        """
        Starts a SdnTopology for the given controller (topology_manager) type.  Used for setting
        routes, clearing flows, etc.
        :return:
        """
        self.topology_adapter = topology_manager.build_topology_adapter(self.topology_adapter_type, controller_ip=self.controller_ip, controller_port=self.controller_port)

    def setup_traffic_generators(self):
        """Each traffic generating host starts an iperf process aimed at
        (one of) the server(s) in order to generate random traffic and create
        congestion in the experiment.  Traffic is all UDP because it sets the bandwidth.

        NOTE: iperf v2 added the capability to tell the server when to exit after some time.
        However, we explicitly terminate the server anyway to avoid incompatibility issues."""

        generators = self._get_mininet_nodes(self._choose_random_hosts(self.n_traffic_generators))

        # TODO: include the cloud_server as a possible traffic generation/reception
        # point here?  could also use other hosts as destinations...
        srv = self.server

        log.info("*** Starting background traffic generators")
        # We enumerate the generators to fill the range of ports so that the server
        # can listen for each iperf client.
        for n, g in enumerate(generators):
            log.info("iperf from %s to %s" % (g, srv))
            # can't do self.net.iperf([g,s]) as there's no option to put it in the background
            i = g.popen('iperf -p %d -t %d -u -b %dM -c %s &' % (IPERF_BASE_PORT + n, EXPERIMENT_DURATION,
                                                                 self.traffic_generator_bandwidth, srv.IP()))
            self.client_iperfs.append(i)
            i = srv.popen('iperf -p %d -t %d -u -s &' % (IPERF_BASE_PORT + n, EXPERIMENT_DURATION))
            self.server_iperfs.append(i)


    def setup_seismic_test(self, sensors, subscribers, server):
        """
        Sets up the seismic sensing test scenario in which each sensor reports
        a sensor reading to the server, which will aggregate them together and
        multicast the result back out to each subscriber.  The server uses RIDE-D:
        a reliable multicast method in which several maximally-disjoint multicast
        trees (MDMTs) are installed in the SDN topology and intelligently
        choosen from at alert-time based on various heuristics.
        :param List[Host] sensors:
        :param List[Host] subscribers:
        :param Host server:
        :returns logs_dir, outputs_dir: the directories (relative to the experiment output
         file) in which the logs and output files, respectively, are stored for this run
        """

        delay = SEISMIC_EVENT_DELAY  # seconds before sensors start picking
        quit_time = EXPERIMENT_DURATION

        # HACK: Need to set PYTHONPATH since we don't install our Python modules directly and running Mininet
        # as root strips this variable from our environment.
        env = os.environ.copy()
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))

        # The logs and output files go in nested directories rooted
        # at the same level as the whole experiment's output file.
        # We typically name the output file as results_$PARAMS.json, so cut off the front and extension
        root_dir = os.path.dirname(self.output_filename)
        base_dirname = os.path.splitext(os.path.basename(self.output_filename))[0]
        if base_dirname.startswith('results_'):
            base_dirname = base_dirname[8:]
        if WITH_LOGS:
            logs_dir = os.path.join(root_dir, 'logs_%s' % base_dirname, 'run%d' % self.current_run_number)
            try:
                os.makedirs(logs_dir)
            except OSError:
                pass
        else:
            logs_dir = None
        outputs_dir =  os.path.join(root_dir, 'outputs_%s' % base_dirname, 'run%d' % self.current_run_number)
        try:
            os.makedirs(outputs_dir)
        except OSError:
            pass

        ##############################
        ### SETUP EDGE / CLOUD SERVERS
        ##############################

        server_ip = server.IP()
        assert server_ip != '127.0.0.1', "ERROR: server.IP() returns localhost!"

        log.info("Seismic server on host %s with IP %s" % (server.name, server_ip))

        #### COMPARISON CONFIGS

        # First, we need to set static unicast routes to subscribers for unicast comparison config.
        # This HACK avoids the controller recovering failed paths too quickly due to Mininet's zero latency
        # control plane network.
        # NOTE: because we only set static routes when not using RideD multicast, this shouldn't
        # interfere with other routes.
        use_multicast = True
        if self.comparison is not None and self.comparison == 'unicast':
            use_multicast = False
            for sub in subscribers:
                try:
                    # HACK: we get the route from the NetworkxTopology in order to have the same
                    # as other experiments, but then need to convert these paths into one
                    # recognizable by the actual SDN Controller Topology manager.
                    # HACK: since self.server is a new Mininet Host not in original topo, we do this:
                    original_server_name = self.topo.get_servers()[0]
                    route = self.topo.get_path(original_server_name, sub.name, weight=DISTANCE_METRIC)
                    # Next, convert the NetworkxTopology nodes to the proper ID
                    route = self._get_mininet_nodes(route)
                    route = [self.get_node_dpid(n) for n in route]
                    # Then we need to modify the route to account for the real Mininet server 'hs0'
                    route.insert(0, self.get_host_dpid(self.server))
                    log.debug("Installing static route for subscriber %s: %s" % (sub, route))

                    flow_rules = self.topology_adapter.build_flow_rules_from_path(route)
                    for r in flow_rules:
                        self.topology_adapter.install_flow_rule(r)
                except Exception as e:
                    log.error("Error installing flow rules for static subscriber routes: %s" % e)
                    raise e
        # For the oracle comparison config we just extend the quit time so the controller has plenty
        # of time to detect and recover from the failures.
        elif self.comparison is not None and self.comparison == 'oracle':
            use_multicast = False
            # TODO: base this quit_time extension on the Coap timeout????
            # quit_time += 20

        sdn_topology_cfg = (self.topology_adapter_type, self.controller_ip, self.controller_port)

        ride_d_cfg = None if not self.with_ride_d else make_scale_config_entry(name="RideD", multicast=use_multicast,
                                                                  class_path="seismic_warning_test.ride_d_event_sink.RideDEventSink",
                                                                  # RideD configurations
                                                                  addresses=self.mcast_address_pool, ntrees=self.ntrees,
                                                                  tree_construction_algorithm=self.tree_construction_algorithm,
                                                                  tree_choosing_heuristic=self.tree_choosing_heuristic,
                                                                  dpid=self.get_host_dpid(self.server),
                                                                  topology_mgr=sdn_topology_cfg,
                                                                  )
        seismic_alert_server_cfg = '' if not self.with_ride_d else make_scale_config_entry(
            class_path="seismic_warning_test.seismic_alert_server.SeismicAlertServer",
            output_events_file=os.path.join(outputs_dir, 'srv'),
            name="SeismicServer")

        _srv_apps = seismic_alert_server_cfg
        if self.with_ride_c:
            # TODO: verify this is right?  maybe we want to shorten the DataPath name so it isn't whole GW DPID?
            data_paths = [[self.get_switch_dpid(gw), self.get_switch_dpid(gw),
                           self.get_host_dpid(self.cloud)] for gw in self.cloud_gateways]
            log.debug("RideC-managed DataPaths are: %s" % data_paths)

            _srv_apps += make_scale_config_entry(class_path="seismic_warning_test.ride_c_application.RideCApplication",
                                                 name="RideC", topology_mgr=sdn_topology_cfg, data_paths=data_paths,
                                                 edge_server=self.get_host_dpid(server),
                                                 cloud_server=self.get_host_dpid(self.cloud),
                                                 publishers=[self.get_host_dpid(h) for h in sensors],
                                                 )

        srv_cfg = make_scale_config(sinks=ride_d_cfg,
                                    networks=None if not self.with_ride_d else \
                                        make_scale_config_entry(name="CoapServer", events_root="/events/",
                                                                class_path="coap_server.CoapServer"),
                                    # TODO: also run a publisher for that bugfix?
                                    applications=_srv_apps
                                    )

        base_args = "-q %d --log %s" % (quit_time, self.debug_level)
        cmd = SCALE_CLIENT_BASE_COMMAND % (base_args + srv_cfg)

        if WITH_LOGS:
            cmd += " > %s 2>&1" % os.path.join(logs_dir, 'srv')

        log.debug(cmd)
        p = server.popen(cmd, shell=True, env=env)
        self.popens.append(p)

        if self.with_cloud:
            # Now for the cloud, which differs only by the fact that it doesn't run RideC and is always unicast alerting
            ride_d_cfg = None if not self.with_ride_d else make_scale_config_entry(name="RideD", multicast=False,
                                                                  class_path="seismic_warning_test.ride_d_event_sink.RideDEventSink",
                                                                  dpid=self.get_host_dpid(self.cloud), addresses=None,
                                                                  )
            seismic_alert_cloud_cfg = '' if not self.with_ride_d else make_scale_config_entry(
                class_path="seismic_warning_test.seismic_alert_server.SeismicAlertServer",
                output_events_file=os.path.join(outputs_dir, 'cloud'),
                name="SeismicServer")
            cloud_apps = seismic_alert_cloud_cfg

            # TODO: UdpEchoServer????

            cloud_cfg = make_scale_config(applications=cloud_apps, sinks=ride_d_cfg,
                                          networks=None if not self.with_ride_d else \
                                          make_scale_config_entry(name="CoapServer", events_root="/events/",
                                                                  class_path="coap_server.CoapServer"),
                                          )

            cmd = SCALE_CLIENT_BASE_COMMAND % (base_args + cloud_cfg)
            if WITH_LOGS:
                cmd += " > %s 2>&1" % os.path.join(logs_dir, 'cloud')

            log.debug(cmd)
            p = self.cloud.popen(cmd, shell=True, env=env)
            self.popens.append(p)

        # XXX: to prevent the 0-latency control plane from resulting in the controller immediately routing around
        # quake-induced failures, we set static routes from each cloud gateway to the subscribers based on their
        # destination IP address.
        # TODO: what to do about cloud server --> cloud gateway?????? how do we decide which gateway (DP) should be used?
        # NOTE: see comments above when doing static routes for unicast comparison configuration about why we
        # should be careful about including a server in the path...
        for sub in subscribers:

            sub_ip = sub.IP()
            matches = self.topology_adapter.build_matches(ipv4_dst=sub_ip, ipv4_src=self.cloud.IP())

            for gw in self.cloud_gateways:
                path = self.topo.get_path(gw.name, sub.name)
                log.debug("installing static route for subscriber: %s" % path)
                path = [self.get_node_dpid(n) for n in self._get_mininet_nodes(path)]
                # XXX: since this helper function assumes the first node is a host, it'll skip over installing
                # rules on it.  Hence, we add the cloud switch serving that gateway as the 'source'...
                path.insert(0, self.get_node_dpid(self.cloud_switches[0]))
                # TODO: what to do with this?  we can't add the cloud or the last gw to be handled will be the one routed through...
                # path.insert(0, self.get_node_dpid(self.cloud))
                frules = self.topology_adapter.build_flow_rules_from_path(path, matches)

                for f in frules:
                    self.topology_adapter.install_flow_rule(f)

        ####################
        ###  SETUP CLIENTS
        ####################

        sensors = set(sensors)
        subscribers = set(subscribers)
        # BUGFIX HACK: server only sends data to subs if it receives any, so we run an extra
        # sensor client on the server host so the server process always receives at least one
        # publication.  Otherwise, if no publications reach it the reachability is 0 when it
        # may actually be 1.0! This is used mainly for comparison vs. NetworkxSmartCampusExperiment.
        # TODO: just run the same seismic sensor on the server so that it will always publish SOMETHING
        # sensors.add(server)

        log.info("Running seismic test client on %d subscribers and %d sensors" % (len(subscribers), len(sensors)))

        # If we aren't using the cloud, publishers will just send to the edge and subscribers only have 1 broker
        cloud_ip = server_ip
        alerting_brokers = [server_ip]
        if self.with_cloud:
            cloud_ip = self.cloud.IP()
            alerting_brokers.append(cloud_ip)

        for client in sensors.union(subscribers):
            client_id = client.name

            # Build the cli configs for the two client types
            subs_cfg = make_scale_config(
                networks=make_scale_config_entry(name="CoapServer", class_path="coap_server.CoapServer",
                                                 events_root="/events/"),
                applications=make_scale_config_entry(
                    class_path="seismic_warning_test.seismic_alert_subscriber.SeismicAlertSubscriber",
                    name="SeismicSubscriber", remote_brokers=alerting_brokers,
                    output_file=os.path.join(outputs_dir, 'subscriber_%s' % client_id)))
            pubs_cfg = make_scale_config(
                sensors=make_scale_config_entry(name="SeismicSensor", event_type="seismic",
                                                dynamic_event_data=dict(seq=0),
                                                class_path="dummy.dummy_virtual_sensor.DummyVirtualSensor",
                                                output_events_file=os.path.join(outputs_dir,
                                                                                'publisher_%s' % client_id),
                                                start_delay=delay, sample_interval=5),
                sinks=make_scale_config_entry(class_path="remote_coap_event_sink.RemoteCoapEventSink",
                                              name="CoapEventSink", hostname=cloud_ip))

            # Build up the final cli configs, merging the individual ones built above if necessary
            args = base_args
            if client in sensors:
                args += pubs_cfg
            if client in subscribers:
                args += subs_cfg
            cmd = SCALE_CLIENT_BASE_COMMAND % args

            if WITH_LOGS:
                unique_filename = ''
                if client in sensors and client in subscribers:
                    unique_filename = 'ps'
                elif client in sensors:
                    unique_filename = 'p'
                elif client in subscribers:
                    unique_filename = 's'
                unique_filename = '%s_%s' % (unique_filename, client_id)
                cmd += " > %s 2>&1" % os.path.join(logs_dir, unique_filename)

            # the node.sendCmd option in mininet only allows a single
            # outstanding command at a time and cancels any current
            # ones when net.CLI is called.  Hence, we need popen.
            log.debug(cmd)
            p = client.popen(cmd, shell=True, env=env)
            self.popens.append(p)

        # make the paths relative to the root directory in which the whole experiment output file is stored
        # as otherwise the paths are dependent on where the cwd is
        logs_dir = os.path.relpath(logs_dir, root_dir)
        outputs_dir = os.path.relpath(outputs_dir, root_dir)
        return logs_dir, outputs_dir

    def teardown_experiment(self):
        log.info("*** Experiment complete! Waiting for all host procs to exit...")

        # need to check if the programs have finished before we exit mininet!
        # NOTE: need to wait more than 10 secs, which is default 'timeout' for CoapServer.listen()
        # TODO: set wait_time to 30? 60?
        def wait_then_kill(proc, timeout = 5, wait_time = 20):
            assert isinstance(proc, Popen)  # for typing
            ret = None
            for i in range(wait_time/timeout):
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(timeout)
            else:
                log.error("process never quit: killing it...")
                proc.kill()
                ret = proc.wait()
                log.error("now it exited with code %d" % ret)
            return ret

        # Inspect the clients first, then the server so it has a little more time to finish up closing
        client_popen_start_idx = 1 if not self.with_cloud else 2

        for p in self.popens[client_popen_start_idx:]:
            ret = wait_then_kill(p)
            if ret is None:
                log.error("Client proc never quit!")
            elif ret != 0:
                # TODO: we'll need to pipe this in from the scale client?
                if ret == errno.ENETUNREACH:
                    # TODO: handle this error appropriately: record failed clients in results?
                    log.error("Client proc failed due to unreachable network!")
                else:
                    log.error("Client proc exited with code %d" % p.returncode)

        ret = wait_then_kill(self.popens[0])
        if ret != 0:
            log.error("server proc exited with code %d" % ret)

        if self.with_cloud:
            ret = wait_then_kill(self.popens[1])
            if ret != 0:
                log.error("cloud proc exited with code %d" % ret)

        # Clean up traffic generators:
        # Clients should terminate automatically, but the server won't do so unless
        # a high enough version of iperf is used so we just do it explicitly.
        for p in self.client_iperfs:
            p.wait()
        for p in self.server_iperfs:
            try:
                wait_then_kill(p)
            except OSError:
                pass  # must have already terminated
        self.popens = []
        self.server_iperfs = []
        self.client_iperfs = []

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


# TODO: import these from somewhere rather than repeating them here... BUT, note that we've done some bugfixes with these ones


def make_scale_config(applications=None, sensors=None, sinks=None, networks=None):
    """
    Builds a string to be used on the command line in order to run a scale client with the given configurations.
    NOTE: make sure to properly space your arguments and wrap any newlines in quotes so they aren't interpreted
    as the end of the command by the shell!
    """
    cfg = ""
    if applications is not None:
        cfg += ' --applications %s ' % applications
    if sensors is not None:
        cfg += ' --sensors %s ' % sensors
    if networks is not None:
        cfg += ' --networks %s ' % networks
    if sinks is not None:
        cfg += ' --event-sinks %s ' % sinks
    return cfg


def make_scale_config_entry(class_path, name, **kwargs):
    """Builds an individual entry for a single SCALE client module that can be fed to the CLI.
    NOTE: don't forget to add spaces around each entry if you use multiple!"""
    # d = dict(name=name, **kwargs)
    d = dict(**kwargs)
    # XXX: can't use 'class' as a kwarg in call to dict, so doing it this way...
    d['class'] = class_path
    # need to wrap the raw JSON in single quotes for use on command line as json.dumps wraps strings in double quotes
    # also need to escape these double quotes so that 'eval' (su -c) actually sees them in the args it passes to the final command
    return "'%s' " % json.dumps({name: d}).replace('"', r'\"')
    # return "'%s'" % json.dumps(d)


if __name__ == "__main__":
    import sys
    exp = MininetSmartCampusExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()

