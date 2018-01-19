#! /usr/bin/env python

# @author: Kyle Benson
# (c) Kyle Benson 2017

import logging
log = logging.getLogger(__name__)

import os
import random
from collections import OrderedDict
import argparse
import time
import ipaddress

from mininet.node import Host

from topology_manager.networkx_sdn_topology import NetworkxSdnTopology
from smart_campus_experiment import SmartCampusExperiment, DISTANCE_METRIC
from mininet_sdn_experiment import MininetSdnExperiment

from seismic_warning_test.seismic_alert_common import SEISMIC_PICK_TOPIC, IOT_GENERIC_TOPIC
from scale_client.core.client import make_scale_config_entry, make_scale_config

from config import *
from ride.config import *


class MininetSmartCampusExperiment(MininetSdnExperiment, SmartCampusExperiment):
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

    NOTE: there are 3 different versions of the network topology stored in this object:
      1) self.topo is the NetworkxSdnTopology that we read from a file and use to populate the networks
      2) self.net is the Mininet topology consisting of actual Mininet Hosts and Links
      3) self.topology_adapter is the concrete SdnTopology (e.g. OnosSdnTopology) instance that interfaces with the
         SDN controller responsible for managing the emulated Mininet nodes.
      Generally, the nodes stored as fields of this class are Mininet Switches and the hosts are Mininet Hosts.  Links
      are expressed simply as (host1.name, host2.name) pairs.  Various helper functions exist to help convert between
      these representations and ensure the proper formatting is used as arguments to e.g. launching processes on hosts.
    """

    def __init__(self, n_traffic_generators=0, traffic_generator_bandwidth=10, comparison=None,
                 # need to save these two params to pass to RideD
                 tree_choosing_heuristic=DEFAULT_TREE_CHOOSING_HEURISTIC, max_alert_retries=None,
                 *args, **kwargs):
        """
        Mininet and the SdnTopology adapter will be started by this constructor.
        NOTE: you must start the remote SDN controller before constructing/running the experiment!
        :param controller_ip: IP address of SDN controller that we point RideD towards: it must be accessible by the server Mininet host!
        :param controller_port: REST API port of SDN controller
        :param tree_choosing_heuristic: explicit in this version since we are running an
         actual emulation and so cannot check all the heuristics at once
        :param max_alert_retries: passed to Ride-D to control # times it retries sending alerts
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
        self.results['params']['max_alert_retries'] = self.max_alert_retries = max_alert_retries
        self.results['params']['n_traffic_generators'] = self.n_traffic_generators = n_traffic_generators
        self.results['params']['traffic_generator_bandwidth'] = self.traffic_generator_bandwidth = traffic_generator_bandwidth

        self.cloud_gateways = []
        # explicitly track these switches separately from the server ones since we do special stuff with them...
        self.cloud_switches = []
        self.edge_server_switch = None

        # HACK: We just manually allocate IP addresses rather than adding a controller API to request them.
        # NOTE: we also have to specify a unique UDP src port for each tree so that responses can be properly routed
        # back along the same tree (otherwise each MDMT would generate the same flow rules and overwrite each other!).
        base_addr = ipaddress.IPv4Address(MULTICAST_ADDRESS_BASE)
        self.mcast_address_pool = [(str(base_addr + i), MULTICAST_ALERT_BASE_SRC_PORT + i) for i in range(kwargs['ntrees'])]

        # This gets passed to seismic hosts
        self.debug_level = kwargs.get('debug', 'error')

    @classmethod
    def get_arg_parser(cls, parents=(SmartCampusExperiment.get_arg_parser(), MininetSdnExperiment.get_arg_parser()), add_help=True):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        # argument parser that can be combined with others when this class is used in a script
        # need to not add help options to use that feature, though
        # WARNING: this inheritance isn't as smart as object inheritance, so it will cause conflicts when multiply
        #    inheriting argparers!  Hence, we use resolve to just pick the first; be careful not to use overlapping
        #    commands to mean different things!!
        # TODO: document some behavior that changes with the Mininet version:
        # -- pubs/subs are actual client processes
        arg_parser = argparse.ArgumentParser(parents=parents, add_help=add_help, conflict_handler='resolve')
        # experimental treatment parameters: all taken from parents
        # background traffic generation
        arg_parser.add_argument('--ngenerators', '-g', default=0, dest='n_traffic_generators', type=int,
                                help='''number of hosts that generate random traffic to cause congestion (default=%(default)s)''')
        arg_parser.add_argument('--generator-bandwidth', '-bw', default=10, dest='traffic_generator_bandwidth', type=float,
                                help='''bandwidth (in Mbps) of iperf for congestion traffic generating hosts (default=%(default)s)''')
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
        """

        super(MininetSmartCampusExperiment, self).setup_topology()

        # import the switches, hosts, and server(s) from our specified file
        self.topo = NetworkxSdnTopology(self.topology_filename)

        for switch in self.topo.get_switches():
            mac = get_mac_for_switch(switch)
            s = self.add_switch(switch, mac)
            if self.topo.is_cloud_gateway(switch):
                self.cloud_gateways.append(s)

        for host in self.topo.get_hosts():
            _ip, _mac = get_ip_mac_for_host(host)
            self.add_host(host, _ip, _mac)

        for server in self.topo.get_servers():
            server_switch_name = server.replace('s', 'e')
            server_switch_dpid = get_mac_for_switch(server_switch_name, is_server=True)
            host = 'h' + server
            _ip, _mac = get_ip_mac_for_host(host)

            # Keep server name for switch so that the proper links will be added later.
            srv, server_switch = self.add_server(name=host, ip=_ip, mac=_mac,
                                                 server_switch_name=server, server_switch_dpid=server_switch_dpid)

            # XXX: this experiment only uses a single server
            self.edge_server = srv
            self.edge_server_switch = server_switch

        for cloud in self.topo.get_clouds():
            # Only consider the cloud special if we've enabled doing so
            if self.with_cloud:
                cloud_switch_name = cloud.replace('x', 'f')
                cloud_switch_dpid = get_mac_for_switch(cloud_switch_name, is_cloud=True)
                # ENHANCE: handle multiple clouds
                host = 'h' + cloud
                _ip, _mac = get_ip_mac_for_host(host)

                # Keep server name for switch so that the proper links will be added later.
                _cloud, cloud_switch = self.add_server(name=host, ip=_ip, mac=_mac,
                                                     server_switch_name=cloud, server_switch_dpid=cloud_switch_dpid)

                self.cloud_switches.append(cloud_switch)
                self.cloud = _cloud
            # otherwise just add a host to prevent topology errors
            else:
                self.net.addHost(cloud)
                self.cloud = self.net.addHost(cloud)

        for link in self.topo.get_links():
            from_link = link[0]
            to_link = link[1]

            # Get link attributes for configuring realistic traffic control settings
            attributes = link[2]
            _bw = attributes.get('bw')  # in Mbps
            _delay = attributes.get('latency')
            # TODO: increase jitter for cloud!
            self.add_link(from_link, to_link, bandwidth=_bw, latency=_delay)

        self.add_nat(self.edge_server)

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

    def _choose_random_hosts(self, nhosts, from_list=None):
        return super(MininetSmartCampusExperiment, self)._choose_random_hosts(nhosts, from_list=self.devices)

    # XXX: filter out servers from list of hosts
    @property
    def devices(self):
        # NOTE: the server host is e.g. hs0!!
        return [h for h in self.hosts if (h.name.startswith('h') and '-' in h.name)]

    def choose_server(self):
        """
        Choose the actual Mininet Host (rather than just strings) that will
        be the server.
        :return Host server:
        """
        # HACK: call the super version of this so that we increment the random number generator correctly
        super(MininetSmartCampusExperiment, self).choose_server()
        return self.edge_server

    def get_failed_nodes_links(self):
        fnodes, flinks = super(MininetSmartCampusExperiment, self).get_failed_nodes_links()
        # NOTE: we can just pass the links as strings
        return self._get_mininet_nodes(fnodes), flinks

    def setup_experiment(self):

        # Need to select the publishers first before configuring the mininet network
        SmartCampusExperiment.setup_experiment(self)
        MininetSdnExperiment.setup_experiment(self)

        # We also have to manually configure the routes for the multicast addresses
        # the server will use.
        for a, p in self.mcast_address_pool:
            self.edge_server.setHostRoute(a, self.edge_server.intf().name)

        # this needs to come after starting network or no interfaces/IP addresses will be present
        log.debug("\n".join("added host %s at IP %s" % (host.name, host.IP()) for host in self.net.hosts))
        log.debug('links: %s' % [(l.intf1.name, l.intf2.name) for l in self.net.links])

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

        self.setup_traffic_generators()

        # NOTE: it takes a second or two for the clients to actually start up!
        # log.debug('*** Starting clients at time %s' % time.time())
        self.setup_seismic_test(self.publishers, self.subscribers, self.edge_server)
        # log.debug('*** Done starting clients at time %s' % time.time())

        ####    FAILURE MODEL     ####

        exp_start_time = time.time()
        log.info('*** Configuration done!  Experiment started at %f; now waiting for failure events...' % exp_start_time)
        # ENCHANCE: instead of just 1 sec before, should try to figure out how long
        # it'll take for different machines/configurations and time it better...
        time.sleep(SEISMIC_EVENT_DELAY)
        quake_time = None

        ###    FAIL / RECOVER DATA PATHS
        # According to the specified configuration, we update each requested DataPath link to the specified status
        # up(recover) / down(fail), sleeping between each iteration to let the system adapt to the changes.

        # XXX: because RideC assigns all publishers to the 'highest priority' (lowest alphanumerically) DP,
        # each iteration should just fail the one with highest priority here to observe the fail-over.
        #
        # Fail ALL DataPaths!  then recover one...
        data_path_changes = [(dpl[0], dpl[1], 'down', TIME_BETWEEN_SEISMIC_EVENTS)
                             for dpl in sorted(self.data_path_links)[1:]]
        # XXX: the first one should happen immediately!
        first_dpl = sorted(self.data_path_links)[0]
        data_path_changes.insert(0, (first_dpl[0], first_dpl[1], 'down', 0))
        data_path_changes.append((first_dpl[0], first_dpl[1], 'up', TIME_BETWEEN_SEISMIC_EVENTS))
        # XXX: since the failure configs can sometimes take a little bit, we should explicitly record when each happened
        output_dp_changes = []

        # We'll fail the first DataPath, then fail the second along with the local links (main earthquake),
        # then eventually recover one of the DataPaths
        for i, (cloud_gw, cloud_switch, new_status, delay) in enumerate(data_path_changes):

            log.debug("waiting for DataPath change...")
            time.sleep(delay)
            dp_change_time = time.time()
            output_dp_changes.append((cloud_gw, new_status, dp_change_time))
            log.debug("%s DataPath link (%s--%s) at time %f" %
                      ("failing" if new_status == 'down' else "recovering", cloud_gw, cloud_switch, dp_change_time))
            self.net.configLinkStatus(cloud_gw, cloud_switch, new_status)

            # First DataPath failure wasn't a 'local earthquake', the second is and will fail part of local topology
            if i == 1:
                # Apply actual failure model: we schedule these to fail when the earthquake hits
                # so there isn't time for the topology to update on the controller,
                # which would skew the results incorrectly. Since it may take a few cycles
                # to fail a lot of nodes/links, we schedule the failures for a second before.
                quake_time = time.time()
                log.info('*** Earthquake at %s!  Applying failure model...' % quake_time)
                for link in self.failed_links:
                    log.debug("failing link: %s" % str(link))
                    self.net.configLinkStatus(link[0], link[1], 'down')
                for node in self.failed_nodes:
                    node.stop(deleteIntfs=False)
                log.debug("done applying failure model at %f" % time.time())

        # wait for the experiment to finish by sleeping for the amount of time we haven't used up already
        remaining_time = exp_start_time + EXPERIMENT_DURATION - time.time()
        log.info("*** Waiting %f seconds for experiment to complete..." % remaining_time)
        if remaining_time > 0:
            time.sleep(remaining_time)

        return {'quake_start_time': quake_time,
                'data_path_changes': output_dp_changes,
                'publishers': {p.IP(): p.name for p in self.publishers},
                'subscribers': {s.IP(): s.name for s in self.subscribers}}

    def setup_traffic_generators(self):
        """Each traffic generating host starts an iperf process aimed at
        (one of) the server(s) in order to generate random traffic and create
        congestion in the experiment.  Traffic is all UDP because it sets the bandwidth.

        NOTE: iperf v2 added the capability to tell the server when to exit after some time.
        However, we explicitly terminate the server anyway to avoid incompatibility issues."""

        generators = self._choose_random_hosts(self.n_traffic_generators)
        bandwidth = self.traffic_generator_bandwidth

        # TODO: include the cloud_server as a possible traffic generation/reception
        # point here?  could also use other hosts as destinations...
        srv = self.edge_server

        log.info("*** Starting background traffic generators")
        # We enumerate the generators to fill the range of ports so that the server
        # can listen for each iperf client.
        for n, g in enumerate(generators):
            self.iperf(g, srv, bandwidth=bandwidth, port=IPERF_BASE_PORT+n)

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
        """

        delay = SEISMIC_EVENT_DELAY  # seconds before sensors start picking
        quit_time = EXPERIMENT_DURATION

        # HACK: Need to set PYTHONPATH since we don't install our Python modules directly and running Mininet
        # as root strips this variable from our environment.
        env = os.environ.copy()
        ride_dir = os.path.dirname(os.path.abspath(__file__))
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = ride_dir + ':'
        else:
            env['PYTHONPATH'] = env['PYTHONPATH'] + ':' + ride_dir

        outputs_dir, logs_dir = self.build_outputs_logs_dirs()

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
                    # HACK: since self.edge_server is a new Mininet Host not in original topo, we do this:
                    original_server_name = self.topo.get_servers()[0]
                    route = self.topo.get_path(original_server_name, sub.name, weight=DISTANCE_METRIC)
                    # Next, convert the NetworkxTopology nodes to the proper ID
                    route = self._get_mininet_nodes(route)
                    route = [self.get_node_dpid(n) for n in route]
                    # Then we need to modify the route to account for the real Mininet server 'hs0'
                    route.insert(0, self.get_host_dpid(self.edge_server))
                    log.debug("Installing static route for subscriber %s: %s" % (sub, route))

                    flow_rules = self.topology_adapter.build_flow_rules_from_path(route, priority=STATIC_PATH_FLOW_RULE_PRIORITY)
                    if not self.topology_adapter.install_flow_rules(flow_rules):
                        log.error("problem installing batch of flow rules for subscriber %s: %s" % (sub, flow_rules))
                except Exception as e:
                    log.error("Error installing flow rules for static subscriber routes: %s" % e)
                    raise e
        # For the oracle comparison config we just extend the quit time so the controller has plenty
        # of time to detect and recover from the failures.
        elif self.comparison is not None and self.comparison == 'oracle':
            use_multicast = False
            # TODO: base this quit_time extension on the Coap timeout????
            # quit_time += 20

        sdn_topology_cfg = self._get_topology_manager_config()
        # XXX: use controller IP specified in config.py if the default localhost was left
        if sdn_topology_cfg['controller_ip'] == '127.0.0.1':
            sdn_topology_cfg['controller_ip'] = CONTROLLER_IP

        ride_d_cfg = None if not self.with_ride_d else make_scale_config_entry(name="RideD", multicast=use_multicast,
                                                                               class_path="seismic_warning_test.ride_d_event_sink.RideDEventSink",
                                                                               # RideD configurations
                                                                               addresses=self.mcast_address_pool, ntrees=self.ntrees,
                                                                               tree_construction_algorithm=self.tree_construction_algorithm,
                                                                               tree_choosing_heuristic=self.tree_choosing_heuristic,
                                                                               max_retries=self.max_alert_retries,
                                                                               dpid=self.get_host_dpid(self.edge_server),
                                                                               topology_mgr=sdn_topology_cfg,
                                                                               )
        seismic_alert_server_cfg = '' if not self.with_ride_d else make_scale_config_entry(
            class_path="seismic_warning_test.seismic_alert_server.SeismicAlertServer",
            output_events_file=os.path.join(outputs_dir, 'srv'),
            name="EdgeSeismicServer")

        _srv_apps = seismic_alert_server_cfg
        if self.with_ride_c:
            # To run RideC, we need to configure it with the necessary information to register each DataPath under
            # consideration: an ID, the gateway switch DPID, the cloud server's DPID, and the probing source port.
            # The source port will be used to distinguish the different DataPathMonitor probes from each other and
            # route them through the correct gateway using static flow rules.
            # NOTE: because we pass these parameters as tuples in a list, with each tuple containing all info
            # necessary to register a DataPath, we can assume the order remains constant.

            src_ports = range(PROBE_BASE_SRC_PORT, PROBE_BASE_SRC_PORT + len(self.cloud_gateways))
            data_path_args = [[gw.name, self.get_switch_dpid(gw), self.get_host_dpid(self.cloud), src_port] for
                          gw, src_port in zip(self.cloud_gateways, src_ports)]
            log.debug("RideC-managed DataPath arguments are: %s" % data_path_args)

            # We have two different types of IoT data flows (generic and seismic) so we use two different CoAP clients
            # on the publishers to distinguish the traffic, esp. since generic data is sent non-CON!
            publisher_args = [(h.IP(), pub_port) for h in sensors for pub_port in (COAP_CLIENT_BASE_SRC_PORT, COAP_CLIENT_BASE_SRC_PORT+1)]

            _srv_apps += make_scale_config_entry(class_path="seismic_warning_test.ride_c_application.RideCApplication",
                                                 name="RideC", topology_mgr=sdn_topology_cfg, data_paths=data_path_args,
                                                 edge_server=self.get_host_dpid(server),
                                                 cloud_server=self.get_host_dpid(self.cloud),
                                                 publishers=publisher_args,
                                                 reroute_policy=self.reroute_policy,
                                                 )

            # Now set the static routes for probes to travel through the correct DataPath Gateway.
            for gw, src_port in zip(self.cloud_gateways, src_ports):
                gw_dpid = self.get_switch_dpid(gw)
                edge_gw_route = self.topology_adapter.get_path(self.get_host_dpid(self.edge_server), gw_dpid,
                                                               weight=DISTANCE_METRIC)
                gw_cloud_route = self.topology_adapter.get_path(gw_dpid, self.get_host_dpid(self.cloud),
                                                                weight=DISTANCE_METRIC)
                route = self.topology_adapter.merge_paths(edge_gw_route, gw_cloud_route)

                # Need to modify the 'matches' used to include the src/dst_port!
                dst_port = ECHO_SERVER_PORT

                matches = dict(udp_src=src_port, udp_dst=dst_port)
                frules = self.topology_adapter.build_flow_rules_from_path(route, add_matches=matches, priority=STATIC_PATH_FLOW_RULE_PRIORITY)

                # NOTE: need to do the other direction to ensure responses come along same path!
                route.reverse()
                matches = dict(udp_dst=src_port, udp_src=dst_port)
                frules.extend(self.topology_adapter.build_flow_rules_from_path(route, add_matches=matches, priority=STATIC_PATH_FLOW_RULE_PRIORITY))

                # log.debug("installing probe flow rules for DataPath (port=%d)\nroute: %s\nrules: %s" %
                #           (src_port, route, frules))
                if not self.topology_adapter.install_flow_rules(frules):
                    log.error("problem installing batch of flow rules for RideC probes via gateway %s: %s" % (gw, frules))

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
            cmd = self.redirect_output_to_log(cmd, 'srv')

        self.run_proc(cmd, server, env=env)

        if self.with_cloud:
            # Now for the cloud, which differs only by the facts that it doesn't run RideC, is always unicast alerting
            # via RideD, and also runs a UdpEchoServer to respond to RideC's DataPath probes
            ride_d_cfg = None if not self.with_ride_d else make_scale_config_entry(name="RideD", multicast=False,
                                                                  class_path="seismic_warning_test.ride_d_event_sink.RideDEventSink",
                                                                  dpid=self.get_host_dpid(self.cloud), addresses=None,
                                                                  )
            seismic_alert_cloud_cfg = '' if not self.with_ride_d else make_scale_config_entry(
                class_path="seismic_warning_test.seismic_alert_server.SeismicAlertServer",
                output_events_file=os.path.join(outputs_dir, 'cloud'),
                name="CloudSeismicServer")
            cloud_apps = seismic_alert_cloud_cfg

            cloud_net_cfg = make_scale_config_entry(class_path='udp_echo_server.UdpEchoServer',
                                                    name='EchoServer', port=ECHO_SERVER_PORT)
            if self.with_ride_d:
                cloud_net_cfg += make_scale_config_entry(name="CoapServer", events_root="/events/",
                                                         class_path="coap_server.CoapServer")

            cloud_cfg = make_scale_config(applications=cloud_apps, sinks=ride_d_cfg, networks=cloud_net_cfg,)

            cmd = SCALE_CLIENT_BASE_COMMAND % (base_args + cloud_cfg)
            if WITH_LOGS:
                cmd = self.redirect_output_to_log(cmd, 'cloud')

            self.run_proc(cmd, self.cloud, env=env)

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
                frules = self.topology_adapter.build_flow_rules_from_path(path, matches, priority=STATIC_PATH_FLOW_RULE_PRIORITY)

                if not self.topology_adapter.install_flow_rules(frules):
                    log.error("problem installing batch of flow rules for subscriber %s via gateway %s: %s" % (sub, gw, frules))

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
                sensors=make_scale_config_entry(name="SeismicSensor", event_type=SEISMIC_PICK_TOPIC,
                                                dynamic_event_data=dict(seq=0),
                                                class_path="dummy.dummy_virtual_sensor.DummyVirtualSensor",
                                                output_events_file=os.path.join(outputs_dir,
                                                                                'publisher_%s' % client_id),
                                                # Need to start at specific time, not just delay, as it takes a few
                                                # seconds to start up each process.
                                                # TODO: Also spread out the reports a little bit, but we should spread
                                                # out the failures too if we do so: + random.uniform(0, 1)
                                                start_time=time.time() + delay,
                                                sample_interval=TIME_BETWEEN_SEISMIC_EVENTS) +
                # for congestion traffic
                        make_scale_config_entry(name="IoTSensor", event_type=IOT_GENERIC_TOPIC,
                                                dynamic_event_data=dict(seq=0),
                                                class_path="dummy.dummy_virtual_sensor.DummyVirtualSensor",
                                                output_events_file=os.path.join(outputs_dir,
                                                                                'congestor_%s' % client_id),
                                                # give servers a chance to start; spread out their reports too
                                                start_delay=random.uniform(5, 10),
                                                sample_interval=IOT_CONGESTION_INTERVAL)
                ,  # always sink the picks as confirmable, but deliver the congestion traffic best-effort
                sinks=make_scale_config_entry(class_path="remote_coap_event_sink.RemoteCoapEventSink",
                                              name="SeismicCoapEventSink", hostname=cloud_ip,
                                              src_port=COAP_CLIENT_BASE_SRC_PORT,
                                              topics_to_sink=(SEISMIC_PICK_TOPIC,)) +
                      make_scale_config_entry(class_path="remote_coap_event_sink.RemoteCoapEventSink",
                                              name="GenericCoapEventSink", hostname=cloud_ip,
                                              # make sure we distinguish the coapthon client instances from each other!
                                              src_port=COAP_CLIENT_BASE_SRC_PORT + 1,
                                              topics_to_sink=(IOT_GENERIC_TOPIC,), confirmable_messages=False)
                # Can optionally enable this to print out each event in its entirety.
                # + make_scale_config_entry(class_path="log_event_sink.LogEventSink", name="LogSink")
            )

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
                cmd = self.redirect_output_to_log(cmd, unique_filename)

            self.run_proc(cmd, client, env=env)

    def record_result(self, result):
        """Save additional results outputs (or convert to the right format) before outputting them."""
        # Need to save node names rather than actual Mininet nodes for JSON serializing.
        self.failed_nodes = [n.name for n in self.failed_nodes]

        # We'll also record the 'oracle' heuristic now so that we know how many pubs/subs should have been reachable
        # by/to the edge/cloud servers
        ftopo = self.get_failed_topology(self.topo.topo, self.failed_nodes, self.failed_links)
        subscriber_names = result['subscribers'].values()
        publisher_names = result['publishers'].values()
        # XXX: just hard-coding the names since we made them e.g. hs0
        server_name = 's0'
        cloud_name = 'x0'

        result['oracle_edge_subs'] = SmartCampusExperiment.get_oracle_reachability(subscriber_names, server_name, ftopo)
        result['oracle_edge_pubs'] = SmartCampusExperiment.get_oracle_reachability(publisher_names, server_name, ftopo)
        if self.with_cloud:
            # we need to remove the first DP link since it'd be failed:
            # XXX: we can just hack the gateway off that we know is always there
            ftopo.remove_node('g0')
            result['oracle_cloud_subs'] = SmartCampusExperiment.get_oracle_reachability(subscriber_names, cloud_name, ftopo)
            result['oracle_cloud_pubs'] = SmartCampusExperiment.get_oracle_reachability(publisher_names, cloud_name, ftopo)

        super(MininetSmartCampusExperiment, self).record_result(result)

    def cleanup_procs(self):
        # NOTE: need to wait more than 10 secs for clients to have a chance to naturally finish,
        # which is default 'timeout' for CoapServer.listen()
        log.debug("*** sleeping to give client procs a chance to finish...")
        time.sleep(20)
        # Additionally, we'll clean up the hosts first then the servers to give them even more time to finish.
        self.popens = OrderedDict(reversed(self.popens.items()))
        super(MininetSmartCampusExperiment, self).cleanup_procs()

    @property
    def data_path_links(self):
        """Returns a collection of (gateway Switch, cloud Switch) pairs to represent DataPath links
        or None if no clouds/DataPath exist"""
        if not self.cloud_switches:
            return None
        # XXX: since we only have one cloud server, we don't need to figure out which one corresponds to each GW
        return [(gw.name, self.cloud_switches[0].name) for gw in self.cloud_gateways]


if __name__ == "__main__":
    import sys
    exp = MininetSmartCampusExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()

