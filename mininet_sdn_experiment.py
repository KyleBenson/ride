# @author: Kyle Benson
# (c) Kyle Benson 2018
import logging
log = logging.getLogger(__name__)

import argparse
from subprocess import Popen
import errno
import time

from mininet.net import Mininet
from mininet.node import RemoteController, Switch, Host
from mininet.cli import CLI

import topology_manager.sdn_topology
from config import *


class MininetSdnExperiment(object):
    """
    An abstract object used for running automated experiments in Mininet.  The basic idea is that you'll subclass it
    (see MininetSmartCampusExperiment as an example) to set up additional configurations and control the overall
    flow of the experiment.  This just helps get a lot of the boilerplate out of the way and integrates it with the
    SdnTopology Python-based framework for interacting with the SDN controller.  See run_experiment for an outline of
    your expected workflow.  Also, expect to override some of the helper methods (e.g. setup_topology()) in your
    subclass in order to add your experiment's logic.
    """

    def __init__(self, controller_ip=CONTROLLER_IP, controller_port=CONTROLLER_REST_API_PORT,
                 topology_adapter=DEFAULT_TOPOLOGY_ADAPTER, show_cli=False, **kwargs):

        try:
            super(MininetSdnExperiment, self).__init__(**kwargs)
        except TypeError:
            super(MininetSdnExperiment, self).__init__()

        # set later as it needs resetting between runs and must be created after the network starts up
        self.topology_adapter = None
        self.topology_adapter_type = topology_adapter
        self.controller = None
        self.controller_port = controller_port
        self.controller_ip = controller_ip

        # These will all be filled in by calling setup_topology()
        # NOTE: make sure you reset these between runs so that you don't collect several runs worth of e.g. hosts!
        self.net = None
        self.switches = []
        self.links = []
        self.hosts = []
        self.servers = list()
        # XXX: see note in setup_topology() about replacing server hosts with a switch to ease multi-homing
        # These dicts maps the server Mininet node to the serving Mininet switch/link
        self.server_switches = dict()
        self.server_switch_links = dict()
        self.nats = dict()  # NAT host --> connection point host mapping
        # Save Popen objects to later ensure procs terminate before exiting Mininet or we'll end up with hanging procs.
        # The key should be a string representing what this proc is running e.g. some name or even the whole command.
        # Note that these may include iperf commands too!
        self.popens = dict()

        # We'll optionally drop to a CLI after the experiment completes for further poking around
        self.show_cli = show_cli

    def run_experiment(self):
        """
        Configures all appropriate settings, runs the experiment, and finally tears it down before returning the results.

        This is not actually implemented, but here's an outline of how to use the helper functions to get started:

        setup_topology()
        start_network()
        ensure_network_setup()
        run your custom experiment!
        teardown_experiment()
        """

        raise NotImplementedError("You must implement your experimental workflow yourself!  However, see docstring"
                                  "for an outline of how you can use the provided helper functions to get started.")

    @classmethod
    def get_arg_parser(cls, parents=(topology_manager.sdn_topology.SdnTopology.get_arg_parser(),), add_help=False):
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
        arg_parser = argparse.ArgumentParser(add_help=add_help)

        arg_parser.add_argument('--cli', '-cli', dest='show_cli', action='store_true',
                                help='''displays the Mininet CLI after running the experiment. This is useful for
                                debugging problems as it prevents the OVS/controller state from being wiped after
                                the experiment and keeps the network topology up.''')
        return arg_parser

    def setup_topology(self):
        """
        Builds the Mininet network, including all hosts, servers, switches, links, and NATs.
        NOTE: you must override this as it currently just sets up the empty Mininet network and RemoteController
        """

        self.net = Mininet(topo=None,
                           build=False,
                           ipBase=IP_SUBNET,
                           autoSetMacs=True,
                           # autoStaticArp=True
                           )

        log.debug('*** Adding controller')
        self.controller = self.net.addController(name='c0',
                                         controller=RemoteController,
                                         ip=self.controller_ip,
                                         port=OPENFLOW_CONTROLLER_PORT,
                                         )

    def add_nat(self, connection_point):
        """Add a NAT to the specified host/server.  We use this primarily so the server can communicate with the
        SDN controller's REST API. """
        # NOTE: because we didn't add it to the actual SdnTopology, we don't need
        # to worry about it getting failed.  However, we do need to ensure it
        # connects directly to the server to avoid failures disconnecting it.
        # HACK: directly connect NAT to the server, set a route for it, and
        # handle this hacky IP address configuration
        nat_ip = NAT_SERVER_IP_ADDRESS % (len(self.nats) + 2)
        srv_ip = NAT_SERVER_IP_ADDRESS % (len(self.nats) + 3)
        nat = self.net.addNAT(connect=connection_point)
        nat.configDefault(ip=nat_ip)
        # Now we set the IP address for the server's new interface.
        # NOTE: we have to set the default route after starting Mininet it seems...
        srv_iface = sorted(connection_point.intfNames())[-1]
        connection_point.intf(srv_iface).setIP(srv_ip)

        self.nats[nat] = connection_point
        return nat

    def start_network(self):
        """Actually start the Mininet network."""
        log.info('*** Starting network')
        log.debug("Building Network...")
        self.net.build()
        log.debug("Network built; starting...")
        self.net.start()
        log.debug("Started!  Waiting for switch connections...")
        self.net.waitConnected()  # ensure switches connect
        log.debug("Switches connected!")

        # set the default routes for each NAT'ed host, which MUST be done after the network starts
        for nat, host in self.nats.items():
            nat_ip = nat.IP()
            srv_iface = host.intfNames()[-1]
            host.setDefaultRoute('via %s dev %s' % (nat_ip, srv_iface))

    def ensure_network_setup(self):
        """
        To ensure the controller can interact with the Mininet topology, we need to run pingall between them so the
        controller becomes aware of them.  To ensure it is now aware, we also connect the topology_adapter and ensure
        the # hosts/switches are as expected.  If not, we'll re-try until they are.

        NOTE: based on the ALL_PAIRS config option, we likely really just ping from each host to each server to save time.
        """
        # give controller time to converge topology so pingall works
        time.sleep(5)
        # May need to ping the hosts again if links start up too late...
        hosts = self.hosts

        def ping_hosts(hosts):
            log.info('*** Pinging hosts so controller can gather IP addresses...')
            # don't want the NAT involved as hosts won't get a route to it
            # comms and the whole point of this is really just to establish the hosts in the
            # controller's topology.  ALSO: we need to either modify this or call ping manually
            # because having error_rate > 0 leads to ping loss, which could results in a host
            # not being known!
            loss = 0
            if ALL_PAIRS:
                loss = self.net.ping(hosts=hosts, timeout=2)
            else:
                for h in hosts:
                    for s in self.servers:
                        loss += self.net.ping((h, s), timeout=2)
                loss /= len(hosts) * len(self.servers)

            if loss > 0:
                log.warning("ping had a loss of %f" % loss)

            # This needs to occur AFTER pingAll as the exchange of ARP messages
            # is used by the controller (ONOS) to learn hosts' IP addresses
            # Similarly to ping, we don't need all-pairs... just for the hosts to/from edge/cloud servers
            if ALL_PAIRS:
                self.net.staticArp()
            else:
                for s in self.servers:
                    server_ip = s.IP()
                    server_mac = s.MAC()
                    for src in hosts:
                        src.setARP(ip=server_ip, mac=server_mac)
                        s.setARP(ip=src.IP(), mac=src.MAC())

        ping_hosts(hosts)
        # Need to sleep so that the controller has a chance to converge its topology again...
        time.sleep(5)
        # Now connect the SdnTopology and verify that all the non-NAT hosts, links, and switches are available through it
        expected_nhosts = len(hosts) + len(self.servers)  # ignore NAT, but include servers
        # Don't forget that we added switches for the servers to easily multi-home them
        expected_nlinks = len(self.links) + len(self.server_switch_links)
        expected_nswitches = len(self.switches) + len(self.server_switches)
        n_sdn_links = 0
        n_sdn_switches = 0
        n_sdn_hosts = 0
        ntries = 1
        while n_sdn_hosts != expected_nhosts or n_sdn_links != expected_nlinks or n_sdn_switches != expected_nswitches:
            self.setup_topology_manager()

            n_sdn_hosts = len(self.topology_adapter.get_hosts())
            n_sdn_links = self.topology_adapter.topo.number_of_edges()
            n_sdn_switches = len(self.topology_adapter.get_switches())

            success = True
            if n_sdn_hosts < expected_nhosts:
                log.warning(
                    "topology adapter didn't find all the hosts!  It only got %d/%d.  Trying topology adapter again..." % (
                    n_sdn_hosts, len(hosts)))
                success = False
            if expected_nlinks > n_sdn_links:
                log.warning(
                    "topology adapter didn't find all the links!  Only got %d/%d.  Trying topology adapter again..." % (
                    n_sdn_links, expected_nlinks))
                success = False
            if expected_nswitches > n_sdn_switches:
                log.warning(
                    "topology adapter didn't find all the switches!  Only got %d/%d.  Trying topology adapter again..." % (
                    n_sdn_switches, expected_nswitches))
                success = False

            time.sleep(2 if success else 10)

            # Sometimes this hangs forever... we should probably try configuring hosts again
            if ntries % 5 == 0 and not success:
                log.warning("pinging hosts again since we still aren't ready with the complete topology...")
                ping_hosts(hosts)
            ntries += 1

            # TODO: should probably exit with an error and clean everything up if we try too many times unsuccessfully

        log.info('*** Network set up!\n*** Configuring experiment...')

    def setup_topology_manager(self):
        """
        Starts a SdnTopology for the given controller (topology_manager) type.  Used for setting
        routes, clearing flows, etc.
        :return:
        """
        kwargs = self._get_topology_manager_config()
        self.topology_adapter = topology_manager.build_topology_adapter(**kwargs)

    def _get_topology_manager_config(self):
        """Get configuration parameters for the topology adapter as a dict."""
        kwargs = dict(topology_adapter_type=self.topology_adapter_type,
                      controller_ip=self.controller_ip, controller_port=self.controller_port)
        if self.topology_adapter_type == 'onos':
            kwargs['username'] = ONOS_API_USER
            kwargs['password'] = ONOS_API_PASSWORD
        return kwargs

    def teardown_experiment(self):
        log.info("*** Experiment complete! Waiting for all host procs to exit...")

        # need to check if the programs have finished before we exit mininet!
        self.cleanup_procs()

        log.debug("*** All processes exited!")

        # But first, give a chance to inspect the experiment state before quitting Mininet.
        if self.show_cli:
            CLI(self.net)

        self.cleanup_mininet()

        self.cleanup_sdn_controller()

        # ENHANCE: in order to support multiple runs, which we could never get working properly in a single process,
        # you'll need to reset all of the members e.g. hosts, switches, net, popens, etc.

        # Sleep for a bit so the controller/OVS can finish resetting
        log.debug("*** Done cleaning up the run!  Waiting %dsecs for changes to propagate to OVS/SDN controller..." % SLEEP_TIME_BETWEEN_RUNS)
        # TODO: only do this if we have more runs left? run.py would need to do it then...
        time.sleep(SLEEP_TIME_BETWEEN_RUNS)

    def cleanup_procs(self):
        """Verify that each of the processes we spawned exit properly.  If they haven't exited after a while, kill
        them explicitly and then make sure they exited after that.  We do this because Mininet can leak open processes
        if they aren't properly closed."""

        def wait_then_kill(proc, timeout=1, wait_time=2):
            assert isinstance(proc, Popen)  # for typing
            for i in range(wait_time / timeout):
                ret = proc.poll()
                if ret is not None:
                    break
                time.sleep(timeout)
            else:
                log.error("process never quit: killing it...")
                try:
                    proc.kill()
                except OSError:
                    pass  # must have already terminated
                ret = proc.wait()
                log.error("now it exited with code %d" % ret)

            return ret

        for cmd, p in self.popens.items():
            ret = wait_then_kill(p)
            if ret is None:
                log.error("Proc never quit: %s" % cmd)
            elif ret != 0:
                # NOTE: you may need to pipe this in from your client manually
                if ret == errno.ENETUNREACH:
                    # TODO: handle this error appropriately: record failed clients in results?
                    log.error("Proc failed due to unreachable network: %s" % cmd)
                else:
                    log.error("Exit code %d from proc %s" % (p.returncode, cmd))

        # TODO: this should stay in RIDE version!
        # XXX: somehow there still seem to be client processes surviving the .kill() commands; this finishes them off:
        p = Popen(CLEANUP_SCALE_CLIENTS, shell=True)
        p.wait()

    def cleanup_mininet(self):
        # BUG: This might error if a process (e.g. iperf) didn't finish exiting.
        try:
            log.debug("Stopping Mininet...")
            self.net.stop()
        except OSError as e:
            log.error("Stopping Mininet failed, but we'll keep going.  Reason: %s" % e)

        # We seem to still have process leakage even after the previous call to stop Mininet,
        # so let's do an explicit clean between each run.
        log.debug("Cleaning up Mininet...")
        p = Popen('sudo mn -c > /dev/null 2>&1', shell=True)
        time.sleep(10 if not TESTING else 2)
        p.wait()

    def cleanup_sdn_controller(self):
        # Clear out all the flows/groups from controller
        # XXX: this method is quicker/more reliable than going through the REST API since that requires deleting each
        # group one at a time!
        if self.topology_adapter_type == 'onos':
            log.debug("Resetting controller for next run...")
            # XXX: for some reason, doing 'onos wipe-out please' doesn't actually clear out switches!  Hence, we need to
            # fully reset ONOS before the next run and wait for it to completely restart by checking if the API is up.
            p = Popen("%s %s" % (CONTROLLER_RESET_CMD, IGNORE_OUTPUT), shell=True)
            p.wait()

            p = Popen(CONTROLLER_SERVICE_RESTART_CMD, shell=True)
            p.wait()
            onos_running = False

            # We also seem to need to fully reset OVS sometimes for larger topologies
            p = Popen(RESET_OVS, shell=True)
            p.wait()
            p = Popen(RUN_OVS, shell=True)
            p.wait()

            while not onos_running:
                try:
                    # wait first so that if we get a 404 error we'll wait to try again
                    time.sleep(10)
                    ret = self.topology_adapter.rest_api.get_hosts()
                    # Once we get back from the API an empty list of hosts, we know that ONOS is fully-booted.
                    if ret == []:
                        onos_running = True
                        log.debug("ONOS fully-booted!")

                        # Check to make sure the switches were actually cleared...
                        uncleared_switches = self.topology_adapter.rest_api.get_switches()
                        if uncleared_switches:
                            log.error(
                                "Why do we still have switches after restarting ONOS??? they are: %s" % uncleared_switches)
                    else:
                        log.debug("hosts not cleared out of ONOS yet...")
                except IOError:
                    log.debug("still waiting for ONOS to fully restart...")

        elif self.topology_adapter is not None:
            log.debug(
                "Removing groups and flows via REST API.  This could take a while while we wait for the transactions to commit...")
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
        else:
            log.warning("No topology adapter!  Cannot reset it between runs...")

    ####   Helper functions for working with Mininet nodes/links    ####

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