# @author: Kyle Benson
# (c) Kyle Benson 2018

import logging
log = logging.getLogger(__name__)

import os
import argparse
from subprocess import Popen
import errno
import time
from collections import OrderedDict
import subprocess

from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import RemoteController, Switch, Host, OVSKernelSwitch
from mininet.cli import CLI

import topology_manager.sdn_topology
from network_experiment import NetworkExperiment
from config import *


class MininetSdnExperiment(NetworkExperiment):
    """
    An abstract object used for running automated experiments in Mininet.  The basic idea is that you'll subclass it
    (see MininetSmartCampusExperiment as an example) to set up additional configurations and control the overall
    flow of the experiment.  This just helps get a lot of the boilerplate out of the way and integrates it with the
    SdnTopology Python-based framework for interacting with the SDN controller.  See run_experiment for an outline of
    your expected workflow.  Also, expect to override some of the helper methods (e.g. setup_topology()) in your
    subclass in order to add your experiment's logic.
    """

    def __init__(self, controller_ip=CONTROLLER_IP, controller_port=CONTROLLER_REST_API_PORT,
                 topology_adapter=DEFAULT_TOPOLOGY_ADAPTER, show_cli=False,
                 experiment_duration=EXPERIMENT_DURATION, **kwargs):
        """
        :param controller_ip:
        :param controller_port:
        :param topology_adapter:
        :param show_cli:
        :param kwargs:
        """

        # XXX: BUG: we couldn't get multiple runs per process working completely for Mininet, so issue a warning:
        if kwargs.get('nruns', 1) > 1:
            raise NotImplementedError("Cannot support >1 runs per process in Mininet currently!")

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

        # The Mininet network, which will be created in setup_topology()
        self.net = None

        # XXX: see note in setup_topology() about replacing server hosts with a switch to ease multi-homing
        # These dicts maps the server Mininet node to the serving Mininet switch/link
        self.server_switches = dict()
        self.server_switch_links = dict()

        self.nats = dict()  # NAT host --> connection point host mapping
        # we actually build the NATs during start_network, so they'll be saved as (connection_point, nat_name) pairs until then
        self._nats_to_add = []

        # Save Popen objects to later ensure procs terminate before exiting Mininet or we'll end up with hanging procs.
        # The key should be a string representing what this proc is running e.g. some name or even the whole command.
        # Note that these may include iperf commands too!
        # We use an OrderedDict so as to preserve the order we clean them up at the end
        self.popens = OrderedDict()

        self.experiment_duration = experiment_duration
        # We'll optionally drop to a CLI after the experiment completes for further poking around
        self.show_cli = show_cli

        # used to store log files (for debugging) and output files (results) from host processes ran during experiments
        # they'll be accessed without the leading '_', which will dynamically build them if necessary
        self._logs_dir = None
        self._outputs_dir = None

        # Disable some of the more verbose and unnecessary loggers
        for _logger_name in LOGGERS_TO_DISABLE:
            l = logging.getLogger(_logger_name)
            l.setLevel(logging.ERROR)

    @classmethod
    def get_arg_parser(cls, parents=(topology_manager.sdn_topology.SdnTopology.get_arg_parser(),
                                     NetworkExperiment.get_arg_parser()), add_help=False):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        # argument parser that can be combined with others when this class is used in a script
        # need to not add help options to use that feature, though
        arg_parser = argparse.ArgumentParser(add_help=add_help, parents=parents, conflict_handler='resolve')

        # Since we'll almost always be running this experiment in a VM, we change the default controller IP so that the
        # Mininet test nodes can contact the controller from within Mininet:
        arg_parser.set_defaults(controller_ip=CONTROLLER_IP)

        arg_parser.add_argument('--cli', '-cli', dest='show_cli', action='store_true',
                                help='''displays the Mininet CLI after running the experiment. This is useful for
                                debugging problems as it prevents the OVS/controller state from being wiped after
                                the experiment and keeps the network topology up.''')
        arg_parser.add_argument('--duration', '-q', dest='experiment_duration', type=int, default=EXPERIMENT_DURATION,
                                help='''duration to run the actual experiment for, which defaults to a value set
                                in config.py''')
        return arg_parser

    def setup_experiment(self):
        """Finishes setting up the network by default."""
        self.start_network()
        self.ensure_network_setup()

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

    def add_switch(self, switch, dpid):
        s = self.net.addSwitch(switch, dpid=dpid, cls=OVSKernelSwitch)
        log.debug("adding switch %s at DPID %s" % (switch, s.dpid))
        self.switches.append(s)
        return s

    def add_host(self, hostname, ip, mac):
        log.debug("Adding host %s with IP=%s and MAC=%s" % (hostname, ip, mac))
        h = self.net.addHost(hostname, ip=ip, mac=mac)
        self.hosts.append(h)
        return h

    def add_server(self, name, ip, mac, with_switch=True, server_switch_name=None, server_switch_dpid=None):
        """
        HACK: adds a server host with the specified attributes attached to a switch (unless otherwise specified).
        This additional switch enables easier multi-homing e.g. older ONOS can only handle a single MAC address per host.
        :param name:
        :param ip:
        :param mac:
        :param with_switch:
        :param server_switch_name:
        :param server_switch_dpid:
        :return: (server, server_switch) if with_switch is True, else just server
        """

        server_host = self.add_host(name, ip=ip, mac=mac)
        self.servers.append(server_host)

        if with_switch:
            if server_switch_name is None:
                server_switch_name = name + '_sw'
            if server_switch_dpid is None and mac is not None:
                server_switch_dpid = '55:55:' + mac

            server_switch = self.add_switch(server_switch_name, dpid=server_switch_dpid)
            self.server_switches[server_host] = server_switch

            # NOTE: we don't treat this like the other links
            l = self.add_link(name, server_switch_name, use_tc=False)
            self.server_switch_links[server_host] = l

            return server_host, server_switch
        else:
            return server_host

    def add_link(self, from_link, to_link, use_tc=True, bandwidth=None, latency=None, jitter=None, error_rate=None):
        """
        Adds a link from the specified node ID (or Mininet node) to the destination and (unless otherwise specified) sets
        channel characteristics appropriately, deferring to the defaults set in self.

        :param from_link:
        :param to_link:
        :param use_tc: whether to use Linux traffic control (TC) to emulate channel characteristics
        :param bandwidth:
        :param latency:
        :param jitter:
        :param error_rate: link loss rate expressed as an integer percentage
        :return:
        """

        # XXX: handle either names or actual Mininet nodes
        if isinstance(from_link, basestring):
            from_link = self.net.get(from_link)
        if isinstance(to_link, basestring):
            to_link = self.net.get(to_link)

        if use_tc:
            bw = bandwidth if bandwidth is not None else self.bandwidth
            delay = '%fms' % (latency if latency is not None else self.latency)
            jitter = '%fms' % (jitter if jitter is not None else self.jitter)
            loss = error_rate if error_rate is not None else self.error_rate

            log.debug("adding link from %s to %s with channel: latency=%s, jitter=%s, loss=%d, BW=%s" % \
                      (from_link, to_link, delay, jitter, loss, bw))

            # For configuration options, see mininet.link.TCIntf.config()
            l = self.net.addLink(from_link, to_link, cls=TCLink,
                                 bw=bw, delay=delay, jitter=jitter, loss=loss)
        else:
            log.debug("adding link from %s to %s without TC" % (from_link, to_link))
            l = self.net.addLink(from_link, to_link)
        self.links.append(l)
        return l

    def update_link_params(self, from_link, to_link, **params):
        """
        Update the channel characteristics of an existing link, keeping the original configuration of each parameter
        when not specified.  Accepts all channel parameters handled in add_link().
        :param from_link:
        :param to_link:
        :param params: the parameters to update with new values (see add_link() for possible options)
        :return:
        """

        # need to transform our naming scheme to Mininet's
        if 'latency' in params:
            params['delay'] = '%fms' % params.pop('latency')
        if 'jitter' in params:
            params['jitter'] = '%fms' % params.pop('jitter')
        if 'bandwidth' in params:
            params['bw'] = params.pop('bandwidth')
        if 'error_rate' in params:
            params['loss'] = params.pop('error_rate')

        log.debug("updating link parameters from %s to %s: %s" % \
                  (from_link.name, to_link.name, params))

        # Need to set the actual interfaces' parameters rather than the whole links
        for from_iface, to_iface in from_link.connectionsTo(to_link):
            # keep the previous parameters where possible
            new_params = from_iface.params.copy()
            new_params.update(params)
            from_iface.config(**new_params)

            new_params = to_iface.params.copy()
            new_params.update(params)
            to_iface.config(**new_params)

    def add_nat(self, connection_point, nat_name=None, nat_ip=None):
        """Add a NAT to the specified node in the network.  It will actually be built in 'start_network' since we have
        to build the network first in order to properly configure hosts.  Hence, self.nats will not contain nats until
        that point.
        NOTE: We use this primarily so the server can communicate with the SDN controller's REST API.
        WARNING: it's unclear that this will handle multiple NATs properly, so be wary of doing so!

        :param connection_point: the Mininet node to connect the NAT to, which can be a Switch or even Host
            WARNING: if you connect the NAT to a host node, only that node will get a default route that lets it connect via the NAT!
        :param nat_name: you can optionally name the NAT, or let Mininet do it for you
        :param nat_ip: you can optionally specify the IP address or we'll use the subnet given in config.py
            WARNING: if you specify it and are connecting to a server make sure they're in the same subnet!
        """

        self._nats_to_add.append((connection_point, nat_name, nat_ip))

    def _build_nats(self):
        """Actually starts the NATs.  Override this to do some different configuration..."""

        # RIDE-specific NOTE: because we didn't add it to the actual SdnTopology, we don't need
        # to worry about it getting failed.  However, we do need to ensure it
        # connects directly to the server to avoid failures disconnecting it.
        # HACK: directly connect NAT to the server, set a route for it, and
        # handle this hacky IP address configuration

        for connection_point, nat_name, nat_ip in self._nats_to_add:
            # we have a few different options of kwargs to specify, so just build a dict
            kwargs = dict()
            if not nat_ip:
                # we should only generate one if we're going to do the same for the host interface or else we could be outside the subnet!
                if isinstance(connection_point, Host):
                    nat_ip = NAT_SERVER_IP_ADDRESS % (len(self.nats) + 2)
            if nat_ip:
                kwargs['ip'] = nat_ip

            if nat_name:
                kwargs['name'] = nat_name

            nat = self.net.addNAT(connect=connection_point, **kwargs)

            # Now we have to handle things differently depending on whether this connection_point is a switch or host
            if isinstance(connection_point, Host):

                # configure the host connection point and set its default route so that it can route via the NAT
                srv_ip = NAT_SERVER_IP_ADDRESS % (len(self.nats) + 3)
                srv_iface = sorted(connection_point.intfNames())[-1]
                connection_point.intf(srv_iface).setIP(srv_ip)

            # This will set up the nat host
            if nat_ip:
                nat.configDefault(ip=nat_ip)
            else:
                nat.configDefault()

            self.nats[nat] = connection_point

    def start_network(self):
        """Actually start the Mininet network."""
        log.info('*** Starting network')
        log.debug("Building Network...")
        self.net.build()

        # Now that the network is built (and hosts configured), we can add the NAT.  By doing this, Mininet will handle
        # setting default routes for the hosts.
        self._build_nats()

        log.debug("Network built; starting...")
        self.net.start()
        log.debug("Started!  Waiting for switch connections...")
        self.net.waitConnected()  # ensure switches connect
        log.debug("Switches connected!")

        # XXX: if we attach the nat to a host rather than switch, the default gateway won't be set properly so we now
        # set the default routes for each NAT'ed host, which MUST be done after the network starts
        for nat, host in self.nats.items():
            if isinstance(host, Host):
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
                losses = []
                if not hosts or not self.servers:
                    log.warning("Empty hosts/servers list!  Can't do non-ALL_PAIRS ping!")

                for h in hosts:
                    for s in self.servers:
                        if s != h:
                            losses.append(self.net.ping((h, s), timeout=2))
                loss = sum(losses)/float(len(losses)) if losses else 0

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
                        if s != src:
                            src.setARP(ip=server_ip, mac=server_mac)
                            s.setARP(ip=src.IP(), mac=src.MAC())

        ping_hosts(hosts)
        # Need to sleep so that the controller has a chance to converge its topology again...
        log.info("*** waiting for controller to recognize all hosts...")
        time.sleep(5)
        # Now connect the SdnTopology and verify that all the non-NAT hosts, links, and switches are available through it
        expected_nhosts = len(hosts)   # ignore NAT, but include servers
        expected_nlinks = len(self.links)
        expected_nswitches = len(self.switches)
        n_sdn_links = 0
        n_sdn_switches = 0
        n_sdn_hosts = 0
        ntries = 1
        while n_sdn_hosts < expected_nhosts or n_sdn_links < expected_nlinks or n_sdn_switches < expected_nswitches:
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

    @classmethod
    def wait_then_kill_proc(cls, proc, name, timeout=1, wait_time=2):
        """
        Repeatedly poll the process until wait_time is up, then kill it if it doesn't finish
        :param proc:
        :type proc: subprocess.Popen
        :param name:
        :param timeout:
        :param wait_time:
        :return:
        """
        assert isinstance(proc, Popen)  # for typing
        for i in range(wait_time / timeout):
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(timeout)
        else:
            log.error("process never quit: killing popen %s..." % name)
            try:
                proc.kill()
            except OSError:
                log.debug("proc.kill() gave error %s, which is usually just it already having terminated...")
            ret = proc.wait()
            log.error("now it exited with code %d" % ret)

        return ret

    def cleanup_procs(self):
        """Verify that each of the processes we spawned exit properly.  If they haven't exited after a while, kill
        them explicitly and then make sure they exited after that.  We do this because Mininet can leak open processes
        if they aren't properly closed."""

        log.debug("cleaning up procs: %s" % list(self.popens))

        for name, p in self.popens.items():
            ret = self.wait_then_kill_proc(p, name)
            if ret is None:
                log.error("Proc never quit: %s" % name)
            elif ret != 0:
                # NOTE: you may need to pipe this in from your client manually
                if ret == errno.ENETUNREACH:
                    # TODO: handle this error appropriately: record failed clients in results?
                    log.error("Proc failed due to unreachable network: %s" % name)
                else:
                    log.error("Exit code %d from proc %s" % (p.returncode, name))

        # XXX: somehow there still seem to be client processes surviving the .kill() commands; this finishes them off:
        # NOTE: SCALE-based experiment-specific!
        p = Popen(CLEANUP_SCALE_CLIENTS, shell=True)
        p.wait()
        p = Popen(CLEANUP_CLIENTS_COMMAND % 'iperf', shell=True)
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

    @property
    def outputs_dir(self):
        if self._outputs_dir is None:
            self.build_outputs_logs_dirs()
        return self._outputs_dir

    @property
    def logs_dir(self):
        if self._logs_dir is None:
            self.build_outputs_logs_dirs()
        return self._logs_dir

    def build_outputs_logs_dirs(self, output_filename=None):
        """
        Creates, stores in self, and returns the path to two directories:
            outputs_dir starts with 'outputs_' for putting the outputs of whatever host processes you run in
            logs_dir starts with 'logs_' for redirecting the raw output from those processes for debugging purposes.
        The latter may be disabled by setting WITH_LOGS=False in config.py, but this method will just return None for
        that directory (stil returns a 2-tuple).

        NOTE: we used to try doing multiple runs per process, which is why these directories also have nested 'run#'
            dirs inside of them, but we opt not to remove this as too much else would need to change...

        :param output_filename: optionally specify the output filename, which will have its file extension removed and
                will be prepended as per above for the dir names (default=self.output_filename)
        :returns outputs_dir, logs_dir: the directories (relative to the experiment output
         file) in which the output and log files, respectively, are stored for this run
        """

        if output_filename is None:
            output_filename = self.output_filename

        # The logs and output files go in nested directories rooted
        # at the same level as the whole experiment's output file.
        # We typically name the output file as results_$PARAMS.json, so cut off the front and extension
        root_dir = os.path.dirname(output_filename)
        base_dirname = os.path.splitext(os.path.basename(output_filename))[0]
        if base_dirname.startswith('results_'):
            base_dirname = base_dirname[8:]
        if WITH_LOGS:
            logs_dir = os.path.join(root_dir, 'logs_%s' % base_dirname, 'run%d' % self.current_run_number)
            try:
                os.makedirs(logs_dir)
                # XXX: since root is running this, we need to adjust the permissions, but using mode=0777 in os.mkdir()
                # doesn't work for some systems...
                os.chmod(logs_dir, 0777)
            except OSError:
                pass
        else:
            logs_dir = None
        outputs_dir =  os.path.join(root_dir, 'outputs_%s' % base_dirname, 'run%d' % self.current_run_number)
        try:
            os.makedirs(outputs_dir)
            os.chmod(outputs_dir, 0777)
        except OSError:
            pass

        self._logs_dir = logs_dir
        self._outputs_dir = outputs_dir

        return outputs_dir, logs_dir

    def record_result(self, result):
        """Override to also record the outputs/logs_dirs"""

        # make the paths relative to the root directory in which the whole experiment output file is stored
        # as otherwise the paths are dependent on where the cwd is
        # WARNING: if a custom path was specified this may cause an error!
        root_dir = os.path.dirname(self.output_filename)
        logs_dir = os.path.relpath(self.logs_dir, root_dir) if self.logs_dir else None
        outputs_dir = os.path.relpath(self.outputs_dir, root_dir) if self.outputs_dir else None
        result['outputs_dir'] = outputs_dir
        result['logs_dir'] = logs_dir

        return super(MininetSdnExperiment, self).record_result(result)

    def redirect_output_to_log(self, cmd, filename):
        """Returns a modified version of the command string that redirects all output to a log file composed of the
        logs_dir and the specified output filename."""
        logs_dir = self.logs_dir
        if logs_dir is None:
            _, logs_dir = self.build_outputs_logs_dirs()
        return self.redirect_output(cmd, filename, dirname=logs_dir)

    def redirect_output(self, cmd, filename, dirname=None):
        """
        Returns a modified version of the command string that redirects all output to a file composed of the
        the specified dirname (default=outputs_dir) and the specified output filename.
        :param cmd:
        :param filename:
        :param dirname: the directory to put the output in, which is self.outputs_dir by default
        """

        if dirname is None:
            dirname = self.outputs_dir
            if dirname is None:
                dirname, _ = self.build_outputs_logs_dirs()

        return cmd + " > %s 2>&1" % os.path.join(dirname, filename)

    def run_proc(self, cmd, host, name=None, **kwargs):
        """Runs the specified command on the requested Mininet host.  Saves the popen object to later ensure all procs
        quit completely before exiting mininet.
        :param cmd: the command to run, which may contain shell symbols
        :param host: the Mininet host to run the command on
        :param name: the name to store this command under in the popens attribute; must be unique! (default=host.name)
        :param kwargs: optional additional kwargs passed to Mininet.Host.popen
        """

        # the node.sendCmd option in mininet only allows a single
        # outstanding command at a time and cancels any current
        # ones when net.CLI is called.  Hence, we need popen.
        log.debug("cmd@%s: %s" % (host.name, cmd))
        p = host.popen(cmd, shell=True, **kwargs)

        # save the popen to ensure it quits later, but ensure there's not already a popen by that name
        if name is None:
            name = host.name
        if name in self.popens:
            name = "%s[cmd=%s]" % (name, cmd)
            if name in self.popens:
                log.warning("Two identical commands being run on host!  Can't generate unique name...\n%s" % name)
        self.popens[name] = p

        return p

    def make_host_cmd(self, cmd, hostname=None):
        """
        This unnecessary helper function is implemented specifically for use with the SCALE client software.
        It formats the specified command so that it includes various configurations (quit time, debug level, etc.),
        formats the shell command appropriately, and optionally redirects output to a log file.

        :param cmd:
        :param hostname: name to use for log file redirection (can be ignored if logging not in use)
        :return:
        """

        base_args = "-q %d --log %s" % (self.experiment_duration, self.debug_level)
        cmd = SCALE_CLIENT_BASE_COMMAND % (base_args + cmd)

        if WITH_LOGS:
            cmd = self.redirect_output_to_log(cmd, hostname)
        else:
            assert hostname is not None, "you must specify the hostname for log file output redirection!"

        return cmd

    def iperf(self, client, server, port=None, bandwidth=None, duration=None,
              output_results=False, pipe_results=False, use_mininet=False):
        """Runs iperf (UDP) between the specified hosts.
        :param client: source of iperf traffic
        :param server: destination of iperf traffic
        :param port: the port number to use (default=IPERF_BASE_PORT)
        :param bandwidth: the requested bandwidth (default=self.bandwidth)
        :param duration: seconds to run it for (default=self.experiment_duration)
               WARNING: server doesn't quit! use_mininet version for now...
        :param output_results: if specified, outputs the results to a file whose name is either the string specified
            (prepended with the client/server name) or 'iperf_<client/server name>'
            Returns these file names
            WARNING: careful to make them unique and don't try to communicate() with the popen objects!
        :param pipe_results: if True, sets the stderr/stdout=PIP flags so you can communicate() with the return Popens
        :param use_mininet: if True, just calls the Mininet.net.iperf() on those hosts, which BLOCKS! Returns parsed results
        :returns client_results, srv_results
        """

        if not use_mininet and pipe_results:
            raise NotImplementedError("piping results doesn't really work because we need to"
                                      " gracefully kill the server to get its results"
                                      " (maybe just use the -o option of iperf and parse them later?)")

        if bandwidth is None:
            bandwidth = self.bandwidth
        if duration is None:
            duration = self.experiment_duration

        # We typically don't do this because it blocks!
        if use_mininet:
            _, srv_res, cli_res = self.net.iperf([client, server], l4Type='UDP', udpBw="%fM" % bandwidth, seconds=duration, port=port)
            # return results so that our interface is consistent: invert cli/srv and cut out the UDP BW value that's just for target BW anyway
            return cli_res, srv_res

        log.info("iperf from %s to %s" % (client, server))
        # TODO: figure out if we can get this working? we had never tried to parse results before...
        # it seems to run okay, but the server never quits so we'll have to do so gracefully
        # NOTE: we used to have ' & ' at the end to run these in background, but probably don't want to do that

        client_cmd = 'iperf -p %d -t %d -c %s -u -b %fM' % (port, duration, server.IP(), bandwidth)
        client_name = "iperf %s -> %s" % (client, server)
        srv_cmd = 'iperf -p %d -u -s' % port
        srv_name = "iperf %s <- %s" % (server, client)

        if output_results:
            if not isinstance(output_results, basestring):
                output_results = "iperf_%s"
            else:
                output_results = "%s_" + output_results

            out_dir = self.outputs_dir
            if out_dir is None:
                out_dir, _ = self.build_outputs_logs_dirs()

            # TODO: try using this -o option?
            client_fname = output_results % client.name
            client_fname = os.path.join(out_dir, client_fname)
            # client_cmd += " -o %s" % client_fname
            srv_fname = output_results % server.name
            srv_fname = os.path.join(out_dir, srv_fname)
            # srv_cmd += " -o %s" % srv_fname

            # OLD but working way of doing this
            client_cmd = self.redirect_output(client_cmd, client_fname, dirname='')
            srv_cmd = self.redirect_output(srv_cmd, srv_fname, dirname='')

        if pipe_results:
            p1 = self.run_proc(client_cmd, client, client_name, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            p2 = self.run_proc(srv_cmd, server, srv_name, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        else:
            p1 = self.run_proc(client_cmd, client, client_name)
            p2 = self.run_proc(srv_cmd, server, srv_name)

        if output_results:
            return client_fname, srv_fname
        return p1, p2

    def parse_iperf(self, data):
        # Mininet's iperf API has a helper method!
        return self.net._parseIperf(data)

    def parse_iperf_from_popen(self, popen_handle, timeout=10):
        """Given a Popen handle, gets the results and returns them (using Mininet helper func) if able to parse properly,
         else logs errors and returns None.
         :param timeout: default of 10 because we expect this process to have already finished
         :returns bandwidth: parsed from results
         """

        try:
            ret = self.wait_then_kill_proc(popen_handle, "iperf result", wait_time=timeout)
            if True:
            # if ret == 0:
                print 'waiting'
                popen_handle.wait()
                print 'communicating'
                (stdoutdata, stderrdata) = popen_handle.communicate()
                print 'yay'
            else:
                log.warning("iperf proc never finished; gave error code %d" % ret)
                return None
        except BaseException as e:
            log.warning("reading iperf results failed with exception: %s\n"
                        "NOTE: did you remember to set stdout=PIPE, stderr=PIPE to the popen args?" % e)
            return None

        if stderrdata:
            log.warning("iperf stderr: %s" % stderrdata)

        return self.parse_iperf(stdoutdata)

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