# Resilient IoT Data Exchange - Collection middleware
import logging

from ride.data_path_monitor import DATA_PATH_UP, DATA_PATH_DOWN
from config import *

import topology_manager
from topology_manager.sdn_topology import SdnTopology
from scale_client.networks.util import DEFAULT_COAP_PORT

log = logging.getLogger(__name__)

# In case we are running RideC outside the original experimental framework that defined this default
try:
    from smart_campus_experiment import DISTANCE_METRIC
except ImportError:
    DISTANCE_METRIC = 'latency'


class RideC(object):
    """
    Middleware layer for managing the collection (upload) of IoT data to cloud/edge servers.  It monitors network
    topology and characteristics to determine the best routes for IoT 'publishers' to get their data to a server.
    Specifically, it manages 'DataPaths' (gateways at the edge of the local network and their abstract routes to
    the cloud server(s)) to ensure that the relevant data is delivered to the cloud via the best available path,
    if possible, and to the edge server if not possible.

    NOTE: many parameters are a DPID (data plane ID), which is expected to be formatted as returned by the SdnTopology
    instance specified in __init__
    """

    def __init__(self, edge_server=None, cloud_server=None, topology_mgr='onos',
                 reroute_policy=DEFAULT_REROUTE_POLICY, distance_metric=DISTANCE_METRIC, **kwargs):
        """
        :param edge_server: DPID of the managed edge server
        :param cloud_server: DPID of the managed cloud server
        :param topology_mgr: used as adapter to SDN controller for maintaining topology and routing information;
        optional with default 'onos'
        :type topology_mgr: SdnTopology|str
        :param reroute_policy: specifies the strategy for rerouting registered publishers to the edge server;
        can be one of: 'disjoint' (default; choose maximally-disjoint shortish paths), 'shortest' (regular shortest paths)
        :param distance_metric: the distance metric determines the length of the paths used when managing
         routing in the local network since these paths are chosen to be minimal (default='latency')
        :param kwargs: ignored (just present so we can pass args from other classes without causing errors)
        """
        # XXX: even though we KNOW an object takes no __init__ args, multiple inheritance may cause us to need
        # another super call before reaching broker; hence we should still pass the (possibly-empty) kwargs along...
        try:
            super(RideC, self).__init__(**kwargs)
        except TypeError:
            super(RideC, self).__init__()

        if not isinstance(topology_mgr, SdnTopology):
            # only adapter type specified: use default other args
            if isinstance(topology_mgr, basestring):
                self.topology_manager = topology_manager.build_topology_adapter(topology_adapter_type=topology_mgr)
            # we expect a dict to have the kwargs
            elif isinstance(topology_mgr, dict):
                self.topology_manager = topology_manager.build_topology_adapter(**topology_mgr)
            # hopefully it's a tuple!
            else:
                try:
                    self.topology_manager = topology_manager.build_topology_adapter(*topology_mgr)
                except TypeError:
                    raise TypeError("topology_mgr parameter (%s) is not of type SdnTopology and couldn't extract"
                                    " further parameters from it!" % topology_mgr)

        # ENHANCE: verify that it's a SdnTopology?  maybe accept a dict of args to its constructor?
        else:
            self.topology_manager = topology_mgr

        # RideC instances will only ever manage one of these
        # ENHANCE: relax this assumption somehow?
        self.edge_server = edge_server
        self.cloud_server = cloud_server

        # these fields will be set dynamically by their relevant add/remove methods
        # NOTE: DP just means the DataPath ID, GW is represented by its SDN DPID, and host is an (address, port) tuple
        self._gateway_for_data_path = dict()  # DP --> GW
        self._data_path_status = dict()       # DP --> status
        self._data_path_for_host = dict()     # host --> DP
        self._host_routes = dict()

        self._distance_metric = distance_metric
        self._reroute_policy = reroute_policy

        # Save the switches currently holding redirection flow rules so we can delete them later upon recovery.
        self.__redirecting_switches = set()

    ## Helper functions

    @property
    def hosts(self):
        return self._data_path_for_host.keys()

    def hosts_for_data_path(self, data_path):
        return [host for host, dp in self._data_path_for_host.items() if dp == data_path]

    def _get_host_ip_address(self, host_address):
        """Extracts the IP address component from the specified host_address.  This is intended to allow future
        migration to a different (or multiple) address format."""
        return host_address[0]

    def _get_host_port(self, host_address):
        """Extracts the port component from the specified host_address.  This is intended to allow future
        migration to a different (or multiple) address format."""
        # XXX: we assume everything is IPv4!!!
        return host_address[1]

    def get_host_dpid(self, host):
        """Returns the DPID corresponding with the given host specified in its internal format i.e. (address, port)"""
        return self.topology_manager.get_host_by_ip(self._get_host_ip_address(host))

    def _get_server_ip_address(self, server_dpid):
        """Extracts the IP address component from the specified server_dpid.  This is intended to allow future
        migration to a different (or multiple) address format."""
        return self.topology_manager.get_ip_address(server_dpid)

    def _get_server_port(self, server_dpid):
        """Extracts the port component from the specified server_dpid.  This is intended to allow future
        migration to a different (or multiple) address format."""
        # XXX: we assume everything CoAP!!!
        return DEFAULT_COAP_PORT

    @property
    def gateways(self):
        return self._gateway_for_data_path.values()

    @property
    def data_paths(self):
        return self._data_path_status.keys()

    @property
    def available_data_paths(self):
        return [dp for dp in self.data_paths if self.is_data_path_up(dp)]

    def is_data_path_up(self, data_path_id):
        return self._data_path_status[data_path_id] == DATA_PATH_UP

    def _choose_data_path(self, host_address=None):
        """
        Choose a DataPath from those currently up that's well-suited for the specified host.
        Currently, we just choose the 'highest priority' (as determined by DataPathID order low-to-high) DataPath
        that is currently functional.
        :param host_address: optional and currently ignored
        :return:
        """
        dp_choices = [dp for dp in self.data_paths if self.is_data_path_up(dp)]
        # TODO: how to handle none being available??? random choice? random.choice(self.data_paths)
        dp_choices = sorted(dp_choices)
        chosen_dp = dp_choices[0]
        log.debug("assigning host %s to DP %s" % (host_address, chosen_dp))
        return chosen_dp

    ## the main public API: control, registration and notification functions

    def update(self):
        """
        Tells RideC to update itself by getting the latest topology and changing the assigned host routes if necessary.
        :return: dict of update host routes (host: route)
        """

        # ENHANCE: extend the REST APIs to support updating the topology rather than getting a whole new one.
        self.topology_manager.build_topology(from_scratch=True)

        # Update all the routes
        # ENHANCE: only update some of them? possibly based on updates to topology?
        updated_routes = dict()
        for host in self.hosts:
            route = self._update_host_route(host)
            updated_routes[host] = route

        return updated_routes

    def on_data_path_status_change(self, data_path_id, status):
        """
        Called to notify the change of a DataPath's status; updates the internal data structures that track it and
        reroutes hosts to a different DataPath if needed (i.e. it goes down).
        :param data_path_id: ID used to originally register the DataPath
        :param status: 0 for down, 1 for up
        :return:
        """

        log.debug("DataPath %s status change: %s" % (data_path_id, status))

        # only need to do anything if status actually changes
        if self._data_path_status[data_path_id] == status:
            return

        self._data_path_status[data_path_id] = status
        if status == DATA_PATH_DOWN:
            if self.available_data_paths:
                self._failover_data_path(data_path_id)
            else:
                self._on_all_data_paths_down()
        elif status == DATA_PATH_UP:
            self._recover_data_path(data_path_id)
        else:
            log.error("unrecognized DataPath status %s for DP %s" % (status, data_path_id))

    # ENHANCE: unregister versions of these?

    def register_data_path(self, data_path_id, gateway_id, cloud_id):
        """
        Registers the specified DataPath under RideC's management.
        :param data_path_id: a unique ID representing this DataPath
        :param gateway_id: DPID of the local gateway that this DataPath passes through (originates at)
        :param cloud_id: DPID of the cloud server that this DataPath terminates at
        :return:
        :raises ValueError: if data_path_id is already registered or gateway_id is not found in the topology
        """
        if gateway_id not in self.topology_manager.topo:
            raise ValueError("gateway_id %s not found in our topology!  Cannot register DataPath..." % gateway_id)
        # ENHANCE: support these?
        if data_path_id in self._data_path_status:
            raise ValueError("DataPath with id %s already registered!  We do not currently support updating it..." % data_path_id)
        assert cloud_id == self.cloud_server, "cloud_id specified that isn't the same as our cloud_server!  this is not yet supported..."

        self._data_path_status[data_path_id] = DATA_PATH_UP
        self._gateway_for_data_path[data_path_id] = gateway_id
        # ENHANCE: implement this, which might include calling some remote node's API to start up a probe to this cloud...
        # self._cloud_for_data_path = cloud_id

    # ENHANCE: should accept
    def register_host(self, host_address, use_data_path=None):
        """
        Registers the specified host as an IoT data publisher managed by RideC.
        :param host_address: address of the host formatted as per the family of the socket the registration was
         received from (e.g. (ipv4_add, src_port) for IPv4).  NOTE: we need the host's source port # too so we can
          properly distinguish traffic of one application from another on the same host!
        :param use_data_path: optional argument that sets the initial DataPath the host is assigned to.  Default
        behavior (or if the requested DataPath is not UP) is to pick an UP DataPath arbitrarily.
        :raises ValueError: if host or DataPath aren't found or if the host is already registered
        :return: the DataPath the registered host is assigned to
        """

        try:
            # can only use a functional DataPath
            if use_data_path is not None and not self.is_data_path_up(use_data_path):
                use_data_path = None
        except KeyError:
            raise ValueError("DataPath %s not found!  Can't assign the registered host to it..." % use_data_path)

        try:
            self.topology_manager.get_host_by_ip(self._get_host_ip_address(host_address))
        except (KeyError, ValueError):
            raise ValueError("host %s not found!  Cannot register it for RideC..." % host_address)
        if host_address in self.hosts:
            raise ValueError("host %s already registered!  We currently do not support updating registrations..." % host_address)

        if use_data_path is None:
            use_data_path = self._choose_data_path(host_address)

        self._data_path_for_host[host_address] = use_data_path
        self._update_host_route(host_address)

        return use_data_path

    ## DataPath and routing management APIs: should really be considered protected methods

    def _update_host_route(self, host_address, route=None):
        """
        Update the given host's route to the optionally-specified one, which by default is chosen for you;
         install flow rules if necessary.
        :param host_address:
        :return: the assigned route
        """

        if route is None:
            route = self._get_host_route(host_address)

        # only update flow rules if necessary
        if host_address not in self._host_routes or route != self._host_routes[host_address]:
            try:
                flow_rules = self.topology_manager.build_flow_rules_from_path(route, priority=STATIC_PATH_FLOW_RULE_PRIORITY)
                if not self.topology_manager.install_flow_rules(flow_rules):
                    log.error("problem installing batch of flow rules for host %s: %s" % (host_address, flow_rules))

                # do this last in case we failed to install flow rules
                self._host_routes[host_address] = route
            except BaseException as e:
                log.error("building/installing flow rules for path %s failed with error: %s" % (route, e))

        return route

    def _get_host_route(self, host_address, dest=None):
        """Return the route from the specified host to the specified destination. By default, this route goes through
        the gateway responsible for its assigned DataPath and eventually to the cloud server."""

        # If the destination wasn't specified, we need to extend the route to the cloud server while ensuring
        # it goes through the right gateway, hence two steps...
        cloud_gw_route = None
        if dest is None:
            data_path = self._data_path_for_host[host_address]
            gateway = self._gateway_for_data_path[data_path]
            # ENHANCE: choose the assigned cloud if we support multiple!
            cloud_gw_route = self.topology_manager.get_path(gateway, self.cloud_server)
            dest = gateway

        host_dpid = self.topology_manager.get_host_by_ip(self._get_host_ip_address(host_address))
        route = self.topology_manager.get_path(host_dpid, dest, weight=self._distance_metric)
        if cloud_gw_route:
            route = self.topology_manager.merge_paths(route, cloud_gw_route)
        return route

    def _failover_data_path(self, data_path):
        """
        Fails over any routes using the specified DataPath to one of the other available ones.
        :param data_path:
        :return:
        """
        impacted_hosts = self.hosts_for_data_path(data_path)
        for h in impacted_hosts:
            self._data_path_for_host[h] = self._choose_data_path(h)
            self._update_host_route(h)

    def _recover_data_path(self, data_path):
        """
        Reacts to the specified DataPath recovering by recomputing host assignments and possibly updating them if
        they're assigned to a different (possibly this newly-recovered) DataPath.  Note that we also have to remove
        any flow rules doing redirection here so that they don't prevent communication with the primary data sink (cloud).
        :param data_path: currently ignored
        :return:
        """

        log.info("DataPath %s recovered! Clearing redirection flows and reassigning hosts..." % data_path)

        self.clear_redirection_flows()

        for h in self.hosts:
            old_dp = self._data_path_for_host[h]
            new_dp = self._data_path_for_host[h] = self._choose_data_path(h)
            if old_dp is None or old_dp != new_dp:
                self._update_host_route(h)

    def _on_all_data_paths_down(self):
        """
        When no DataPaths are available, our default behavior is to reroute all hosts to the edge server.
        :return:
        """

        log.info("All DataPaths down!  Re-routing hosts to edge server...")

        # ENHANCE: choose from several cloud/edge servers
        old_dest = self.cloud_server
        new_dest = self.edge_server

        # Comparing two strategies: rerouting via shortest path routing VS. rerouting via maximally-disjoint paths
        # Based on the policy, we'll collect the routes to be used for the hosts into a dict so we can reference them later
        if self._reroute_policy == 'shortest':
            routes = {self.get_host_dpid(h): self._get_host_route(h, new_dest) for h in self.hosts}
        else:
            if self._reroute_policy != 'disjoint':
                log.error("unknown reroute_policy '%s'; defaulting to 'disjoint'...")
            # since we can have a host registered with multiple ports, we should just make this a unique list so their
            # flows take the same path, though in the future we may want to assign different paths for different flows...
            host_dpids = set(self.get_host_dpid(h) for h in self.hosts)
            # ENHANCE: pre-compute these for faster re-route
            routes = {p[0]: p for p in self.topology_manager.get_multi_source_disjoint_paths(host_dpids, new_dest, weight=self._distance_metric)}
            assert list(sorted(routes.keys())) == list(sorted(host_dpids)), "not all hosts accounted for in disjoint paths!" \
                                                                            " Got: %s\nMissing: %s" % (routes, set(host_dpids) - set(routes.keys()))

        # TODO: skip over ones that are already routing there?  or just adjust the weights used to choose between edge/cloud?
        flow_rules = []
        for h in self.hosts:
            host_dpid = self.get_host_dpid(h)
            assert host_dpid != old_dest and host_dpid != new_dest
            route = routes[host_dpid]
            log.debug("host re-route path: %s" % route)

            # ENHANCE: may need to handle other address families?  Or transport layers?
            host_src_port = self._get_host_port(h)
            old_dst_port = self._get_server_port(old_dest)
            new_dst_port = self._get_server_port(new_dest)
            flow_rules.extend(self.topology_manager.build_redirection_flow_rules(host_dpid, old_dest, new_dest,
                                                                            route=route, tp_protocol='udp',
                                                                            source_port=host_src_port,
                                                                            old_dest_port=old_dst_port,
                                                                            new_dest_port=new_dst_port,
                                                                            priority=REDIRECTION_FLOW_RULE_PRIORITY))

            # XXX: Save the switches that are doing actual redirection translations of addresses/ports.  Upon recovery,
            # we will later remove these flow rules from them.  Note that the current implementation just uses the
            # two switches saved below for this translation.
            trans_switch1 = route[1]
            trans_switch2 = route[-2]
            log.debug("saving redirection switches %s and %s for clearing flow rules later..." % (trans_switch1, trans_switch2))
            self.__redirecting_switches.add(trans_switch1)
            self.__redirecting_switches.add(trans_switch2)

            self._host_routes[h] = route
            self._data_path_for_host[h] = None

        log.debug("installing redirection flow rules")
        if not self.topology_manager.install_flow_rules(flow_rules):
            log.error("failed to install redirection flow rules: %s" % flow_rules)

        log.debug("finished re-routing hosts to edge!")

    def clear_redirection_flows(self):
        """
        Clears all redirection flow rules using a serious HACK:
        We assume that we can leave the flow rules that simply forward traffic since it shouldn't cause any problems.
        :return:
        """

        # XXX: could get all flows for the switches that do actual translation; find all that have the redirect
        # priority, get their flowId, and then delete all those flows?  Super hacky but might actually be easier than
        # trying to look through returned flow rules to find those with the same matches/actions...

        for switch_id in self.__redirecting_switches:
            # XXX: these are all ONOS api-specific methods for digging into the flows' details!
            flows = self.topology_manager.rest_api.get_flow_rules(switch_id)
            flow_ids_to_kill = []
            for f in flows:
                # XXX: we find the target translation flow rules by just looking for those
                if f['priority'] == REDIRECTION_FLOW_RULE_PRIORITY:
                    flow_ids_to_kill.append(f['id'])

            for f in flow_ids_to_kill:
                if not self.topology_manager.remove_flow_rule(switch_id, f):
                    log.error("Failed to remove swithc %s's redirection flow: %s" % (switch_id, f))

        self.__redirecting_switches.clear()