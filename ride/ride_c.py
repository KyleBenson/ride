# Resilient IoT Data Exchange - Collection middleware
import logging
import random

import topology_manager
from ride.data_path_monitor import DATA_PATH_UP, DATA_PATH_DOWN
from topology_manager.sdn_topology import SdnTopology

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

    def __init__(self, edge_server=None, cloud_server=None, topology_mgr='onos', distance_metric=DISTANCE_METRIC, **kwargs):
        """
        :param edge_server: DPID of the managed edge server
        :param cloud_server: DPID of the managed cloud server
        :param topology_mgr: used as adapter to SDN controller for maintaining topology and routing information;
        optional with default 'onos'
        :type topology_mgr: SdnTopology|str
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
        # NOTE: DP just means the DataPath ID, whereas host/GW are represented by their SDN DPID
        self._gateway_for_data_path = dict()  # DP --> GW
        self._data_path_status = dict()       # DP --> status
        self._data_path_for_host = dict()     # host --> DP
        self._host_routes = dict()

        self._distance_metric = distance_metric

    ## Helper functions

    @property
    def hosts(self):
        return self._data_path_for_host.keys()

    def hosts_for_data_path(self, data_path):
        return [host for host, dp in self._data_path_for_host.items() if dp == data_path]

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

    def _choose_data_path(self, host_id=None):
        """
        Choose a DataPath from those currently up that's well-suited for the specified host.
        Currently, we just choose the 'highest priority' (as determined by DataPathID order low-to-high) DataPath
        that is currently functional.
        :param host_id: optional and currently ignored
        :return:
        """
        dp_choices = [dp for dp in self.data_paths if self.is_data_path_up(dp)]
        # TODO: how to handle none being available??? random choice? random.choice(self.data_paths)
        dp_choices = sorted(dp_choices)
        chosen_dp = dp_choices[0]
        log.debug("assigning host %s to DP %s" % (host_id, chosen_dp))
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

        self._data_path_status[data_path_id] = status
        if status == DATA_PATH_DOWN:
            if self.available_data_paths:
                self._failover_data_path(data_path_id)
            else:
                self._on_all_data_paths_down()

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

    def register_host(self, host_id, use_data_path=None):
        """
        Registers the specified host as an IoT data publisher managed by RideC.
        :param host_id: DPID of the host
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

        if not host_id in self.topology_manager.topo:
            raise ValueError("host %s not found!  Cannot register it for RideC..." % host_id)
        if host_id in self.hosts:
            raise ValueError("host %s already registered!  We currently do not support updating registrations..." % host_id)

        if use_data_path is None:
            use_data_path = self._choose_data_path(host_id)

        self._data_path_for_host[host_id] = use_data_path
        self._update_host_route(host_id)

        return use_data_path

    ## DataPath and routing management APIs: should really be considered protected methods

    def _update_host_route(self, host_id, route=None):
        """
        Update the given host's route to the optionally-specified one, which by default is chosen for you;
         install flow rules if necessary.
        :param host_id:
        :return: the assigned route
        """

        if route is None:
            route = self._get_host_route(host_id)

        # only update flow rules if necessary
        if host_id not in self._host_routes or route != self._host_routes[host_id]:
            try:
                flow_rules = self.topology_manager.build_flow_rules_from_path(route)
                for r in flow_rules:
                    self.topology_manager.install_flow_rule(r)

                # do this last in case we failed to install flow rules
                self._host_routes[host_id] = route
            except BaseException as e:
                log.error("building/installing flow rules for path %s failed with error: %s" % (route, e))

        return route

    def _get_host_route(self, host_id, dest=None):
        """Return the route from the specified host to the specified destination. By default, this route goes through
        the gateway responsible for its assigned DataPath and eventually to the cloud server."""

        # If the destination wasn't specified, we need to extend the route to the cloud server while ensuring
        # it goes through the right gateway, hence two steps...
        cloud_gw_route = None
        if dest is None:
            data_path = self._data_path_for_host[host_id]
            gateway = self._gateway_for_data_path[data_path]
            # ENHANCE: choose the assigned cloud if we support multiple!
            cloud_gw_route = self.topology_manager.get_path(gateway, self.cloud_server)
            dest = gateway

        route = self.topology_manager.get_path(host_id, dest, weight=self._distance_metric)
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

    def _on_all_data_paths_down(self):
        """
        When no DataPaths are available, our default behavior is to reroute all hosts to the edge server.
        :return:
        """

        log.info("All DataPaths down!  Re-routing hosts to edge server...")

        # TODO: skip over ones that are already routing there?  or just adjust the weights used to choose between edge/cloud?
        for h in self.hosts:
            # ENHANCE: choose from several cloud/edge servers
            old_dest = self.cloud_server
            new_dest = self.edge_server
            route = self._get_host_route(h, new_dest)

            flow_rules = self.topology_manager.build_redirection_flow_rules(h, old_dest, new_dest, route=route)
            for f in flow_rules:
                self.topology_manager.install_flow_rule(f)
            self._host_routes[h] = route
            # ENHANCE: may want to set a translation for the other direction so that the host receives a response and
            # thinks it's from old_dest rather than new_dest
