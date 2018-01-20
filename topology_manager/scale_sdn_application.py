import logging
log = logging.getLogger(__name__)

from topology_manager.sdn_topology import SdnTopology
from topology_manager import build_topology_adapter
from scale_client.core.threaded_application import ThreadedApplication


class ScaleSdnApplication(ThreadedApplication):
    """
    Abstract implementation that allows SCALE clients to interact with an SDN controller via our RIDE REST API adapter.
    """

    def __init__(self, broker, topology_mgr='onos', maintenance_interval=10, **kwargs):
        """
        :param broker:
        :param topology_mgr: an SdnTopology or args to create it (dict, string to specify type, or tuple/list)
        :type topology_mgr: SdnTopology
        :param maintenance_interval: periodically update the topology manager every this many seconds; set to 0/None to disable
        :param kwargs:
        """

        super(ScaleSdnApplication, self).__init__(broker=broker, **kwargs)

        self.topology_manager = topology_mgr
        self.maintenance_interval = maintenance_interval

    def on_start(self):
        """Starts the SdnTopology manager instance according to the previously-specified args"""

        super(ScaleSdnApplication, self).on_start()

        if not isinstance(self.topology_manager, SdnTopology):
            # only adapter type specified: use default other args
            if isinstance(self.topology_manager, basestring):
                self.topology_manager = build_topology_adapter(topology_adapter_type=self.topology_manager)
            # we expect a dict to have the kwargs
            elif isinstance(self.topology_manager, dict):
                self.topology_manager = build_topology_adapter(**self.topology_manager)
            # hopefully it's a tuple!
            else:
                try:
                    self.topology_manager = build_topology_adapter(*self.topology_manager)
                except TypeError:
                    raise TypeError("topology_mgr parameter is not of type SdnTopology and couldn't extract further parameters from it!")

        if self.maintenance_interval:
            self.timed_call(self.maintenance_interval, self.__class__.__maintain_topology, repeat=True)

        log.error("topology contains: %s" % list(self.net.nodes()))

    def __maintain_topology(self):
        """Updates the topology manager by connecting to the SDN controller and rebuilding the topology.
        Meant to be run periodically."""

        log.debug("updating topology via SDN controller...")
        self.topology_manager.build_topology()
        # TODO: add some sort of 'topology_updated' event that we can publish in case others want that info?

    @property
    def net(self):
        """Convenience method for accessing the network topology managed within the SdnTopology."""
        return self.topology_manager.topo