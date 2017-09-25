CLASS_DESCRIPTION = """Failure model for the campus network resilience experiment.
Chooses network components from an SdnTopology to fail based on different models."""

import argparse
import random

DEFAULT_MODEL = 'uniform'
DEFAULT_FPROB = 0.1


class SmartCampusFailureModel(object):
    """
    Used in conjunction with a specified SdnTopology object to apply one of
    several failure models to the nodes and links in the topology.
    """

    def __init__(self, model=DEFAULT_MODEL, fprob=DEFAULT_FPROB,
                 failure_rand_seed=None, **kwargs):
        # NOTE: kwargs just used for construction via argparse
        super(SmartCampusFailureModel, self).__init__()
        self.model = model
        self.fprob = fprob
        self.random = random.Random(failure_rand_seed)

    def apply_failure_model(self, topo):
        """Applies the failure model to the topology by choosing failed components.
        :param SdnTopology topo:
        @:return [failed_nodes], [failed_links]"""
        if self.model == 'uniform':
            return self.apply_uniform_failure_model(topo)
        if self.model == 'building':
            return self.apply_building_failure_model(topo)
        if self.model == 'srlg':
            return self.apply_srlg_failure_model(topo)

    def should_fail(self):
        """Helper method for choosing whether to fail a single component or not."""
        return self.random.random() < self.fprob

    def apply_uniform_failure_model(self, topo):
        """
        Fail each component (link/switch node) in topo at uniformly random rate.
        :param topo: the topology to choose failed nodes/links from
        :type topo: NetworkxSdnTopology
        """
        switches = []
        links = []

        # NOTE: we must ensure these switches/links won't completely fail the cloud/edge servers or gateways
        # as we can't fail them and gather meaningful results...
        for s in topo.get_switches():
            if not topo.is_cloud_gateway(s) and self.should_fail():
                switches.append(s)
        for l in topo.get_links(attributes=False):
            # The DataPath link (gateway to cloud) failures will be handled by the SmartCampusExperiment
            is_gw_cloud_link = (topo.is_cloud_gateway(l[0]) and topo.is_cloud(l[1])) or \
                               (topo.is_cloud_gateway(l[1]) and topo.is_cloud(l[0]))
            if not is_gw_cloud_link and self.should_fail():
                links.append(l)

        return switches, links

    def apply_building_failure_model(self, topo):
        """Fail each complete building at uniformly random rate."""
        failures = []
        # TODO: maybe buildings are a group rather than a single node?  essentially a srNg
        try:
            for b in topo.get_buildings():
                if self.should_fail():
                    failures.append(b)
        except AttributeError:
            print "no get_buildings() method for topology %s!" % topo

        return failures, []

    def apply_srlg_failure_model(self, topo):
        """Fail each shared risk link group at uniformly random rate."""
        # TODO: add a second fprob for within a srlg?  how to choose them?
        failures = []
        try:
            for g in topo.get_srlgs():
                if self.should_fail():
                    # TODO: not fail ALL components in a group?
                    failures.extend(g)
        except AttributeError:
            print "no get_buildings() method for topology %s!" % topo

        return failures, []

    def get_params(self):
        """Return params as a string for recording configurations."""
        return "%s/%f" % (self.model, self.fprob)

    # argument parser that can be combined with others when this class is used in a script
    # need to not add help options to use that feature, though
    arg_parser = argparse.ArgumentParser(description=CLASS_DESCRIPTION, add_help=False)
    arg_parser.add_argument('--fprob', '-f', type=float, default=DEFAULT_FPROB,
                        help='''probability of component failure (default=%(default)s)''')
    arg_parser.add_argument('--failure-rand-seed', type=int, default=None, dest='failure_rand_seed',
                        help='''random seed for failure model (default=%(default)s)''')
    arg_parser.add_argument('--model', '-m', type=str, default=DEFAULT_MODEL,
                        help='''failure model to apply for choosing component failures (default=%(default)s)''')

if __name__ == '__main__':
    # NOTE: this should be a topology file that includes cloud servers/gateways so that we properly test not failing them!
    topo_file = 'topos/cloud_campus_topo_3b-3h-1ibl.json'
    from topology_manager.networkx_sdn_topology import NetworkxSdnTopology
    topo = NetworkxSdnTopology(filename=topo_file)
    fail = SmartCampusFailureModel(fprob=1.0)

    print fail.get_params()
    nodes, links = fail.apply_failure_model(topo)
    print "Nodes failed:", nodes
    print "Links failed:", links

    # test to make sure we only chose the right links/nodes for failing
    for n in nodes:
        assert not topo.is_host(n), "shouldn't be failing host nodes!"
        assert not topo.is_server(n), "shouldn't be failing edge server nodes!"
        assert not topo.is_cloud(n), "shouldn't be failing cloud server nodes!"
        assert not topo.is_cloud_gateway(n), "shouldn't be failing cloud gateway nodes!"
    for l in links:
        assert not topo.is_cloud(l[0]) and not topo.is_cloud(l[1]), "shouldn't be failing cloud DataPath!!"

    unfailed_nodes = set(topo.topo.nodes()) - set(nodes)
    unfailed_links = set(topo.topo.edges()) - set(links)
    print 'unfailed nodes:', unfailed_nodes
    print 'unfailed links:', unfailed_links