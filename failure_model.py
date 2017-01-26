CLASS_DESCRIPTION = """Failure model for the campus network resilience experiment.
Chooses network components from an SdnTopology to fail based on different models."""

import argparse
import random

DEFAULT_MODEL = 'uniform'
DEFAULT_FPROB = 0.1


class SmartCampusFailureModel(object):

    def __init__(self, model=DEFAULT_MODEL, fprob=DEFAULT_FPROB,
                 failure_rand_seed=None, **kwargs):
        # NOTE: kwargs just used for construction via argparse
        super(SmartCampusFailureModel, self).__init__()
        self.model = model
        self.fprob = fprob
        # TODO: specify this seed
        self.random = random.Random(failure_rand_seed)

    def apply_failure_model(self, topo):
        """Applies the failure model to the topology by choosing failed components.
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
        """Fail each component (link/switch node) at uniformly random rate."""
        switches = []
        links = []

        for s in topo.get_switches():
            if self.should_fail():
                switches.append(s)
        for l in topo.get_links():
            if self.should_fail():
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
    from networkx_sdn_topology import NetworkxSdnTopology
    st = NetworkxSdnTopology()
    fail = SmartCampusFailureModel()

    print fail.get_params()
    n,l = fail.apply_failure_model(st)
    print "Nodes failed:", n
    print "Links failed:", l