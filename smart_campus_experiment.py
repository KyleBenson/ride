import ride

CLASS_DESCRIPTION = '''Experiment that models failures in a campus network setting
and determines the effectiveness of several SDN/IP multicast tree-constructing and choosing
algorithms in improving data dissemination to subscribing IoT devices around the campus.'''

# @author: Kyle Benson
# (c) Kyle Benson 2017

import argparse
import logging as log
import random
import signal
import time
from abc import abstractmethod, ABCMeta

import networkx as nx
from failure_model import SmartCampusFailureModel

class SmartCampusExperiment(object):
    __metaclass__ = ABCMeta

    def __init__(self, nruns=1, ntrees=4, tree_construction_algorithm=('steiner',), nsubscribers=5, npublishers=5,
                 failure_model=None, topology_filename=None,
                 debug='info', output_filename='results.json',
                 choice_rand_seed=None, rand_seed=None,
                 # TODO: maybe this is specific to netx?  or maybe it's a general error rate...
                 publication_error_rate=0.0,
                 # HACK: kwargs just used for construction via argparse since they'll include kwargs for other classes
                 **kwargs):
        super(SmartCampusExperiment, self).__init__()
        self.nruns = nruns
        self.ntrees = ntrees
        self.nsubscribers = nsubscribers
        self.npublishers = npublishers

        self.topology_filename = topology_filename
        self.topo = None  # built later in setup_topology()

        self.output_filename = output_filename
        self.tree_construction_algorithm = tree_construction_algorithm
        self.publication_error_rate = publication_error_rate

        log_level = log.getLevelName(debug.upper())
        log.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log_level)

        # this is used for choosing pubs/subs/servers ONLY
        self.random = random.Random(choice_rand_seed)
        # this RNG is used for everything else (tie-breakers, algorithms, etc.)
        random.seed(rand_seed)
        # QUESTION: do we need one for the algorithms as well?  probably not because
        # each algorithm could call random() a different number of times and so the
        # comparison between the algorithms wouldn't really be consistent between runs.

        if failure_model is None:
            failure_model = SmartCampusFailureModel()
        self.failure_model = failure_model

        # results are output as JSON to file after the experiment runs
        self.results = {'results': [], # each is a single run containing: {run: run#, heuristic_name: percent_reachable}
                        'params': {'ntrees': ntrees,
                                   'nsubscribers': nsubscribers,
                                   'npublishers': npublishers,
                                   'failure_model': self.failure_model.get_params(),
                                   'heuristic': self.get_mcast_heuristic_name(),
                                   'topo': topology_filename,
                                   'publication_error_rate': self.publication_error_rate,
                                   'choicerandseed': choice_rand_seed,
                                   'randseed': rand_seed,
                                   'failrandseed': kwargs.get('failure_rand_seed', None),
                                   }
                        }

    @classmethod
    def get_arg_parser(cls, parents=(SmartCampusFailureModel.arg_parser,
                                     ride.ride_d.RideD.get_arg_parser())):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :return argparse.ArgumentParser arg_parser:
        """
        arg_parser = argparse.ArgumentParser(description=CLASS_DESCRIPTION,
                                             parents=parents, add_help=False)
        # experimental treatment parameters
        arg_parser.add_argument('--nruns', '-r', type=int, default=1,
                            help='''number of times to run experiment (default=%(default)s)''')
        arg_parser.add_argument('--nsubscribers', '-s', type=int, default=5,
                            help='''number of multicast subscribers (terminals) to reach (default=%(default)s)''')
        arg_parser.add_argument('--npublishers', '-p', type=int, default=5,
                            help='''number of IoT sensor publishers to contact edge server (default=%(default)s)''')
        arg_parser.add_argument('--error-rate', type=float, default=0.0, dest='publication_error_rate',
                            help='''error rate of publications from publishers (chance that we won't
                            include a publisher in the STT even if it's still connected to server) (default=%(default)s)''')
        arg_parser.add_argument('--topology-filename', '--topo', type=str, default='topos/campus_topo.json', dest='topology_filename',
                            help='''file name of topology to use (default=%(default)s)''')

        # experiment interaction control
        arg_parser.add_argument('--debug', '-d', type=str, default='info', nargs='?', const='debug',
                            help='''set debug level for logging facility (default=%(default)s, %(const)s when specified with no arg)''')
        arg_parser.add_argument('--output-file', '-o', type=str, default='results.json', dest='output_filename',
                            help='''name of output file for recording JSON results (default=%(default)s)''')
        arg_parser.add_argument('--choice-rand-seed', type=int, default=None, dest='choice_rand_seed',
                            help='''random seed for choices of subscribers & servers (default=%(default)s)''')
        arg_parser.add_argument('--rand-seed', type=int, default=None, dest='rand_seed',
                            help='''random seed used by other classes via calls to random module (default=%(default)s)''')

        return arg_parser

    @classmethod
    def build_from_args(cls, args):
        """Constructs from command line arguments."""

        args = cls.get_arg_parser().parse_args(args)

        # convert to plain dict
        args = vars(args)
        failure_model = SmartCampusFailureModel(**args)
        args['failure_model'] = failure_model
        return cls(**args)

    def run_all_experiments(self):
        """Runs the requested experimental configuration
        for the requested number of times, saving the results to an output file."""

        # Log progress to a file so that we can check on
        # long-running simulations to see how far they've gotten.
        progress_filename = self.output_filename.replace(".json", ".progress")
        if progress_filename == self.output_filename:
            progress_filename += ".progress"
        try:
            progress_file = open(progress_filename, "w")
            progress_file.write("Starting experiments at time %s\n" % time.ctime())
        except IOError as e:
            log.warn("Error opening progress file for writing: %s" % e)
            progress_file = None

        self.set_interrupt_signal()

        # start the actual experimentation
        for r in range(self.nruns):
            log.info("Starting run %d" % r)
            self.setup_topology()
            subs = self.choose_subscribers()
            pubs = self.choose_publishers()
            # NOTE: this is unnecessary as we only have a single server in our test topos.  If we use multiple, need
            # to actually modify RideD here with updated server.
            server = self.choose_server()
            failed_nodes, failed_links = self.get_failed_nodes_links()

            assert server not in failed_nodes, "shouldn't be failing the server!  useless run...."

            self.setup_experiment(failed_nodes, failed_links, server, pubs, subs)
            result = self.run_experiment(failed_nodes, failed_links, server, pubs, subs)
            self.teardown_experiment()

            result['run'] = r
            self.record_result(result)

            if progress_file is not None:
                try:
                    progress_file.write("Finished run %d at %s\n" % (r, time.ctime()))
                    progress_file.flush()  # so we can tail it
                except IOError as e:
                    log.warn("Error writing to progress file: %s" % e)
        self.output_results()

    def set_interrupt_signal(self):
        # catch termination signal and immediately output results so we don't lose ALL that work
        def __sigint_handler(sig, frame):
            # HACK: try changing filename so we know it wasn't finished
            self.output_filename = self.output_filename.replace('.json', '_UNFINISHED.json')
            log.critical("SIGINT received! Outputting current results to %s and exiting" % self.output_filename)
            self.output_results()
            exit(1)
        signal.signal(signal.SIGINT, __sigint_handler)

    @abstractmethod
    def record_result(self, result):
        """Result is a dict that includes the percentage of subscribers
        reachable as well as metadata such as run #"""
        pass

    @abstractmethod
    def output_results(self):
        """Outputs the results to a file"""
        pass

    def get_failed_nodes_links(self):
        """Returns which nodes/links failed according to the failure model.
        @:return failed_nodes, failed_links"""
        nodes, links = self.failure_model.apply_failure_model(self.topo)
        log.debug("Failed nodes: %s" % nodes)
        log.debug("Failed links: %s" % links)
        return nodes, links

    def choose_subscribers(self):
        # ENHANCE: could sample ALL of the hosts and then just slice off nsubs.
        # This would make it so that different processes with different nhosts
        # but same topology would give complete overlap (smaller nsubs would be
        # a subset of larger nsubs).  This would still cause significant variance
        # though since which hosts are chosen is different and that affects outcomes.
        subs = self._choose_random_hosts(self.nsubscribers)
        log.debug("Subscribers: %s" % subs)
        return subs

    def choose_publishers(self):
        pubs = self._choose_random_hosts(self.npublishers)
        log.debug("Publishers: %s" % pubs)
        return pubs

    def _choose_random_hosts(self, nhosts):
        """
        Chooses a uniformly random sampling of hosts to act as some group.
        If nhosts > total_hosts, will return all hosts.
        :param nhosts:
        :return:
        """
        hosts = self.topo.get_hosts()
        sample = self.random.sample(hosts, min(nhosts, len(hosts)))
        return sample

    def choose_server(self):
        # TODO: maybe this just needs to be an explicit argument...  or an attribute of the graph topology itself?
        server = self.random.choice(self.topo.get_servers())
        log.debug("Server: %s" % server)
        return server

    @abstractmethod
    def setup_topology(self):
        """
        Construct and configure appropriately the topology based on the previously
        specified topology_adapter and topology_filename.
        :return:
        """
        pass

    @staticmethod
    def build_mcast_heuristic_name(*args):
        """The heuristic is given with arguments so we use this function
        to convert it to a compact human-readable form.  This is a
        separate static function for use by other classes."""
        if len(args) > 1:
            interior = ",".join(args[1:])
            return "%s[%s]" % (args[0], interior)
        else:
            return args[0]

    def get_mcast_heuristic_name(self):
        return self.build_mcast_heuristic_name(*self.tree_construction_algorithm)

    @abstractmethod
    def run_experiment(self, failed_nodes, failed_links, server, publishers, subscribers):
        """
        Setup, run, and teardown the experiment before returning the results.

        :param List[str] failed_nodes:
        :param List[str] failed_links:
        :param str server:
        :param List[str] publishers:
        :param List[str] subscribers:
        :returns dict results:
        """
        pass

    def setup_experiment(self, failed_nodes, failed_links, server, publishers, subscribers):
        """
        Set up the experiment and configure it as necessary before run_experiment is called.
        By default does nothing.

        :param List[str] failed_nodes:
        :param List[str] failed_links:
        :param str server:
        :param List[str] publishers:
        :param List[str] subscribers:
        :return:
        """
        pass

    def teardown_experiment(self):
        """
        Cleans up the experiment in preparation for the next call to setup (or being finished).
        By default does nothing.
        """
        pass