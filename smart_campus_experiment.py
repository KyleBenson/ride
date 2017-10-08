CLASS_DESCRIPTION = '''Experiment that models failures in a smart campus network setting
and determines the effectiveness of the Resilient IoT Data Exchange (RIDE) middleware.
RIDE improves data collection using RIDE-C, which routes IoT data publishers to the cloud
server through the best available DataPath or directly to the edge server if they aren't available.
RIDE improves alert dissemination using RIDE-D, which utilizes SDN-enabled IP multicast and several
multicast tree-constructing and choosing algorithms for creation of
 Multiple Maximally-Disjoint Multicast Trees (MDMTs).'''

# @author: Kyle Benson
# (c) Kyle Benson 2017

import json
import os
import argparse
import logging
log = logging.getLogger(__name__)
import random
import signal
import time
from abc import abstractmethod, ABCMeta

import networkx as nx
import ride
from ride.config import *
from failure_model import SmartCampusFailureModel
DISTANCE_METRIC = 'latency'  # for shortest path calculations

class SmartCampusExperiment(object):
    __metaclass__ = ABCMeta

    def __init__(self, nruns=1, nsubscribers=5, npublishers=5,
                 failure_model=None, topology_filename=None,
                 debug='info', output_filename='results.json',
                 choice_rand_seed=None, rand_seed=None,
                 error_rate=0.0,
                 ## RideD params
                 ntrees=4, tree_construction_algorithm=DEFAULT_TREE_CONSTRUCTION_ALGORITHM,
                 ## RideC params
                 reroute_policy=DEFAULT_REROUTE_POLICY,
                 # flags to enable/disable certain features for running different combinations of experiments
                 with_ride_d=True, with_ride_c=True,
                 # HACK: kwargs just used for construction via argparse since they'll include kwargs for other classes
                 **kwargs):
        """
        :param nruns:
        :param ntrees:
        :param tree_construction_algorithm:
        :param nsubscribers:
        :param npublishers:
        :param reroute_policy: used by RideC to determine the publishers' routes to edge server (after re-route from cloud in our real scenario)
        :param failure_model:
        :param topology_filename:
        :param debug:
        :param output_filename:
        :param choice_rand_seed:
        :param rand_seed:
        :param error_rate:
        :param with_ride_d:
        :param with_ride_c:
        :param kwargs:
        """
        super(SmartCampusExperiment, self).__init__()
        self.nruns = nruns
        self.current_run_number = None
        self.ntrees = ntrees
        self.nsubscribers = nsubscribers
        self.npublishers = npublishers

        self.topology_filename = topology_filename
        self.topo = None  # built later in setup_topology()

        self.output_filename = output_filename
        if self.output_filename is None:
            log.warning("output_filename is None!  Using default of results.json")
            self.output_filename = 'results.json'
        self.tree_construction_algorithm = tree_construction_algorithm
        self.reroute_policy = reroute_policy
        self.error_rate = error_rate

        self.with_ride_c = with_ride_c
        self.with_ride_d = with_ride_d
        self.with_cloud = with_ride_c

        log_level = logging.getLevelName(debug.upper())
        logging.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log_level)

        # this is used for choosing pubs/subs/servers/other hosts ONLY
        self.random = random.Random(choice_rand_seed)
        # this RNG is used for everything else (tie-breakers, algorithms, etc.)
        random.seed(rand_seed)
        # QUESTION: do we need one for the algorithms as well?  probably not because
        # each algorithm could call random() a different number of times and so the
        # comparison between the algorithms wouldn't really be consistent between runs.

        if failure_model is None:
            failure_model = SmartCampusFailureModel()
        self.failure_model = failure_model

        # These fields will be set for each run so as to avoid passing them around as parameters.
        self.subscribers = None
        self.publishers = None
        self.server = None
        self.cloud = None
        self.failed_nodes, self.failed_links = (None, None)

        # results are output as JSON to file after the experiment runs
        self.results = {'results': [], # each is a single run containing: {run: run#, heuristic_name: percent_reachable}
                        'params': {'ntrees': ntrees,
                                   'nsubscribers': nsubscribers,
                                   'npublishers': npublishers,
                                   'failure_model': self.failure_model.get_params(),
                                   'heuristic': self.get_mcast_heuristic_name(),
                                   'reroute_policy': self.reroute_policy,
                                   'topo': topology_filename,
                                   'error_rate': self.error_rate,
                                   'choicerandseed': choice_rand_seed,
                                   'randseed': rand_seed,
                                   'failrandseed': kwargs.get('failure_rand_seed', None),
                                   # NOTE: subclasses should store their type here!
                                   'experiment_type': None
                                   }
                        }

    @classmethod
    def get_arg_parser(cls, parents=(SmartCampusFailureModel.arg_parser,
                                     ride.ride_d.RideD.get_arg_parser()),
                       add_help=False):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """
        arg_parser = argparse.ArgumentParser(description=CLASS_DESCRIPTION,
                                             parents=parents, add_help=add_help)
        # experimental treatment parameters
        arg_parser.add_argument('--nruns', '-r', type=int, default=1,
                            help='''number of times to run experiment (default=%(default)s)''')
        arg_parser.add_argument('--nsubscribers', '-s', type=int, default=5,
                            help='''number of multicast subscribers (terminals) to reach (default=%(default)s)''')
        arg_parser.add_argument('--npublishers', '-p', type=int, default=5,
                            help='''number of IoT sensor publishers to contact edge server (default=%(default)s)''')
        arg_parser.add_argument('--error-rate', type=float, default=0.0, dest='error_rate',
                            help='''error rate of links (default=%(default)s)''')
        arg_parser.add_argument('--topology-filename', '--topo', type=str, default='topos/campus_topo.json', dest='topology_filename',
                            help='''file name of topology to use (default=%(default)s)''')

        # experiment interaction control
        arg_parser.add_argument('--debug', '-d', type=str, default='info', nargs='?', const='debug',
                            help='''set debug level for logging facility (default=%(default)s, %(const)s when specified with no arg)''')
        arg_parser.add_argument('--output-file', '-o', type=str, default=None, dest='output_filename',
                            help='''name of output file for recording JSON results
                            (by default we generate a filename located in 'results'
                            directory, with '.json' extension, that includes a summary of experiment parameters:
                            see SmartCampusExperiment.build_default_results_file_name())''')
        arg_parser.add_argument('--choice-rand-seed', type=int, default=None, dest='choice_rand_seed',
                            help='''random seed for choices of subscribers & servers (default=%(default)s)''')
        arg_parser.add_argument('--rand-seed', type=int, default=None, dest='rand_seed',
                            help='''random seed used by other classes via calls to random module (default=%(default)s)''')

        return arg_parser

    # ENHANCE: maybe a version that uses the members rather than being classmethod?
    @classmethod
    def build_default_results_file_name(cls, args, dirname='results'):
        """
        :param args: argparse object (or plain dict) with all args info (not specifying ALL args is okay)
        :param dirname: directory name to place the results files in
        :return: string representing the output_filename containing a parameter summary for easy identification
        """

        # Convert argparse object to dict
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Pass empty args to get the default configurations.
        defaults = cls.get_arg_parser().parse_args(args=[])

        # Extract topology file name
        try:
            topo_fname = args.get('topology_filename', defaults.topology_filename)
            topo_fname = os.path.splitext(os.path.basename(topo_fname).split('_')[2])[0]
        except IndexError:
            # topo_fname must not be formatted as expected: just use it plain but remove _'s to avoid confusing code parsing the topo for its params
            topo_fname = os.path.splitext(os.path.basename(args.get('topology_filename', defaults.topology_filename).replace('_', '')))[0]

        output_filename = 'results_%dt_%0.2ff_%ds_%dp_%s_%s_%0.2fe.json' % \
                          (args.get('ntrees', defaults.ntrees), args.get('fprob', defaults.fprob),
                           args.get('nsubscribers', defaults.nsubscribers), args.get('npublishers', defaults.npublishers),
                           cls.build_mcast_heuristic_name(*args.get('tree_construction_algorithm', defaults.tree_construction_algorithm)),
                           topo_fname, args.get('error_rate', defaults.error_rate))

        output_filename = os.path.join(dirname, output_filename)

        return output_filename

    @classmethod
    def build_from_args(cls, args):
        """Constructs from command line arguments."""

        args = cls.get_arg_parser().parse_args(args)

        # convert to plain dict
        args = vars(args)
        failure_model = SmartCampusFailureModel(**args)
        args['failure_model'] = failure_model

        if args['output_filename'] is None:
            args['output_filename'] = cls.build_default_results_file_name(args)

        return cls(**args)

    def run_all_experiments(self):
        """Runs the requested experimental configuration
        for the requested number of times, saving the results to an output file."""

        # Log progress to a file so that we can check on
        # long-running simulations to see how far they've gotten.
        progress_filename = self.output_filename.replace(".json", ".progress")
        # in case we hadn't specified a .json output file:
        if progress_filename == self.output_filename:
            progress_filename += ".progress"

        # ensure the directory exists...
        try:
            os.mkdir(os.path.dirname(progress_filename))
        except OSError:  # dir exists
            pass

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
            self.current_run_number = r
            # ENHANCE: may only need to set this up once...
            self.setup_topology()

            self.setup_experiment()
            result = self.run_experiment()
            self.teardown_experiment()

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

    def record_result(self, result):
        """Result is a dict that includes the experimental results for the current run i.e. percentage of subscribers
        reachable, what parameters were used, etc.  This method also records additional parameters such as run #,
        failed_nodes/links, etc."""

        # First, add additional parameters used on this run.
        result['failed_nodes'] = self.failed_nodes
        result['failed_links'] = self.failed_links
        result['run'] = self.current_run_number
        self.results['results'].append(result)

    def output_results(self):
        """Outputs the results to a file"""
        log.info("Results: %s" % json.dumps(self.results, sort_keys=True, indent=2))
        if os.path.exists(self.output_filename):
            log.warning("Output file being overwritten: %s" % self.output_filename)
        with open(self.output_filename, "w") as f:
            json.dump(self.results, f, sort_keys=True, indent=2)

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
        server = self.random.choice(self.topo.get_servers())
        log.debug("Server: %s" % server)
        return server

    @abstractmethod
    def setup_topology(self):
        """
        Construct and configure appropriately the topology based on the previously
        specified topology_adapter_type and topology_filename.
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
    def run_experiment(self):
        """
        Run the actual experiment and return the results in a dict to be recorded.

        :returns dict results:
        """
        raise NotImplementedError

    def setup_experiment(self):
        """
        Set up the experiment and configure it as necessary before run_experiment is called.
        By default it chooses the subscribers, publishers, server, and failed nodes/links
        for this experimental run.

        :return:
        """
        self.subscribers = self.choose_subscribers()
        self.publishers = self.choose_publishers()
        # NOTE: this is unnecessary as we only have a single server in our test topos.  If we use multiple, need
        # to actually modify RideD here with updated server.
        self.server = self.choose_server()
        self.failed_nodes, self.failed_links = self.get_failed_nodes_links()

        assert self.server not in self.failed_nodes, "shouldn't be failing the server!  useless run...."

    def teardown_experiment(self):
        """
        Cleans up the experiment in preparation for the next call to setup (or being finished).
        By default does nothing.
        """
        pass

    ### Helper functions for working with failures and calculating reachability in our topology

    @staticmethod
    def get_oracle_reachability(subscribers, server, failed_topology):
        """Gets the reachability for the 'oracle' heuristic, which is simply calculating how many subscribers are
        reachable by ANY path"""
        failed_topology.graph['heuristic'] = 'oracle'
        topos_to_check = [failed_topology]
        reach = SmartCampusExperiment.get_reachability(server, subscribers, topos_to_check)[0]
        return reach

    @staticmethod
    def get_reachability(server, subscribers, topologies):
        """Returns the average probability of reaching any of the subscribers from the
        server in each of the given topologies.  Also includes the result for using
        all topologies at once.
        :returns list: containing reachability for each topology (in order),
        with the last entry representing using all topologies at the same time
        """

        subscribers = set(subscribers)
        subs_reachable_by_tree = []
        all_subscribers_reachable = set()
        for topology in topologies:
            nodes_reachable = set(nx.single_source_shortest_path(topology, server))
            # could also use has_path()
            subscribers_reachable = subscribers.intersection(nodes_reachable)
            subs_reachable_by_tree.append(len(subscribers_reachable) / float(len(subscribers)))
            all_subscribers_reachable.update(subscribers_reachable)

            log.debug("%s heuristic reached %d subscribers in this topo" % (topology.graph['heuristic'], len(subscribers_reachable)))
        log.debug("ALL subscribers reached by these topos: %d" % len(all_subscribers_reachable))
        # Lastly, include all of them reachable
        subs_reachable_by_tree.append(float(len(all_subscribers_reachable)) / len(subscribers))
        return subs_reachable_by_tree

    @staticmethod
    def get_failed_topology(topo, failed_nodes, failed_links):
        """Returns a copy of the graph topo with all of the failed nodes
        and links removed."""
        # since we froze the graph we can't just use .copy()
        topology = nx.Graph(topo)
        topology.remove_edges_from(failed_links)
        topology.remove_nodes_from(failed_nodes)
        return topology