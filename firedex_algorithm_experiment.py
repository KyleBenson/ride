#! /usr/bin/env python

# @author: Kyle Benson
# (c) Kyle Benson 2018

import argparse
import random

# For working with the temp configuration file
import json
import os
import tempfile


from network_experiment import NetworkExperiment
from scifire.firedex_scenario import FiredexScenario
from scale_client.stats.random_variable import RandomVariable

import logging
log = logging.getLogger(__name__)


# TODO: eventually refactor this into a FiredexExperiment class so we can keep the alg/sim-specific stuff just here
class FiredexAlgorithmExperiment(NetworkExperiment, FiredexScenario):
    """
    Simulation-based experiments that
    NOTE: we consider this experiment class as a configuration object plus the experiment, but didn't bother to
    add another inheritance layer since a current configuration includes network elements too.
    """

    def __init__(self, **kwargs):
        super(FiredexAlgorithmExperiment, self).__init__(**kwargs)

        # these will get filled in later
        self.service_rates = []
        self.pub_rates = []
        self.subscriptions = None
        self.advertisements = None

        # FUTURE: mobility models for FFs / devs?

        # update params with any static ones, otherwise the ones generated later need to be updated as we output the file
        # TODO: any static params to update here?
        self.results['params'].update(FiredexScenario.as_dict(self))

    def setup_experiment(self):
        """
        Generate proper configuration for this scenario e.g. subscriptions, publications, etc.
        :return:
        """

        self.generate_experiment_config()
        super(FiredexAlgorithmExperiment, self).setup_experiment()

        self.generate_subscriptions()
        self.generate_advertisements()

    def run_experiment(self):
        """Run the algorithm on our current scenario and feed the configuration to a queuing network simulator
        to determine its performance under these 'ideal' network settings."""

        # Since we're running the queuing simulator for a static configuration, we just assign the topic priorities statically
        # TODO: replace with run_algorithm()?
        prios = self.get_priorities()

        # Setup a compact data model of our formulation that's used to configure external queue simulator experiment.
        cfg = self.get_simulator_input_dict(prios)

        # Since we're running an external queuing simulator, make a temporary file for passing experiment configuration
        cfg_file, cfg_filename = tempfile.mkstemp('firedex_sim_cfg', text=True)
        with os.fdopen(cfg_file, 'w') as f:
            f.write(json.dumps(cfg))

        # TODO: send config to simulator and run it
        with open(cfg_filename) as f:
            print "EXPERIMENT RUNNING CONFIG:", f.read()

        # Delete the temp file since the configuration is saved in the results anyway
        os.remove(cfg_filename)

        #TODO: read results from simulator and feed them into utility functions

        return dict(results="NOT YET IMPLEMENTED!", config=cfg)

    def generate_experiment_config(self):
        """
        Using the specified random variable distributions and static configuration parameters, generates an actual
        configuration for one simulation run of an experiment that includes assigned values (be they static or
        further random distributions) pulled from probability distributions for specific topic publication rates,
        event sizes, utility functions, subscriptions, etc.  This configuration can be used to run a simulation, drive
        the algorithms, or configure clients in an emulated experiment.
        """

        data_size_rvs = [RandomVariable.build(kws) for kws in self.topic_class_data_sizes]
        pub_rate_rvs = [RandomVariable.build(kws) for kws in self.topic_class_pub_rates]
        # TODO: make these dicts instead of lists?
        self.service_rates = []
        self.pub_rates = []
        for topics, size_rv, rate_rv in zip(self.topic_classes, data_size_rvs, pub_rate_rvs):
            for t in topics:
                data_size = size_rv.get_int()
                # TODO: handle specifying different bandwidth values
                srv_rate = self.calculate_service_rate(data_size)
                self.service_rates.append(srv_rate)

                pub_rate = rate_rv.get()
                self.pub_rates.append(pub_rate)

        assert len(self.service_rates) == self.num_topics
        assert len(self.pub_rates) == self.num_topics

    def get_simulator_input_dict(self, priorities=None):
        """
        Generates a dict that represents the system configuration parameters used for running a queuing network-based
        simulation experiment.

        NOTE: lambdas are the total (over all publishers) arrival rates at the broker of each topic

        :return: a dict of configuration parameters e.g.:
          {
            "lambdas": [topic1_pub_rate, topic2_pub_rate, ...],
            "mus": [topic1_service_rate, topic2_service_rate, ...],
            "error_rate": 0.1,
            "subscriptions": [0, 2, 3, 5],  #currently only for a single subscriber!
            "priorities": [0, 0, 1, 2, 3],
          }
        """

        # Since the simulator just takes the arrival rates and doesn't actually model publishers, we need to scale the
        # arrival rates based on the number of publishers on each of those topics i.e. lambda[i] = pub_rate[i]*npubs_on_i

        # NOTE: these must all be floats as the Java queuing simulator's JSON parser has issues casting properly
        lambdas = {top: 0.0 for top in self.topics}
        for pub_class_ads in self.advertisements:
            for pub_ads in pub_class_ads:
                for topic in pub_ads:
                    lambdas[topic] += self.pub_rates[topic]
        lambdas = lambdas.items()
        lambdas.sort()
        lambdas = [v for (k,v) in lambdas]

        return dict(mus=self.service_rates, lambdas=lambdas, subscriptions=self.subscriptions,
                    priorities=priorities, error_rate=self.error_rate)

    def generate_subscriptions(self):
        """
        Generates the topic subscriptions for this scenario configuration.

        NOTE: this is currently only for a single subscriber and the subscriptions are assumed to remain constant
        for the entire experiment's duration.

        :return: list of topics subscribed to
        """

        # TODO: generate subs for multiple subscribers
        # TODO: need to distinguish IC from other FFs when we do this

        # IDEA: for a single subscriber, consider each topic class and generate topics from the given distribution
        # until we have a number of unique subscriptions >= #requested for that class
        # TODO: need to bring in topic start time and duration distributions too... maybe make subscriptions a class???
        # ENHANCE: maybe we should actually have the # subscriptions be a RV so we can vary the subscribers that way?
        subs = []
        for class_idx, (class_topic_rate, rv) in enumerate(zip(self.topic_class_sub_rates, self.topic_class_sub_dists)):
            rv = self.build_sampling_random_variable(rv, len(self.topics_for_class(class_idx)))
            ntopics = int(class_topic_rate * self.ntopics_per_class[class_idx])
            try:
                class_subs = rv.sample(self.topics_for_class(class_idx), ntopics)
                subs.extend(class_subs)
            except ValueError as e:
                log.error("failed to generate topics for class %d due to error: %s" % (class_idx, e))

        self.subscriptions = subs
        return subs

    def generate_advertisements(self):
        """
        Generates the topics each publisher will publish to for this scenario configuration.
        :return: (ff_ads, iot_ads) where ads is a list of lists mapping publisher to its topics advertised
        """

        # ENHANCE: try to skew ads so that FFs tend to publish to the same topics that IoT devs do not e.g. FF health monitoring
        # ENHANCE: use lists nested inside dicts instead of two lists of lists?

        ff_ads = []
        iot_ads = []

        ads_rvs = [self.build_sampling_random_variable(dist, len(self.topics_for_class(tc_idx))) for tc_idx, dist in enumerate(self.topic_class_pub_dists)]

        # For each type of publisher e.g. FF or IoT-dev
        for (num_pubs, tc_num_ads, tc_ads_rvs, ads) in ((self.num_ffs, self.topic_class_advertisements_per_ff, ads_rvs, ff_ads),
                                                        (self.num_iots, self.topic_class_advertisements_per_iot, ads_rvs, iot_ads)):
            # generate ads for each publisher
            for p in range(num_pubs):
                ads_for_pub = []
                # based on each topic class distribution/population
                for tc in range(self.ntopic_classes):
                    rv = ads_rvs[tc]
                    # TODO: pull this num from a RV dist rather than assuming it's a constant
                    num_ads = tc_num_ads[tc]
                    try:
                        ads_for_pub.extend(rv.sample(self.topics_for_class(tc), num_ads))
                    except ValueError as e:
                        log.error("failed to generate advertisements for class %d due to error: %s" % (tc, e))
                ads.append(ads_for_pub)

        log.debug("FF advertisements: %s" % ff_ads)
        log.debug("IoT advertisements: %s" % iot_ads)

        self.advertisements = (ff_ads, iot_ads)
        return ff_ads, iot_ads

    def calculate_service_rate(self, pkt_size, bandwidth=None):
        """
        Calculates the transmission rate of a packet on the network according to the specified bandwidth.
        :param pkt_size: in bytes
        :param bandwidth: defaults to self.bandwidth
        :return:
        """
        if bandwidth is None:
            bandwidth = self.bandwidth
        return self.bandwidth_bytes(bandwidth) / float(pkt_size)

    # TODO: move this to an algorithm class?
    # TODO: should also take system state rather than just configuration...
    def get_priorities(self, configuration=None, algorithm='random'):
        """
        Runs the actual algorithm to determine what the priority levels should be according to the current real-time
        configuration specified.
        :param configuration: actual current network/data exchange state (default=self since an exp. IS a config!)
        :param algorithm: algorithm to use for priority assignment (default=random)
        :return:
        """

        if configuration is None:
            configuration = self

        if algorithm == 'random':
            # TODO: always return lowest prio level for non-subscribed topics?
            # TODO: we should potentially generate a dictionary for the topics if they're strings?
            prios = [random.randrange(self.num_priority_levels) for t in self.topics]
            return prios
        # TODO: assign them based on static utility functions e.g. maybe order topics by utility and break into even prio groups?
        # elif algorithm == 'static':
        else:
            raise ValueError("unrecognized priority-assignment algorithm %s" % algorithm)

    def build_sampling_random_variable(self, rv_dist_cfg, population_size):
        """
        Returns a RandomVariable that is more likely to guarantee it will be able to properly sample from a population of
        the given size.

        It checks if the distribution has an upper bound; if it's the uniform distribution with no upper bound arg
        specified it will set them to the default of [low, population_size] where low is the lower bound specified else 0.
        This allows us to just specify a distribution as uniform when configuring the experiment and let the range be
        dynamically generated according to e.g. num_topics

        Also checks for proper lower bound e.g. zipf gets an additional argument to shift the range down to range [0, inf)

        :param rv_dist_cfg:
        :param population_size:
        :return:
        """

        # ENHANCE: store the random variable so we can keep it between runs?  would need to identify it e.g. with a string...

        rv = RandomVariable.build(rv_dist_cfg)
        if rv.is_upper_bounded():
            if rv.dist == 'uniform':
                args = list(rv_dist_cfg.get('args', [0]))
                if len(args) == 1:
                    low = args[0]
                    args.append(population_size - low)
                    rv_dist_cfg['args'] = args
                    rv = RandomVariable.build(rv_dist_cfg)

        if rv.dist == 'zipf':
            args = rv_dist_cfg.get('args')
            # XXX: shift if with the loc parameter
            if 'loc' not in rv_dist_cfg and (not args or len(args) <= 1):
                args = rv_dist_cfg.copy()
                args['loc'] = -1
                rv = RandomVariable.build(args)

        return rv

    ####  Boiler-plate helper functions for running experiments

    @classmethod
    def get_arg_parser(cls, parents=(FiredexScenario.get_arg_parser(), NetworkExperiment.get_arg_parser()), add_help=True):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        arg_parser = argparse.ArgumentParser(parents=parents, add_help=add_help, conflict_handler='resolve')

        # TODO: add experimental parameters here that don't go in the scenario?  maybe the event-generation models go here?
        # e.g. algorithm goes here since that'll actually use the FiredexScenario

        return arg_parser

    @classmethod
    def build_from_args(cls, args):
        """Constructs from command line arguments."""

        args = cls.get_arg_parser().parse_args(args)

        # convert to plain dict
        args = vars(args)

        return cls(**args)

    def record_result(self, result):
        # TODO: is this even needed?  might need to add some custom info...
        # First, add additional parameters used on this run.
        return super(FiredexAlgorithmExperiment, self).record_result(result)

    def output_results(self):
        # add additional parameters we generated after constructor for this exp. configuration
        extra_params = dict(lams=self.pub_rates, mus=self.service_rates, subs=self.subscriptions)
        for k,v in extra_params.items():
            self.record_parameter(k, v)

        super(FiredexAlgorithmExperiment, self).output_results()


if __name__ == "__main__":
    import sys
    exp = FiredexAlgorithmExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()
