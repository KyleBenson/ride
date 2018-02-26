#! /usr/bin/env python

# @author: Kyle Benson
# (c) Kyle Benson 2018

import argparse
import json
import os
# For working with the temp configuration file
import tempfile

from network_experiment import NetworkExperiment
from scifire.firedex_configuration import FiredexConfiguration
from scifire.firedex_scenario import FiredexScenario
from scifire.algorithms import build_algorithm
from scifire.defaults import *
import logging
log = logging.getLogger(__name__)


# TODO: eventually refactor this into a FiredexExperiment class so we can keep the alg/sim-specific stuff just here
class FiredexAlgorithmExperiment(NetworkExperiment, FiredexConfiguration):
    """
    Simulation-based experiments that run in our Java-based queuing network simulation.
    NOTE: we consider this experiment class as a configuration object plus the experiment, but didn't bother to
    add another inheritance layer since a current configuration includes network elements too.
    """

    def __init__(self, algorithm=DEFAULT_ALGORITHM, **kwargs):
        super(FiredexAlgorithmExperiment, self).__init__(**kwargs)

        if not isinstance(algorithm, dict):
            algorithm = dict(algorithm=algorithm)
        self.algorithm = build_algorithm(**algorithm)

        # FUTURE: mobility models for FFs / devs?

        # update params with any static ones, otherwise the ones generated later need to be updated as we output the file
        # TODO: any static params to update here?
        self.results['params'].update(FiredexScenario.as_dict(self))

        # since we'll be comparing algorithms in plots, we should make it a more compact object i.e. a string:
        alg_rep = algorithm['algorithm']
        if len(algorithm) > 1:
            alg_params = {k: v for k, v in algorithm.items() if k != 'algorithm'}
            alg_params = sorted(alg_params.items())
            alg_params = [str(v) for k, v in alg_params]
            alg_rep += "-" + "-".join(alg_params)
        self.record_parameter('algorithm', alg_rep)

    def setup_experiment(self):
        """
        Generate proper configuration for this scenario e.g. subscriptions, publications, etc.
        :return:
        """

        self.generate_configuration()
        super(FiredexAlgorithmExperiment, self).setup_experiment()

    def run_experiment(self):
        """Run the algorithm on our current scenario and feed the configuration to a queuing network simulator
        to determine its performance under these 'ideal' network settings."""

        # Since we're running the queuing simulator for a static configuration, we just assign the topic priorities statically
        # NOTE: since the exp class IS a config class, just pass self and the alg can ignore exp-specific parts
        prios = self.algorithm.get_topic_priorities(self)

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

        # priorities expected just as a list enumerating the priority class of each topic in order
        if priorities is not None:
            priorities = list(sorted(priorities.items()))
            priorities = [p for t,p in priorities]

        return dict(mus=self.service_rates, lambdas=lambdas, subscriptions=self.subscriptions,
                    priorities=priorities, error_rate=self.error_rate)

    ####  Boiler-plate helper functions for running experiments

    @classmethod
    def get_arg_parser(cls, parents=(FiredexConfiguration.get_arg_parser(), NetworkExperiment.get_arg_parser()), add_help=True):
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
