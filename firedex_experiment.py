#! /usr/bin/env python

# @author: Kyle Benson
# (c) Kyle Benson 2018

import argparse

from network_experiment import NetworkExperiment
from scifire.firedex_configuration import FiredexConfiguration
from scifire.firedex_scenario import FiredexScenario
from scifire.defaults import *
from scifire.algorithms import build_algorithm
from scifire.algorithms.firedex_algorithm import FiredexAlgorithm  # just for type hinting

import logging
log = logging.getLogger(__name__)


class FiredexExperiment(NetworkExperiment, FiredexConfiguration):
    """
    General abstract configuration for experiments that will be run by a concrete child class.
    NOTE: we consider this experiment class as a configuration object plus the experiment, but didn't bother to
    add another inheritance layer since a current configuration includes network elements too.
    """

    def __init__(self, algorithm=DEFAULT_ALGORITHM, **kwargs):
        """
        :param algorithm: configuration for the priority-assignment algorithm
        :param kwargs: passed to super constructor
        """

        super(FiredexExperiment, self).__init__(**kwargs)

        # XXX: ensure algorithm argument is formatted as expected before child class uses it
        if not isinstance(algorithm, dict):
            algorithm = dict(algorithm=algorithm)
        self.algorithm_cfg = algorithm

        # 0 priority levels means we aren't assigning ANY priorities!
        if self.num_priority_levels > 0:
            self.algorithm = build_algorithm(**self.algorithm_cfg)  # type: FiredexAlgorithm
        else:
            self.algorithm = build_algorithm(algorithm='null')  # type: FiredexAlgorithm

        # FUTURE: mobility models for FFs / devs?

        # update params with any static ones, otherwise the ones generated later need to be updated as we output the file
        self.results['params'].update(FiredexScenario.as_dict(self))
        self.record_parameter('ro_tol', self.algorithm_cfg.get('ro_tolerance', DEFAULT_RO_TOLERANCE))

        # since we'll be comparing algorithms in plots, we should make it a more compact object i.e. a string:
        alg_rep = self.algorithm_cfg['algorithm']
        if len(self.algorithm_cfg) > 1:
            alg_params = {k: v for k, v in self.algorithm_cfg.items() if k != 'algorithm'}
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
        super(FiredexExperiment, self).setup_experiment()

    def get_analytical_model_results(self):
        """
        Calculates the expected performance using the analytical model in order to determine its accuracy.
        :return: a dict of resulting expectations to be saved in 'results'
        """

        expected_service_delays = dict()
        for sub, delay in zip(self.subscriptions, self.algorithm.service_delays(self)):
            subscriber = expected_service_delays.setdefault(sub.subscriber, dict())
            subscriber[sub.topic] = delay

        expected_total_delays = dict()
        for sub, delay in zip(self.subscriptions, self.algorithm.total_delays(self)):
            subscriber = expected_total_delays.setdefault(sub.subscriber, dict())
            subscriber[sub.topic] = delay

        expected_delivery_rates = dict()
        for sub, rate in zip(self.subscriptions, self.algorithm.delivery_rates(self)):
            subscriber = expected_delivery_rates.setdefault(sub.subscriber, dict())
            subscriber[sub.topic] = rate

        expected_utilities = dict()
        for sub, util in zip(self.subscriptions, self.algorithm.estimate_utilities(self)):
            subscriber = expected_utilities.setdefault(sub.subscriber, dict())
            subscriber[sub.topic] = util

        utility_weights = dict()
        for sub, uw in zip(self.subscriptions, self.subscription_utility_weights):
            subscriber = utility_weights.setdefault(sub.subscriber, dict())
            subscriber[sub.topic] = uw

        return dict(exp_srv_delay=expected_service_delays, exp_total_delay= expected_total_delays,
                    exp_delivery=expected_delivery_rates, utility_weights=utility_weights,
                    exp_utilities=expected_utilities)

    def get_run_config_for_results_dict(self):
        """
        :return:  a dict of configuration parameters for this run
        """

        # QUESTION: should we add mus?  not clear we'd use it for anything though...

        # TODO: this should probably account for publisher error rates since mn version will have them?
        # needed for max delivery rate/utility
        lambdas = {topic: rate for topic, rate in zip(self.topics, self.algorithm.broker_arrival_rates(self))}

        # TODO: maybe we'll end up needing this but for now we can just total up utility so don't need to know about
        #   subscriptions for which we never received notifications
        # WARNING: will need to convert from mininet.Host to its name as well!
        # XXX: need to format this appropriately for conversion by the stats parser
        # subscriptions = [(sub.subscriber, sub.topic) for sub in self.subscriptions]

        # XXX: convert the subscription-to-prio map to one indexed by subscriber and then topic:
        priorities = dict()
        for sub, prio in self.algorithm.get_subscription_priorities(self).items():
            subscriber = priorities.setdefault(sub.subscriber, dict())
            subscriber[sub.topic] = prio

        # XXX: need to convert net flow map to a subscription map i.e. indexed by subscriber and then topic:
        _drop_rates = self.algorithm.get_drop_rates(self)
        drop_rates = dict()
        for sub in self.subscriptions:
            subscriber = drop_rates.setdefault(sub.subscriber, dict())
            net_flow = self.algorithm.get_subscription_net_flows(self, sub.subscriber)[sub]
            subscriber[sub.topic] = _drop_rates[net_flow]

        return dict(lambdas=lambdas, priorities=priorities, drop_rates=drop_rates,
                    # subscriptions=subscriptions,
                    )

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

        if args['output_filename'] is None:
            log.warning("NO OUTPUT FILENAME SPECIFIED!")

        return cls(**args)

    def record_result(self, result):
        # TODO: is this even needed?  might need to add some custom info...
        # First, add additional parameters used on this run.
        return super(FiredexExperiment, self).record_result(result)

    def output_results(self):
        # add additional parameters we generated after constructor for this exp. configuration
        extra_params = dict(lams=self.pub_rates, mus=self.service_rates)
        for k,v in extra_params.items():
            self.record_parameter(k, v)

        super(FiredexExperiment, self).output_results()
