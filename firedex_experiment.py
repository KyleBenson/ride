#! /usr/bin/env python

# @author: Kyle Benson
# (c) Kyle Benson 2018

import argparse

from network_experiment import NetworkExperiment
from scifire.firedex_configuration import FiredexConfiguration
from scifire.firedex_scenario import FiredexScenario
from scifire.defaults import *
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
        self.algorithm = algorithm

        # FUTURE: mobility models for FFs / devs?

        # update params with any static ones, otherwise the ones generated later need to be updated as we output the file
        self.results['params'].update(FiredexScenario.as_dict(self))
        self.record_parameter('ro_tol', self.algorithm.get('ro_tolerance', DEFAULT_RO_TOLERANCE))

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
        super(FiredexExperiment, self).setup_experiment()

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
