#! /usr/bin/env python

# @author: Kyle Benson
# (c) Kyle Benson 2018

import argparse
import logging
log = logging.getLogger(__name__)

from network_experiment import NetworkExperiment
from scifire.config import *
from scifire.firedex_scenario import FiredexScenario


class FiredexAlgorithmExperiment(NetworkExperiment, FiredexScenario):

    def __init__(self, **kwargs):
        super(FiredexAlgorithmExperiment, self).__init__(**kwargs)

        self.results['params'].update({'nffs': self.num_ffs,
                                       'niots': self.num_iots,
                                       'nflows': self.num_net_flows,
                                       'nprios': self.num_priority_levels,
                                       'ntopics': self.num_topics,
                                       })

    def run_experiment(self):
        # TODO: this will actually setup a compact data model of our formulation and run the algorithm on it to determine its performance under these 'ideal' network settings.
        # rather than raise a warning, just yell so we can use this to test the super class integration for now
        log.warning("NOT YET IMPLEMENTED!")
        return dict(results="NOT YET IMPLEMENTED!")

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


if __name__ == "__main__":
    import sys
    exp = FiredexAlgorithmExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()
