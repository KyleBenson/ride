#!/usr/bin/python

STATISTICS_DESCRIPTION = '''Gathers statistics from the campus_net_experiment.py output files in order to determine how
resilient different multicast tree-generating algorithms are under various scenarios.
Also handles printing them in an easily-read format as well as creating plots.'''

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import json


def parse_args(args):
##################################################################################
#################      ARGUMENTS       ###########################################
# ArgumentParser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
# action is one of: store[_const,_true,_false], append[_const], count
# nargs is one of: N, ?(defaults to const when no args), *, +, argparse.REMAINDER
# help supports %(var)s: help='default value is %(default)s'
##################################################################################

    parser = argparse.ArgumentParser(description=STATISTICS_DESCRIPTION,
                                     #formatter_class=argparse.RawTextHelpFormatter,
                                     #epilog='Text to display at the end of the help print',
                                     )

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dirs', '-d', type=str, nargs="+",
                        help='''directories containing files from which to read outputs
                        (default=%(default)s)''')
    group.add_argument('--files', '-f', type=str, nargs="+", default=['results.json'],
                        help='''files from which to read output results
                        (default=%(default)s)''')

    parser.add_argument('--x-axis', '-x', type=str, default='fprob', dest='x_axis',
                        help='''name of parameter to place on the x-axis (ordered for increasing reachability)
                        (default=%(default)s)''')

    # TODO: filter heuristics, label x-axis, add title, x-axis is a list?


    return parser.parse_args(args)


class SeismicStatistics(object):
    """Parse results and visualize reachability rate."""

    def __init__(self, config):
        super(self.__class__, self).__init__()
        self.x_axis = config.x_axis
        self.dirs = config.dirs
        self.files = config.files
        self.parsing_dirs = config.dirs is not None

        # store all the parsed stats indexed by x-axis parameter values
        self.stats = dict()

    def parse_all(self):
        """Parse either all directories (if specified) or all files."""
        if self.parsing_dirs:
            for dirname in self.dirs:
                self.parse_dir(dirname)
        else:
            for fname in self.files:
                self.parse_file(fname)

    def parse_dir(self, dirname):
        for filename in os.listdir(dirname):
            self.parse_file(os.path.join(dirname, filename))

    def parse_file(self, fname):
        with open(fname) as f:
            data = json.load(f)
        # store the results along with any others grouped according to the x-axis parameter
        param_value = data['params'][self.x_axis]
        self.stats.setdefault(param_value, []).extend(data['results'])

    def get_reachability(self, group):
        """Averages the reachabilities over this collection of results
        (previously grouped by x-axis parameter name).
        @:param group - list of {heuristic1: reachability, heuristic2: reachability} dicts
        @:return dict with each {heuristic1: avg_reachability, ...}
        """

        # First add up the total reachability for each heuristic, and
        # then divide by the count to get the average
        reachabilities = {}
        heuristic_counts = {}
        for run in group:
            for heuristic, reachability in run.items():
                # HACK to skip other parameters for this run
                if heuristic == 'run':
                    continue
                reachabilities[heuristic] = reachabilities.get(heuristic, 0) + reachability
                heuristic_counts[heuristic] = heuristic_counts.get(heuristic, 0) + 1

        for heuristic, count in heuristic_counts.items():
            reachabilities[heuristic] /= count

        return reachabilities

    def plot_reachability(self):
        '''Plots the average reachability of subscribers by each heuristic versus the
        specified x-axis parameter, ordered ascending.'''

        # TODO: use these markers
        # markers = 'x.*+do^s1_|'
        # plot(..., marker=markers[i%len(markers)])

        # First, we need to rotate the dict to group by heuristic in order to plot a curve for each
        new_stats = dict()
        xvalues = []  # TODO: how to plot these if they're labels not values????
        for (xvalue, group) in self.stats.items():
            reachabilities = self.get_reachability(group)
            print reachabilities
            xvalues.append(xvalue)
            for heuristic, reachability in reachabilities.items():
                new_stats.setdefault(heuristic, []).append(reachability)

        print 'x=', xvalues
        print 'stats:', self.stats
        print 'new_stats:', new_stats
        # TODO: may need to store these with the xvalue in order to order everything consistently

        # TODO: figure out how to label x-axis properly
        for (heuristic, yvalues) in new_stats.items():
            plt.plot(range(len(xvalues)), yvalues, label=heuristic)

        plt.xlabel(self.x_axis)
        plt.ylabel("avg % hosts reached")
        plt.title('Subscriber hosts reached')
        plt.show()

    def print_statistics(self):
        """Prints summary statistics for all groups and heuristics,
        in particular the mean and standard deviation of reachability."""

        for group_name, group in self.stats.items():
            reachabilities = self.get_reachability(group)
            for heur, reach in reachabilities.items():
                reach = np.array(reach)
                print "Group %s heuristic %s's Mean: %f; stdev: %f" % (group_name, heur, np.mean(reach), np.std(reach))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    stats = SeismicStatistics(args)
    stats.parse_all()
    stats.print_statistics()
    stats.plot_reachability()