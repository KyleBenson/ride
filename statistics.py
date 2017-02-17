#!/usr/bin/python

STATISTICS_DESCRIPTION = '''Gathers statistics from the campus_net_experiment.py output files in order to determine how
resilient different multicast tree-generating algorithms are under various scenarios.
Also handles printing them in an easily-read format as well as creating plots.'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import argparse
import sys
import os
import json


# skip over these metrics when looking for results from heuristics
METRICS_TO_SKIP = {'run', 'nhops', 'overlap', 'cost'}


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
                        help='''name of parameter to plot reachability against:
                        places it on the x-axis (ordered for increasing reachability)
                        (default=%(default)s)''')
    parser.add_argument('--title', '-t', type=str, default='Subscriber hosts reached',
                        help='''title of the plot (default=%(default)s)''')
    parser.add_argument('--y-axis', '-y', type=str, default="avg host reach ratio", dest='y_axis',
                        help='''label to place on the y-axis (default=%(default)s)''')

    # TODO: filter heuristics / handle chosens
    # TODO: graph metrics as box and whisker plot


    return parser.parse_args(args)


class SeismicStatistics(object):
    """Parse results and visualize reachability rate."""

    def __init__(self, config):
        super(self.__class__, self).__init__()
        self.x_axis = config.x_axis
        self.dirs = config.dirs
        self.files = config.files
        self.parsing_dirs = config.dirs is not None
        self.config = config

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

    def get_reachability(self, results):
        """Averages the reachabilities over this collection of results
        (previously grouped by x-axis parameter name; hence the results
        can be grouped together since they've received the same treatment).
        @:param group - list of {heuristic1: reachability, heuristic2: reachability} dicts
        @:return dict with each {heuristic1: avg_reachability, ...}
        """

        # First add up the total reachability for each heuristic, and
        # then divide by the count to get the average
        reachabilities = {}
        heuristic_counts = {}
        for run in results:
            for heuristic, reachability in run.items():
                # HACK to skip other parameters / metrics for this run
                # TODO: if we specify one of the metrics as treatment groups, don't skip it!
                if heuristic in METRICS_TO_SKIP:
                    continue

                # Some heuristics have nested results with further parameters.
                # Here we add the heuristic name to those parameters to make a
                # new unique heuristic group.
                # TODO: filter these
                if isinstance(reachability, dict):
                    reachability = {"%s-%s" % (heuristic, k): v for k, v in reachability.items()}
                else:
                    reachability = {heuristic: reachability}
                # Now we can iterate over all of them (or the original one)
                for _heuristic, _reachability in reachability.items():
                    reachabilities[_heuristic] = reachabilities.get(_heuristic, 0) + _reachability
                    heuristic_counts[_heuristic] = heuristic_counts.get(_heuristic, 0) + 1

        # TODO: include error bars?
        # Finally, convert totals to averages and return them
        for heuristic, count in heuristic_counts.items():
            reachabilities[heuristic] /= float(count)

        return reachabilities

    def plot_reachability(self):
        """Plots the average reachability of subscribers by each heuristic versus the
        specified x-axis parameter, ordered ascending."""

        # First, we need to rotate the dict, which is currently indexed
        # by x-axis parameter, to index by heuristic (values are lists
        # of reachabilities) instead. Then we can plot a curve for each.
        new_stats = dict()
        xvalues = []
        for (xvalue, results) in self.stats.items():
            reachabilities = self.get_reachability(results)
            xvalues.append(xvalue)
            # Gather the groups and their respective y-values that
            # will be plotted for this x-value
            for heuristic, reachability in reachabilities.items():
                new_stats.setdefault(heuristic, []).append(reachability)

        # TODO: may need to store these with the xvalue in order to order everything consistently
        # especially if some files include some heuristics and others don't

        # If x-axis contains strings (str or unicode), need to request numerics instead
        # TODO: try to space these according to any numbers found in them rather than evenly
        if any(isinstance(xv, basestring) for xv in xvalues):
            plt.xticks(range(len(xvalues)), xvalues)
            xvalues = range(len(xvalues))

        # Plot each group (heuristic) with different markers / colors
        markers = 'x.*+do^s1_|'
        colors = 'rbgycm'
        linestyles = ['solid','dashed','dashdot','dotted']
        
        i = 0
        for (heuristic, yvalues) in new_stats.items():
            plt.plot(xvalues, yvalues, label=heuristic, marker=markers[i%len(markers)],
                     color=colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)])
            i += 1

        plt.xlabel(self.x_axis)
        plt.ylabel(self.config.y_axis)
        plt.title(self.config.title)
        plt.legend(loc=4)  # bottom right
        # adjust the left and right of the plot to make them more visible
        xmin, xmax = plt.xlim()
        plt.xlim(xmin=(xmin - 0.05 * (xmax - xmin)), xmax=(xmax + 0.05 * (xmax - xmin)))
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