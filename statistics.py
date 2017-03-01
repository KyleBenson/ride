#!/usr/bin/python

STATISTICS_DESCRIPTION = '''Gathers statistics from the campus_net_experiment.py output files in order to determine how
resilient different multicast tree-generating algorithms are under various scenarios.
Also handles printing them in an easily-read format as well as creating plots.'''

import logging as log
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

    # Which files to parse?  Only specify either dirs or files
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dirs', '-d', type=str, nargs="+",
                        help='''directories containing files from which to read outputs
                        (default=%(default)s)''')
    group.add_argument('--files', '-f', type=str, nargs="+", default=['results.json'],
                        help='''files from which to read output results
                        (default=%(default)s)''')

    # Controlling what data is plotted
    parser.add_argument('--x-axis', '-x', type=str, default='failure_model', dest='x_axis',
                        help='''name of parameter to plot reachability against:
                        places it on the x-axis (ordered for increasing reachability)
                        (default=%(default)s)''')
    parser.add_argument('--include-choices', '-c', default=None, nargs='+', dest='include_choices',
                        help='''name of heuristics' choice to include in analysis (default includes all)''')
    parser.add_argument('--include-heuristics', '-i', default=None, nargs='+', dest='include_heuristics',
                        help='''name of heuristics (before tree choice heuristic name is added)
                        to include in analysis (default includes all)''')

    # Controlling plots
    parser.add_argument('--title', '-t', type=str, default='Subscriber hosts reached',
                        help='''title of the plot (default=%(default)s)''')
    parser.add_argument('--ylabel', '-yl', type=str, default="avg host reach ratio",
                        help='''label to place on the y-axis (default=%(default)s)''')
    parser.add_argument('--xlabel', '-xl', type=str, default=None,
                        help='''label to place on the x-axis (default=%(default)s)''')
    parser.add_argument('--save', nargs='?', default=False, const='fig.png',
                        help='''save the figure to file automatically (default=%(default)s)''')
    parser.add_argument('--skip-plot', '-s', action='store_true', dest='skip_plot',
                        help='''disables showing the plot''')
    parser.add_argument('--no-legend', '-l', action='store_false', dest='legend',
                        help='''disables showing the legend; useful if you have too many groups
                        but still want to look at general trends''')


    # Misc control params
    parser.add_argument('--debug', '--verbose', '-v', type=str, default='info', nargs='?', const='debug',
                        help='''set verbosity level for logging facility (default=%(default)s, %(const)s when specified with no arg)''')

    # TODO: graph metrics as box and whisker plot.  perhaps we specify what gets put on the y axis (default=reach)

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

        log_level = log.getLevelName(config.debug.upper())
        log.basicConfig(format='%(levelname)s:%(message)s', level=log_level)


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
        try:
            param_value = data['params'][self.x_axis]
        except KeyError as e:
            # HACK: nhosts is actually two parameters, so need to create an aggregate for it
            if self.x_axis != 'nhosts':
                raise e
            param_value = (data['params']['nsubscribers'], data['params']['npublishers'])
            # param_value = "%s,%s" % param_value
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
            # Skip any results with complete failure
            if run['oracle'] == 0.0:
                continue

            for heuristic, reachability in run.items():
                # HACK to skip other parameters / metrics for this run
                # Also, skip any heuristics we don't want included (if this option was specified)
                # TODO: actually handle metrics?  they should probably go on y-axis, though plotting them against reach could be useful too.
                if (self.config.include_heuristics is not None\
                        and heuristic not in self.config.include_heuristics)\
                        or (self.config.include_heuristics is None\
                                and heuristic in METRICS_TO_SKIP):
                    continue

                # Some heuristics have nested results with further parameters.
                # Here we add the heuristic name to those parameters to make a
                # new unique heuristic group.
                # We also filter these by 'choice' (tree-choosing heuristic).
                # If only one choice, don't include choice name.
                if isinstance(reachability, dict):
                    reachability_dict = {}
                    for choice, v in reachability.items():
                        if self.config.include_choices is not None:
                            if choice not in self.config.include_choices:
                                continue
                            if len(self.config.include_choices) > 1:
                                reachability_dict["%s (%s)" % (heuristic, choice)] = v
                            else:
                                reachability_dict[heuristic] = v
                        else:
                            # TODO: something similar where we trim off the heuristic name if only including one of them?
                            reachability_dict["%s (%s)" % (heuristic, choice)] = v
                else:
                    reachability_dict = {heuristic: reachability}
                # Now we can iterate over all of them (or the original one)
                for _heuristic, _reachability in reachability_dict.items():
                    reachabilities[_heuristic] = reachabilities.get(_heuristic, 0) + _reachability
                    heuristic_counts[_heuristic] = heuristic_counts.get(_heuristic, 0) + 1

        # TODO: include error bars?  count would be done differently then
        # Finally, convert totals to averages and return them
        for heuristic, count in heuristic_counts.items():
            reachabilities[heuristic] /= float(count)

        return reachabilities

    def plot_reachability(self):
        """Plots the average reachability of subscribers by each heuristic versus the
        specified x-axis parameter, ordered ascending by x-axis param.
        NOTE: we try to extract numerical values from the x-axis parameter strings if possible."""

        # First, we need to rotate the dict, which is currently indexed
        # by x-axis parameter, to index by heuristic (values are lists
        # of reachabilities that correspond to the list of xvalues) instead.
        # Then we can plot a curve for each.
        new_stats = dict()
        xvalues = []
        for (xvalue, results) in self.stats.items():
            reachabilities = self.get_reachability(results)
            xvalues.append(xvalue)
            # Gather the groups and their respective y-values that
            # will be plotted for this x-value
            for heuristic, reachability in reachabilities.items():
                log.debug("Reach for x=%s, heur[%s]: %f" % (xvalue, heuristic, reachability))
                new_stats.setdefault(heuristic, []).append(reachability)

        # Extract numerical xvalues from strings
        # NOTE: don't forget to sort them since we'll do that when plotting!
        try:
            xvalues = [float(x) for x in xvalues]
        except ValueError:
            # failure_model looks like: uniform/0.100000
            if self.x_axis == 'failure_model':
                xvalues = [float(x.split('/')[1]) for x in xvalues]
            # If x-axis contains general strings (str or unicode),
            # need to request numerics instead
            elif any(isinstance(xv, basestring) for xv in xvalues):
                plt.xticks(range(len(xvalues)), sorted(xvalues))
                xvalues = range(len(xvalues))
        except TypeError as e:
            # nhosts will format them as tuples, which is good for sorting,
            # but bad for labels and actual plotting
            if self.x_axis == 'nhosts':
                _xval = tuple("%s,%s" % (s,p) for s,p in sorted(xvalues))
                plt.xticks(range(len(xvalues)), _xval)
            else:
                raise e

        # Plot each group (heuristic) with different markers / colors
        # TODO: figure out how to consistently color different heuristics/groups
        markers = 'x.*+do^s1_|'
        colors = 'rbgycm'
        linestyles = ['solid','dashed','dashdot','dotted']
        i = 0
        # TODO: order the heuristics appropriately (oracle first, unicast last)
        for (heuristic, yvalues) in new_stats.items():
            log.debug("plotting for %s: %s vs. %s" % (heuristic, xvalues, yvalues))
            # sort by xvalues
            xval, yval = zip(*sorted(zip(xvalues, yvalues)))
            if not (len(xval) == len(xvalues) and len(yval) == len(yvalues)):
                log.warn("We seem to be missing some y or x values for heuristic %s" % heuristic)

            # HACK: tuples will cause pyplot to think it's multi-dimensional data
            if self.x_axis == 'nhosts':
                xval = range(len(xval))

            # TODO: optional bar graph?
            plt.plot(xval, yval, label=heuristic, marker=markers[i%len(markers)],
                     color=colors[i%len(colors)], linestyle=linestyles[i%len(linestyles)])
            i += 1

        # Adjust the plot visually, including labelling and legends.
        plt.xlabel(self.config.xlabel if self.config.xlabel is not None else self.config.x_axis)
        plt.ylabel(self.config.ylabel)
        plt.title(self.config.title)
        if self.config.legend:
            plt.legend(loc=6)  # loc=4 --> bottom right
        # adjust the left and right of the plot to make them more visible
        xmin, xmax = plt.xlim()
        plt.xlim(xmin=(xmin - 0.05 * (xmax - xmin)), xmax=(xmax + 0.05 * (xmax - xmin)))

        if not self.config.skip_plot:
            plt.show()
        if self.config.save:
            # TODO: finish this!
            raise NotImplementedError("Can't save plots automatically yet")
            #savefig(os.path.join(args.output_directory, )

    def print_statistics(self):
        """Prints summary statistics for all groups and heuristics,
        in particular the mean and standard deviation of reachability."""

        for group_name, group in self.stats.items():
            reachabilities = self.get_reachability(group)
            for heur, reach in reachabilities.items():
                reach = np.array(reach)
                log.info("Group %s heuristic %s's Mean: %f; stdev: %f" % (group_name, heur, np.mean(reach), np.std(reach)))

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    stats = SeismicStatistics(args)
    stats.parse_all()
    stats.print_statistics()
    stats.plot_reachability()
