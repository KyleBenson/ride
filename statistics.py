#!/usr/bin/python
import numbers

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
# these two were added so that we can choose a consistent color/linestyle for plots
# TODO: we should probably just build up a list as we read them in order to handle more heuristics, esp. with params
CONSTRUCTION_HEURISTICS = ['red-blue', 'steiner', 'diverse-paths', 'ilp',
                           'steiner[double]', 'steiner[max]',]
CHOICE_HEURISTICS = ['min-missing-chosen', 'max-overlap-chosen', 'max-reachable-chosen', 'importance-chosen']

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
    parser.add_argument('--include-choices', '-c', default=None, nargs='*', dest='include_choices',
                        help='''name of multicast tree choice heuristics to include in analysis (default includes all).
                        Specify no args to filter out all tree-choosing heuristics.''')
    # TODO: determine how to plot ONLY choices for a given heuristic without plotting that heuristic's stats
    parser.add_argument('--include-heuristics', '-i', default=None, nargs='+', dest='include_heuristics',
                        help='''name of heuristics (before tree choice heuristic name is added)
                        to include in analysis (default includes all)''')
    parser.add_argument('--stats-to-plot', '-st', dest='stats_to_plot', default=None, nargs='+',
                        help='''rather than plotting the mean values of heuristics' reachability
                        (or complete error bars if that's not disabled), plot the given stats
                        instead.  Options are: (mean, stdev, min, max)''')
    # TODO: specify what gets put on the y axis (default=reach)

    ### Controlling plots

    # labelling
    parser.add_argument('--title', '-t', nargs='?', default='Subscriber hosts reached', const=None,
                        help='''title of the plot (default=%(default)s; no title if specified with no arg)''')
    parser.add_argument('--ylabel', '-yl', type=str, default="avg host reach ratio",
                        help='''label to place on the y-axis (default=%(default)s)''')
    parser.add_argument('--xlabel', '-xl', type=str, default=None,
                        help='''label to place on the x-axis (default=%(default)s)''')
    # We aren't plotting non-int/float values on y-axis currently so no need for this yet
    # parser.add_argument('--ynames', '-yn', type=str, default=None, nargs='+',
    #                     help='''replace y-axis parameter names with these values.
    #                     Specify them in the order the original graph put the parameter values in;
    #                     it will replace the parameter value names and then re-sort again''')
    parser.add_argument('--xnames', '-xn', type=str, default=None, nargs='+',
                        help='''replace x-axis parameter names with these values.
                        Specify them in the order the original graph put the parameter values in;
                        it will replace the parameter value names and then re-sort again''')

    # how to do the plot?
    parser.add_argument('--legend', '-l', nargs='?', dest='legend', type=int, const=None, default=0,
                        help='''disables showing the legend if specified with no args,
                        which is useful if you have too many groups but still want to look
                        at general trends.  Can optionally specify an integer passed to
                        matplotlib for determining the legend's location. Specifying 0 asks
                        matplotlib to find the best location. (default=%(default)s)''')
    parser.add_argument('--error-bars', '-err', action='store_true', dest='error_bars',
                        help='''show the error bars and max/min values on curves''')
    parser.add_argument('--sort-curves-by-name', '-byname', dest='sort_curves_by_name', action='store_true',
                        help='''sort curves by group name rather than default method,
                        which puts oracle on top, unicast on bottom, and the others in
                        approximate order of y-value''')


    # Misc control params
    parser.add_argument('--save', '-s', nargs='?', default=False, const='fig.png',
                        help='''save the figure to file automatically
                        (default=%(default)s or %(const)s when switch specified with no arg)''')
    parser.add_argument('--skip-plot', action='store_true', dest='skip_plot',
                        help='''disables showing the plot''')
    parser.add_argument('--debug', '--verbose', '-v', type=str, default='info', nargs='?', const='debug',
                        help='''set verbosity level for logging facility (default=%(default)s, %(const)s when specified with no arg)''')

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

        # Determine how we name the metrics as we gather them up.
        # If <=1 tree choice heuristic is requested, the name will
        # be solely the tree construction heuristic.
        # If <=1 tree construction heuristic (other than oracle/unicast)
        # is requested, the name will be solely the tree choice heuristic.
        # When both of these cases apply, we only use the heuristic name.
        # If we requested plotting certain statistical metrics (min, mean, etc.),
        # we should always include choice names or else it won't be clear what
        # value is being referred to by the heuristic's name.
        omnipresent_heuristics = {'oracle', 'unicast'}
        self.include_choice_name = True
        self.include_construction_name = True
        self.include_stats_name = False

        # if we're plotting stats other than just mean, we should label them as such even if just one of them
        if self.config.stats_to_plot is not None:
            self.include_stats_name = True
        if self.config.include_choices is not None and len(self.config.include_choices) <= 1\
                and not self.include_stats_name:
            # if we label with stats we should definitely include choice name even if only one to avoid confusion
            self.include_choice_name = False
        elif self.config.include_heuristics is not None and \
                        len(set(self.config.include_heuristics) - omnipresent_heuristics) <= 1:
            self.include_construction_name = False

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
        # TODO: support grouping by something other than the x_axis arg? use python itertools.groupby
        with open(fname) as f:
            # this try statement was added because the -d <dir> option didn't work with .progress files
            try:
                data = json.load(f)
            except ValueError as e:
                log.debug("Skipping file %s that raised error: %s" % (fname, e))
                return

        # store the results along with any others grouped according to the x-axis parameter
        try:
            param_value = data['params'][self.x_axis]
        except KeyError as e:
            # HACK: nhosts is actually two parameters, so need to create an aggregate for it
            if self.x_axis != 'nhosts':
                raise e
            param_value = (data['params']['nsubscribers'], data['params']['npublishers'])
            # param_value = "%s,%s" % param_value

        # HACK: topo is stored as a list, so convert it to a tuple
        # Actually maybe we want to just extract the [1:] strings?
        try:
            self.stats.setdefault(param_value, []).extend(data['results'])
        except TypeError:
            self.stats.setdefault(tuple(param_value), []).extend(data['results'])


    def gather_yvalues_from_raw_results(self, results):
        """Averages the reachabilities (or other value) over this collection of results
        (previously grouped by x-axis parameter name; hence the results
        can be grouped together since they've received the same treatment).

        @:param results - list of {heuristic1: reachability, heuristic2: {'max': max_reachability...}} dicts
        @:return dict stats_metrics, dict raw_values:
         where stats_metrics = {heuristic1: {'mean': np.array(avg_reachability), 'max': np.array(max_reachability), ...}}
         and raw_values simply contains {heuristic2: np.array(values), ...}
        """

        # Now gather up the y-values (typically reachability) for each heuristic/metric
        # in the proper data structure depending on what the results contain (dict vs. scalar).
        yvalues_dict = {}
        yvalues_array = {}
        for run in results:
            # Skip any results with complete failure
            if run['oracle'] == 0.0:
                continue

            for metric_name, yvalue in run.items():
                # HACK to skip other parameters / metrics for this run
                # Also, skip any heuristics we don't want included (if this option was specified)
                # TODO: actually handle metrics?  they should probably go on y-axis, though plotting them against reach could be useful too.
                if (self.config.include_heuristics is not None
                    and metric_name not in self.config.include_heuristics)\
                        or (self.config.include_heuristics is None
                            and metric_name in METRICS_TO_SKIP):
                    continue

                # Actual results may have nested results with further parameters.
                # As an example, consider a single run:
                # {
                #     "cost": {
                #         "max": 596.3999999999999,
                #         "mean": 585.92499999999995,
                #         "min": 579.0,
                #         "stdev": 5.1380322108760055,
                #         "unicast": 1481.4000000000071
                #     },
                #     "nhops": {
                #         "max": 9,
                #         "mean": 3.9896875000000001,
                #         "min": 3,
                #         "stdev": 1.3643519166050841
                #     },
                #     "oracle": 0.7975,
                #     "overlap": 31628,
                #     "run": 29,
                #     "steiner": {                        <------  metric_name=steiner, yvalue=this dict
                #         "all": 0.78,
                #         "importance-chosen": 0.7125,
                #         "max": 0.7125,
                #         "max-overlap-chosen": 0.7125,
                #         "max-reachable-chosen": 0.7125,
                #         "mean": 0.56031249999999999,
                #         "min": 0.135,
                #         "min-missing-chosen": 0.7125,
                #         "stdev": 0.17726409279306962
                #     },
                #     "unicast": 0.605
                # }
                #
                # Here we add the heuristic name to those parameters to make a
                # new unique heuristic group after filtering these by 'choice'
                # (tree-choosing heuristic).  We also extract the min, max, mean, stdev
                # for the heuristic (or other treatment/metric) in question.
                if isinstance(yvalue, dict):

                    metrics_to_gather_in_dict = ('max', 'min', 'mean', 'stdev')
                    if not all(key in yvalue for key in metrics_to_gather_in_dict):
                        log.warn("Did not find statistical metrics in results dictionary: must be results from older version?")

                    for metric_result_key, metric_results_value in yvalue.items():
                        # Collect the statistical metrics for this heuristic or metric
                        if metric_result_key in metrics_to_gather_in_dict:
                            yvalues_dict.setdefault(metric_name, {}).setdefault(metric_result_key, []).append(metric_results_value)

                        # Collect the reachabilities for the tree-choosing heuristics
                        # metric_result_key is a tree-choosing heuristic
                        else:
                            # TODO: need a hack for unicast cost metric when we expand to include metrics
                            if self.config.include_choices is not None and metric_result_key not in self.config.include_choices:
                                continue
                            # Build the name for the metric based on tree-choosing/construction heuristic
                            if self.include_choice_name:
                                if self.include_construction_name:
                                    name = "%s (%s)" % (metric_name, metric_result_key)
                                else:
                                    name = metric_result_key
                            else:
                                name = metric_name
                            yvalues_array.setdefault(name, []).append(metric_results_value)
                else:
                    yvalues_array.setdefault(metric_name, []).append(yvalue)

        # Before returning the values, we need to convert the lists into np.arrays
        return {k: {k2: np.array(v) for k2, v in d.items()} for k,d in yvalues_dict.items()},\
               {k: np.array(v) for k,v in yvalues_array.items()}
        # ENHANCE: cache the result so we don't recompute for print_statistics

    def plot_reachability(self):
        """Plots the average reachability of subscribers by each heuristic versus the
        specified x-axis parameter, ordered ascending by x-axis param.
        NOTE: we try to extract numerical values from the x-axis parameter strings if possible."""

        # First, we need to rotate the stats dict, which is currently indexed
        # by x-axis parameter value, to index by heuristic/metric (where values are lists
        # of reachabilities/metric values that correspond in order to the list of xvalues) instead.
        stats_by_group = dict()
        xvalues = []
        for (xvalue, results) in self.stats.items():
            yvalues_dicts, yvalues_arrays = self.gather_yvalues_from_raw_results(results)
            # TODO: ensure that all of the results are present for this xvalue, else put a placeholder
            xvalues.append(xvalue)

            # Gather the groups and their respective y-values that
            # will be plotted for this x-value.  The dicts already
            # have min, max, mean, stdev so we'll need to gather each
            # of those up and average them, whereas the arrays will
            # be directly converted to arrays and have those metrics
            # pulled form them using numpy.
            #
            # Each time we run this loop to completion, we've appended
            # each group's entry for this x-axis value i.e. a single
            # point in the plot.
            for group_name, group_values_dict in yvalues_dicts.items():
                log.debug("Mean value for x=%s, group[%s]: %f" % (xvalue, group_name, group_values_dict['mean'].mean()))

                # Heuristics' results are stored in these dicts so if we requested specific
                # stats this is where we should gather the ones we want
                if self.include_stats_name:
                    for stat in self.config.stats_to_plot:
                        _name = '%s'
                        if self.include_construction_name:
                            _name = group_name + ' (%s)'
                        stats_by_group.setdefault(_name % stat, []).append(group_values_dict[stat])
                else:
                    stats_by_group.setdefault(group_name, []).append(group_values_dict)

            for group_name, group_values_array in yvalues_arrays.items():
                log.debug("Mean value for x=%s, group[%s]: %f" % (xvalue, group_name, group_values_array.mean()))
                stats_by_group.setdefault(group_name, []).append(group_values_array)

        # Extract numerical xvalues from strings for plotting
        # NOTE: don't forget to sort them before applying names since we'll do that when plotting!
        # We'll relabel the x-axis to explicitly show the x-values rather than a uniform spread.
        xtick_values = xvalues
        xtick_locations = xvalues

        # HACKS: first try to extract proper values for known parameters
        # TODO: extract numeric values from topo?
        if self.x_axis == 'failure_model':
            # failure_model looks like: uniform/0.100000
            xvalues = [float(x.split('/')[1]) for x in xvalues]

        try:
            xvalues = [float(x) for x in xvalues]
            # now try to take this to an integer, which it may have been from the beginning
            if all(int(x) == x for x in xvalues):
                xvalues = [int(x) for x in xvalues]
            # updates these since we may have updated to numerics now
            xtick_values = xvalues
            xtick_locations = xvalues
        except ValueError as e:
            # If x-axis contains general strings (str or unicode),
            # need to request numerics instead
            if any(isinstance(xv, basestring) for xv in xvalues):
                xtick_values = sorted(xvalues)
                xtick_locations = range(len(xvalues))
        except TypeError as e:
            # HACKS: lists/tuples will cause this error
            # nhosts will format them as tuples, which is good for sorting,
            # but bad for labels and actual plotting
            if self.x_axis == 'nhosts':
                xtick_values = tuple("%s,%s" % (s,p) for s,p in sorted(xvalues))
                xtick_locations = range(len(xvalues))
            # topo is a list containing topology reader and filename
            elif self.x_axis == 'topo':
                xtick_values = tuple(topo[1].split('.')[0].split('_')[-1] for topo in sorted(xvalues))
                xtick_locations = range(len(xvalues))
            else:
                raise e
        finally:
            # optionally re-label the x-axis parameter values with new names
            # (ordered by original sorting).
            if self.config.xnames is not None:
                if len(xvalues) != len(self.config.xnames):
                    raise ValueError("Specified new labels don't have same length as original xvalues!")

                # try to extract numeric values from the user-provided labels
                try:
                    xtick_locations = [float(x) for x in self.config.xnames]
                    xtick_values = xtick_locations
                except ValueError:
                    xtick_locations = range(len(xvalues))
                    xtick_values = self.config.xnames

                # ENHANCE: optionally sort by new labels?  causes issues e.g. 200 < 40 since they're strings
                # now we've extracted floats if they specified compatible strings, so just need to update xticks in try statement above
                # new_xvalues = [x[0] for x in sorted(zip(self.config.xnames, sorted(xvalues)))]
                # may also need to adjust the sorting thing below as the strings might have been sorted differently than how your new xvalues will be

            plt.xticks(xtick_locations, xtick_values)

            # Verify we now have numeric values after all this hacking...
            if not all(isinstance(x, numbers.Number) for x in xvalues):
                # need to set xvalues to index values, but maintain ordering
                xvalues = [sorted(xvalues).index(x) for x in xvalues]

        # order the heuristics appropriately (oracle first, unicast last, rest alphabetical or by y-value)
        def __heuristic_sorter(tup):
            _group_name = tup[0]
            _yvalues = tup[1]
            if _group_name == 'unicast':
                return '~' if self.config.sort_curves_by_name else float("inf")  # highest ASCII letter
            if _group_name == 'oracle':
                return ' ' if self.config.sort_curves_by_name else -float("inf")  # lowest ASCII letter
            if self.config.sort_curves_by_name:
                return _group_name  # else alphabetical
            else:  # by y-value
                # average in case we're missing some y-values for certain x-values
                # don't forget about possible dicts with stats in them!
                try:
                    return -sum(y.mean() for y in _yvalues) / float(len(_yvalues))
                except AttributeError:
                    return -sum(y['mean'].mean() for y in _yvalues) / float(len(_yvalues))
        stats_to_plot = sorted(stats_by_group.items(), key=__heuristic_sorter)

        for (group_name, yvalues) in stats_to_plot:
            # We need to extract the actual yvalues from the yvalues' np.arrays' raw data,
            # which might mean getting the arrays from a dict of mean, min, max, stdev arrays.
            # Keep them as lists for now because we'll be sorting them with the xvalues.
            # After this try statement, yvalues will be a dict('mean': ...) if config.error_bars is true
            # TODO: perhaps we only want error-bars on SOME of the heuristics?  too many is too crowded...
            try:
                if self.config.error_bars:
                    yvalues = [{'mean': y.mean(), 'stdev': y.std(),
                                'min': y.min(), 'max': y.max()} for y in yvalues]
                else:
                    yvalues = [y.mean() for y in yvalues]
            except AttributeError:
                # must be a dict then...
                if self.config.error_bars:
                    yvalues = [{'mean': y['mean'].mean(), 'stdev': y['stdev'].mean(),
                                'min': y['min'].mean(), 'max': y['max'].mean()} for y in yvalues]
                else:
                    yvalues = [y['mean'].mean() for y in yvalues]
            log.debug("plotting for %s: %s vs. %s" % (group_name, xvalues, yvalues))

            # sort by xvalues
            xval, yval = zip(*sorted(zip(xvalues, yvalues)))
            if not (len(xval) == len(xvalues) and len(yval) == len(yvalues)):
                log.warn("We seem to be missing some y or x values for heuristic %s" % group_name)

            plot_kwargs = self.get_curve_style(group_name)

            # Optionally plot errorbars and min/max by overlaying two
            # different errorbars plots: one thicker than the other.
            # This gives us more flexibility than, looks better than,
            # and would not be correct if we used box-and-whisker.
            # Taken from http://stackoverflow.com/questions/33328774/box-plot-with-min-max-average-and-standard-deviation
            if self.config.error_bars:
                means = np.array([y['mean'] for y in yval])
                stdevs = np.array([y['stdev'] for y in yval])
                mins = np.array([y['min'] for y in yval])
                maxes = np.array([y['max'] for y in yval])

                plt.errorbar(xval, means, [means - mins, maxes - means], lw=1, **plot_kwargs)
                del plot_kwargs['label']  # to avoid 2 copies showing up in the legend
                plt.errorbar(xval, means, stdevs, lw=2, **plot_kwargs)
            else:
                plt.plot(xval, yval, **plot_kwargs)

        # Adjust the plot visually, including labelling and legends.
        plt.xlabel(self.config.xlabel if self.config.xlabel is not None else self.config.x_axis)
        plt.ylabel(self.config.ylabel)
        if self.config.title is not None:
            plt.title(self.config.title)
        if self.config.legend is not None:
            plt.legend(loc=self.config.legend)  # loc=4 --> bottom right
        # adjust the left and right of the plot to make them more visible
        xmin, xmax = plt.xlim()
        plt.xlim(xmin=(xmin - 0.05 * (xmax - xmin)), xmax=(xmax + 0.05 * (xmax - xmin)))

        if not self.config.skip_plot:
            plt.show()
        if self.config.save:
            if '.' not in self.config.save:
                log.warn("No file extension specified in filename we're saving to: %s; using .png" % self.config.save)
                self.config.save += '.png'
            plt.savefig(self.config.save, bbox_inches=0)  # may need to use 'tight' on some systems for bbox

    def get_curve_style(self, group_name):
        """ We want to plot each group (heuristic) with different markers / colors, but
        consistently color heuristics/groups to easily compare them.
        :param group_name:
        :return dict: containing label, marker, color, and linestyle
        """

        # markers = 'x.*+do^s1_|'  # set explicityly currently
        colors = 'rbgy'  # 'rbgycm'  # c/m are reserved!
        linestyles = ['solid','dashed','dashdot','dotted']

        # We determine what groups have been requested to be varied
        # and assign them consistent values based on the arguments'
        ret =  {'label': group_name}

        # COLORS go with construction heuristics
        # First, return immediately for these special cases to avoid trying to include markers or linestyles on them
        if group_name == 'unicast':
            ret['color'] = 'c'
            return ret
        elif group_name == 'oracle':
            ret['color'] = 'm'
            return ret
        # Next, always assign colors if we're explicitly labelling which construction algorithm was used
        elif self.include_construction_name:
            heur_name = group_name.split(' ')[0]
            color_idx = CONSTRUCTION_HEURISTICS.index(heur_name)
            color = colors[color_idx % (len(colors))]
        else:
            color = None

        # LINESTYLES go with tree-choosing heuristics
        # TODO: make this use markers instead if we add more tree-choosing heuristics?
        linestyle = None
        if self.include_choice_name and 'chosen' in group_name:
            # NOTE: only try to slice the choice name out if the group name has multiple parts!
            if '(' in group_name:
                choice_name = group_name[group_name.find('(')+1 : group_name.find(')')]
            else:
                choice_name = group_name
                assert not self.include_construction_name, "we don't currently support any construction algorithms with 'chosen' in the name: time to fix this"
            linestyle_idx = CHOICE_HEURISTICS.index(choice_name)
            linestyle = linestyles[linestyle_idx % len(linestyles)]

        # MARKERS go with stats (min, max, etc.)
        # NOTE: we do not assign a marker AND a linestyle to the same curve as they're different labels!
        # TODO: change that fact if we add more choice heuristics as it'll result in re-used colors anyway
        marker = None
        marker_map = {'max': '*', 'mean': 'd', 'min': '^', 'std': 's'}
        if self.include_stats_name and linestyle is None:
            # NOTE: only try to slice the choice name out if the group name has multiple parts!
            if '(' in group_name:
                stat_name = group_name[group_name.find('(')+1 : group_name.find(')')]
            else:
                assert not self.include_construction_name
                stat_name = group_name

            marker = marker_map[stat_name]
            # this only used if we're going to convert marker to a color
            marker_idx = self.config.stats_to_plot.index(stat_name)

        # Colors are the best way to distinguish curves, so default to that
        # if we didn't include many heuristics.
        if color is None:
            # NOTE: prioritize using colors for tree-choices and using our hard-coded markers for stats
            if linestyle is not None:
                color = colors[linestyle_idx % len(colors)]
                linestyle = None
            elif marker is not None:
                if self.include_choice_name:
                    color = 'k'  # need to ensure a color chosen or pyplot will assign them for you
                else:
                    color = colors[marker_idx % len(colors)]
                    marker = None
            else:
                # didn't assigned colors, markers, OR linestyles
                color = 'k'
                log.warn("didn't assigned colors, markers, OR linestyles: this is not well-supported; defaulting to black")
        assert color, "need to ensure a color chosen or pyplot will assign them for you"

        ret['color'] = color
        if marker is not None:
            ret['marker'] = marker
        if linestyle is not None:
            ret['linestyle'] = linestyle
        return ret

    def print_statistics(self):
        """Prints summary statistics for all groups and heuristics,
        in particular the mean and standard deviation of reachability."""

        msg = "Group %s heuristic %s's Mean: %f; stdev: %f; min: %f; max: %f"
        for group_name, group in self.stats.items():
            reachabilities_dict, reachabilities_array = self.gather_yvalues_from_raw_results(group)

            print "reach_dicts:"
            for heur, reach_dict in reachabilities_dict.items():
                assert all(isinstance(v, np.ndarray) for v in reach_dict.values())
                log.info(msg % (group_name, heur, reach_dict['mean'].mean(),
                                reach_dict['stdev'].mean(), reach_dict['min'].mean(),
                                reach_dict['max'].mean()))

            print "reach_arrays:"
            for heur, reach_array in reachabilities_array.items():
                assert isinstance(reach_array, np.ndarray)
                log.info(msg % (group_name, heur, reach_array.mean(), reach_array.std(),
                                reach_array.min(), reach_array.max()))


def run_tests():
    dummy_args = parse_args([])
    stats = SeismicStatistics(dummy_args)
    # create some dummy results with really simple values for testing
    nresults = 4
    results = [
        {
            "cost": {
                "max": 2000.0*i,
                "mean": 1000.0*i,
                "min": 10.0*i,
                "stdev": 20.0*i,
                "unicast": 4000.0*i
            },
            "nhops": {
                "max": 10.0*i,
                "mean": 5.0*i,
                "min": 1.0*i,
                "stdev": 2.0*i
            },
            "oracle": 1.0*i,
            "overlap": 10000*i,
            "run": i-1,
            ["steiner", "red-blue", "fake_heuristic"][i%3]: {  # note that since i starts at 1 this means red-blue repeats first
                "all": 0.7*i,
                "importance-chosen": 0.55*i,
                "max": 0.6*i,
                "max-overlap-chosen": 0.45*i,
                "max-reachable-chosen": 0.4*i,
                "mean": 0.5*i,
                "min": 0.1*i,
                "min-missing-chosen": 0.3*i,
                "stdev": 0.2*i
            },
            "unicast": 0.25*i
        } for i in range(1, nresults+1)
        ]

    # validate correctness of gather_yvalues_from_raw_results()
    stats_dicts, stats_arrays = stats.gather_yvalues_from_raw_results(results)
    print stats_dicts
    print stats_arrays
    # assert stats_dicts['']

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_tests()
        exit()

    args = parse_args(sys.argv[1:])
    stats = SeismicStatistics(args)
    stats.parse_all()
    stats.print_statistics()
    stats.plot_reachability()
