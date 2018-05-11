import os
import pandas as pd
import parse
import json

from scale_client.stats.statistics import ScaleStatistics

import logging
log = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    log.error("failed to import pyplot: don't call plotting functions! Error: %s" % e)
    plt = None


class NetworkExperimentStatistics(ScaleStatistics):
    """Parse the results.json-style output files from a NetworkExperiment, possibly by parsing the output files specified
    in each run.  Used to build a 'database' (really a CSV will be output) of the individual results for visualization
    of various statistics e.g. message delivery rate, latency, etc."""

    ### Helper funcs you'll likely want to override

    @property
    def varied_params(self):
        """Returns a set of the parameter names (post-extract_parameters()) that may be varied in different
        experimental treatments.  This is used predominately for average_over_runs() to know which columns we should
        group by and not drop after averaging over the 'run' column."""
        # TODO: figure out how to do this based on the NetworkExperiment / NetworkChannelState objects???
        return {'loss', 'bw', 'lat', 'jitter', 'exp_type'}

    @property
    def param_col_map(self):
        """
        Returns a dict mapping full parameter names to shorter column names that make viewing the DataFrame extracted
        from the results easier.  Mostly this is just used by extract_parameters()
        :return:
        """

        # WARNING: latency/delay may end up being used as a column if you calculate e.g. event-delivery delay
        # Similarly, loss could be used for something like propagation_loss in the future...
        return dict(error_rate='loss', experiment_type='exp_type', bandwidth='bw', latency='lat')

    # TODO: probably also want to do a drop_param() (not property as can do some easy pattern matching)
    # so we can cut out columns that don't go into DataFrames well.

    def collate_outputs_results(self, *results):
        """Combines the parsed results from an outputs_dir into a single DataFrame.  By default just uses _collate_results()"""
        return self.collate_results(*results)

    def get_treatment_str(self, results_filename, **params):
        """Build a string representing the experimental treatment these results come from, which is by default
        just the name of the results file without the '.json' extension or run number ('results.1.json')."""
        if results_filename.endswith('.json'):
            treatment = results_filename[:-len('.json')]
        else:
            log.warning("why does results file not end with .json???  hopefully everything parses okay...")
            treatment = results_filename

        # trim off run #, but ensure the trailing text is a run# before trimming it!
        if parse.parse('.{:d}', treatment[treatment.rfind('.'):]):
            treatment = treatment[:treatment.rfind('.')]

        return treatment

    # NOTE: you'll probably also need to override the choose_parser() function

    ### Helper functions that should be well-suited to most NetworkExperiments

    def extract_parameters(self, exp_params):
        """
        Extracts the relevant parameters from the specified ones, possibly changing some of their names to a shorter or
        more distinct one.  Removes any parameters with null values.
        :param exp_params:
        :type exp_params: dict
        :return: dict of extracted params
        """

        # XXX: this function should only be run once per extracted exp_params dict as it directly modifies it and will
        # generate warnings if you run it again.
        exp_params = exp_params.copy()

        for k, v in self.param_col_map.items():
            if v not in exp_params:
                exp_params[v] = exp_params.pop(k, None)
            ## Actually, this should be fine as we already store some of the params in a shortened form
            # else:
            #     log.warning("shorter column name %s already appears in exp_params! skipping it..." % v)

        # XXX: clear any null params e.g. unspecified random seeds (do this last in case anything above set None values)
        for k, v in exp_params.items():
            if v is None:
                log.debug("deleting null parameter: %s" % k)
                del exp_params[k]

        return exp_params

    def extract_stats_from_results(self, results, filename, **exp_params):
        """
        With the correct ParsedSensedEvent objects (you may need to override this to choose the right parser object),
        parse the output files in the given results and return them as an aggregated DataFrame.

        :param results: 'results' json list[dict] taken directly from the results file with each dict being a run
        :type results: list[dict]
        :param filename: name of the file these results were read from: its path is used to build the actual
         path of output files that will be further parsed!
        :param exp_params:
        :return: the stats
        :rtype: pd.DataFrame
        """

        treatment = self.get_treatment_str(filename, **exp_params)

        # The outputs_dir is specified relative to the results file we're currently processing
        this_path = os.path.dirname(filename)
        dirs_to_parse = [(os.path.join(this_path, run['outputs_dir']), run) for run in results]

        # parse each dir and combine all the parsed results into a single data frame
        stats = []
        for d, r in dirs_to_parse:
            try:
                this_run_params = self.extract_run_params(r, filename, **exp_params)
                o = self.parse_outputs_dir(d, treatment=treatment, **this_run_params)
                if self.is_results_good(o):
                    stats.append(o)
            except BaseException as e:
                if self.config.raise_errors:
                    raise
                else:
                    log.warning("skipping output directory %s that generated error: %s" % (d, e))
        if stats:
            stats = self.merge_all(*stats)
            return stats
        else:
            return None

    def extract_run_params(self, run_results, filename, **exp_params):
        """
        Extracts any relevant per-run parameters and includes them in the returned modified version of the specified
        experiment parameters.  Base implementation just saves the run number ('run').

        :param run_results:
        :param filename:
        :param exp_params:
        :return:
        """
        exp_params['run'] = run_results['run']
        return exp_params

    def parse_results(self, results, filename, **params):
        """
        This version has to parse a top-level results.json-style file by parsing the individual output files from outputs_dir
        :param results:
        :type results: str
        :param filename:
        :param params:
        :return:
        """

        # this is a results.json file we're parsing at the 'top level', so we need to extract all the outputs files from it first
        assert filename in self.files

        results = json.loads(results)
        params = results['params']
        results = results['results']

        # Extract the properly-formatted results dict by combining the parameters as static data columns into the
        # results DataFrame
        params = self.extract_parameters(params)
        results = self.extract_stats_from_results(results, filename=filename, **params)

        return results

    def parse_outputs_dir(self, out_dir, treatment, **params):
        """
        Parse the individual results files in the specified outputs_dir and
        return the resulting data combined into a single pd.DataFrame.
        :param out_dir:
        :param treatment:
        :param params:
        :return:
        :rtype: pd.DataFrame
        """

        res = []
        for fname in os.listdir(out_dir):
            # need to determine how to parse this file before we can do it
            fname = os.path.join(out_dir, fname)
            parser = self.choose_parser(fname, treatment=treatment, **params)
            if parser is None:
                continue

            data = self.read_file(fname)

            # NOTE: this should include the treatment as a column in the resulting DataFrame
            data = parser(data, filename=fname, treatment=treatment, **params)
            if self.is_results_good(data):
                res.append(data)

        # may need to override this in order to merge the different DataFrames in a particular way specific to your application
        if not res:
            return None
        res = self.collate_outputs_results(*res)
        return res

    def average_over_runs(self, df, column_to_drop='run'):
        """
        Averages the given DataFrame's values over all the runs for each unique treatment grouping.

        NOTE: you can also use this function (or its alias) to average over and drop a different column.

        :type df: pd.DataFrame
        :param column_to_drop: averages all other non-treatment columns over this one and drops it (default='run')
        :rtype: pd.DataFrame
        """
        # XXX: need to ensure we have all these parameters available
        cols = set(df.columns.tolist())
        group_params = cols.intersection(self.varied_params)
        group_params.discard(column_to_drop)
        group_params = list(group_params)

        df = df.groupby(group_params).mean().reset_index()
        # XXX: if we're trying to average over a column with string values, it will have already been dropped!
        try:
            df = df.drop(column_to_drop, axis=1)
        except ValueError:
            pass
        return df
    average_over_column = average_over_runs

    def plot(self, x, y, groupby=None, average_over=('run',), stats=None,
             show_plot=True, xlabel=None, ylabel=None, **kwargs):
        """
        Plots the given column names of the specified stats DataFrame (self.stats by default)
        :param x:
        :param y: specify a list of column names to plot multiple curves
        :param groupby: group DF by this column and plot each group as a subplot
        :param average_over: average over all these parameters and drop them first
        :type average_over: iterable
        :param stats:
        :param show_plot: if False, don't show the plot (so you can add other lines to it first)
        :param xlabel: use the specified label on x axis instead of the default (name of x column)
        :param ylabel: use the specified label on y axis instead of the default (name of y column)
        :param kwargs: passed to DataFrame.plot()
        :return:
        """

        if not plt:
            log.error("pyplot wasn't imported!  skipping over plot...")

        if stats is None:
            stats = self.stats

        # ensure we don't average over a column being plotted on an axis!
        average_over = set(average_over)
        average_over.discard(x)
        if not isinstance(y, basestring):
            for _y in y:
                average_over.discard(_y)

        print "STATS BEFORE AVG:", stats

        for col in average_over:
            stats = self.average_over_column(stats, column_to_drop=col)

        log.info("Plotting stats:\n%s" % stats)

        if groupby:
            fig, ax = plt.subplots()
            for g, df in stats.groupby(groupby):
                if not isinstance(y, basestring):
                    label = ["%s(%s=%s)" % (_y, groupby, g) for _y in y]
                else:
                    label = "%s=%s" % (groupby, g)
                ax = df.plot(x=x, y=y, label=label, ax=ax, **kwargs)
        else:
            ax = stats.plot(x=x, y=y, **kwargs)

        plt.xlabel(x if xlabel is None else xlabel)
        plt.ylabel(y if ylabel is None else ylabel)
        if show_plot:
            plt.show()

        return ax

    def __iadd__(self, other):
        self.stats = pd.concat((self.stats, other.stats), ignore_index=True)
        return self


if __name__ == '__main__':
    stats = NetworkExperimentStatistics.main()

    # now you can do something with the stats to e.g. get your own custom experiment results
    final_stats = stats.stats

    if stats.config.output_file:
        stats.output_stats(stats=final_stats)
