#!/usr/bin/python

STATISTICS_DESCRIPTION = '''Gathers statistics from the networkx_smart_campus_experiment.py output files in order to determine how
resilient different multicast tree-generating algorithms are under various scenarios.'''

import logging
log = logging.getLogger(__name__)
import argparse
import sys
import os
import json
import pandas as pd

from seismic_warning_test.statistics import SeismicStatistics as MininetSeismicStatistics

# skip over these metrics when looking for reachability results from heuristics
AVAILABLE_METRICS = {'run', 'nhops', 'overlap', 'cost'}

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

    parser.add_argument('--output-file', '-o', metavar='output_file', type=str, default=False,
                        help='''if specified, output the resulting DataFrame to a CSV file''')
    parser.add_argument('--debug', '--verbose', '-v', type=str, default='info', nargs='?', const='debug',
                        help='''set verbosity level for logging facility (default=%(default)s, %(const)s when specified with no arg)''')

    return parser.parse_args(args)


class SmartCampusExperimentStatistics(object):
    """Parse results and visualize statistics e.g. reachability rate, latency (mininet exps only)."""

    def __init__(self, config):
        super(self.__class__, self).__init__()
        self.dirs = config.dirs
        self.files = config.files
        self.parsing_dirs = config.dirs is not None
        self.config = config

        log_level = logging.getLevelName(config.debug.upper())
        log.setLevel(log_level)

        # mininet version stores a SeismicStatistics object to hold stats extracted from results
        self.stats = None

        # This will be set when parsing files to determine the type of experimental
        # results file we're working with.  Can be either 'mininet' or 'networkx'
        self.experiment_type = None

    def parse_all(self):
        """Parse either all directories (if specified) or all files."""
        if self.parsing_dirs:
            for dirname in self.dirs:
                self.parse_dir(dirname)
        else:
            for fname in self.files:
                self.parse_file(fname)

    def parse_dir(self, dirname):
        log.debug("parsing dir: %s" % dirname)
        for filename in os.listdir(dirname):
            filename = os.path.join(dirname, filename)
            if filename.endswith('.progress') or not os.path.isfile(filename):
                continue
            self.parse_file(filename)

    def parse_file(self, fname):
        log.debug("parsing file: %s" % fname)

        with open(fname) as f:
            # this try statement was added because the -d <dir> option didn't work with .progress files
            try:
                data = json.load(f)
            except ValueError as e:
                log.debug("Skipping file %s that raised error: %s" % (fname, e))
                return

        params = data['params']
        results = data['results']

        # Extract the properly-formatted results dict
        # NOTE: old netx simulations didn't include this field!
        self.experiment_type = data['params'].get('experiment_type', 'networkx')
        params = self.extract_parameters(params)
        results = self.extract_stats_from_results(results, filename=fname, **params)

        return results

    def extract_parameters(self, exp_params, experiment_type=None):
        """
        Extracts the relevant parameters from the specified ones, possibly changing some of their names to a shorter or
        more distinct one.
        :param exp_params:
        :type exp_params: dict
        :param experiment_type: whether its a 'mininet' or 'networkx' experiment, default is self.experiment_type
        :return: dict of extracted params
        """

        if experiment_type is None:
            experiment_type = self.experiment_type

        ##### First, make some modifications to the parameters so we can e.g. view columns easier

        ## These modifications are common across all exp types
        # XXX: rename some params, esp. for printing cols
        exp_params['const_alg'] = exp_params.pop("heuristic")
        exp_params['exp_type'] = exp_params.pop("experiment_type", "networkx")  # old version didn't have this
        # XXX: since the only failure model we currently use is uniform, let's just extract the 'fprob'
        exp_params['fprob'] = float(exp_params.pop('failure_model').split('/')[1])
        # XXX: old version included topo type first, but we never made additional types
        if not isinstance(exp_params['topo'], basestring):
            exp_params['topo'] = exp_params['topo'][1]
            assert isinstance(exp_params['topo'], basestring)
        # XXX: clear any null params e.g. unspecified random seeds
        for k,v in exp_params.items():
            if v is None:
                log.debug("deleting null parameter: %s" % k)
                del exp_params[k]

        if experiment_type == 'mininet':
            # handle additional parameters in mininet version
            exp_params['select_policy'] = exp_params.pop('tree_choosing_heuristic')
            # XXX: we don't use traffic generation anymore since it's just done by publishers
            if not exp_params.get("n_traffic_generators"):
                del exp_params["n_traffic_generators"]
                del exp_params["traffic_generator_bandwidth"]

        elif experiment_type == 'networkx':
            pass  # nothing different
        else:
            log.error("unrecognized 'experiment_type' %s. Aborting." % experiment_type)
            exit(1)

        return exp_params

    def extract_stats_from_results(self, results, filename=None, experiment_type=None, **exp_params):
        """
        With our SeismicStatistics object, parse the given results, which contains paths to the actual output
        files that will need to be further parsed. Return the parsed data as a SeismicStatistics object that can
        be used to get the actual DataFrames
        :param results: raw 'results' json list[dict] taken directly from the results file with each dict being a run
        :type results: list[dict]
        :param filename: name of the file these results were read from: used to build the actual
         path of output files that will be further parsed!
        :param experiment_type:
        :param exp_params:
        :return: the stats
        :rtype: SeismicStatistics
        """

        if experiment_type is None:
            experiment_type = self.experiment_type

        # For results from a Mininet experiment, we need to parse the specified client files.
        if experiment_type == 'mininet':

            if filename is None:
                log.warning("no filename specified! assuming the 'outputs_dir' for each run in results is "
                            "relative to current working directory!")
                filename = ''

            # The outputs_dir is specified relative to the results file we're currently processing
            this_path = os.path.dirname(filename)
            dirs_to_parse = [os.path.join(this_path, run['outputs_dir']) for run in results]

            # We want to combine all of our results into one SeismicStatistics object:
            if self.stats is None:
                self.stats = MininetSeismicStatistics(dirs=dirs_to_parse)
                self.stats.parse_all(**exp_params)
            else:
                self.stats.parse_all(dirs_to_parse, **exp_params)

            stats = self.stats

            # QUESTION: how are we going to keep the experiment parameters on the data frames if we do a bunch of filtering operations that might remove them?

            # TODO: save data path change times and quake start time?!

            # TODO: now that we have the dp change times, we can slice up the results to get just the alerts received in a timely manner

            # TODO: save the oracles results!

            # TODO: maybe save failures?
            # probably don't need the pubs/subs since we just use the host_id to directly compute it


        elif experiment_type == 'networkx':
            # just use the file name without .json as treatment
            assert filename.endswith('.json'), "why does results file not end with .json???"
            treatment = filename[:-len('.json')]
            stats = NetworkxSeismicStatistics(results, treatment=treatment, **exp_params)

            # we'll combine all the parsed results into a single data frame
            if self.stats is None:
                self.stats = stats
            else:
                self.stats += stats
        else:
            log.error("unrecognized 'experiment_type' %s. Aborting." % experiment_type)
            exit(1)

        return stats


class NetworkxSeismicStatistics(object):
    """
    Parses the raw results (reachabilities) into a DataFrame.
    """

    def __init__(self, results, **exp_params):
        """
        :param results:
        :type results: list[dict]
        :param exp_params:
        """
        super(NetworkxSeismicStatistics, self).__init__()

        # Skip over all these metrics/records for now since we aren't using them
        # NOTE: unicast we'll actually use, but we can't add it in the for loop: see XXX note down below
        # Same with cost
        skip_keys = {'stdev', 'all', 'nhops', 'failed_nodes', 'failed_links', 'unicast', 'cost'}
        # TODO: add back in? 'overlap',
        # Since the 'const_alg' is a key in this dict with all the MDMT-selection policies' reachabilities
        # as the values, we'll actually be making multiple rows for each parsed run here.
        # Hence, we just build a new data frame for each run and concat them at the end.

        data_frames = []
        for run in results:
            # use a dict for DataFrame constructor where most values will be scalar but the ones we're 'unrolling' from
            # 'const_alg' will be a list.
            this_run_data = exp_params.copy()
            for k, v in run.items():
                # Special case: we recorded multiple metrics for this, but we'll just use mean.
                if k == 'cost':
                    this_run_data[k] = v['mean']
                elif k in skip_keys:
                    continue
                elif k == exp_params['const_alg']:
                    for select_policy, reach in v.items():
                        # not keeping ALL selection policies that are actually special recorded values
                        if select_policy not in skip_keys:
                            this_run_data.setdefault('select_policy', []).append(select_policy)
                            this_run_data.setdefault('reach', []).append(reach)
                else:
                    this_run_data[k] = v

            this_df = pd.DataFrame(this_run_data)
            # XXX: unicast/oracle are a weird special case where we treat it as a separate row where ntrees=0,
            # which we need to add as a new row AFTER building the DataFrame since just appending it to a row earlier
            # would have made for different-length indices given to the df constructor!
            special_row = this_run_data.copy()
            # TODO: this?
            # special_row['ntrees'] = 0
            for special_treatment in ('unicast', 'oracle'):
                special_row['select_policy'] = special_treatment
                special_row['const_alg'] = special_treatment
                special_row['reach'] = run[special_treatment]
                special_row['cost'] = run['cost']['unicast'] if special_treatment == 'unicast' else 0
                this_df = this_df.append(special_row, ignore_index=True)

            data_frames.append(this_df)

        self.stats = pd.concat(data_frames, ignore_index=True, copy=False)
        self.stats.reset_index()

    def reachabilities(self, **param_matches):
        """
        Returns the reachability values for the parsed stats filtered over the key/values specified in kwargs.
        This is basically just self.stats since no further computation is necessary.
        :param param_matches: matches the rows for which the column given by the key has the associated value
        :return:
        :rtype pd.DataFrame:
        """

        # can't run empty query
        if not param_matches:
            return self.stats
        query_string = ' & '.join(('%s == "%s"' % (k, v) for k, v in param_matches.items()))
        log.debug("running query: %s" % query_string)
        return self.stats.query(query_string)

    VARIED_PARAMS = ('const_alg', 'select_policy', 'reroute_policy', 'exp_type', 'error_rate', 'fprob', 'ntrees',
                     'npublishers', 'nsubscribers', 'topo', 'treatment')

    @classmethod
    def average_over_runs(cls, df):
        """
        Averages the given DataFrame's values over all the runs for each unique treatment grouping.
        :type df: pd.DataFrame
        """
        # XXX: need to ensure we have all these parameters available
        cols = set(df.columns.tolist())
        group_params = list(set(cls.VARIED_PARAMS).intersection(cols))
        return df.groupby(group_params).mean().reset_index().drop('run', axis=1)

    def __iadd__(self, other):
        self.stats = pd.concat((self.stats, other.stats), ignore_index=True)
        return self


#### EXAMPLE RESULTS (one index):

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
#     "steiner": { #                        <------  metric_name=steiner, yvalue=this dict
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

# TODO: could extract details about the topo file from its name?  mostly we've been doing campus_topo.json though...
# topo is a list containing topology reader and filename, so just extract filename
# and parse the parameters from it
#     param_value = data['params']['topo'][1].split('.')[0].split('_')[-1]
#     _parsed = re.match('(\d+)b-(\d+)h-(\d+)ibl', param_value).groups()
#     param_value = (int(_parsed[0]), int(_parsed[1]), int(_parsed[2]))


def run_tests():
    dummy_args = parse_args([])
    dummy_args.debug = 'debug'
    stats = SmartCampusExperimentStatistics(dummy_args)
    # create some dummy results with really simple values for testing
    nresults = 4
    test_heuristics = ["steiner", "red-blue", "fake_heuristic"]
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
            test_heuristics[i%3]: {  # note that since i starts at 1 this means red-blue repeats first
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

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s')  # if running stand-alone

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        run_tests()
        exit()

    # lets you print the data frames out on a wider screen
    pd.set_option('display.max_columns', 15)  # seismic_events has 14 columns
    pd.set_option('display.width', 2500)

    args = sys.argv[1:]
    args = parse_args(args)
    stats = SmartCampusExperimentStatistics(args)
    stats.parse_all()

    # Now we have to determine what kind of stats we want to collect and output... Can use filtering here
    # assert isinstance(stats.stats, MininetSeismicStatistics)
    # assert isinstance(stats.stats, NetworkxSeismicStatistics)
    df = stats.stats.reachabilities()
    print 'original stats (%d):\n' % len(df), df
    # print stats.stats.stats.info()

    original_stats = df

    # policies_to_keep = ('oracle', 'unicast', 'max', 'mean')
    # policies_to_keep = ('oracle', 'unicast')
    policies_to_keep = ('oracle', 'unicast', 'importance-chosen', 'max-overlap-chosen', 'max-reachable-chosen', 'min-missing-chosen')
    # const_algs_to_keep = ('diverse-paths', 'red-blue', 'steiner')
    # for pol in policies_to_keep:
    #     for alg in const_algs_to_keep:
    query = '|'.join(['select_policy == "%s" ' % pol for pol in policies_to_keep])
            # query = '|'.join(['const_alg == "%s" ' % alg for alg in const_algs_to_keep])


            # query = "const_alg == '%s' & select_policy == '%s'" % (alg, pol)
    #
    # print 'QUERY:', query
    # df = original_stats.query(query).query('|'.join(['const_alg == "%s" ' % pol for pol in ["red-blue", "unicast", "oracle"]]))
    df = original_stats.query('select_policy == "mean"')
    # print 'filtered stats (%d):\n' % len(df), df.head()

    # final_stats = stats.stats.average_over_runs(df)
    final_stats = df

    print 'final stats:\n', final_stats

    if args.output_file:
        fname = args.output_file
        # fname = 'netx_results/ntrees.csv'
        # final_stats.to_csv(fname, index=False)
        original_stats.to_csv(fname, index=False)
