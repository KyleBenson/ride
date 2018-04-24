#!/usr/bin/env python

from network_experiment_statistics import NetworkExperimentStatistics
from scale_client.stats.parsed_sensed_events import ParsedSensedEvents
import pandas as pd
from scifire.utilities import calculate_utility

import logging
log = logging.getLogger(__name__)


class FireStatistics(NetworkExperimentStatistics):
    """Output parser and statistics analyzer for the SciFire structure fire scenario."""

    @property
    def varied_params(self):
        vp = super(FireStatistics, self).varied_params

        more_vp = {
            ### keep these after we average across runs
            "algorithm",
            "treatment", # in case we group param treatments and assign a specific filename, it's easier to groupby this
            "ro_tol",
            ## these are data points!  explicitly added from e.g. params
            "topic",
            "prio",
            "prio_prob"
            ## these should stay as ints, not get averaged into floats!
            "nprios", "nffs", "niots", "nflows", "ntopics",
        }
        vp.update(more_vp)
        return vp

    # XXX: instead of two different classes, just explicitly checking exp_type
    def is_mininet(self, **params):
        return params.get('exp_type', 'sim') == 'mininet'

    def choose_parser(self, filename, **params):

        if filename.endswith('.csv'):
            def CsvParser(results_str, **params):
                df = pd.read_csv(filename, names=('delay', 'rcv_rate', 'sim_exp_delay'))

                # to track subscriptions to topics, create a bit vector and add that as a column
                # similar for utilities, expected values, etc.
                # need to create a vector for all topics from a subscriptions-only vector
                # ENHANCE: just call to FdxConfig.topics_to_subscriptions??? really basic logic though...
                # ENHANCE: generalize all this in case we add some more fields later?

                vec_len = params['ntopics']
                subs = params.pop('subscriptions')
                subs_vec = [0] * vec_len
                # for utilities, we will calculate the actual and max possible utility given the weights, lambdas, delays, etc.
                uws = params.pop('uws')
                uws_vec = [0] * vec_len
                utils_vec = [0] * vec_len
                max_utils_vec = [0] * vec_len
                exp_utils = params.pop('exp_utils')
                exp_utils_vec = [0] * vec_len
                exp_rcv = params.pop('exp_rcv')
                exp_rcv_vec = [0] * vec_len
                exp_delay = params.pop('exp_delay')
                exp_delay_vec = [0] * vec_len

                for sub, uw, exp_util, rcv, delay in zip(subs, uws, exp_utils, exp_rcv, exp_delay):
                    subs_vec[sub] = 1
                    uws_vec[sub] = uw
                    # max rcv rate is the rate events are published:
                    max_rcv = params['lams'][sub]
                    utils_vec[sub] = calculate_utility(rcv, max_rcv, delay, uw)
                    max_utils_vec[sub] = calculate_utility(max_rcv, max_rcv, delay, uw)
                    exp_utils_vec[sub] = exp_util
                    exp_rcv_vec[sub] = rcv
                    exp_delay_vec[sub] = delay

                df['subd'] = subs_vec
                df['utils'] = utils_vec
                df['max_utils'] = max_utils_vec
                df['exp_utils'] = exp_utils_vec
                df['exp_rcv'] = exp_rcv_vec
                df['exp_delay'] = exp_delay_vec

                # store all the parameters as columns
                for k, v in params.items():
                    try:
                        df[k] = v
                    # Skip anything that's not a constant or the same length as #topics
                    except ValueError as e:
                        pass
                    # XXX: basic attempt to bring in columns that include things like topic classes, distributions, etc.
                        try:
                            df[k] = str(v)
                        except BaseException as e2:
                            log.warning('failed to add col %s: %s\nError1: %s\nError2: %s' % (k, v, e, e2))

                # Need to scale rcv_rate by the lambdas since the value in the .csv is just a proportion in [0,1]
                df['rcv_lams'] = df.rcv_rate * df.lams

                # XXX: need to set the index to be topic ID so each new parser we bring in will match up the topics
                df = df.reset_index().rename(columns=dict(index='topic'))

                # now to actually calculate the utilities from the util weights we got above:
                df['utils'] = calculate_utility(df.rcv_lams, df.lams, df.delay, df.utils)

                return df

            return CsvParser

        else:
            raise ValueError("unrecognized output file type (expected .csv) for filename: %s" % filename)

    def extract_run_params(self, run_results, filename, **exp_params):
        exp_params = super(FireStatistics, self).extract_run_params(run_results, filename, **exp_params)

        if run_results['return_code'] != 0:
            raise ValueError("skipping run %d from results file %s with non-0 return code..." % (exp_params['run'], filename))

        exp_params['prio'] = run_results['sim_config']['priorities']
        exp_params['subscriptions'] = run_results['sim_config']['subscriptions']

        # get actual per-run lambda values (based on # publishers on each topic) rather than per-topic ones
        exp_params['lams'] = run_results['sim_config']['lambdas']
        exp_params['mus'] = run_results['sim_config']['mus']

        # extract drop_rate policy configuration: need to assign them to each row based on that topic's priority
        prio_probs = run_results['sim_config']['prio_probs']
        prios = exp_params['prio']
        exp_params['prio_prob'] = [prio_probs[p] for p in prios]

        # incorporate our analytical model
        exp_params['exp_delay'] = run_results['exp_srv_delay']
        exp_params['exp_rcv'] = run_results['exp_delivery']

        # for calculating/plotting utility, record utility functions (also assigned per-row based on topic for subscriptions)
        exp_params['uws'] = run_results['utility_weights']
        exp_params['exp_utils'] = run_results['exp_utilities']

        return exp_params

    def collate_outputs_results(self, *results):
        """
        Combine the parsed results from an outputs_dir and calculate statistics.
        :param results:
        :type results: list[ParsedSensedEvents]
        :return:
        """

        results = super(FireStatistics, self).collate_outputs_results(*results)

        # TODO: used for mininet version when we start it up again
        # XXX: for now, let's just keep the broker results so we can extract latencies
        # results = [r for r in results if 'time_rcvd' in r]
        # results = super(FireStatistics, self).collate_outputs_results(*results)
        #
        # # rename some columns
        # results.rename_columns(value='seq')
        #
        # self.calc_latencies(results)
        return results

    @property
    def param_col_map(self):
        m = dict(num_fire_fighters='nffs', num_iot_devices='niots', num_priority_levels='nprios',
                 num_net_flows='nflows', num_topics='ntops', topic_class_weights='tc_weights',
                 topic_class_data_sizes='tc_sizes', topic_class_pub_rates='tc_pub_rates', topic_class_pub_dists='tc_pub_dists',
                 topic_class_advertisements_per_ff='tc_ff_ads', topic_class_advertisements_per_iot='tc_iot_ads',
                 topic_class_sub_dists='tc_sub_dists', topic_class_sub_rates='tc_sub_rates',
                 ic_sub_rate_factor='ic_sub_rate', topic_class_sub_start_times='tc_sub_start',
                 topic_class_sub_durations='tc_sub_time', reliable_publication='tc_pub_retx'
                 )
        m.update(super(FireStatistics, self).param_col_map)
        return m

    def extract_parameters(self, exp_params):
        """
        Splits each of the topic class parameters into nclasses different params (i.e. columns).
        :param exp_params:
        :type exp_params: dict
        :return: dict of extracted params
        """

        exp_params = super(FireStatistics, self).extract_parameters(exp_params)

        for k,v in exp_params.items():
            # TODO: splits into nclasses different columns as opposed to just dropping
            # ENHANCE: could also convert them to strings, find differences, and just keep those so as to get rid of e.g. {'dist': ...
            if k.startswith('tc_'):
                del exp_params[k]
            # TODO: probably bring at least ads back in? if we vary it...
            elif 'subs' in k or 'ads' in k:
                del exp_params[k]

        return exp_params

    # We should also average over topics
    def plot(self, x, y, groupby=None, average_over=('run', 'topic'), stats=None, **kwargs):
        super(FireStatistics, self).plot(x, y, groupby=groupby, average_over=average_over, stats=stats, **kwargs)


if __name__ == '__main__':
    stats = FireStatistics.main()

    # now you can do something with the stats to e.g. get your own custom experiment results
    final_stats = stats.stats

    # drop topics with no subscription OR advertisement (i.e. pub rate is 0)
    final_stats = final_stats[(final_stats.subd != 0) & (final_stats.lams != 0)]

    # doing this for now to view columns easier...
    ignored_cols = ('nffs', 'niots',
                    'bw', 'jitter',
                    # 'treatment',
                    )
    for col in ignored_cols:
        del final_stats[col]

    # print "STATS:\n", final_stats

    # TODO: calculate_utilities()

    ####   SIMULATION RESULTS

    ## Show that bandwidth affects delay
    # stats.plot(x='bw', y='delay', groupby='prio', stats=final_stats)

    ## Show that lower priority topics have increased delay
    # stats.plot(x='prio', y='delay', groupby='nprios')

    ## Show that error rate results in lower delivery rate
    # stats.plot(x='loss', y=['rcv_lams', 'rcv_rate'], stats=final_stats, average_over=('run', 'topic', 'prio'))
    # this one shows the curve perfectly as expected:
    # stats.plot(x='loss', y='rcv_rate', stats=final_stats, average_over=('run', 'topic', 'prio'))
    ## this one sometimes looks wonky due to randomness of lambdas
    # stats.plot(x='loss', y='rcv_lams', stats=final_stats, average_over=('run', 'topic', 'prio'))


    ####    ANALYTICAL MODEL  vs.   SIM RESULTS

    final_stats = final_stats[final_stats.ro_tol == 0.1]
    ## error rate affects delivery
    # stats.plot(x='loss', y=['lams', 'rcv_lams', 'exp_rcv'], stats=final_stats, average_over=('run', 'topic', 'prio'))

    ## Show that bandwidth affects delay
    # stats.plot(x='bw', y=['delay', 'exp_delay'], average_over=('run', 'topic', 'prio'), stats=final_stats)

    ## Show that lower priority topics have increased delay
    # stats.plot(x='prio', y=['delay', 'exp_delay'], groupby='treatment', stats=final_stats)
    # stats.plot(x='prio', y='delay', groupby=['nprios', 'algorithm'], stats=final_stats)

    # Plot actual delay difference
    # final_stats = final_stats[final_stats.run == 75]
    # final_stats['delay_diff'] = final_stats.delay - final_stats.exp_delay
    # stats.plot(x='prio', y='delay_diff', groupby='treatment', stats=final_stats)
    # stats.plot(x='topic', y='delay_diff', groupby='treatment', average_over=('run', 'prio'), stats=final_stats)
    # stats.plot(x='topic', y='delay_diff', groupby='nprios', average_over=('run', 'prio'), stats=final_stats)
    # stats.plot(x='prio', y='delay_diff', groupby='nprios', stats=final_stats)
    # stats.plot(x='topic', y=['delay', 'exp_delay'], average_over=('run', 'prio'), groupby='nprios', stats=final_stats)
    # stats.plot(x='topic', y=['delay', 'sim_exp_delay'], stats=final_stats)

    final_stats = final_stats.sort_values('topic')
    print final_stats
    final_stats.plot(x='topic', y=['delay', 'sim_exp_delay'])
    import matplotlib.pyplot as plt
    plt.show()
    # stats.plot(x='topic', y='delay', stats=final_stats)
    # stats.plot(x='prio', y=['lams', 'delay_diff'], groupby='nprios', stats=final_stats)

    #### Plotting utilities
    ##     Show that we achieve more of max possible utility for higher priority subscription groups:
    # May be more helpful to plot utility as a percent achieved of max possible
    # TODO: may be even better to just sum up utilities instead of averaging! still want to scale by max though...
    # final_stats['util_perc'] = final_stats.utils / final_stats.max_utils
    # stats.plot(x='prio', y='util_perc', groupby=['nprios', 'algorithm'], stats=final_stats)
    # stats.plot(x='prio', y='utils', groupby=['nprios', 'algorithm'], stats=final_stats)

    ###   Explicitly save results to file
    # stats.config.output_file = "out.csv"

    if stats.config.output_file:
        stats.output_stats(stats=final_stats)
