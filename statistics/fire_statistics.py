#!/usr/bin/env python

from network_experiment_statistics import NetworkExperimentStatistics
from scale_client.stats.parsed_sensed_events import ParsedSensedEvents
import pandas as pd
from scifire.utilities import calculate_utility
from parse import parse

import os
import logging
log = logging.getLogger(__name__)


class FiredexParsedSensedEvents(ParsedSensedEvents):
    """
    Custom version of SensedEvent parser that first parses all the events, arranges them according to subscriptions,
    and then joins together the events output by publishers/brokers/subscribers as end-to-end events (i.e. from the
    publisher to broker to subscriber, with the latter two having null values if the event didn't reach them).
    """

    def __init__(self, data, filename=None, **kwargs):
        """
        Parses the given data into a dict of events recorded by the client
         and passes the resulting data into a pandas.DataFrame with additional columns specified by kwargs,
        which can include e.g. experimental treatments, host IP address, etc.

        :param data: raw string containing JSON object-like data e.g. nested dicts/lists
        :type data: str
        :param filename: name of the file from which this data is being parsed (not used by default)
        :param timezone: the timezone to use for converting time columns (default='America/Los_Angeles'); set to None to disable conversion
        :param kwargs: additional static values for columns to distinguish this group of events from others
            e.g. host_id, host_ip
        """

        # XXX: extract subscriber's hostname from the filename
        filename = os.path.split(filename)[-1]
        sub_name = filename.split('_')[-1].split('-')[-1]
        kwargs['sub'] = sub_name

        super(FiredexParsedSensedEvents, self).__init__(data, filename=filename, **kwargs)

    def combine_params_columns(self, columns, **params):
        """
        To add the relevant params to columns, we need to convert each per-topic or per-subscription param into a
        per-event format.
        :param columns:
        :type columns: dict
        :param params:
        :return:
        :rtype: dict
        """

        # TODO: this will have to be calculated based on rates once we convert events to rates!
        # columns['utils'] = []
        # max_rcv = params['lams'][sub]
        # columns['max_utils'] = []

        # per-topic ones:
        lambdas = params.pop('lams')
        mus = params.pop('mus')
        lambdas_col = []
        mus_col = []
        for top in columns['topic']:
            # XXX: the lambdas are just a list where each index corresponds to that topic i.e. idx 0 == topic '0'
            top = int(top)
            lambdas_col.append(lambdas[top])
            mus_col.append(mus[top])

        columns['lams'] = lambdas_col
        columns['mus'] = mus_col

        # Now, per-subscription columns:
        columns['exp_rcv'] = []
        columns['exp_delay'] = []
        columns['prio'] = []
        columns['drop'] = []
        columns['uws'] = []
        columns['exp_utils'] = []

        exp_rcv = params.pop('exp_rcv')
        exp_delay = params.pop('exp_delay')
        priorities = params.pop('prio')
        drop_rates = params.pop('drop')
        util_weights = params.pop('uws')
        exp_utils = params.pop('exp_utils')

        # Now we go through each row of the events and extract the relevant additional columns wanted
        subscriber = params['sub']
        for topic in columns['topic']:
            columns['exp_rcv'].append(exp_rcv[subscriber][topic])
            columns['exp_delay'].append(exp_delay[subscriber][topic])
            columns['prio'].append(priorities[subscriber][topic])
            columns['drop'].append(drop_rates[subscriber][topic])
            columns['uws'].append(util_weights[subscriber][topic])
            columns['exp_utils'].append(exp_utils[subscriber][topic])

        return super(FiredexParsedSensedEvents, self).combine_params_columns(columns, **params)

    def extract_columns(self, data, parse_metadata=True):
        """Modify some of the SensedEvent columns here before creating the DataFrame"""

        res = super(FiredexParsedSensedEvents, self).extract_columns(data, parse_metadata=parse_metadata)
        res.pop('local_resource_uri')
        res.pop('relay_uri')  # only using a single broker currently!

        # XXX: extract publisher's ID from source URI that looks like: "coap://10.128.20.1:7777/scale/sensors/IoTSensor_iot1_17"
        sources = res.pop('source')
        pub_ids = []
        for src in sources:
            pub_ids.append(src.split('_')[1])
        res['pub'] = pub_ids

        # MAYBE: convert topic to int?

        # XXX: value is a sequence number as a long string to make up the right payload length
        values = res.pop('value')
        seqs = []
        for v in values:
            seqs.append(int(v))
        res['seq'] = seqs

        return res


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
            "pub",
            "sub",
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

                # XXX: since these are all in dict format, need to index by this output file's subscriber first:
                subscriber = parse("sim_output_{:w}.csv", os.path.split(filename)[-1])[0]

                params = self.extract_run_results_analysis(subscriber, **params)

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

        elif self.is_mininet(**params):

            # for now we only extract stats from subscriber(s)

            # TODO: add publishers (and maybe brokers?) later on?
            # NOTE: will need to skip publications without matching subscriptions...
            # NOTE: publications that weren't received MAY need to be counted for each relevant subscription!

            filename = os.path.split(filename)[-1]

            if filename.startswith('subscriber'):
                return FiredexParsedSensedEvents
            else:
                log.debug("skipping outputs file: %s" % filename)
                return None

            # OPTIONS:
            # - publisher: need to know source (from filename?), topic, seq#: match up with subs if possible
            # - subscriber: extract time_rcvd and source (original, but maybe also broker eventually?)
            # - events_broker: extract metadata['time_rcvd'] and seq#

        else:
            raise ValueError("unrecognized output file type (expected .csv) for filename: %s" % filename)

    def extract_run_params(self, run_results, filename, **exp_params):
        exp_params = super(FireStatistics, self).extract_run_params(run_results, filename, **exp_params)
        ret = exp_params

        # first get those common across all versions:

        # incorporate our analytical model
        # TODO: also get the total_delay? for now we don't include prop. delay (latency) since sim doesn't....
        ret['exp_delay'] = run_results['exp_delay']
        ret['exp_rcv'] = run_results['exp_delivery']

        # for calculating/plotting utility, record utility functions (also assigned per-row based on topic for subscriptions)
        ret['uws'] = run_results['utility_weights']
        ret['exp_utils'] = run_results['exp_utilities']

        # Then, get some params specific to the different sim versions
        if self.is_mininet(**exp_params):
            # TODO: anything else here?
            # def _extract_mininet_run_params(self, run_results, filename, **exp_params):
            ret['prio'] = run_results.pop('priorities')
            ret['drop'] = run_results.pop('drop_rates')

            # TODO: use this to filter events so we have a warm-up period!
            ret['time_sim_start'] = run_results['start_time']

        elif exp_params.get('exp_type', 'sim') == 'sim':
            # XXX: since each subscriber will have a different simulation run, let's just save the overall results and
            # then later when we inspect that run's output file we can actually gather the specific params for that subscriber.
            ret['sim_results'] = run_results['sim_results']

        # Since analysis-only version has no output files, we need to extract all the relevant information NOW:
        elif exp_params['exp_type'] == 'analysis':
            ret['sim_results'] = run_results['sim_results']
            dfs = []
            for subscriber in ret['sim_results'].keys():
                # XXX: is expecting a filename arg...
                run_stats = self.extract_run_results_analysis(subscriber, filename=None, **ret)
                df = pd.DataFrame(run_stats)

                # XXX: need to set the index to be topic ID so each new parser we bring in will match up the topics
                df = df.reset_index().rename(columns=dict(index='topic'))

                # seems as though the utils are already there? this is just analytical model anyway so exp_utils is fine...
                # now to actually calculate the utilities from the util weights we got above:
                # df['utils'] = calculate_utility(df.rcv_lams, df.lams, df.delay, df.utils)
                dfs.append(df)

            # XXX: since we have no actual output file to parse, we have to manually DIRECTLY store the stats...
            if self.stats is None:
                self.stats = self.merge_all(*dfs)
            else:
                self.stats = self.merge_all(self.stats, *dfs)

        return ret

    def _extract_queue_sim_run_params(self, run_results, filename, **exp_params):

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

        return exp_params

    def extract_run_results_analysis(self, subscriber, **params):
        """
        Extracts the configuration info (e.g. subscriptions) and analytical model results for a particular run.
        :param subscriber:
        :param params:
        :return:
        """

        # to track subscriptions to topics, create a bit vector and add that as a column
        # similar for utilities, expected values, etc.
        # need to create a vector for all topics from a subscriptions-only vector
        # ENHANCE: just call to FdxConfig.topics_to_subscriptions??? really basic logic though...
        # ENHANCE: generalize all this in case we add some more fields later?
        vec_len = params['ntopics']
        subs_vec = [0] * vec_len
        # for utilities, we will calculate the actual and max possible utility given the weights, lambdas, delays, etc.
        sim_results = params.pop('sim_results')[subscriber]
        # NOTE: this will overwrite the 'mus' with the per-subscriber (i.e. bandwidth-sliced) ones!
        #   Unclear that's actually what we want...
        params = self._extract_queue_sim_run_params(sim_results, **params)  # params includes 'filename'!
        subs = params.pop('subscriptions')
        uws = params.pop('uws')[subscriber]
        uws_vec = [0] * vec_len
        utils_vec = [0] * vec_len
        max_utils_vec = [0] * vec_len
        exp_utils = params.pop('exp_utils')[subscriber]
        exp_utils_vec = [0] * vec_len
        exp_rcv = params.pop('exp_rcv')[subscriber]
        exp_rcv_vec = [0] * vec_len
        exp_delay = params.pop('exp_delay')[subscriber]
        exp_delay_vec = [0] * vec_len
        for sub in subs:
            # XXX: since topics are strings in the dicts but ints in the 'subscriptions' brought in for queue sim,
            #   we need to convert for here and then get the fields we want:
            sub = str(sub)

            uw = uws[sub]
            exp_util = exp_utils[sub]
            rcv = exp_rcv[sub]
            delay = exp_delay[sub]

            # XXX: back to int for list indexing
            sub = int(sub)

            subs_vec[sub] = 1
            uws_vec[sub] = uw
            # max rcv rate is the rate events are published:
            max_rcv = params['lams'][sub]
            utils_vec[sub] = calculate_utility(rcv, max_rcv, delay, uw)
            max_utils_vec[sub] = calculate_utility(max_rcv, max_rcv, delay, uw)
            exp_utils_vec[sub] = exp_util
            exp_rcv_vec[sub] = rcv
            exp_delay_vec[sub] = delay

        params['subd'] = subs_vec
        params['sub'] = subscriber
        params['utils'] = utils_vec
        params['max_utils'] = max_utils_vec
        params['exp_utils'] = exp_utils_vec
        params['exp_rcv'] = exp_rcv_vec
        params['exp_delay'] = exp_delay_vec

        return params

    def collate_outputs_results(self, *results):
        """
        Combine the parsed results from an outputs_dir and calculate statistics.
        :param results:
        :type results: list[ParsedSensedEvents]
        :return:
        """

        results = super(FireStatistics, self).collate_outputs_results(*results)

        # TODO: used for mininet version... maybe bring back in later?
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
            elif k in ('exp_duration',):
                del exp_params[k]

        return exp_params

    # We should also average over topics
    def plot(self, x, y, groupby=None, average_over=('run', 'topic'), stats=None, **kwargs):
        return super(FireStatistics, self).plot(x, y, groupby=groupby, average_over=average_over, stats=stats, **kwargs)


if __name__ == '__main__':
    stats = FireStatistics.main()

    # now you can do something with the stats to e.g. get your own custom experiment results
    final_stats = stats.stats

    # drop topics with no subscription OR advertisement (i.e. pub rate is 0)
    final_stats = final_stats[(final_stats.subd != 0) & (final_stats.lams != 0)]

    # doing this for now to view columns easier...
    ignored_cols = ('niots',
                    'bw', 'jitter',
                    # 'treatment',
                    )
    for col in ignored_cols:
        if col in final_stats:
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

    # final_stats = final_stats[final_stats.ro_tol == 0.1]
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

    ## example for plotting across specific subscribers
    # stats.plot(x='sub', y=['delay', 'sim_exp_delay'], average_over=('run', 'topic'), stats=final_stats)
    # stats.plot(x='prio', y=['delay', 'sim_exp_delay'], average_over=('sub', 'run', 'topic',), stats=final_stats)

    # final_stats = final_stats.sort_values('topic')
    # print final_stats
    # final_stats.plot(x='topic', y=['delay', 'sim_exp_delay'])
    # import matplotlib.pyplot as plt
    # plt.show()
    # stats.plot(x='topic', y='delay', stats=final_stats)
    # stats.plot(x='prio', y=['lams', 'delay_diff'], groupby='nprios', stats=final_stats)

    #### Plotting utilities
    ##     Show that we achieve more of max possible utility for higher priority subscription groups:

    # To properly organize the treatment column, it may be helpful to apply a function over it that changes the string.
    # In this case, we want to get rid of everything except the very last bit that captures the utility weight dist used:
    final_stats['treatment'] = final_stats['treatment'].apply(lambda t: int(t.split('_')[-1][1:]))

    # need to sum utility, not average over it! also more helpful to plot utility as a percent achieved of max possible
    final_stats = final_stats.groupby(['treatment', 'algorithm', 'run']).sum().reset_index()
    final_stats['util_perc'] = final_stats.utils / final_stats.max_utils

    # XXX: somehow one of the configs resulted in super high delay despite ro < .95.... investigate further later.
    # ff3 = final_stats.query('sub == "ff3" & treatment == 8 & run == 4')
    # final_stats.drop(ff3.index, inplace=True)

    # XXX: then average over runs (not sure why but this wasn't working via plot call...
    final_stats = final_stats.groupby(['treatment', 'algorithm']).mean().reset_index()

    # stats.plot(x='treatment', y='utils', groupby='algorithm', average_over=(), stats=final_stats)
    # stats.plot(x='treatment', y='exp_delay', groupby='algorithm', average_over=(), stats=final_stats)
    stats.plot(x='treatment', y='util_perc', groupby='algorithm', average_over=(), stats=final_stats)
    # stats.plot(x='treatment', y='util_perc', groupby='algorithm', average_over=('topic', 'run', 'prio', 'sub'), stats=final_stats)
    # stats.plot(x='prio', y='utils', groupby=['nprios', 'algorithm'], stats=final_stats)


    ################################################################################################################
    ############################        FINAL PLOTTING FOR PAPER               #####################################
    ################################################################################################################

    ###### EXAMPLE CONFIGURATIONS for plotting function:
    # style: list of line styles for columns
    # logx/logy/loglog
    # xticks/yticks: values to use for ticks
    # fontsize: for ticks
    # yerr/xerr: error bars (see https://pandas.pydata.org/pandas-docs/stable/visualization.html#visualization-errorbars)
    # colormap: name of colormap to load from matplotlib
    # secondary_y: boolean or list of columns to plot on secondary y axis
    # kwargs: additional arguments (see matplotlib docs)
    #
    ### matplotlib options to try: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html
    # marker/markersize: can't seem to select one for each column... probably need to plot one column at a time for it?

    ### validation
    # stats.plot(x='sub', y=['delay', 'sim_exp_delay'], average_over=('run', 'topic'), stats=final_stats)
    # stats.plot(x='prio', y=['delay', 'sim_exp_delay'], average_over=('sub', 'run', 'topic',), stats=final_stats)

    ## XXX: attempt at using matplotlib that we probably won't use....
    # ax = stats.plot(x='prio', y='sim_exp_delay', marker='o', average_over=('sub', 'run', 'topic',), stats=final_stats, show_plot=False)
    # stats.plot(ax=ax, x='prio', y='delay', marker='x',
    #            legend=3, # move around legend by specifying an integer for the location
    #            average_over=('sub', 'run', 'topic',), stats=final_stats)

    ### firedex approach evaluation
    # not done...

    ### algorithms comparison
    # not done...

    ################################################################################################################
    ############################        COLUMN EXTRACTION FOR MATLAB PLOTTING               ########################
    ################################################################################################################

    # first, need to average over the columns we want to drop out
    # for col in ('run', 'topic',
    #             'sub',
    #             ):
    #     final_stats = stats.average_over_runs(final_stats, column_to_drop=col)

    # then, print out a list of each column we want as shown here:
    # print "PRIOS:", list(final_stats.prio)
    # print "DELAYS:", list(final_stats.delay)
    # print "RCVS:", list(final_stats.rcv_rate)
    # TODO: figure this out??? doesn't work probably cuz subs are strings...
    # print "SUBS:", list(final_stats.sub)

    ###   Explicitly save results to file
    # stats.config.output_file = "out.csv"

    if stats.config.output_file:
        stats.output_stats(stats=final_stats)
