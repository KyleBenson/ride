#!/usr/bin/env python

from network_experiment_statistics import NetworkExperimentStatistics
from scale_client.stats.parsed_sensed_events import ParsedSensedEvents


class FireStatistics(NetworkExperimentStatistics):
    """Output parser and statistics analyzer for the SciFire structure fire scenario."""

    # TODO: varied_params, choose_parser?

    def collate_outputs_results(self, *results):
        """
        Combine the parsed results from an outputs_dir and calculate statistics.
        :param results:
        :type results: list[ParsedSensedEvents]
        :return:
        """

        # XXX: for now, let's just keep the broker results so we can extract latencies
        results = [r for r in results if 'time_rcvd' in r]
        results = super(FireStatistics, self).collate_outputs_results(*results)

        # rename some columns
        results.rename_columns(value='seq')

        self.calc_latencies(results)
        return results

    def extract_parameters(self, exp_params):
        exp_params = super(FireStatistics, self).extract_parameters(exp_params)

        # Shorten some of our scenario-specific params
        exp_params['nffs'] = exp_params.pop("num_fire_fighters")
        exp_params['niots'] = exp_params.pop("num_iot_devices")

        return exp_params

if __name__ == '__main__':
    stats = FireStatistics.main()

    # now you can do something with the stats to e.g. get your own custom experiment results
    final_stats = stats.stats

    if stats.config.output_file:
        stats.output_stats(stats=final_stats)
