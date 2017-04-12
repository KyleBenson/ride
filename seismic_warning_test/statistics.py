#!/usr/bin/python

STATISTICS_DESCRIPTION = '''Gathers statistics from the output files in order to determine how long it
    took the events to reach the interested clients. Also handles printing them in
     an easily-read format as well as creating plots.'''

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

    parser.add_argument('--dirs', '-d', type=str, nargs="+", default=['output'],
                        help='''directories containing files from which to read outputs
                        (default=%(default)s)''')

    return parser.parse_args(args)


class SeismicStatistics(object):
    """Gathers statistics from the output files in order to determine how long it
    took the events to reach the interested clients. Also handles printing them in
    an easily-read format as well as creating plots.

    General strategy:
    -- Each directory represents a single experiment
    -- Each file in that directory represents the readings received by a single subscriber
    -- Each reading tells us when it was first sent and when it was received
    -- So each time step has a number of readings received by this client
    -- Hence we should average over this # readings across all clients
    -- We should then plot the CDF of these averages to see how well the experiment performed
    """

    def __init__(self, config):
        super(self.__class__, self).__init__()
        self.config = config

        # store all the parsed stats indexed by directory name, then by filename
        self.stats = dict()

    def parse_all(self):
        for dirname in self.config.dirs:
            self.parse_dir(dirname)

    def parse_dir(self, dirname):
        self.stats[dirname] = dict()
        for filename in os.listdir(dirname):
            parsed = self.parse_file(os.path.join(dirname, filename))
            self.stats[dirname][filename] = parsed

    def parse_file(self, fname):
        with open(fname) as f:
            return json.loads(f.read())

    def get_latencies(self, group):
        """Averages the latencies over this group of results (directory).
        @:param group - dictionary of {filename: parsed_results} pairs
        @:return latencies, num_sensors
        """

        # track which hosts anyone received data from to establish an upper bound
        hosts_rcvd = set()
        # QUESTION: should we get the min of all time_sents in order to avoid
        # inaccuracy due to clients starting at different times?  Probably not
        # since there isn't really anything we can do at this point to fix it...

        latencies = []
        for client in group.values():
            for (sensor, results) in client.items():
                hosts_rcvd.add(sensor)
                latencies.append(results['time_rcvd'] - results['time_sent'])

        return np.array(latencies), len(hosts_rcvd)

    def plot_cdf(self, num_bins=10):
        '''Plots the CDF of the number of seismic events
        received at interested clients. Averaged over all subscribers.'''

        # TODO: is this assuming that all sensors are also subscribers????

        # Can't just normalize because some experiments may have overall
        # better performance than others (or even different # sensors).
        # Instead, we estimate the total # sensors by counting all unique occurrences,
        # and then each actual subscriber contributes its appropriate weight of counts
        # to each bin.  This results in the total counts being the number of subscribers.
        # So then we could further reduce the contributed weights by the # subs in order
        # to get the total being 1.0 iff all subs eventually received all events.

        # TODO: use these markers
        # markers = 'x.*+do^s1_|'
        # plot(..., marker=markers[i%len(markers)])

        for (group_name, group) in self.stats.items():
            latencies, nsensors = self.get_latencies(group)
            # adjust the weight as per above paragraph
            weight_adjustment = [1.0/nsensors/len(group)] * len(latencies)
            counts, bin_edges = np.histogram(latencies, bins=num_bins, weights=weight_adjustment)
            cdf = np.cumsum(counts)
            plt.plot(bin_edges[1:], cdf, label=group_name)

        plt.xlabel("time(secs)")
        # TODO: put x scale as log? maybe y too?
        plt.ylabel("avg % readings rcvd")
        plt.title('Sensor readings received over time')
        plt.show()


    def plot_time(self):
        '''Plots the events' latencies over time.  Useful for
        determining if event processing slows down over time or spikes
        at a certain point.'''

        # Rather than raw datetimes, we want to use the total seconds since the
        # simulation's start
        timestamps = [e.timestamp for e in self.events]
        tmin = min(timestamps)
        timestamps = [(t-tmin).total_seconds() for t in timestamps]

        plt.plot(timestamps, self.get_latencies())
        plt.xlabel("time(s)")
        plt.ylabel("latency(ms)")
        plt.title('Event latency over time')
        plt.show()

    def print_statistics(self):
        '''Prints summary statistics for all groups, in particular
        the mean latency and standard deviation.'''

        for group_name, group in self.stats.items():
            latencies, _ = self.get_latencies(group)
            print "Group %s's Mean: %fs; stdev: %fs" % (group_name, np.mean(latencies), np.std(latencies))

    # could use this if we ever put the events into some kind of object
    # def __repr__(self):
    #     s = "SensedEvent (%s) with value %s" % (self.get_type(), str(self.get_raw_data()))
    #     s += pprint.pformat(self.data, width=1)
    #     return s

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    stats = SeismicStatistics(args)
    stats.parse_all()
    stats.print_statistics()
    stats.plot_cdf()