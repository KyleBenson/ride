CLASS_DESCRIPTION = """
A generic network experiment interface that configures the network, sets up the experiment, runs it, tears it down, and
records the results for a number of runs.  It's intended to provide a unified interface for some other class that
manages running it or at least a set of generic parameters to be configured via command line and methods to be
overriden by derived classes.
"""

import argparse
import json
import os
import random
import signal
import time
from abc import abstractmethod
from abc import ABCMeta
import logging
log = logging.getLogger(__name__)

from config import *


class NetworkChannelState(object):
    """
    Basic network channel characteristics used across various communications network classes.
    """
    __metaclass__ = ABCMeta

    def __init__(self, latency=DEFAULT_LATENCY, jitter=DEFAULT_JITTER,
                 error_rate=DEFAULT_ERROR_RATE, bandwidth=DEFAULT_BANDWIDTH, **kwargs):
        """
        :param latency: in ms
        :param jitter:
        :param error_rate: of packets, as a rate in range [0 .. 1.0]
        :type error_rate: float
        :param bandwidth: in Mbps
        :param kwargs:
        """

        self.error_rate = error_rate
        self.latency = latency
        self.jitter = jitter
        self.bandwidth = bandwidth

    def bandwidth_bytes(self, bw=None):
        """
        Convert bandwidth to bytes/second.
        :param bw: defaults to self.bandwidth
        :return:
        """

        if bw is None:
            bw = self.bandwidth
        return bw * 1000000 / 8.0  # originally Mbps

    def latency_seconds(self, lat=None):
        """
        Convert latency to seconds from the expected default units (e.g. milliseconds)
        :param lat: defaults to self.latency
        :return:
        """
        if lat is None:
            lat = self.latency
        return lat / 100.0  # originally in ms


class NetworkExperiment(NetworkChannelState):
    """Inherits from NetworkChannelState to use its channel characteristics as default values for the network."""

    def __init__(self, nruns=1, run_start_num=0, debug='info', output_filename='results.json',
                 choice_rand_seed=None, rand_seed=None, **kwargs):
        """
        :param nruns:
        :param run_start_num:
        :param debug:
        :param output_filename:
        :param choice_rand_seed:
        :param rand_seed:
        :param error_rate: default error rate (as a %) for links that don't specify one during construction
        :type error_rate: int
        :param latency: default latency (in ms) for links that don't specify one during construction
        :param jitter: default jitter (in ms) for links that don't specify one during construction
        :param bandwidth: default bandwidth (in Mbps) for links that don't specify one during construction
        :param kwargs:
        """

        try:
            super(NetworkExperiment, self).__init__(**kwargs)
        except TypeError:
            super(NetworkExperiment, self).__init__()

        log_level = logging.getLevelName(debug.upper())
        logging.basicConfig(format='%(levelname)s:%(module)s:%(message)s', level=log_level)
        self.debug_level = debug  # save this so we can pass it to host procs ran during experiments

        self.nruns = nruns
        self.current_run_number = run_start_num

        # These will all be filled in by calling setup_topology()
        # NOTE: make sure you reset these between runs so that you don't collect several runs worth of e.g. hosts!
        self.switches = []
        self.links = []
        self.hosts = []
        self.servers = list()

        # this is used for choosing hosts ONLY; try to ensure it gets called the same # times regardless of
        # what type of experiment you run for fair comparisons between e.g. simulated/emulated
        self.random = random.Random(choice_rand_seed)
        # this RNG is used for everything else (tie-breakers, algorithms, etc.)
        random.seed(rand_seed)

        # results are output as JSON to file after the experiment runs
        self.output_filename = output_filename
        if self.output_filename is None:
            log.warning("output_filename is None!  Using default of results.json")
            self.output_filename = "results.json"

        # If running separate processes, they might output some results or write their stdout to logs:
        self._outputs_dir = None
        self._logs_dir = None

        self.results = {'results': [], # each is a single run containing: {run: run#, <custom_data>: ...}
                        'params': {'error_rate': self.error_rate,
                                   'latency': self.latency,
                                   'jitter': self.jitter,
                                   'bandwidth': self.bandwidth,
                                   'choicerandseed': choice_rand_seed,
                                   'randseed': rand_seed,
                                   # NOTE: subclasses should store their type here!
                                   'experiment_type': None
                                   }
                        }

    @classmethod
    def get_arg_parser(cls, parents=(), add_help=False):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        arg_parser = argparse.ArgumentParser(description=CLASS_DESCRIPTION, parents=parents, add_help=add_help, conflict_handler='resolve')

        # experimental treatment parameters
        arg_parser.add_argument('--nruns', '-r', type=int, default=1,
                            help='''number of times to run experiment (default=%(default)s)''')
        arg_parser.add_argument('--run-start-num', type=int, default=0, dest='run_start_num',
                            help='''run number to start at.  Used for doing several runs with the same treatment
                            but in different processes rather than all in one (default=%(default)s)''')
        arg_parser.add_argument('--choice-rand-seed', type=int, default=None, dest='choice_rand_seed',
                            help='''random seed for choices of subscribers & servers (default=%(default)s)''')
        arg_parser.add_argument('--rand-seed', type=int, default=None, dest='rand_seed',
                            help='''random seed used by other classes via calls to random module (default=%(default)s)''')

        # network channel characteristics, which may or may not be used depending on implementation's level of realism
        arg_parser.add_argument('--error-rate', type=int, default=DEFAULT_ERROR_RATE, dest='error_rate',
                            help='''default error rate (expressed as an integer percentage) of links that don't specify one (default=%(default)s)''')
        arg_parser.add_argument('--latency', type=float, default=DEFAULT_LATENCY,
                            help='''default latency (in ms) of links that don't specify one (default=%(default)s)''')
        arg_parser.add_argument('--jitter', type=float, default=DEFAULT_JITTER,
                            help='''default jitter (in ms) of links that don't specify one (default=%(default)s)''')
        arg_parser.add_argument('--bandwidth', type=float, default=DEFAULT_BANDWIDTH,
                            help='''default bandwidth (in Mbps) of links that don't specify one (default=%(default)s)''')

        # experiment interaction control
        arg_parser.add_argument('--debug', '-d', type=str, default='info', nargs='?', const='debug',
                            help='''set debug level for logging facility (default=%(default)s, %(const)s when specified with no arg)''')
        arg_parser.add_argument('--output-file', '-o', type=str, default="results.json", dest='output_filename',
                            help='''name of output file for recording JSON results (default=%(default)s''')

        return arg_parser

    def run_all_experiments(self):
        """Runs the requested experimental configuration
        for the requested number of times, saving the results to an output file."""

        # ensure the output directory exists...
        try:
            os.mkdir(os.path.dirname(self.output_filename))
        except OSError:  # dir exists
            pass

        # Log progress to a file so that we can check on long-running simulations to see how many runs they have left.
        progress_file = None
        if self.nruns > 1:
            progress_filename = self.output_filename.replace(".json", ".progress")
            # in case we hadn't specified a .json output file:
            if progress_filename == self.output_filename:
                progress_filename += ".progress"

            try:
                progress_file = open(progress_filename, "w")
                progress_file.write("Starting experiments at time %s\n" % time.ctime())
            except IOError as e:
                log.warn("Error opening progress file for writing: %s" % e)

        self.set_interrupt_signal()

        # start the actual experimentation
        for r in range(self.nruns):
            log.info("Starting run %d" % self.current_run_number)

            self.setup_topology()

            self.setup_experiment()
            result = self.run_experiment()
            self.teardown_experiment()

            self.record_result(result)

            if progress_file is not None:
                try:
                    progress_file.write("Finished run %d at %s\n" % (self.current_run_number, time.ctime()))
                    progress_file.flush()  # so we can tail it
                except IOError as e:
                    log.warn("Error writing to progress file: %s" % e)

            self.current_run_number += 1
        self.output_results()

    def set_interrupt_signal(self):
        # catch termination signal and immediately output results so we don't lose ALL that work
        def __sigint_handler(sig, frame):
            # HACK: try changing filename so we know it wasn't finished
            self.output_filename = self.output_filename.replace('.json', '_UNFINISHED.json')
            log.critical("SIGINT received! Outputting current results to %s and exiting" % self.output_filename)
            self.output_results()
            exit(1)
        signal.signal(signal.SIGINT, __sigint_handler)

    def record_result(self, result):
        """Result is a dict that includes the experimental results for the current run i.e. percentage of subscribers
        reachable, what parameters were used, etc.  This method also records additional parameters such as run #,
        failed_nodes/links, etc."""

        # First, add additional parameters used on this run.
        result['run'] = self.current_run_number

        # make the paths relative to the root directory in which the whole experiment output file is stored
        # as otherwise the paths are dependent on where the cwd is
        # WARNING: if a custom path was specified this may cause an error!
        # NOTE: we first check to see if we even required this feature i.e. generated these directories
        if self._logs_dir or self._outputs_dir:
            root_dir = os.path.dirname(self.output_filename)
            logs_dir = os.path.relpath(self._logs_dir, root_dir) if self._logs_dir else None
            outputs_dir = os.path.relpath(self._outputs_dir, root_dir) if self._outputs_dir else None
            result['outputs_dir'] = outputs_dir
            result['logs_dir'] = logs_dir

        self.results['results'].append(result)

    def record_parameter(self, param_name, value):
        """Records a parameter that might be varied in different experiments."""
        self.results['params'][param_name] = value

    def output_results(self):
        """Outputs the results to a file"""
        log.info("Results: %s" % json.dumps(self.results, sort_keys=True, indent=2))
        if os.path.exists(self.output_filename):
            log.warning("Output file being overwritten: %s" % self.output_filename)
        with open(self.output_filename, "w") as f:
            json.dump(self.results, f, sort_keys=True, indent=2)

    def _choose_random_hosts(self, nhosts, from_list=None):
        """
        Chooses a uniformly random sampling of hosts to act as some group.
        If nhosts > total_hosts, will return all hosts.
        :param nhosts:
        :param from_list: optionally specify a list of hosts to choose from instead of self.hosts
        :return:
        """
        hosts = self.hosts if from_list is None else from_list
        sample = self.random.sample(hosts, min(nhosts, len(hosts)))
        return sample

    def setup_topology(self):
        """
        Construct and configure appropriately the network topology.  By default it does nothing.
        :return:
        """
        pass

    @abstractmethod
    def run_experiment(self):
        """
        Run the actual experiment and return the results in a dict to be recorded.

        :returns dict results:
        """
        raise NotImplementedError

    def setup_experiment(self):
        """
        Set up the experiment and configure it as necessary before run_experiment is called.  By default does nothing.
        :return:
        """
        pass

    def teardown_experiment(self):
        """
        Cleans up the experiment in preparation for the next call to setup (or being finished).
        By default does nothing.
        """
        pass

    ## Working with log and sub-output (i.e. from a simulation process our experiment runs) files

    def build_outputs_logs_dirs(self, output_filename=None):
        """
        Creates, stores in self, and returns the path to two directories:
            outputs_dir starts with 'outputs_' for putting the outputs of whatever host processes you run in
            logs_dir starts with 'logs_' for redirecting the raw output from those processes for debugging purposes.
        The latter may be disabled by setting WITH_LOGS=False in config.py, but this method will just return None for
        that directory (stil returns a 2-tuple).

        NOTE: we used to try doing multiple runs per process, which is why these directories also have nested 'run#'
            dirs inside of them, but we opt not to remove this as too much else would need to change...

        :param output_filename: optionally specify the output filename, which will have its file extension removed and
                will be prepended as per above for the dir names (default=self.output_filename)
        :returns outputs_dir, logs_dir: the directories (relative to the experiment output
         file) in which the output and log files, respectively, are stored for this run
        """

        if output_filename is None:
            output_filename = self.output_filename

        # The logs and output files go in nested directories rooted
        # at the same level as the whole experiment's output file.
        # We typically name the output file as results_$PARAMS.json, so cut off the front and extension
        root_dir = os.path.dirname(output_filename)
        base_dirname = os.path.splitext(os.path.basename(output_filename))[0]
        if base_dirname.startswith('results_'):
            base_dirname = base_dirname[8:]
        if WITH_LOGS:
            logs_dir = os.path.join(root_dir, 'logs_%s' % base_dirname, 'run%d' % self.current_run_number)
            try:
                os.makedirs(logs_dir)
                # XXX: since root is running this, we need to adjust the permissions, but using mode=0777 in os.mkdir()
                # doesn't work for some systems...
                os.chmod(logs_dir, 0777)
            except OSError:
                pass
        else:
            logs_dir = None
        outputs_dir =  os.path.join(root_dir, 'outputs_%s' % base_dirname, 'run%d' % self.current_run_number)
        try:
            os.makedirs(outputs_dir)
            os.chmod(outputs_dir, 0777)
        except OSError:
            pass

        self._logs_dir = logs_dir
        self._outputs_dir = outputs_dir

        return outputs_dir, logs_dir

    def redirect_output(self, cmd, filename, dirname=None):
        """
        Returns a modified version of the command string that redirects all output to a file composed of the
        the specified dirname (default=outputs_dir) and the specified output filename.
        :param cmd:
        :param filename:
        :param dirname: the directory to put the output in, which is self.outputs_dir by default
        """

        if dirname is None:
            dirname = self.outputs_dir
            if dirname is None:
                dirname, _ = self.build_outputs_logs_dirs()

        return cmd + " > %s 2>&1" % os.path.join(dirname, filename)

    def redirect_output_to_log(self, cmd, filename):
        """Returns a modified version of the command string that redirects all output to a log file composed of the
        logs_dir and the specified output filename."""
        logs_dir = self.logs_dir
        if logs_dir is None:
            _, logs_dir = self.build_outputs_logs_dirs()
        return self.redirect_output(cmd, filename, dirname=logs_dir)

    @property
    def outputs_dir(self):
        self.build_outputs_logs_dirs()
        return self._outputs_dir

    @property
    def logs_dir(self):
        self.build_outputs_logs_dirs()
        return self._logs_dir


NetworkExperiment.__doc__ = CLASS_DESCRIPTION