#!/usr/bin/env python
# need above to support virtualenv and simple command './run.py'
# to turn off assertions, enable optimized python by instead explicitly calling 'python -OO run.py'

"""
This script runs a whole batch of experiments in multiple processes
for parallelization.  Mostly it's meant for personal use: defining
the configurations to experiment with and just (un)commenting which
ones we're currently doing so we can easily re-run them.
"""

import random
import sys, os
import getpass
from os import getpid
from multiprocessing import Pool, Manager
from multiprocessing.managers import ValueProxy
import signal
import traceback

from scifire.firedex_scenario import FiredexScenario
from scifire.algorithms import ALL_ALGORITHMS


# when True, this flag causes run.py to only print out the commands rather than run them each
# testing = True
testing = False
debug_level = 'info'  # for the actual experiment
# debug_level = 'warn'
verbose = True
print_cmd = True
nruns = 1
run_start_num = 0  # change this to add additional runs to experiments e.g. after each 10 runs set to 10, 20, etc.
using_mininet = False
# Mininet can't handle multiple runs per process instance (weird OS-level errors occur sometimes after a few runs)
# WARNING: don't mix this with a version that doesn't as the smart_campus_experiment will be using a different random #
# for each run due to the times the RNG was used between each.
one_proc_per_run = using_mininet
if using_mininet:
    if getpass.getuser() != 'root' and not testing:
        print "ERROR: Mininet must be run as root!"
        exit(1)

    from firedex_mininet_experiment import FiredexMininetExperiment as TheExperiment
    from config import CONTROLLER_IP, CONTROLLER_RESET_CMD, IGNORE_OUTPUT
else:
    from firedex_algorithm_experiment import FiredexAlgorithmExperiment as TheExperiment

# we'll explore each of these when running experiments
ntopics = [10, 100, 1000]
nprios = [1, 3, 6, 9]  # 1 means no priorities!
errs = [0, 0.001, 0.01, 0.1]
nffs = [3, 6, 12, 18]
sub_dists = (({'dist': 'uniform'}, {'dist': 'uniform'}), ({'dist': 'zipf', 'args': [2]}, {'dist': 'zipf', 'args': [2]}))
bandwidths = [10, 100, 1000]  # in Mbps
algs = ALL_ALGORITHMS
ro_tolerances = [0.001, 0.01, 0.1, 0.2, .4]
# NOTE: if a parameter is a dict (e.g. single RV dist. or alg.), you need to wrap it in a dict keyed by its parameter
# name so that the runner doesn't treat it as a collection of parameters but a single parameter!
# algs = [dict(algorithm={'algorithm': 'random', 'seed': 567678383})]

# Here is where you can define experiments to run.  A list of values as dict value will vary the key's
# parameter for each of those values; a list of dicts as the dict value will explicitly set each of
# those parameters, which means you can name the key anything you want.  Note that you can specify
# the random seeds here if you want to re-run previous experiments with new parameters and have the
# same pubs/subs/topics etc.  Also note that the keys for this dict will correspond with the output directory.
EXPERIMENTAL_TREATMENTS = {
    ## this is a good setup for testing that most things are working okay
    # 'num_topics': ntopics,
    ## this is for explicitly forcing just a single run to quickly test the simulator itself
    'testing': [{'bandwidth': 0.1, 'nruns': 1, 'testing': False}],
    ## Actual varied parameter explorations:
    'nprios': [{'num_priority_levels': p, 'num_net_flows': p} for p in nprios],  # currently assume nprios==nflows always!
    'error_rate': errs,
    'bandwidth': bandwidths,
    # 'algorithm': algs,
    'prio_probs': [{'bandwidth': 0.2, 'nruns': 10, 'algorithm': dict(algorithm='random', ro_tolerance=ro_tol),
                 'output_filename': 'results_%fro.json' % ro_tol} for ro_tol in ro_tolerances],
    ## NOTE: the rest of these parameter explorations do not have the parameter included in the default output_filename
    # 'topic_dists': [{'num_topics': t, 'num_ffs': f, 'num_iots': f*2, 'topic_class_sub_dists': sub_dist,
    #                  'output_filename': 'results_%dt_%df_sub-%s.json' % (t, f, sub_dist[0]['dist'])}\
    #                 for t in ntopics for f in nffs for sub_dist in sub_dists]
    # TODO: subscriptions/utilities, advertisements(publications), pub rates/sizes, topic classes?
}

# these aren't passed to the experiment class
nprocs = 2  # queuing simulator is multi-threaded
# nprocs = None if not testing else 1  # None uses cpu_count()

def makecmds(output_dirname=''):
    """Generator for each process (parallel call to the experiment)."""

    # Experiments to run: this orders the parameter lists from above
    # so that we explore each parameter by varying its arguments
    # against a common 'background' treatment of default parameters.

    for param, treatments in EXPERIMENTAL_TREATMENTS.items():
        # we output everything for this exploration into a directory by this name and use it for other book-keeping
        experiment_name = param

        for treat in treatments:
            # NOTE: we have to make a copy of the args dict or else
            # the next iteration of the loops will overwrite the value
            # of the previous one!
            args = dict()

            # treat is a dict when it brings in > 1 params
            if isinstance(treat, dict):
                args.update(treat)
            # otherwise we just set the 1 param
            else:
                args[param] = treat

            # make the directory tell which treatment is being explored currently
            this_dirname = os.path.join(output_dirname, experiment_name)
            if this_dirname and not testing:
                try:
                    os.mkdir(this_dirname)
                except OSError:
                    pass

            args3 = getargs(output_dirname=this_dirname, **args)
            # ensure we actually output everything to this directory (mostly for when fname is manually specified)
            if not args3['output_filename'].startswith(this_dirname):
                args3['output_filename'] = os.path.join(this_dirname, args3['output_filename'])

            # When we spawn a new process for each run, we need to specify the run# and increment the seeds correctly!
            if one_proc_per_run:

                for run_num in range(args3['nruns']):
                    run_num += run_start_num
                    args4 = args3.copy()
                    args4['nruns'] = 1
                    args4['run_start_num'] = run_num

                    # We also need to change the 'output_filename' to avoid overwriting it with each new run!
                    fname = args4['output_filename']
                    if not fname.endswith('.json'):
                        print "WARNING: output_filename %s doesn't end with '.json'!!  Appending run number instead, which may break things..."
                        fname += '.%d' % run_num
                    else:
                        fname = fname.replace('.json', '.%d.json' % run_num)
                    args4['output_filename'] = fname

                    args4 = get_args_with_seeds(args4, experiment_name, run_num)
                    yield args4

            else:
                if run_start_num > 0:
                    args3['run_start_num'] = run_start_num
                args3 = get_args_with_seeds(args3, experiment_name)
                yield args3


def getargs(output_dirname='', **kwargs):
    """Builds the argument list with defaults defined up top but
    overwritten by any kwargs passed in."""

    _args = kwargs
    _args.setdefault('nruns', nruns)
    _args.setdefault('debug', debug_level)

    # Build output filename, which by default explicitly lists the params being varied
    if 'output_filename' not in _args:
        # XXX: if the output directory IS a parameter, just include that one since that's (most likely) all we're varying
        exp_name = os.path.split(output_dirname)[-1]
        if exp_name in _args:
            params_summary = {exp_name: _args[exp_name]}
        else:
            ignored_params = ('run', 'run_start_num', 'nruns', 'debug')
            params_summary = {k:v for k,v in _args.items() if ('seed' not in k) and (k not in ignored_params)}
        params_summary = '_'.join(["%s-%s" % (v, k) for k,v in params_summary.items()])
        _args['output_filename'] = os.path.join(output_dirname, 'results_%s.json' % params_summary)

    return _args

_seeds_for_exps = dict()
def _get_seed():
    return random.randint(-sys.maxsize - 1, sys.maxsize)
def get_args_with_seeds(args, exp_name, run=0):
    """
    Sorts through all the arguments and sets seeds for any that are random variable distributions (or other config
    dicts that accept seeds).  Guarantees the same seeds for each arg when passed the same exp_name for the purpose of
    ensuring a fair comparison across all treatments in that experiment.

    :param args:
    :param exp_name:
    :param run: if specified, bit shifts the seed by the run number to ensure different seeds for each run
    :return:
    """

    # TODO finish get_args_with_seeds method!

    return args

    # Need to filter all the parameters that specify random distributions/algorithms/etc.
    # seeded_params = [p for p in args if p in FiredexScenario.RANDOM_VARIABLE_DISTRIBUTION_PARAMETERS]
    #
    # # For each of these, we need to set the seeds for EACH distribution this parameter specifies (i.e. some are nested in lists)
    # for param in seeded_params:
    #     param_value = args[param]
    #     seeds = _seeds_for_exps.setdefault(exp_name, dict()).setdefault(param, [])
    #
    #     def set_seed(kwarg):
    #         pass
    #
    #     # Since some params may be constants or nested lists containing RV dicts, we need to generate seeds as we first
    #     # need them to ensure if another treatment parameter comes around with more RV dicts (e.g. more topic classes),
    #     # we won't get an IndexError
    #     if isinstance(param_value, dict):
    #         pass
    #     elif not isinstance(param_value, basestring):
    #         try:
    #             for d in param_value:
    #                 pass
    #             # PROBLEMS: does dict.copy() recurse and copy the values? probs not... how to actually modify the dict then and return it????
    #         except TypeError:
    #             pass  # probably just a number/constant


def run_tests_on_cmd(**kwargs):
    if os.path.exists(kwargs['output_filename']):
        print "WARNING: file %s already exists!" % kwargs['output_filename']


def run_experiment(jobs_finished, total_jobs, kwargs):
    """
    :param ProxyValue jobs_finished:
    :param int total_jobs:
    :param dict kwargs:
    :return:
    """

    if using_pool and not using_mininet:
        # Ignore ctrl-c in worker processes
        # Need to process it when using Mininet since we don't apply_async
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    if verbose:
        print "Proc", getpid(), "starting", kwargs['output_filename'], "..."
    if print_cmd:
        print kwargs

    # re-raise any errors after we register that the job completed
    err = None
    if not testing:
        try:
            if using_mininet:
                # probably don't need this as it's already done in the experiment....
                # Clean SDN Controller (ONOS) and Mininet just in case
                # p = subprocess.Popen('%s %s' % (CONTROLLER_RESET_CMD, IGNORE_OUTPUT), shell=True)
                # p.wait()
                # p = subprocess.Popen('sudo mn -c %s' % IGNORE_OUTPUT, shell=True)
                # p.wait()

                # Need to set params used for real system configs.
                # ENHANCE: include port #, topology_adapter_type, etc...
                kwargs['controller_ip'] = CONTROLLER_IP

            exp = TheExperiment(**kwargs)
            exp.run_all_experiments()
        except BaseException as e:
            err = (e, traceback.format_exc())
    else:
        run_tests_on_cmd(**kwargs)
        return

    if verbose:
        if isinstance(jobs_finished, ValueProxy):
            jobs_finished.set(jobs_finished.get() + 1)
            jobs_finished = jobs_finished.value
        else:
            jobs_finished += 1
        # ENHANCE: use a lock instead of a counter and update a progressbar (that's the package name)
        print "Proc", getpid(), "finished" if err is None else "FAILED!!", kwargs['output_filename'],\
            "-- %f%% complete" % (jobs_finished*100.0/total_jobs)
        if err is not None:
            print "FAILURE TRACEBACK:\n", err[1]

    if err is not None:
        raise err[0]


if __name__ == '__main__':

    # store files in a different directory if requested
    dirname = 'results'
    if len(sys.argv) > 1:
        dirname = sys.argv[1]
        if dirname == 'testing':
            print 'testing flag set to True from user request'
            testing = True
    if dirname and not testing:
        try:
            os.mkdir(dirname)
        except OSError:
            pass

    all_cmds = list(makecmds(output_dirname=dirname))
    total_jobs = len(all_cmds)

    # use a process pool to run jobs in parallel for the Networkx Experiment
    # NOTE: we still use it for Mininet, but only because we need to spawn a new process for each run
    using_pool = (nprocs != 1) or using_mininet
    # track the returned (empty) results to see if any processes crash
    results = []
    if using_pool:
        # XXX: we seem to need a new process for each Mininet run as if we just directly use the built-in apply() we end up
        # with weird buffer overflow errors after the first experiment completes.  Hence, we make each process in the pool
        # only execute one command (using non-async apply()) before being replaced.
        pool = Pool(processes=nprocs, maxtasksperchild=1 if using_mininet else None)
        # shared variable to track progress
        # NOTE: need to use Manager as directly using Value with pool causes RuntimeException...
        _mgr = Manager()
        jobs_completed = _mgr.Value('i', 0)
    else:
        pool = None
        jobs_completed = 0

    def __sigint_handler(sig, frame):
        """Called when user presses Ctrl-C to kill whole process.
        Kills the pool to end the program."""
        if using_pool:
            try:
                _mgr.shutdown()
                pool.terminate()
            except BaseException as e:
                print "Error trying to terminate pool:", e
            # ENHANCE: gracefully close the jobs_completed shared counter
            # and print out any incompleted jobs for easy manual restart
        exit(1)

    signal.signal(signal.SIGINT, __sigint_handler)

    # Now we actually run the commands using the right version of 'apply()'
    for i, cmd in enumerate(all_cmds):
        cmd = [jobs_completed, total_jobs, cmd]
        if using_pool:
            if using_mininet:
                result = pool.apply(run_experiment, cmd)
            else:
                result = pool.apply_async(run_experiment, cmd)
            results.append((result, cmd))
        else:
            apply(run_experiment, cmd)
            jobs_completed += 1

    # clean up the pool and print out any failed commands for later manual re-run
    # ENHANCE: have this get called even if we Ctrl+C terminate?
    if using_pool:
        pool.close()

        if not using_mininet:

            # These only work with pool.apply_async()
            failed_cmds = []
            # wait for results to finish first so that we don't
            # interleave failure reports with progress reports
            for res, cmd in results:
                res.wait()
            for res, cmd in results:
                if res.ready() and not res.successful():
                    # slice off first two since they're just metadata
                    failed_cmds.append(cmd[2:])
                    try:
                        print "COMMAND FAILED with result:", res.get()
                    except BaseException as e:
                        print "COMMAND FAILED:", cmd
                        print "REASON:", e.__class__, e.message, e.args

            if failed_cmds:
                failed_cmds_filename = os.path.join(dirname, "failed_cmds")
                with open(failed_cmds_filename, "w") as f:
                    f.write(str(failed_cmds))
                print "Failed commands written to file", failed_cmds_filename
            else:
                print "All commands successful!"

        pool.join()
