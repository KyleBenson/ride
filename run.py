#!/usr/bin/python

"""
This script runs a whole batch of experiments in multiple processes
for parallelization.  Mostly it's meant for personal use: defining
the configurations to experiment with and just (un)commenting which
ones we're currently doing so we can easily re-run them.
"""
import random

import sys, os
from multiprocessing import Pool
import signal
from campus_net_experiment import SmartCampusNetworkxExperiment
from failure_model import SmartCampusFailureModel
from itertools import chain

# debug_level = 'debug'
debug_level = 'warn'
verbose = False

DEFAULT_PARAMS = {
    'fprob': 0.1,
    'ntrees': 4,
    'nsubscribers': 80,
    'npublishers': 40,
    'topo': ['networkx', 'campus_topo_200b-20h.json'], # 200
    'mcast_heuristic': ['networkx', 'paths'],  # always a list!  we run all of them for each treatment
}

# we'll explore each of these when running experiments
nsubscribers = [10, 20, 40, 80, 160]
npublishers = [10, 20, 40, 80, 160]
EXPERIMENTAL_TREATMENTS = {
    'ntrees': [2, 4, 8, 16],
    'fprob': [DEFAULT_PARAMS['fprob'], 0.2, 0.35, 0.5],
    # TODO: vary topology for inter-building connectivity
    # 'topo': [DEFAULT_PARAMS['topo']],
}

CONTROL_FLOW_PARAMS = {
    'nruns': 50, # 100
}
# these aren't passed to the experiment class
nprocs = None  # uses cpu_count()

# The treatment for pubs/subs requires interleaving them else we'd
# have e.g. 5 pubs and 200 subs.
# We zip together the lists defined above with some offset from each
# other (in both directions) in order to get all the pairs we want.
nhosts = []
for i in range(3):
    # Prevent duplicates for first iteration
    if i == 0:
        tups = zip(nsubscribers, npublishers)
    else:
        tups = chain(zip(nsubscribers[i:], npublishers), zip(nsubscribers, npublishers[i:]))
    for tup in tups:
        nhosts.append({k: v for k,v in zip(('nsubscribers', 'npublishers'), tup)})
EXPERIMENTAL_TREATMENTS['nhosts'] = nhosts


def makecmds(output_dirname=''):
    """Generator for each process (parallel call to the experiment)."""

    # Experiments to run: this orders the parameter lists from above
    # so that we explore each parameter by varying its arguments
    # against a common 'background' treatment of default parameters.


    for param, treatments in EXPERIMENTAL_TREATMENTS.items():
        # Generate new random seeds between each parameter exploration.
        # This ensures parallel runs of the same treatment will have
        # the same publishers, subscribers, failures, etc.
        _args = {k: random.randint(-sys.maxsize-1, sys.maxsize) for k in\
                 [
                     'choice_rand_seed',
                     'rand_seed',
                     'failure_rand_seed',
                 ]}

        for treat in treatments:
            # treat is a dict when it brings in > 1 params
            if isinstance(treat, dict):
                _args.update(treat)
            # otherwise we just set the 1 param
            else:
                _args[param] = treat

            # We always want to run all the heuristics for each treatment
            for heur in DEFAULT_PARAMS['mcast_heuristic']:
                _args['mcast_heuristic'] = heur
                # make the directory tell which treatment is being explored currently
                this_dirname = os.path.join(output_dirname, param)
                if this_dirname:
                    try:
                        os.mkdir(this_dirname)
                    except OSError:
                        pass

                _args = getargs(output_dirname=this_dirname, **_args)
                yield _args


def getargs(output_dirname='', **kwargs):
    """Builds the argument list with defaults defined up top but
    overwritten by any kwargs passed in."""

    _args = DEFAULT_PARAMS.copy()
    _args.update(CONTROL_FLOW_PARAMS)
    _args['debug'] = debug_level
    _args.update(**kwargs)

    # label the file with a parameter summary and optionally place in a directory
    _args['output_filename'] = os.path.join(output_dirname, 'results_%dt_%0.2ff_%ds_%dp_%s.json' % \
                                           (_args['ntrees'], _args['fprob'], _args['nsubscribers'],
                                            _args['npublishers'], _args['mcast_heuristic']))

    return _args


def run_experiment(kwargs):
    # Ignore ctrl-c in worker
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    if verbose:
        print kwargs['output_filename']

    failure_model = SmartCampusFailureModel(**kwargs)
    exp = SmartCampusNetworkxExperiment(failure_model=failure_model, **kwargs)
    exp.run_all_experiments()


if __name__ == '__main__':

    # store files in a directory if requested
    args = sys.argv[1:]
    dirname = ''
    if len(args) > 0:
        dirname = args[0]
    if dirname:
        try:
            os.mkdir(dirname)
        except OSError:
            pass
    # use a process pool to run jobs in parallel
    pool = Pool(processes=nprocs)

    def __sigint_handler(sig, frame):
        """Called when user presses Ctrl-C to kill whole process.
        Kills the pool to end the program."""
        pool.terminate()
        exit(1)

    signal.signal(signal.SIGINT, __sigint_handler)

    # map inputs a positional argument, not kwargs
    # pool.map(run_experiment, makecmds(_dirname=dirname), chunksize=1)
    for cmd in makecmds(output_dirname=dirname):
        if nprocs is not None and nprocs > 1:
            pool.apply_async(run_experiment, [cmd])
        else:
            apply(run_experiment, [cmd])

    pool.close()
    pool.join()
