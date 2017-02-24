#!/usr/bin/python -O
# -O flag turns off assertions

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

# when True, this flag causes run.py to only print out the commands rather than run them each
testing = True
# testing = False
# debug_level = 'debug'  # for the actual experiment
debug_level = 'warn'
verbose = True

DEFAULT_PARAMS = {
    'fprob': 0.1,
    'ntrees': 4,
    'nsubscribers': 80,
    'npublishers': 40,
    'topo': ['networkx', 'campus_topo_200b-20h.json'], # 200
    'mcast_heuristic': ['networkx', 'paths'],  # always a list!  we run all of them for each treatment
}

# we'll explore each of these when running experiments
nsubscribers = [20, 40, 80, 160]
npublishers = [10, 20, 40, 80, 160]
# nhosts = None  # build nhosts with the nsubscribers/npublishers parameters
# subs/pubs ratio goes 1:1 thru 1:8, also vary total # hosts
nhosts = [{'nsubscribers': i*ratio, 'npublishers': i} for i in [25, 50, 100] for ratio in [1, 2, 4, 8]
]  # explicitly set the nhosts params
ntrees = [2, 4, 8, 16]
fprobs = [DEFAULT_PARAMS['fprob'], 0.2, 0.35, 0.5]

# now build up the actual dict of parameters
def get_nhosts_treatment(nsubs, npubs):
    # The treatment for pubs/subs requires interleaving them else we'd
    # have e.g. 5 pubs and 200 subs.
    # We zip together the lists defined above with some offset from each
    # other (in both directions) in order to get all the pairs we want.
    nhosts = []
    for i in range(3):
        # Prevent duplicates for first iteration
        if i == 0:
            tups = zip(nsubs, npubs)
        else:
            tups = chain(zip(nsubs[i:], npubs), zip(nsubs, npubs[i:]))
        for tup in tups:
            nhosts.append({k: v for k,v in zip(('nsubscribers', 'npublishers'), tup)})
    return nhosts
EXPERIMENTAL_TREATMENTS = {
    'ntrees': ntrees,
    'fprob': fprobs,
    # built with above func, looks like: [{nsubs:10, npubs:20}, {nsubs:20, npubs:10}]
    'nhosts': nhosts if nhosts is not None else get_nhosts_treatment(nsubscribers, npublishers)

    # TODO: vary topology for inter-building connectivity
    # 'topo': [DEFAULT_PARAMS['topo']],
}

CONTROL_FLOW_PARAMS = {
    'nruns': 50, # 100
}
# these aren't passed to the experiment class
nprocs = None if not testing else 1  # None uses cpu_count()

def makecmds(output_dirname=''):
    """Generator for each process (parallel call to the experiment)."""

    # Experiments to run: this orders the parameter lists from above
    # so that we explore each parameter by varying its arguments
    # against a common 'background' treatment of default parameters.

    for param, treatments in EXPERIMENTAL_TREATMENTS.items():
        # Generate new random seeds between each parameter exploration.
        # This ensures that different treatments and heuristics will
        # be compared completely fairly: parallel heuristic-varied
        # runs of the same treatment or treatment-varied runs within
        # an exploration will have the same publishers, subscribers,
        # failures, etc. unless that is the parameter being explored.
        crs, rs, frs = get_next_seeds()
        args = {'choice_rand_seed': crs,
                'rand_seed': rs,
                'failure_rand_seed': frs
                }

        for treat in treatments:
            # NOTE: we have to make a copy of the args dict or else
            # the next iteration of the loops will overwrite the value
            # of the previous one!
            args = args.copy()

            # treat is a dict when it brings in > 1 params
            if isinstance(treat, dict):
                args.update(treat)
            # otherwise we just set the 1 param
            else:
                args[param] = treat

            # We always want to run all the heuristics for each treatment
            for heur in DEFAULT_PARAMS['mcast_heuristic']:
                # Again, make a copy to avoid overwriting params
                args = args.copy()
                args['mcast_heuristic'] = heur

                # make the directory tell which treatment is being explored currently
                this_dirname = os.path.join(output_dirname, param)
                if this_dirname:
                    try:
                        os.mkdir(this_dirname)
                    except OSError:
                        pass

                args = getargs(output_dirname=this_dirname, **args)
                yield args


def getargs(output_dirname='', **kwargs):
    """Builds the argument list with defaults defined up top but
    overwritten by any kwargs passed in."""

    # Need to copy to avoid corrupting defaults
    _args = DEFAULT_PARAMS.copy()
    _args.update(CONTROL_FLOW_PARAMS)
    _args['debug'] = debug_level
    _args.update(**kwargs)

    # label the file with a parameter summary and optionally place in a directory
    _args['output_filename'] = os.path.join(output_dirname, 'results_%dt_%0.2ff_%ds_%dp_%s.json' % \
                                           (_args['ntrees'], _args['fprob'], _args['nsubscribers'],
                                            _args['npublishers'], _args['mcast_heuristic']))
    return _args


# Setting to None makes random seeds; a list will
# yield the seeds for each new 'experiment'
explicit_seeds = None  #[
    # nhosts
    # (4825234495444926055, 1862866243447831926, -4004346743967612352),
    # ntrees
    # (1901063825387834684, 494229327978780803, -223932071436254171),
    # fprob
    # (5597102490377962295, -1749099356559411320, -2363720760631486061),
# ]
def get_next_seeds(nseeds=3):
    """Typically just randomly generates nseeds random seeds,
    but you can also modify it to explicitly return seeds
    e.g. those pre-chosen by previous runs you want to do again.

    :returns (choice_rand_seed, rand_seed, failure_rand_seed)"""

    global explicit_seeds
    if explicit_seeds is not None:
        seeds = explicit_seeds[0]
        # trim it off for next time
        explicit_seeds = explicit_seeds[1:]
        return seeds
    else:
        return tuple(random.randint(-sys.maxsize-1, sys.maxsize) for i in range(nseeds))


def run_experiment(finished_message, kwargs):
    if using_pool:
        # Ignore ctrl-c in worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    if testing:
        print kwargs['output_filename']
        return

    if verbose:
        print kwargs['output_filename']

    failure_model = SmartCampusFailureModel(**kwargs)
    exp = SmartCampusNetworkxExperiment(failure_model=failure_model, **kwargs)
    exp.run_all_experiments()

    if verbose:
        print finished_message


if __name__ == '__main__':

    # store files in a directory if requested
    dirname = ''
    if len(sys.argv) > 1:
        dirname = sys.argv[1]
    if dirname:
        try:
            os.mkdir(dirname)
        except OSError:
            pass

    # use a process pool to run jobs in parallel
    using_pool = nprocs != 1
    if using_pool:
        pool = Pool(processes=nprocs)

    def __sigint_handler(sig, frame):
        """Called when user presses Ctrl-C to kill whole process.
        Kills the pool to end the program."""
        if using_pool:
            pool.terminate()
        exit(1)

    signal.signal(signal.SIGINT, __sigint_handler)

    # map inputs a positional argument, not kwargs
    # pool.map(run_experiment, makecmds(_dirname=dirname), chunksize=1)
    all_cmds = list(makecmds(output_dirname=dirname))
    for i, cmd in enumerate(all_cmds):
        msg = "%f%% complete (assuming previous ones finished)" % ((i+1)*100.0/(len(all_cmds)))
        cmd = [msg, cmd]
        if using_pool:
            pool.apply_async(run_experiment, cmd)
        else:
            apply(run_experiment, cmd)

    if using_pool:
        pool.close()
        pool.join()
