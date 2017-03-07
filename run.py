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
from os import getpid
from multiprocessing import Pool, Manager
from multiprocessing.managers import ValueProxy
import signal
from time import sleep
import traceback
from campus_net_experiment import SmartCampusNetworkxExperiment
from failure_model import SmartCampusFailureModel
from itertools import chain

# when True, this flag causes run.py to only print out the commands rather than run them each
#testing = True
testing = False
# debug_level = 'debug'  # for the actual experiment
debug_level = 'warn'
verbose = True
print_cmd = False
nruns = 100

DEFAULT_PARAMS = {
    'fprob': 0.1,
    'ntrees': 4,
    'nsubscribers': 400,
    # 'nsubscribers': 40,
    'npublishers': 200,
    # 'npublishers': 20,
    'topo': ['networkx', 'campus_topo_200b-20h-20ibl.json'],
    # 'topo': ['networkx', 'campus_topo_20b-8h-3ibl.json'],
    # always a list of tuples!  we run all of them for each treatment and
    # each heuristic optionally takes arguments
    # 'mcast_heuristic': [('steiner',), ('diverse-paths',), ('red-blue',)],
    'mcast_heuristic': [('steiner',), ('red-blue',)],  # skip diverse-paths since it's slowest
    # 'mcast_heuristic': [('red-blue',)],  # diverse-paths is really slow and steiner almost always performs worse
    # 'mcast_heuristic': [('steiner', 'max'), ('steiner', 'double')],
}

# we'll explore each of these when running experiments
nsubscribers = [20, 40, 80, 160]
npublishers = [10, 20, 40, 80, 160]
# nhosts = None  # build nhosts with the nsubscribers/npublishers parameters
# subs/pubs ratio goes 1:1 thru 1:8 and vice-versa, also vary total # hosts
nhosts = [{'nsubscribers': s, 'npublishers': p} for s,p in
          sorted(set([(50 * (ratio if vary_subs else 4), 50 * (ratio if vary_pubs else 4))  # set one param to 200, the other varies from 50-800
           for vary_subs, vary_pubs in ((0,1), (1,0))
           for ratio in [1, 2, 4, 8, 16]]), reverse=True)
]  # explicitly set the nhosts params
# nhosts = [{'nsubscribers': s, 'npublishers': p,
#            "choicerandseed": 7683823364746221991, "failrandseed": -7234762391813259413, "randseed": 737923788253431206,}
#           for s,p in [(400, 800), (400, 25), (800, 200), (800, 50), (800, 800), (800, 1600)]]
ntrees = [1, 2, 4, 8, 16]
fprobs = [0.05, 0.15, 0.25, 0.35, 0.5]
# nhosts.reverse()  # put larger jobs up front to make for easier sharing across procs
ntrees.reverse()

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
# Here is where you can define experiments to run.  A list of values as dict value will vary the key's
# parameter for each of those values; a list of dicts as the dict value will explicitly set each of
# those parameters, which means you can name the key anything you want.  Note that you can specify
# the random seeds here if you want to re-run previous experiments with new parameters and have the
# same pubs/subs/failures etc.
EXPERIMENTAL_TREATMENTS = {
    #    'ilp': [{'mcast_heuristic': [('ilp',)], 'npublishers': p, 'nsubscribers': p*2} for p in [10, 20, 40]]
    # NOTE: TRY itertools.product HERE FOR CROSS PRODUCTS

    # 'ntrees': ntrees,
    # look at varying fprobs too as 0.1 may be too low for >2-4 trees
    # 'ntrees': [{'ntrees': t, 'fprob': f} for t in ntrees for f in [0.2, 0.4]],
    # 'fprob': fprobs,
    # built with above func, looks like: [{nsubs:10, npubs:20}, {nsubs:20, npubs:10}]
    'nhosts': nhosts if nhosts is not None else get_nhosts_treatment(nsubscribers, npublishers),
    # we want to vary ntrees and fprobs together to see how the versions of the heuristic perform
    # 'steiner-double': [{'ntrees': t, 'fprob': f} for t in [8, 4, 2] for f in fprobs[:3]]
    # vary topology for inter-building connectivity
    # 'topo-ibl': [{'topo': ['networkx', 'campus_topo_200b-20h-%dibl.json' % ibl],
    #               "choicerandseed": 8968339335534376984, "failrandseed": -4400980186153869600,
    #               "randseed": -6760040867077965717,} for ibl in [200, 400, 800]],
    # vary topology size (need to vary nhosts along with it)
    # 'topo-sizes': [{'topo': ['networkx', 'campus_topo_%db-%dh-%dibl.json' % (nbuilds, nhosts, ibl)],
    #                 'npublishers': nbuilds, 'nsubscribers': nbuilds*2,
    #                 # 'output_filename': "%db-%dh-%d.json" % (nbuilds, nhosts, ibl),
    #                 }
    #                for nbuilds, nhosts, ibl in (
    #         # might want to check if lower numbers with repeats improves
    #         # if this big topo with ncores=8 makes a difference, try mid-range ibls
    #         # (400, 80, 400),
    #         # (200, 8, 20),
    #         # (80, 16, 8), (200, 40, 20),  # see if nhosts makes a difference
    #     )],  # (200, 20, 20)
    # 'topo-redundant': [{'topo': ['networkx', fname]} for fname in
    #                    ['campus_topo_200b-20h-1000ibl-redundant.json', 'campus_topo_200b-20h-1000ibl-redundant2.json']]
}

CONTROL_FLOW_PARAMS = {
    'nruns': nruns,
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
        start_args = {'choice_rand_seed': crs,
                'rand_seed': rs,
                'failure_rand_seed': frs
                }

        for treat in treatments:
            # NOTE: we have to make a copy of the args dict or else
            # the next iteration of the loops will overwrite the value
            # of the previous one!
            args2 = start_args.copy()

            # treat is a dict when it brings in > 1 params
            if isinstance(treat, dict):
                args2.update(treat)
            # otherwise we just set the 1 param
            else:
                args2[param] = treat

            # We always want to run all the heuristics for each treatment
            for heur in args2.get('mcast_heuristic', DEFAULT_PARAMS['mcast_heuristic']):
                # Again, make a copy to avoid overwriting params
                args3 = args2.copy()
                args3['mcast_heuristic'] = heur

                # make the directory tell which treatment is being explored currently
                this_dirname = os.path.join(output_dirname, param)
                if this_dirname and not testing:
                    try:
                        os.mkdir(this_dirname)
                    except OSError:
                        pass

                args3 = getargs(output_dirname=this_dirname, **args3)
                yield args3


def getargs(output_dirname='', **kwargs):
    """Builds the argument list with defaults defined up top but
    overwritten by any kwargs passed in."""

    # Need to copy to avoid corrupting defaults
    _args = DEFAULT_PARAMS.copy()
    _args.update(CONTROL_FLOW_PARAMS)
    _args['debug'] = debug_level
    _args.update(**kwargs)

    # label the file with a parameter summary and optionally place in a directory
    topo_fname = _args['topo'][1].split('_')[2].split('.')[0]
    _args['output_filename'] = os.path.join(output_dirname, _args.get('output_filename', 'results_%dt_%0.2ff_%ds_%dp_%s_%s.json' % \
                                                                      (_args['ntrees'], _args['fprob'], _args['nsubscribers'], _args['npublishers'],
                                                                       SmartCampusNetworkxExperiment.build_mcast_heuristic_name(*_args['mcast_heuristic']),
                                                                       topo_fname)))
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


def run_tests_on_cmd(**kwargs):
    if os.path.exists(kwargs['output_filename']):
        print "WARNING: file %s already exists!" % kwargs['output_filename']
    assert os.path.exists(kwargs['topo'][1])


def run_experiment(jobs_finished, total_jobs, kwargs):
    """
    :param ProxyValue jobs_finished:
    :param int total_jobs:
    :param dict kwargs:
    :return:
    """

    if using_pool:
        # Ignore ctrl-c in worker processes
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    if verbose:
        print "Proc", getpid(), "starting", kwargs['output_filename'], "..."
    if print_cmd:
        print kwargs

    # re-raise any errors after we register that the job completed
    err = None
    if not testing:
        try:
            failure_model = SmartCampusFailureModel(**kwargs)
            exp = SmartCampusNetworkxExperiment(failure_model=failure_model, **kwargs)
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

    # store files in a directory if requested
    dirname = ''
    if len(sys.argv) > 1:
        dirname = sys.argv[1]
    if dirname and not testing:
        try:
            os.mkdir(dirname)
        except OSError:
            pass

    # use a process pool to run jobs in parallel
    using_pool = nprocs != 1
    # track the returned (empty) results to see if any processes crash
    results = []
    if using_pool:
        pool = Pool(processes=nprocs)
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

    # map inputs a positional argument, not kwargs
    # pool.map(run_experiment, makecmds(_dirname=dirname), chunksize=1)
    all_cmds = list(makecmds(output_dirname=dirname))
    total_jobs = len(all_cmds)
    for i, cmd in enumerate(all_cmds):
        cmd = [jobs_completed, total_jobs, cmd]
        if using_pool:
            result = pool.apply_async(run_experiment, cmd)
            results.append((result, cmd))
        else:
            apply(run_experiment, cmd)
            jobs_completed += 1

    # clean up the pool and print out any failed commands for later manual re-run
    # ENHANCE: have this get called even if we Ctrl+C terminate?
    if using_pool:
        pool.close()

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
