#!/usr/bin/python -O
# -O flag turns off assertions

"""
This script runs a whole batch of experiments in multiple processes
for parallelization.  Mostly it's meant for personal use: defining
the configurations to experiment with and just (un)commenting which
ones we're currently doing so we can easily re-run them.
"""

import random
import subprocess
import sys, os
import getpass
from os import getpid
from multiprocessing import Pool, Manager
from multiprocessing.managers import ValueProxy
import signal
import traceback
from itertools import chain

from ride.ride_d import RideD
from failure_model import SmartCampusFailureModel

# when True, this flag causes run.py to only print out the commands rather than run them each
# testing = True
testing = False
# debug_level = 'debug'  # for the actual experiment
debug_level = 'warn'
verbose = True
print_cmd = True
nruns = 10
run_start_num = 0  # change this to add additional runs to experiments e.g. after each 10 runs set to 10, 20, etc.
reverse_cmds = False
using_mininet = True
# Mininet can't handle multiple runs per process instance (weird OS-level errors occur sometimes after a few runs)
# WARNING: don't mix this with a version that doesn't as the smart_campus_experiment will be using a different random #
# for each run due to the times the RNG was used between each.
# You also can't set seeds explicitly through the params this way!  See the 'explicit_seeds' variable instead....
one_proc_per_run = using_mininet
if using_mininet:
    if getpass.getuser() != 'root' and not testing:
        print "ERROR: Mininet must be run as root!"
        exit(1)

    from mininet_smart_campus_experiment import MininetSmartCampusExperiment as TheSmartCampusExperiment
    from config import CONTROLLER_IP, CONTROLLER_RESET_CMD, IGNORE_OUTPUT
else:
    from networkx_smart_campus_experiment import NetworkxSmartCampusExperiment as TheSmartCampusExperiment

DEFAULT_PARAMS = {
    'fprob': 0.1,
    'ntrees': 4,
    'nsubscribers': 20 if using_mininet else 400,
    'npublishers': 10 if using_mininet else 200,
    'topology_filename': 'topos/campus_topo_20b-2h-5ibl.json' if using_mininet else 'topos/campus_topo_200b-20h-20ibl.json',
    # NOTE: the construction algorithm is specified as a list since each config is run for each experimental treatment!
    # Used to compare each algorithm for each set of parameters, but now we run a specific one
    #'tree_construction_algorithm': [('steiner',), ('diverse-paths',), ('red-blue',)],
    'tree_construction_algorithm': ('red-blue',),  # diverse-paths is really slow and steiner almost always performs worse
    # 'tree_construction_algorithm': [('steiner', 'max'), ('steiner', 'double')],
}
# for smaller topology (quicker run):
# if not using_mininet:
#     DEFAULT_PARAMS.update(dict(topology_filename='topos/cloud_campus_topo_20b-10h-5ibl.json',
#                                nsubscribers=40, npublishers=20))

# we'll explore each of these when running experiments
nsubscribers = [20, 40, 80, 160]
npublishers = [10, 20, 40, 80, 160]
# nhosts = None  # build nhosts with the nsubscribers/npublishers parameters
# subs/pubs ratio goes 1:1 thru 1:8 and vice-versa, also vary total # hosts
nhosts = [{'nsubscribers': s, 'npublishers': p, "choicerandseed": -5732823796696650875,
    "failrandseed": 2648076232431673581,
    "randseed": -7114345580798557657,} for s,p in
    sorted(set([(50 * (ratio if vary_subs else 4), 50 * (ratio if vary_pubs else 4))  # set one param to 200, the other varies from 50-800
        for vary_subs, vary_pubs in ((0,1), (1,0))
        for ratio in [1, 2, 4, 8, 16]]), reverse=True)
]  # explicitly set the nhosts params
# nhosts = [{'nsubscribers': s, 'npublishers': p,
#            "choicerandseed": 7683823364746221991, "failrandseed": -7234762391813259413, "randseed": 737923788253431206,}
#           for s,p in [(400, 800), (400, 25), (800, 200), (800, 50), (800, 800), (800, 1600)]]
# ntrees = [1, 2, 4, 8, 16]
# fprobs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5]
ntrees = [0,1,2,4,8]
fprobs = [0.0, 0.05, 0.1, 0.2, 0.3]
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
    # NOTE: TRY itertools.product HERE FOR CROSS PRODUCTS
    # 'construction-reroute': [
    # 'npubs-reroute': [
    # # 'construction-selection': [
    #     {
    #         # 'tree_construction_algorithm': alg,
    #         # 'tree_choosing_heuristic': choice,
    #         # TODO: not p*2 just static 20 or maybe 10?
    #         'npublishers': p, 'nsubscribers': 400,
    #         # 'ntrees': t,
    #         'fprob': f,
    #         # 'topology_filename': 'topos/campus_topo_20b-2h-5ibl.json',
    #         'reroute_policy': rrp,
    #     }
    #     for rrp in ['disjoint', 'shortest']
    #     for p in [100, 200, 400, 800]
    #     # for t in [2, 4]
    #     for f in [0.1, 0.2]
    #     # for choice in RideD.MDMT_SELECTION_POLICIES
    #     # for alg in [[('steiner',), ('diverse-paths',), ('red-blue',)]]
    #     ],
    # 'tree_choosing_heuristic': RideD.MDMT_SELECTION_POLICIES,
    # 'reroute_policy': ['disjoint', 'shortest'],
    # look at varying fprobs too as 0.1 may be too low for >2-4 trees
    # 'ntrees': [{'ntrees': t, 'fprob': f} for t in ntrees for f in [0.2, 0.4]],
    # 'ntrees': ntrees,
    # 'fprob': fprobs,
    # built with above func, looks like: [{nsubs:10, npubs:20}, {nsubs:20, npubs:10}]
    # 'nhosts': nhosts if nhosts is not None else get_nhosts_treatment(nsubscribers, npublishers),
    ## NOTE: the rest of these parameter explorations do not have the parameter included in the default output_filename
    'nretries': [{'max_alert_retries': rt, 'output_filename': 'results_%dretries.json' % rt} for rt in [0, 1, 3, 7]]
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
            for heur in args2.get('tree_construction_algorithm', (DEFAULT_PARAMS['tree_construction_algorithm'],)):
                # Again, make a copy to avoid overwriting params
                args3 = args2.copy()
                args3['tree_construction_algorithm'] = heur

                # make the directory tell which treatment is being explored currently
                this_dirname = os.path.join(output_dirname, param)
                if this_dirname and not testing:
                    try:
                        os.mkdir(this_dirname)
                    except OSError:
                        pass

                args3 = getargs(output_dirname=this_dirname, **args3)
                # ensure we actually output everything to this directory (mostly for when fname is manually specified)
                if not args3['output_filename'].startswith(this_dirname):
                    args3['output_filename'] = os.path.join(this_dirname, args3['output_filename'])

                # When we spawn a new process for each run, we need to specify the run# and increment the seeds correctly!
                if one_proc_per_run:
                    # XXX: HACK to ensure that we get the same sequence of seeds for each treatment we have to seed the
                    # RNG explicitly and re-seed it for each treatment to reset the sequence.
                    # When we move on to the next parameter exploration, we'll get a new random seed that's based on the
                    # previous value but this won't give us a continuation of the sequence since it's now a seed!
                    random.seed(rs)

                    for run_num in range(args3['nruns']):
                        run_num += run_start_num
                        args4 = args3.copy()
                        args4['nruns'] = 1
                        args4['run_start_num'] = run_num

                        # We're automatically generating new seeds, but we want to keep the originals around for resetting.
                        new_crs, new_rs, new_frs = get_next_seeds()
                        args4['choice_rand_seed'] = new_crs
                        args4['rand_seed'] = new_rs
                        args4['faiure_rand_seed'] = new_frs

                        # We also need to change the 'output_filename' to avoid overwriting it with each new run!
                        fname = args4['output_filename']
                        if not fname.endswith('.json'):
                            print "WARNING: output_filename %s doesn't end with '.json'!!  Appending run number instead, which may break things..."
                            fname += '.%d' % run_num
                        else:
                            fname = fname.replace('.json', '.%d.json' % run_num)
                        args4['output_filename'] = fname

                        yield args4

                else:
                    if run_start_num > 0:
                        args3['run_start_num'] = run_start_num
                    yield args3


def getargs(output_dirname='', **kwargs):
    """Builds the argument list with defaults defined up top but
    overwritten by any kwargs passed in."""

    # Need to copy to avoid corrupting defaults
    _args = DEFAULT_PARAMS.copy()
    _args.update(CONTROL_FLOW_PARAMS)
    _args['debug'] = debug_level
    _args.update(**kwargs)

    # Build output filename
    _args['output_filename'] = _args.get('output_filename', TheSmartCampusExperiment.build_default_results_file_name(_args, output_dirname))
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
    assert os.path.exists(kwargs['topology_filename']), "topology file %s doesn't exist!" % kwargs['topology_filename']


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

            failure_model = SmartCampusFailureModel(**kwargs)
            exp = TheSmartCampusExperiment(failure_model=failure_model, **kwargs)
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
    using_pool = (nprocs != 1)
    using_pool = True
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
    # TODO: sort cmds to place diverse-paths first since it runs longest
    if reverse_cmds:
        all_cmds.reverse()
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
