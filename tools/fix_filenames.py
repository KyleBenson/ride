#!/usr/bin/python

import sys
import os
import json

dirname = sys.argv[1]

new_files = set()

for filename in os.listdir(dirname):
    if "mislabeled" not in filename:
        continue

    filename = os.path.join(dirname, filename)
    with open(filename) as f:
        params = json.load(f)['params']

    # old looks like: results_mislabeled_4t_0.10f_20s_20p_networkx.json
    old_nsubs = filename.split('_')[4]
    old_npubs = filename.split('_')[5]
    old_heur = filename.split('_')[6]
    # print "old nsubs:", old_nsubs, "npubs:", old_npubs, "heur:", old_heur


    new_nsubs = "%ds" % params['nsubscribers']
    new_npubs = "%dp" % params['npublishers']
    new_heur = "%s.json" % params['heuristic']
    new_fname = filename.replace(old_nsubs, new_nsubs).replace(old_npubs, new_npubs).replace("_mislabeled", "").replace(old_heur, new_heur)

    # print "nsubs:", new_nsubs, "npubs:", new_npubs
    # print filename, "-->", new_fname

    if os.path.exists(new_fname):
        new_fname = new_fname.replace("results_", "results2_")
    # if new_fname in new_files:
    #     new_fname = new_fname.replace("results_", "results2_")

    # Tracking new files cuz some overwrote each other
    if new_fname in new_files:
        print "ALREADY GOT ONE!"
        exit()
    new_files.add(new_fname)

    os.rename(filename, new_fname)