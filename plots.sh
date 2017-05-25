#!/bin/bash

TODO:





            # TODO: plot metrics with this:

# ax = plt.gca()
# ax2 = ax.twinx()
# plt.axis('normal')
# ax2.axvspan(74, 76, facecolor='g', alpha=1)
# ax.plot(range(50), 'b',linewidth=1.5)
# ax.set_ylabel("name",fontsize=14,color='blue')
# ax2.set_ylabel("name2",fontsize=14,color='blue')
# ax.set_ylim(ymax=100)
# ax.set_xlim(xmax=100)
# ax.grid(True)
# plt.title("name", fontsize=20,color='black')
# ax.set_xlabel('xlabel', fontsize=14, color='b')
# plt.show()




# create symlinks to other results with different fprob values so we can explore more options
cd results/fprob; for i in `ls ../ntrees-fprob/results_4t*.json`; do ln -s $i; done

# use env vars to easily change what exactly we're looking at without manually going thru cmd line args
export HEURISTIC=red-blue;
# never got this working right
#export SAVEFIG=--skip-plot\ -s\

##############################
## What actually went in Middleware paper submission:

# use these for everything unless otherwise specified
export RESULTS_ROOT=old_results_networkx
export RESULTS_DIR="$RESULTS_ROOT/results4"
export FPROB=0.10
export CHOICES="-c importance-chosen max-overlap-chosen max-reachable-chosen min-missing-chosen"

# To improve readability, we removed the legends and included them as separate subfigures.
# First, generate a plot for the group of subfigures (3 groups), cut out the legend as another subfigure,
# then re-generate without the legend.
export LEGEND="--legend "
export LEGEND=""

# ntrees for constr. algs.
./statistics.py -st max mean min -x ntrees -xl '#MDMTs (k)' -f $RESULTS_DIR/ntrees/results_*t_"$FPROB"f*.json -c --title $LEGEND
# fprob
./statistics.py -c -st max mean min -d $RESULTS_DIR/fprob/ --title -xl "failure probability" $LEGEND

# ntrees for selection policies
# all 3 construction algorithms for this one:
export ALGORITHM=red-blue #results4
export ALGORITHM=diverse-paths #results2
export ALGORITHM=steiner
./statistics.py -x ntrees -xl '#MDMTs (k)' -f $RESULTS_DIR/ntrees/results_*t_"$FPROB"f*_"$ALGORITHM"_*.json -i unicast oracle $ALGORITHM $CHOICES -st max mean --title $LEGEND

# npublishers for red-blue
export RESULTS_DIR="$RESULTS_ROOT/results3"
./statistics.py -x npublishers -xl '#sensor-publishers' -f $RESULTS_DIR/nhosts/results_*_200s_*p*_red-blue_*.json -i unicast oracle red-blue -st max mean --title $CHOICES $LEGEND


# publication loss rate
export RESULTS_DIR="$RESULTS_ROOT/results4"
./statistics.py -i unicast red-blue -st max -xl "publication loss rate" -x publication_error_rate -d $RESULTS_DIR/pub*rate/ --title $CHOICES $LEGEND

##############################


# NHOSTS
# we want to display the choosing heuristics as npubs should affect their reach
#export NSUBS=200
export NSUBS=400
export CHOICES="-c importance-chosen max-overlap-chosen max-reachable-chosen min-missing-chosen"
#export NSUBS=800
./statistics.py -st max mean -x npublishers -f results3/nhosts/results_*_"$NSUBS"s_*p*.json -i unicast oracle $HEURISTIC $CHOICES --title "$NSUBS subscribers, $HEURISTIC heuristic reachability" # $SAVEFIG "npubs_$HEURISTIC.png"
export NPUBS="50 100 400"  # for loop; 100 looks good
./statistics.py -st max mean -x nsubscribers -f results*/nhosts/results_*_"$NPUBS"p_*.json -i unicast oracle $HEURISTIC $CHOICES --title "$NPUBS publishers, $HEURISTIC heuristic reachability"
export HEURISTIC="red-blue diverse-paths steiner"

# NTREES
export FPROB=0.10
export RESULTS_DIR=results2
./statistics.py -st max -x ntrees -f $RESULTS_DIR/ntrees/results_*t_"$FPROB"f*.json $CHOICES -st max mean min --title #$SAVEFIG "ntrees.png"
# results 3 has two different seeds so just used results 3 for now
# when its done:
# results4/ntrees/results_*t_"$FPROB"f*.json

# FPROB
# though of doing max/mean, but min looks a little better and shows the sharp drop faster
./statistics.py -c -st max min -d $RESULTS_DIR/fprob/ --title -xl "failure probability"

# TOPOLOGIES
# inconclusive results from this one that just looks at different sizes  (do one for all 8h, and one for all others)
./statistics.py -x topo -c -xl "topology size (nbuildings, nhosts, nIBLs)" -st max min -d $RESULTS_DIR/topo-sizes
./statistics.py -x topo -c -xl "topology size (nbuildings, nhosts, nIBLs)" -st max min -d $RESULTS_DIR/topo-sizes8h
# this one needs more runs; probably wants to be a bar graph with errbars
./statistics.py -x topo -c -xl "topology size (nbuildings, nhosts, nIBLs)" -st max min -d $RESULTS_DIR/topo-ibl


# HEURISTICS (construction)
# comparison of all the heuristics?  fprob does this well...
# variances on steiner
./statistics.py -x ntrees -c -d results/steiner-double/ -st min max mean --title "comparing Steiner approx. strategies"

# HEURISTICS (choosing)
export CHOICES="-c importance-chosen max-overlap-chosen max-reachable-chosen min-missing-chosen"
# vary ntrees
./statistics.py -i unicast oracle $HEURISTIC -st max mean -x ntrees -d results4/ntrees/ $CHOICES --title
export HEURISTIC=red-blue;
# vary npubs: same as NHOSTS above???
./statistics.py -st max mean -x npublishers -f results3/nhosts/results_*_"$NSUBS"s_*p*.json -i unicast oracle $HEURISTIC -c importance-chosen max-overlap-chosen max-reachable-chosen min-missing-chosen  --title "choice heuristics comparison for $HEURISTIC heuristic"


# PUB ERROR RATE
./statistics.py -i unicast $HEURISTIC -st max mean -xl "publication loss rate" -x publication_error_rate -d results4/pub*rate/ $CHOICES --title


#######  METRICS

### Overlap
export XPARAM=overlap
export HEURISTIC="-i red-blue steiner diverse-paths oracle unicast"
export FILES=results3/nhosts/results_4t_0.10f_400s_200p_*.json results4/ntrees/results_4t_0.10f_400s_200p_*.json
./statistics.py -x $XPARAM --title -st max min -c -f $FILES
# also need to get overlap vs. ntrees working again...


# Cost


# n-hops




