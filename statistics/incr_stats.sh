#!/bin/bash

## Used this file to generate .csv files incrementally since it takes so long to create them with a lot of runs.
## Use statistics/collate_csvs.py to cat them back together

for NPUBS in 10 20 40 80;
do
  for POLICY in disjoint shortest;
  do
    echo $NPUBS pubs and $POLICY
		python statistics/smart_campus_statistics.py -f final_results/results/npubs-reroute/results_4t_0.10f_40s_"$NPUBS"p_*_"$POLICY"*.json -o final_results/mininet_results/analyzed/routing_policy_"$NPUBS"_"$POLICY".csv
		#ls final_results/results/npubs-reroute/results_4t_0.10f_40s_"$NPUBS"p_*_"$POLICY"*.[12].json 
  done
done


#for NTREES in 0 1 2 4 8;
#do
#  echo $NTREES trees
#  FNAME=final_results/results/ntrees/results_"$NTREES"t_0.10f_20s_10p_*.json
#  #FNAME=final_results/results/ntrees/results_"$NTREES"t_0.10f_20s_10p_*.[12].json
#  python tstats.py -f $FNAME -o final_results/mininet_results/analyzed/ntrees_"$NTREES".csv
#  #ls $FNAME
#done


#for FPROB in 0.00 0.05 0.10 0.20 0.30;
#do
#  echo $FPROB fprob
#  FNAME=final_results/results/fprob/results_4t_"$FPROB"f_20s_10p_*.json
#  #FNAME=final_results/results/fprob/results_4t_"$FPROB"f_20s_10p_*.[12].json
#  python statistics/smart_campus_statistics.py -f $FNAME -o final_results/mininet_results/analyzed/fprob_"$FPROB".csv
#  #ls $FNAME
#done