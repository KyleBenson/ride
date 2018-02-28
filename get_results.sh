#!/usr/bin/env bash

#BASE_DIR="~"
BASE_DIR="/home/dsm/scifire"

REMOTE='dsm@dsmride.ics.uci.edu'
SSH_CMD="ssh $REMOTE \"cd $BASE_DIR && tar -czvf results.tgz results\""
# alias not working
#SSH_CMD=dsmride...
SCP_CMD="scp $REMOTE:$BASE_DIR/results.tgz ."

# move existing results dir first
if [ -d results ]
then
  if [ -d results.bak ]
  then rm -rf results.bak
  fi
mv results results.bak
fi

cmd=$SSH_CMD
echo $cmd
eval $cmd

cmd=$SCP_CMD
echo $cmd
eval $cmd

# extract everything
tar -xzvf results.tgz