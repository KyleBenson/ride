#!/bin/bash
#results_dir=overnight_results/*
results_dir=results
# check for errors in the logs that aren't the ones we always see when killing the client at end of experiment
grep -i error $results_dir/logs*/*/* | grep -v 'Network is unreach' | grep -v "<type 'exceptions.AttributeError'>: 'NoneType' object has no attribute 'EXCHANGE_LIFETIME'" | grep -v "<type 'exceptions.AttributeError'>: 'NoneType' object has no attribute 'timeout'" | grep -v "raise error(EBADF, 'Bad file descriptor')" | grep -v "list.remove(x): x not in list" | grep -v "'NoneType' object is not callable" | grep -v "'socket.error'>): error(9, 'Bad file descriptor'" | grep -v "error: [Errno 9] Bad file descriptor" | grep -v "\--raise-errors"
