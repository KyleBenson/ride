#### Experimental control configuration parameters, many of which are specific to the SmartCampus seismic scenario.
#    You might need to change some for a specific installation.

# can just change this and not the others if everything is running under same user account...
DEFAULT_USER='vagrant'

# This will control a lot of delays and debugging
TESTING = False
WITH_LOGS = True  # output seismic client/server stdout to a log file

# When True, runs host processes in mininet with -OO command for optimized python code
OPTIMISED_PYTHON = not TESTING

# Disable some of the more verbose and unnecessary loggers
LOGGERS_TO_DISABLE = ('sdn_topology', 'topology_manager.sdn_topology', 'connectionpool', 'urllib3.connectionpool')

# Useful to blow away commands that never quit after Mininet closed
CLEANUP_CLIENTS_COMMAND="ps aux | grep '%s' | grep -v 'grep' | awk '{print $2;}' | xargs -n 1 kill -9"

##### Misc.
IGNORE_OUTPUT = ' > /dev/null 2>&1'

## Delays (in seconds) control when events happen, how long experiment lasts, and waiting for convergence
SEISMIC_EVENT_DELAY = 60 if not TESTING else 40  # seconds before the 'earthquake happens', i.e. sensors start sending data
TIME_BETWEEN_SEISMIC_EVENTS = 20 if not TESTING else 10  # for multiple earthquakes / aftershock events
SLEEP_TIME_BETWEEN_RUNS = 15 if not TESTING else 5 # give Mininet/OVS/ONOS a chance to reconverge after cleanup
# make sure this leaves enough time for all data_path failures, recoveries, and publishing additional
# seismic events so we can see the recovery!  Maybe just configure this dynamically in the future?
EXPERIMENT_DURATION = SEISMIC_EVENT_DELAY + 10 + TIME_BETWEEN_SEISMIC_EVENTS * (3 if TESTING else 4)
# whether to set static ARP and ping between all pairs or just cloud/edge servers
# set this to True if the controller topology doesn't seem to be including all expected hosts
ALL_PAIRS = False
# Since we're using the scale client for background traffic, we need to specify an interval between SensedEvents
# ENHANCE: base this on the traffic_generator_bandwidth parameter
IOT_CONGESTION_INTERVAL = 0.1
DEFAULT_TOPOLOGY_ADAPTER = 'onos'
SHORTEST_PATH_DISTANCE_METRIC = 'latency'

# Enumerate all of the parameters varied by the various experiments mostly for the purpose of grouping results in
# the statistics.py parsers
VARIED_PARAMETERS = ['const_alg', 'select_policy', 'reroute_policy', 'exp_type', 'error_rate', 'fprob', 'ntrees',
                     'npublishers', 'nsubscribers', 'topo', 'treatment', 'max_alert_retries']
