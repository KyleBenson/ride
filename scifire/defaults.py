# NOTE: stick to just IC (no FFs) as long as we're using the external queuing simulator since it doesn't do multiple subscribers!
DEFAULT_NUM_FFS = 0
DEFAULT_NUM_IOTS = 6
DEFAULT_NUM_PRIORITIES = 3
DEFAULT_NUM_NET_FLOWS = DEFAULT_NUM_PRIORITIES
DEFAULT_ALGORITHM = 'random'  # can add 'seed' by putting this in a dict-style config e.g. {'algorithm': 'random', 'seed': 5}
DEFAULT_RO_TOLERANCE = 0.8

DEFAULT_NUM_TOPICS = 20
# NOTES on topic classes: all arguments should be lists/tuples, but you can specify single-element lists to be expanded
#    to match the longest-length args e.g. highest # classes
DEFAULT_TOPIC_CLASS_WEIGHTS = (0.7, 0.3)  # sensor data, async events
# NOTES on DISTributions:
#   The experiment sets default 'args' for uniform and zipf: sets an upper range limit for the former to include
#       all topics; shifts zipf left to include 0 (if you want to not include 0 as the default would do, do e.g. 'args': [2, 0]
#   uniform generates values in range [args0, args0+args1]
#   zipf's second arg shifts the distribution, so make sure you do e.g. args=[2, -1] to e.g. allow selecting topic0!
DEFAULT_TOPIC_CLASS_DATA_SIZES = (10, 200)
# DEFAULT_TOPIC_CLASS_DATA_SIZES = ({'dist': 'expon', 'args': [20], 'lbound': 1, 'ubound': 10000},
#                                   {'dist': 'expon', 'args': [1000], 'lbound': 1, 'ubound': 10000})
# Event Publication rates in 'events/second'
#   this config varies sensor data between updates every 100ms to every 5 mins, async events much less frequent (500ms to 30mins)
# DEFAULT_TOPIC_CLASS_PUB_RATES = ({'dist': 'norm', 'args': [1], 'lbound': 1.0/300, 'ubound': 10},
#                                  {'dist': 'expon', 'args': [0.2], 'lbound': 1.0/(60*30), 'ubound': 2})
DEFAULT_TOPIC_CLASS_PUB_RATES = (1, 0.2)
DEFAULT_TOPIC_CLASS_PUB_DISTS = ({'dist': 'uniform'}, {'dist': 'uniform'})
# TODO: make these into distributions?
# TODO: could easily add a third class only for FF data!
# XXX: since we're currently just running on a single subscriber, let's just subscribe to ALL topics:
DEFAULT_TOPIC_CLASS_ADVERTISEMENTS_PER_FF = (DEFAULT_NUM_TOPICS * DEFAULT_TOPIC_CLASS_WEIGHTS[0],
                                             DEFAULT_NUM_TOPICS * DEFAULT_TOPIC_CLASS_WEIGHTS[1])
DEFAULT_TOPIC_CLASS_ADVERTISEMENTS_PER_IOT = DEFAULT_TOPIC_CLASS_ADVERTISEMENTS_PER_FF
# DEFAULT_TOPIC_CLASS_ADVERTISEMENTS_PER_FF = (10, 4)
# DEFAULT_TOPIC_CLASS_ADVERTISEMENTS_PER_IOT = (2, 1)

DEFAULT_TOPIC_CLASS_SUB_DISTS = ({'dist': 'uniform'}, {'dist': 'uniform'})
# average portion of topics in that class that a subscriber should request
# NOTE: make sure these are high enough that we'll actually generate subscriptions!  We round the actual #subs down...
DEFAULT_TOPIC_CLASS_SUB_RATES = (1, 1)
# the IC (FF #0) subscribes to this times many more topics than regular FFs
# ENHANCE: other skews for the IC?  e.g. all FF-published event topics, more data telemetry than events, etc.?
DEFAULT_IC_SUB_RATE_FACTOR = 1
DEFAULT_TOPIC_CLASS_SUB_DURATIONS = (float('inf'), float('inf'))
DEFAULT_TOPIC_CLASS_SUB_START_TIMES = (0, 0)

DEFAULT_TOPIC_CLASS_UTILITY_WEIGHTS = ({'dist': 'expon', 'args': [0.5], 'lbound': 0.01, 'ubound': 2.0},
                                       {'dist': 'expon', 'args': [1.0], 'lbound': 0.1, 'ubound': 4.0})

# We only really consider service rates for SDN switch outbound queue (i.e. due to bandwidth constraint).
# ENHANCE: consider actual service rates for the other queues
DEFAULT_MU = 64000.0

# todo: make a list of different topics and probably assign them different priorities/utilities?
IOT_DEV_TOPIC = 'sensor_data'

DEFAULT_VIDEO_RATE_MBPS = 1.0
DEFAULT_VIDEO_PORT = 5000

FIRE_EXPERIMENT_DURATION = 200
