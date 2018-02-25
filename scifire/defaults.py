DEFAULT_NUM_FFS = 3
DEFAULT_NUM_IOTS = 6
DEFAULT_NUM_PRIORITIES = 3
DEFAULT_NUM_NET_FLOWS = DEFAULT_NUM_PRIORITIES

DEFAULT_NUM_TOPICS = 5
DEFAULT_TOPIC_CLASS_WEIGHTS = (0.7, 0.3)  # sensor data, async events
# NOTES on DISTributions:
#   The experiment sets default 'args' for uniform and zipf: sets an upper range limit for the former to include
#       all topics; shifts zipf left to include 0 (if you want to not include 0 as the default would do, do e.g. 'args': [2, 0]
#   uniform generates values in range [args0, args0+args1]
#   zipf's second arg shifts the distribution, so make sure you do e.g. args=[2, -1] to e.g. allow selecting topic0!
DEFAULT_TOPIC_CLASS_DATA_SIZES = ({'dist': 'expon', 'args': [100], 'lbound': 1, 'ubound': 10000},
                                  {'dist': 'expon', 'args': [1000], 'lbound': 1, 'ubound': 10000})
DEFAULT_TOPIC_CLASS_PUB_RATES = ({'dist': 'norm', 'args': [1], 'lbound': 0.01, 'ubound': 100},
                                 {'dist': 'expon', 'args': [50], 'lbound': 1, 'ubound': 1000})
DEFAULT_TOPIC_CLASS_PUB_DISTS = ({'dist': 'uniform'}, {'dist': 'uniform'})
# TODO: make these into distributions?
# TODO: could easily add a third class only for FF data!
DEFAULT_TOPIC_CLASS_ADVERTISEMENTS_PER_FF = (2, 4)
DEFAULT_TOPIC_CLASS_ADVERTISEMENTS_PER_IOT = (2, 0)

DEFAULT_TOPIC_CLASS_SUB_DISTS = ({'dist': 'uniform'}, {'dist': 'uniform'})
# average portion of topics in that class that a subscriber should request
DEFAULT_TOPIC_CLASS_SUB_RATES = (0.2, 0.5)
# the IC (FF #0) subscribes to this times many more topics than regular FFs
# ENHANCE: other skews for the IC?  e.g. all FF-published event topics, more data telemetry than events, etc.?
DEFAULT_IC_SUB_RATE_FACTOR = 10
DEFAULT_TOPIC_CLASS_SUB_DURATIONS = (float('inf'), float('inf'))
DEFAULT_TOPIC_CLASS_SUB_START_TIMES = (0, 0)

# TODO: make a list of different topics and probably assign them different priorities/utilities?
IOT_DEV_TOPIC = 'sensor_data'

DEFAULT_VIDEO_RATE_MBPS = 1.0
DEFAULT_VIDEO_PORT = 5000

FIRE_EXPERIMENT_DURATION = 15
