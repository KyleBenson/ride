from scifire.algorithms.null_firedex_algorithm import NullFiredexAlgorithm
from scifire.algorithms.random_firedex_algorithm import RandomFiredexAlgorithm

ALL_ALGORITHMS = ('random', 'null')
# TODO: static, naive, sophisticated ones?


def build_algorithm(algorithm='random', **kwargs):
    """
    Builds and returns a concrete FiredexAlgorithm class for the requested algorithm and configuration.
    :param algorithm:
    :param kwargs:
    :return:
    :rtype: FiredexAlgorithm
    """

    if algorithm == 'random':
        return RandomFiredexAlgorithm(**kwargs)
    elif algorithm == 'null':
        return NullFiredexAlgorithm(**kwargs)
    # TODO: assign them based on static utility functions e.g. maybe order topics by utility and break into even prio groups?
    # elif algorithm == 'static':
    # TODO: assign static priorities without considering the analytical model i.e. just relative utilities?
    # elif algorithm == 'naive':
    else:
        raise ValueError("unrecognized priority-assignment algorithm %s" % algorithm)