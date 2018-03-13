from scifire.algorithms.null_firedex_algorithm import NullFiredexAlgorithm
from scifire.algorithms.random_firedex_algorithm import RandomFiredexAlgorithm
from scifire.algorithms.greedy_split_firedex_algorithm import GreedySplitFiredexAlgorithm


ALL_ALGORITHMS = ('random', 'null', 'split')


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
    elif algorithm in ('greedy-split', 'greedy', 'split', 'static', 'naive'):
        return GreedySplitFiredexAlgorithm(**kwargs)
    # TODO:
    # elif algorithm == 'opt':
    else:
        raise ValueError("unrecognized priority-assignment algorithm %s" % algorithm)