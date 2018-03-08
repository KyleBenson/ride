import numpy as np


# NOTE: keeping this separate so we can re-use it in stats, which is easier to run in a different virtualenv, without
# bringing in extra dependencies
def calculate_utility(delivery_rate, max_delivery_rate, delay, weight):
    """
    Calculates the utility for a particular subscription according to the specified parameters, which may be
    estimates or actual measured values.
    :param delivery_rate: rate of successful notification delivery to the subscriber
    :param max_delivery_rate: rate of original publications that match this subscription
    :param delay: end-to-end delay from publication to subscriber reception (in seconds)
    :param weight:
    :return: a non-negative number
    """

    # avoid division by 0 error
    # XXX: use numpy as truth value of a series is ambiguous
    if np.count_nonzero(max_delivery_rate) == 0:
        return 0.0

    # TODO: incorporate delay!  make sure we keep non-negative numbers!
    # ENHANCE: try other functions out?
    # NOTE: we add 1 to keep from getting domain error with log(0); also keeps result non-negative
    # NOTE: we use numpy.log so we can operate on vectors e.g. columns of a data frame in stats
    return weight * np.log(1.0 + delivery_rate / max_delivery_rate)