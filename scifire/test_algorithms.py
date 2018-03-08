import unittest

from scifire.algorithms.firedex_algorithm import FiredexAlgorithm
from scifire.utilities import calculate_utility


class TestAlgorithms(unittest.TestCase):
    def test_utilities(self):

        alg = FiredexAlgorithm()
        self.assertEqual(0, calculate_utility(100.0, 0, 5.0, 2.0), "utility when max_delivery_rate==0 should be 0!")

        self.assertGreater(2 * calculate_utility(8.0, 20.0, 1.0, 3.0), calculate_utility(16.0, 20.0, 1.0, 3.0),
                           "utility should be sub-linear! i.e. twice a utility should be more than the utility with twice its arrival rate")

        # TODO: what kinds of tests to run?  maybe set static utility weights?
        # verify we don't get errors when 0 subscriptions?  might raise an error for that in the future though...


if __name__ == '__main__':
    unittest.main()
