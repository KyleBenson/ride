import unittest

from firedex_algorithm_experiment import FiredexAlgorithmExperiment
from scifire.algorithms import NullFiredexAlgorithm
from scifire.algorithms.firedex_algorithm import FiredexAlgorithm
from scifire.firedex_configuration import FiredexConfiguration


class TestAlgorithms(unittest.TestCase):

    def test_null(self):
        """Verifies that requesting 0 priority levels generates the null algorithm."""

        exp = FiredexAlgorithmExperiment(algorithm='random', num_priority_levels=0)
        exp.generate_configuration()
        self.assertTrue(isinstance(exp.algorithm, NullFiredexAlgorithm))
        prios = exp.algorithm.get_topic_priorities(exp)
        self.assertTrue(len(prios) > 0, "problem: no subscriptions generated!")
        self.assertTrue(all(p == FiredexConfiguration.NO_PRIORITY for p in prios.values()))

    def test_utilities(self):

        alg = FiredexAlgorithm()
        self.assertEqual(0, alg.calculate_utility(100.0, 0, 5.0, 2.0), "utility when max_delivery_rate==0 should be 0!")

        self.assertGreater(2 * alg.calculate_utility(8.0, 20.0, 1.0, 3.0), alg.calculate_utility(16.0, 20.0, 1.0, 3.0),
                           "utility should be sub-linear! i.e. twice a utility should be more than the utility with twice its arrival rate")

        self.assertEqual(alg.calculate_utility(11.0, 11.0, 5.0, 2.0), alg.calculate_utility(111.0, 111.0, 5.0, 2.0),
                         "utility should be the same when max delivery rate is expected despite different total delivery rates!")

        self.assertEqual(alg.calculate_utility(11.0, 11.0, 5.0, 4.0), 2.0*alg.calculate_utility(111.0, 111.0, 5.0, 2.0),
                         "utility difference should just be the weight when max delivery rate is expected despite"
                         " different total delivery rates!")

        # TODO: what kinds of tests to run?  maybe set static utility weights?
        # verify we don't get errors when 0 subscriptions?  might raise an error for that in the future though...


    ###     GREEDY SPLIT

    def test_info_per_byte(self):
        from scifire.algorithms.greedy_split_firedex_algorithm import GreedySplitFiredexAlgorithm
        alg = GreedySplitFiredexAlgorithm()

        ### First, test that info_per_byte is higher when utility is higher
        class_util_weights = (2.0, 4.0)
        # subscribe to all topics since these tests aren't well-defined for non-subscribed topics
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(3.0, 3.0), topic_class_data_sizes=(100, 100),
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()

        any_sub = exp.arbitrary_subscriber
        self.assertLess(alg.info_per_byte(2, exp, any_sub), alg.info_per_byte(6, exp, any_sub))
        # check equality within a class
        self.assertEqual(alg.info_per_byte(0, exp, any_sub), alg.info_per_byte(1, exp, any_sub))
        self.assertEqual(alg.info_per_byte(6, exp, any_sub), alg.info_per_byte(7, exp, any_sub))

        ### Second, test that info_per_byte is lower when data rate or size is higher
        class_util_weights = (2.0, 2.0)
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(3.1, 3.0), topic_class_data_sizes=(100, 100),
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()

        any_sub = exp.arbitrary_subscriber
        self.assertLess(alg.info_per_byte(2, exp, any_sub), alg.info_per_byte(6, exp, any_sub))
        self.assertEqual(alg.info_per_byte(0, exp, any_sub), alg.info_per_byte(1, exp, any_sub))
        self.assertEqual(alg.info_per_byte(6, exp, any_sub), alg.info_per_byte(7, exp, any_sub))

        # now check data size...
        class_util_weights = (2.0, 2.0)
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(3.0, 3.0), topic_class_data_sizes=(111, 100),
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()

        any_sub = exp.arbitrary_subscriber
        self.assertLess(alg.info_per_byte(2, exp, any_sub), alg.info_per_byte(6, exp, any_sub))
        self.assertEqual(alg.info_per_byte(0, exp, any_sub), alg.info_per_byte(1, exp, any_sub))
        self.assertEqual(alg.info_per_byte(6, exp, any_sub), alg.info_per_byte(7, exp, any_sub))

    def test_even_split_groups(self):
        from scifire.algorithms.greedy_split_firedex_algorithm import GreedySplitFiredexAlgorithm
        alg = GreedySplitFiredexAlgorithm()

        ### check ngroups

        # 1 group
        self.assertEqual(alg._even_split_groups(range(5, 10), 1)[0], range(5, 10))

        # 3 groups
        split = alg._even_split_groups(range(5, 10), 3)
        self.assertLess(len(split[-1]), len(split[-2]),
                        "2nd to last group should have more items than the last group when len(items) is odd,"
                        " but got: %s" % str(split))

        ### check items

        self.assertEqual(len(alg._even_split_groups((), 3)), 3,
                        "empty items should give right dimension iterables")
        self.assertTrue(not any(alg._even_split_groups((), 3)),
                        "empty items should give right dimension iterables, but be empty!")

        #  too few items for ngroups should behave similarly and include only items in the first fill-able groups
        split = alg._even_split_groups((1, 2), 3)
        self.assertEqual(len(split[0]), len(split[1]), "too few items should only include items in first groups")
        self.assertGreater(len(split[0]), len(split[-1]), "too few items should only include items in first groups")
        self.assertEqual(len(split[-1]), 0, "too few items should only include items in first groups")

        # two more than ngroups should give 2 items in first two groups; one in others
        split = alg._even_split_groups(range(5), 3)
        self.assertEqual(len(split[0]), len(split[1]), "> ngroups items should only include items in first groups")
        self.assertGreater(len(split[0]), len(split[-1]), "> ngroups items should only include items in first groups")
        self.assertEqual(len(split[-1]), 1, "> ngroups items should only include items in first groups")
        self.assertEqual(len(split[0]), 2, "> ngroups items should only include items in first groups")

        # 0 groups = no prioritization?
        # NOTE: we just changed the experiment implementation to ensure this never happens by building Null alg instead...
        # self.assertEqual(len(alg._even_split_groups(range(3, 6), 0)), 1, "does no groups mean no prioritization?")

    def test_greedy_split(self):
        class_util_weights = (2.0, 3.0, 4.0)
        # subscribe to all topics since these tests aren't well-defined for non-subscribed topics
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.3, 0.3, 0.4), topic_class_sub_rates=(1.0,),
                                         draw_subscriptions_from_advertisements=False,
                                         num_priority_classes=3, num_net_flows=3,
                                         topic_class_advertisements_per_ff=(3,3,4), topic_class_advertisements_per_iot=(3,3,4),
                                         topic_class_pub_rates=(3.0, 3.0, 3.0), topic_class_data_sizes=(100, 100, 100),
                                         topic_class_utility_weights=class_util_weights,
                                         algorithm='greedy-split')
        exp.generate_configuration()

        prios = exp.algorithm.get_topic_priorities(exp)
        for t in range(6, 10):
            self.assertEqual(prios[t], 0, "higher-utility class topic %d did not have highest priority 0 but rather %d" % (t, prios[t]))
        for t in range(3, 6):
            self.assertEqual(prios[t], 1, "medium-utility class topic %d did not have highest priority 1 but rather %d" % (t, prios[t]))
        for t in range(3):
            self.assertEqual(prios[t], 2, "low-utility class topic %d did not have highest priority 2 but rather %d" % (t, prios[t]))

        # make sure we can handle different # priorities and net flows:

        # subscribe to all topics since these tests aren't well-defined for non-subscribed topics
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.3, 0.3, 0.4), topic_class_sub_rates=(1.0,),
                                         draw_subscriptions_from_advertisements=False,
                                         num_priority_levels=2, num_net_flows=3,
                                         topic_class_advertisements_per_ff=(3,3,4), topic_class_advertisements_per_iot=(3,3,4),
                                         topic_class_pub_rates=(3.0, 3.0, 3.0), topic_class_data_sizes=(100, 100, 100),
                                         topic_class_utility_weights=class_util_weights,
                                         algorithm='greedy-split')
        exp.generate_configuration()

        prios = exp.algorithm.get_topic_priorities(exp)
        for t in range(6, 10):
            self.assertEqual(prios[t], 0, "higher-utility class topic %d did not have highest priority 0 but rather %d" % (t, prios[t]))
        for t in range(3, 6):
            self.assertTrue(prios[t] == 0 or prios[t] == 1, "medium-utility class topic %d did not have priority 0 or 1 but rather %d" % (t, prios[t]))
        for t in range(3):
            self.assertEqual(prios[t], 1, "low-utility class topic %d did not have lowest priority 1 but rather %d" % (t, prios[t]))


        # make sure we can handle different # priorities and net flows:
        class_util_weights = (2.0, 3.0, 4.0, 5.0, 6.0)

        # subscribe to all topics since these tests aren't well-defined for non-subscribed topics
        exp = FiredexAlgorithmExperiment(num_topics=20, topic_class_weights=(0.2,), topic_class_sub_rates=(1.0,),
                                         draw_subscriptions_from_advertisements=False,
                                         num_priority_levels=3, num_net_flows=5,
                                         topic_class_advertisements_per_ff=(4,), topic_class_advertisements_per_iot=(4,),
                                         topic_class_pub_rates=(3.0,), topic_class_data_sizes=(100,),
                                         topic_class_utility_weights=class_util_weights,
                                         algorithm='greedy-split')
        exp.generate_configuration()

        prios = exp.algorithm.get_topic_priorities(exp)
        for t in range(12, 20):
            self.assertEqual(prios[t], 0, "higher-utility classes topic %d did not have highest priority 0 but rather %d" % (t, prios[t]))
        for t in range(4, 12):
            self.assertEqual(prios[t], 1, "medium-utility class topic %d did not have priority 1 but rather %d" % (t, prios[t]))
        for t in range(4):
            self.assertEqual(prios[t], 2, "low-utility class topic %d did not have lowest priority 2 but rather %d" % (t, prios[t]))


if __name__ == '__main__':
    unittest.main()
