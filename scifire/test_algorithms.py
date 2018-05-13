import unittest

from firedex_algorithm_experiment import FiredexAlgorithmExperiment
from scifire.algorithms import NullFiredexAlgorithm, ALL_ALGORITHMS
from scifire.algorithms.firedex_algorithm import FiredexAlgorithm
from scifire.firedex_configuration import FiredexConfiguration


class TestAlgorithms(unittest.TestCase):

    def test_null(self):
        """Verifies that requesting 0 priority levels generates the null algorithm."""

        exp = FiredexAlgorithmExperiment(algorithm='random', num_priority_levels=0)
        exp.generate_configuration()
        self.assertTrue(isinstance(exp.algorithm, NullFiredexAlgorithm))
        prios = exp.algorithm.get_subscription_priorities(exp)
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

    def test_experiment(self):
        """
        Integration test with basic experiment class that doesn't run any actual simulations.
        """
        for alg in ALL_ALGORITHMS:
            exp = FiredexAlgorithmExperiment(testing=True, algorithm=alg)
            exp.generate_configuration()
            # this should not record any results or even take long: it basically just generates configs
            exp.run_experiment()

            self.assertTrue(exp.algorithm.ros_okay(exp), "ro values not okay in exp on algorithm %s" % alg)

            # TODO: what else to verify it worked okay? 'error' not in results?


class TestHelpers(unittest.TestCase):
    """Tests various helper functions in the algorithm class."""

    def test_bandwidth_portions(self):

        ### First, test that even split is correct
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         num_ffs=1,
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(1.0, 1.0), topic_class_data_sizes=(100, 100))
        exp.generate_configuration()

        proportions = exp.algorithm.bandwidth_proportions(exp)
        self.assertEqual(proportions, [0.5, 0.5])


        ### Now try 4 subs
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(0.5, 0.5),
                                         draw_subscriptions_from_advertisements=False,
                                         num_ffs=3, bandwidth=100,  # ensure we don't apply drop rates!
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(1.0, 1.0), topic_class_data_sizes=(100, 100))
        exp.generate_configuration()

        proportions = exp.algorithm.bandwidth_proportions(exp)
        self.assertEqual(proportions, [0.25]*4)

        ### Finally, try an unequal split by increasing # subs for IC
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(0.5, 0.5),
                                         draw_subscriptions_from_advertisements=False, ic_sub_rate_factor=2,
                                         num_ffs=3, bandwidth=100,  # ensure we don't apply drop rates!
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(1.0, 1.0), topic_class_data_sizes=(100, 100))

        exp.generate_configuration()

        proportions = exp.algorithm.bandwidth_proportions(exp)
        self.assertEqual(proportions, [0.2, 0.2, 0.2, 0.4])


###     GREEDY SPLIT

class TestGreedySplit(unittest.TestCase):

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
        subs = exp.get_subscriptions(exp.arbitrary_subscriber)

        self.assertLess(alg.info_per_byte(subs[2], exp), alg.info_per_byte(subs[6], exp))
        # check equality within a class
        self.assertEqual(alg.info_per_byte(subs[0], exp), alg.info_per_byte(subs[1], exp))
        self.assertEqual(alg.info_per_byte(subs[6], exp), alg.info_per_byte(subs[7], exp))

        ### Second, test that info_per_byte is lower when data rate or size is higher
        class_util_weights = (2.0, 2.0)
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(3.1, 3.0), topic_class_data_sizes=(100, 100),
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()

        self.assertLess(alg.info_per_byte(subs[2], exp), alg.info_per_byte(subs[6], exp))
        self.assertEqual(alg.info_per_byte(subs[0], exp), alg.info_per_byte(subs[1], exp))
        self.assertEqual(alg.info_per_byte(subs[6], exp), alg.info_per_byte(subs[7], exp))

        # now check data size...
        class_util_weights = (2.0, 2.0)
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(5,5), topic_class_advertisements_per_iot=(5,5),
                                         topic_class_pub_rates=(3.0, 3.0), topic_class_data_sizes=(111, 100),
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()

        self.assertLess(alg.info_per_byte(subs[2], exp), alg.info_per_byte(subs[6], exp))
        self.assertEqual(alg.info_per_byte(subs[0], exp), alg.info_per_byte(subs[1], exp))
        self.assertEqual(alg.info_per_byte(subs[6], exp), alg.info_per_byte(subs[7], exp))

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

        ### tests on a few different configurations
        for items, ngroups in ((range(20), 7), (range(13), 4), (range(5), 5)):
            split = alg._even_split_groups(items, ngroups)
            self.assertEqual(len(split), ngroups)

            items2 = []
            for s in split:
                items2.extend(s)
            self.assertEqual(set(items), set(items2))

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

        prios = exp.algorithm.get_subscription_priorities(exp)

        for req in prios.keys():
            if req in range(6, 10):
                self.assertEqual(prios[req], 0, "higher-utility class topic %d did not have highest priority 0 but rather %d" % (req, prios[req]))
        for req in prios.keys():
            if req in range(3, 6):
                self.assertEqual(prios[req], 1, "medium-utility class topic %d did not have highest priority 1 but rather %d" % (req, prios[req]))
        for req in prios.keys():
            if req in range(3):
                self.assertEqual(prios[req], 2, "low-utility class topic %d did not have highest priority 2 but rather %d" % (req, prios[req]))

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

        prios = exp.algorithm.get_subscription_priorities(exp)
        for req in prios.keys():
            if req in range(6, 10):
                self.assertEqual(prios[req], 0, "higher-utility class topic %d did not have highest priority 0 but rather %d" % (req, prios[req]))
        for req in prios.keys():
            if req in range(3, 6):
                self.assertTrue(prios[req] == 0 or prios[req] == 1, "medium-utility class topic %d did not have priority 0 or 1 but rather %d" % (req, prios[req]))
        for req in prios.keys():
            if req in range(3):
                self.assertEqual(prios[req], 1, "low-utility class topic %d did not have lowest priority 1 but rather %d" % (req, prios[req]))


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

        prios = exp.algorithm.get_subscription_priorities(exp)
        for req in prios.keys():
            if req in range(12, 20):
                self.assertEqual(prios[req], 0, "higher-utility classes topic %d did not have highest priority 0 but rather %d" % (req, prios[req]))
        for req in prios.keys():
            if req in range(4, 12):
                self.assertEqual(prios[req], 1, "medium-utility class topic %d did not have priority 1 but rather %d" % (req, prios[req]))
        for req in prios.keys():
            if req in range(4):
                self.assertEqual(prios[req], 2, "low-utility class topic %d did not have lowest priority 2 but rather %d" % (req, prios[req]))


class TestDropPolicies(unittest.TestCase):

    def test_opt_alg_internal(self):
        """Directly runs the internal algorithm with matrix-based inputs."""
        import numpy as np
        import algorithms.opt_firedex_algorithm as opt

        ncopies = 1  # used for testing scalability
        alphas = np.array([1., 2., 4., 3., 2.] * ncopies)
        sub_flow_map = np.mat([
                                 [0, 1],
                                 [0, 1],
                                 [0, 1],
                                 [1, 0],
                                 [1, 0],
                                 ] * ncopies)
        e = 0.01

        ### First, with unsaturated queues
        lambdas = np.array([1., 2., 2., .25, 1.3])
        mus = np.array([4., 6., 8., 10., 12.])
        # print "RO SUM:", sum(l / m for l, m in zip(lambdas, mus))

        drop_rates = opt.do_run_opt_alg(alphas, lambdas, mus, e, sub_flow_map=sub_flow_map, ro_tolerance=0)
        drop_rates_per_sub = sub_flow_map * drop_rates
        drop_rates_per_sub = np.squeeze(np.asarray(drop_rates_per_sub.flatten()))

        final_lambdas = np.multiply((1.0 - drop_rates_per_sub), lambdas)
        # print "final lambdas:", final_lambdas, lambdas
        # print "final ros sum:", np.sum(final_lambdas / mus)
        self.assertTrue(np.allclose(final_lambdas, lambdas))

        ### Next, saturated queues
        lambdas = np.array([1., 2., 2., 2.5, 3.] * ncopies)
        mus = np.array([1.5, 3., 4., 5., 4.5] * ncopies)
        old_ro_sum = sum(l / m for l, m in zip(lambdas, mus))
        self.assertGreater(old_ro_sum, 1.0)
        # print "RO SUM:", old_ro_sum

        drop_rates = opt.do_run_opt_alg(alphas, lambdas, mus, e, sub_flow_map=sub_flow_map, ro_tolerance=0.01)
        drop_rates_per_sub = sub_flow_map * drop_rates
        drop_rates_per_sub = drop_rates_per_sub.flatten()

        final_lambdas = np.multiply((1.0 - drop_rates_per_sub), lambdas)
        ros_sum = np.sum(final_lambdas / mus)
        self.assertTrue(0.97 < ros_sum < 1.0)
        # print "final lambdas:", final_lambdas
        # print "final ros sum:", ros_sum


    # def test_opt_input_format(self):
    def test_opt_drop_rates(self):

        ### First, check unsaturated case
        class_util_weights = (2.0, 2.0)
        nsubs = 1
        npubs = 10
        ntopics = 10
        nprios = nflows = 2
        ro_tol = 0.0
        # to saturate, need to have bw/8 (to mbps) < ntopics*npubs*pub_rate*data_size = 3MB/S
        bw = 25
        pub_rates = (30.0, 30.0)
        data_sizes = (1000, 1000)

        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
            num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs-1, num_iots=npubs, bandwidth=bw,
            num_topics=ntopics, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
            draw_subscriptions_from_advertisements=False,
            topic_class_advertisements_per_ff=(0,0), topic_class_advertisements_per_iot=(ntopics/2,ntopics/2),
            topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
            topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        for dr in drops.values():
            self.assertEqual(dr, 0.0)

        ### Test for saturated queues, even drop rates
        bw = 12
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
            num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs-1, num_iots=npubs, bandwidth=bw,
            num_topics=ntopics, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
            draw_subscriptions_from_advertisements=False,
            topic_class_advertisements_per_ff=(0,0), topic_class_advertisements_per_iot=(ntopics/2,ntopics/2),
            topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
            topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        for dr in drops.values():
            self.assertAlmostEqual(dr, 0.5)

        ### Another test of even drop rates, but in this case due to increase in packet size being
        #  equaled out by increase in utility weight
        #   WARNING: this might be a bad test case to keep around long-term if we change e.g. utility function to include delay
        data_sizes = (500, 1500)
        class_util_weights = (1.0, 3.0)  # need this to force topic classes into separate priorities
        bw = 12
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
                                         num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs - 1,
                                         num_iots=npubs, bandwidth=bw,
                                         num_topics=ntopics, topic_class_weights=(0.5, 0.5),
                                         topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(0, 0),
                                         topic_class_advertisements_per_iot=(ntopics / 2, ntopics / 2),
                                         topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        for dr in drops.values():
            self.assertAlmostEqual(dr, 0.5, 4)

        ### Test uneven drop rates
        data_sizes = (500, 1500)
        class_util_weights = (1.0, 4.0)  # need this to force topic classes into separate priorities
        bw = 12
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
                                         num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs - 1,
                                         num_iots=npubs, bandwidth=bw,
                                         num_topics=ntopics, topic_class_weights=(0.5, 0.5),
                                         topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(0, 0),
                                         topic_class_advertisements_per_iot=(ntopics / 2, ntopics / 2),
                                         topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        self.assertGreater(drops[1], drops[0])

        ### Test uneven drop rates: so much more utility from second topic class that first is totally dropped
        data_sizes = (500, 1500)
        class_util_weights = (1.0, 400000.0)  # need this to force topic classes into separate priorities
        bw = 12
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
                                         num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs - 1,
                                         num_iots=npubs, bandwidth=bw,
                                         num_topics=ntopics, topic_class_weights=(0.5, 0.5),
                                         topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(0, 0),
                                         topic_class_advertisements_per_iot=(ntopics / 2, ntopics / 2),
                                         topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        self.assertAlmostEqual(drops[1], 1.0)

        ### Test for more flows/prios
        data_sizes = (500, 1500)
        class_util_weights = (1.0, 4.0)  # need this to force topic classes into separate priorities
        bw = 36
        ntopics = 30
        nflows = nprios = 3
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
                                         num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs - 1,
                                         num_iots=npubs, bandwidth=bw,
                                         num_topics=ntopics, topic_class_weights=(0.5, 0.5),
                                         topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(0, 0),
                                         topic_class_advertisements_per_iot=(ntopics / 2, ntopics / 2),
                                         topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        self.assertTrue(drops[0] < drops[1] < drops[2])

        ### Test for unequal flows/prios
        nflows = 4
        nprios = 2
        ntopics = 14
        bw = 12
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
                                         num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs - 1,
                                         num_iots=npubs, bandwidth=bw,
                                         num_topics=ntopics, topic_class_weights=(0.5, 0.5),
                                         topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(0, 0),
                                         topic_class_advertisements_per_iot=(ntopics / 2, ntopics / 2),
                                         topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        # since just two topic classes split up, should get about same drop rate within a topic class/priority class
        # HOWEVER, since the #flows causes an unequal distribution of topic classes into flows, we see a low-utility
        # topic in flow 1, hence it having a slightly higher drop rate than flow 0 but a much lower drop rate than flow 2
        self.assertTrue(drops[0] < drops[1])
        self.assertTrue(drops[1] < drops[2])
        self.assertEqual(drops[3], drops[2])

        # ENHANCE: need to configure with more topic classes so each flow is different, then can do this:
        # XXX: compare them pair-wise down the line
        # for dr1, dr2 in zip()
        # self.assertTrue(drops[0] < drops[1] < drops[2])

        ### Test for multiple subscribers: use uneven drop rates and compare flows within a subscriber
        data_sizes = (500, 1500)
        class_util_weights = (1.0, 4.0)  # need this to force topic classes into separate priorities
        nflows = nprios = 2
        bw = 24
        nsubs = 2
        ntopics = 10
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='opt'),
                                         num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs - 1,
                                         num_iots=npubs, bandwidth=bw,
                                         num_topics=ntopics, topic_class_weights=(0.5, 0.5),
                                         topic_class_sub_rates=(1.0, 1.0),
                                         draw_subscriptions_from_advertisements=False,
                                         topic_class_advertisements_per_ff=(0, 0),
                                         topic_class_advertisements_per_iot=(ntopics / 2, ntopics / 2),
                                         topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
                                         topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        sub0_flows = exp.net_flows_for_subscriber(exp.subscribers[0])
        sub1_flows = exp.net_flows_for_subscriber(exp.subscribers[1])

        self.assertGreater(drops[sub0_flows[1]], drops[sub0_flows[0]])
        self.assertGreater(drops[sub1_flows[1]], drops[sub1_flows[0]])

        ### Test not all topics subscribed?
        # ENHANCE: not sure how to make this deterministic... probably this path doesn't need to be checked?

        # print 'UTILS:', sum(exp.algorithm.estimate_utilities(exp))
        # print "DROPS:", drops
        # print exp.algorithm.get_req_flows(exp)

    def test_linear_drop_rates(self):
        ### First, check unsaturated case
        class_util_weights = (2.0, 2.0)
        nsubs = 1
        npubs = 10
        ntopics = 10
        nprios = nflows = 2
        ro_tol = 0.0
        # to saturate, need to have bw/8 (to mbps) < ntopics*npubs*pub_rate*data_size = 3MB/S
        bw = 25
        pub_rates = (30.0, 30.0)
        data_sizes = (1000, 1000)

        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='linear'),
            num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs-1, num_iots=npubs, bandwidth=bw,
            num_topics=ntopics, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
            draw_subscriptions_from_advertisements=False,
            topic_class_advertisements_per_ff=(0,0), topic_class_advertisements_per_iot=(ntopics/2,ntopics/2),
            topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
            topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        for dr in drops.values():
            self.assertEqual(dr, 0.0)

        ### Test for saturated queues
        # Note that we can only drop HALF of the traffic at most since prio0 is always drop rate 0!
        bw = 12.1  # needs to be just over half BW...
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='linear'),
            num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs-1, num_iots=npubs, bandwidth=bw,
            num_topics=ntopics, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
            draw_subscriptions_from_advertisements=False,
            topic_class_advertisements_per_ff=(0,0), topic_class_advertisements_per_iot=(ntopics/2,ntopics/2),
            topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
            topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        self.assertEqual(drops[0], 0.0)
        self.assertGreater(drops[1], 0.95)

    def test_flat_drop_rates(self):
        ### First, check unsaturated case
        class_util_weights = (2.0, 2.0)
        nsubs = 1
        npubs = 10
        ntopics = 10
        nprios = nflows = 2
        ro_tol = 0.0
        # to saturate, need to have bw/8 (to mbps) < ntopics*npubs*pub_rate*data_size = 3MB/S
        bw = 25
        pub_rates = (30.0, 30.0)
        data_sizes = (1000, 1000)

        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='flat'),
            num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs-1, num_iots=npubs, bandwidth=bw,
            num_topics=ntopics, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
            draw_subscriptions_from_advertisements=False,
            topic_class_advertisements_per_ff=(0,0), topic_class_advertisements_per_iot=(ntopics/2,ntopics/2),
            topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
            topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        for dr in drops.values():
            self.assertEqual(dr, 0.0)

        ### Test for saturated queues
        # we expect to drop half of packets if we set bw to half that required
        bw = 12
        exp = FiredexAlgorithmExperiment(algorithm=dict(algorithm='greedy', ro_tolerance=ro_tol, drop_policy='flat'),
            num_priority_levels=nprios, num_net_flows=nflows, num_ffs=nsubs-1, num_iots=npubs, bandwidth=bw,
            num_topics=ntopics, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(1.0, 1.0),
            draw_subscriptions_from_advertisements=False,
            topic_class_advertisements_per_ff=(0,0), topic_class_advertisements_per_iot=(ntopics/2,ntopics/2),
            topic_class_pub_rates=pub_rates, topic_class_data_sizes=data_sizes,
            topic_class_utility_weights=class_util_weights)
        exp.generate_configuration()
        drops = exp.algorithm.get_drop_rates(exp)

        for dr in drops.values():
            self.assertAlmostEqual(dr, 0.5, 3)


if __name__ == '__main__':
    unittest.main()
