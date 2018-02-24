import unittest

from firedex_algorithm_experiment import FiredexAlgorithmExperiment


class TestExperimentConfiguration(unittest.TestCase):
    def test_subscriptions(self):
        # only one topic for each class
        exp = FiredexAlgorithmExperiment(num_topics=10, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(0.2, 0.2))
        subs = exp.generate_subscriptions()

        # ensure we have the right amount for each class
        class0 = list(exp.topics_for_class(0))
        class1 = list(exp.topics_for_class(1))
        self.assertEqual(len([t for t in subs if t in class0]), 1)
        self.assertEqual(len([t for t in subs if t in class1]), 1)

        # half topics for each class
        exp = FiredexAlgorithmExperiment(num_topics=20, topic_class_weights=(0.5, 0.5), topic_class_sub_rates=(0.5, 0.5),
                                         topic_class_sub_dists=({"dist": "uniform", "args": [0, 10]},
                                                                {"dist": "uniform", "args": [0, 10]}))
        subs = exp.generate_subscriptions()

        # ensure we have the right amount for each class
        class0 = list(exp.topics_for_class(0))
        class1 = list(exp.topics_for_class(1))
        self.assertEqual(len([t for t in subs if t in class0]), 5)
        self.assertEqual(len([t for t in subs if t in class1]), 5)

        # all topics for each class
        exp = FiredexAlgorithmExperiment(num_topics=20, topic_class_weights=(0.25, 0.75), topic_class_sub_rates=(1.0, 1.0),
                                         topic_class_sub_dists=({"dist": "uniform", "args": [0, 5]},
                                                                {"dist": "uniform", "args": [0, 15]}))
        subs = exp.generate_subscriptions()

        # ensure we have the right amount for each class
        class0 = list(exp.topics_for_class(0))
        class1 = list(exp.topics_for_class(1))
        self.assertEqual(len([t for t in subs if t in class0]), 5)
        self.assertEqual(len([t for t in subs if t in class1]), 15)

        # try Zipf distribution, which many pub-sub papers say is well-representative
        exp = FiredexAlgorithmExperiment(num_topics=20, topic_class_weights=(0.25, 0.75), topic_class_sub_rates=(0.5, 0.2),
                                         topic_class_sub_dists=({"dist": "zipf", "args": [2, -1]},
                                                                {"dist": "zipf", "args": [2, -1]}))
        subs = exp.generate_subscriptions()

        # ensure we have the right amount for each class
        class0 = list(exp.topics_for_class(0))
        class1 = list(exp.topics_for_class(1))
        self.assertEqual(len([t for t in subs if t in class0]), 2)
        self.assertEqual(len([t for t in subs if t in class1]), 3)

        # TODO: play with class weights and sub rates to check edge cases!


if __name__ == '__main__':
    unittest.main()
