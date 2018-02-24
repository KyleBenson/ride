import unittest
from firedex_scenario import FiredexScenario


class TestTopics(unittest.TestCase):
    def test_topic_class_sizes(self):
        ntopics = 10
        topic_class_weights = (0.5, 0.5)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        self.assertEqual(scen.ntopics_per_class[0], 5)
        self.assertEqual(scen.ntopics_per_class[1], 5)

        ntopics = 10
        topic_class_weights = (0.2, 0.8)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        self.assertEqual(scen.ntopics_per_class[0], 2)
        self.assertEqual(scen.ntopics_per_class[1], 8)

        # uneven number of topics should add remainders to first topic class
        ntopics = 11
        topic_class_weights = (0.5, 0.5)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        self.assertEqual(scen.ntopics_per_class[0], 6)
        self.assertEqual(scen.ntopics_per_class[1], 5)

        # edge cases:

        # all in one class
        ntopics = 11
        topic_class_weights = (1.0,)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        self.assertEqual(scen.ntopics_per_class[0], 11)

        # non-1.0 weights
        ntopics = 5
        topic_class_weights = (0.3, 0.5)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        self.assertEqual(scen.ntopics_per_class[0], 3)
        self.assertEqual(scen.ntopics_per_class[1], 2)

        ntopics = 5
        topic_class_weights = (0.0, 0.6)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        self.assertEqual(scen.ntopics_per_class[0], 2)
        self.assertEqual(scen.ntopics_per_class[1], 3)

    def test_topic_generation(self):
        """Tests the 'names' of topics when we just treat them as ID #s starting at 0."""

        ntopics = 10
        topic_class_weights = (0.2, 0.8)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        for t_exp, t_act in zip(range(ntopics), scen.topics):
            self.assertEqual(t_exp, t_act, "all topics not a list of [0..ntopics-1] !!")

        for t_exp, t_act in zip(range(2), list(scen.topic_classes)[0]):
            self.assertEqual(t_exp, t_act, "topic class %s not containing expected topic ID range %s" % (t_act, t_exp))
        for t_exp, t_act in zip(range(2, ntopics), list(scen.topic_classes)[1]):
            self.assertEqual(t_exp, t_act, "topic class %s not containing expected topic ID range %s" % (t_act, t_exp))

        # edge case with uneven #s
        ntopics = 11
        topic_class_weights = (0.2, 0.8)
        scen = FiredexScenario(num_topics=ntopics, topic_class_weights=topic_class_weights)
        for t_exp, t_act in zip(range(ntopics), scen.topics):
            self.assertEqual(t_exp, t_act, "all topics not a list of [0..ntopics-1] !!")

        for t_exp, t_act in zip(range(3), list(scen.topic_classes)[0]):
            self.assertEqual(t_exp, t_act, "topic class %s not containing expected topic ID range %s" % (t_act, t_exp))
        for t_exp, t_act in zip(range(3, ntopics), list(scen.topic_classes)[1]):
            self.assertEqual(t_exp, t_act, "topic class %s not containing expected topic ID range %s" % (t_act, t_exp))

    # TODO:
    # def test_topic_string_generation(self):


if __name__ == '__main__':
    unittest.main()
