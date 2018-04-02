import unittest

from scale_client.applications.event_storing_application import EventStoringApplication
from scale_client.core.client import ScaleClient, configure_logging
from scale_client.event_sinks.local_coap_event_sink import LocalCoapEventSink
from scale_client.networks.coap_server import CoapServer
from scale_client.sensors.dummy.random_virtual_sensor import RandomVirtualSensor
from scifire.scale.firedex_coap_subscriber import FiredexCoapSubscriber

# XXX: run the clients in separate threads since the client.run() call blocks
import threading

# XXX: if True, the tests will have shorter durations but this may cause some errors in the # events expected
#   due to e.g. CoAP forwarding events as an eventually consistent guarantee rather than per-message forwarding.
FAST_TESTS = False


class TestSubscriber(unittest.TestCase):
    """
    Test the FiredexSubscriber integration by running multiple ScaleClients: a subscriber and one publisher for each
    network flow. Each publisher runs an app to store all the published events (as does the subscriber).
    We verify the published events are received by the subscriber via the proper network connection (net flow).
    """

    # XXX: since a closed server doesn't always release the address instantly, don't re-use port #s!  Better testing anyways...
    TEST_PORT_NUM = 9000

    def setUp(self):
        # to test this, we need to build a client and have a stats app to subscribe
        self.quit_time = 15
        self.event_interval = 1.0
        self.nevents = 10
        args = ScaleClient.parse_args()  # default args
        args.log_level = 'debug'
        configure_logging(args)

        self.all_topics = ('fire', 'ice', 'wind')
        self.ports = [TestSubscriber.TEST_PORT_NUM + i for i in range(len(self.all_topics))]
        TestSubscriber.TEST_PORT_NUM = self.ports[-1] + 1
        self.publishers = []
        self.publishers_stats = []

        # each topic gets its own flow in (most of) the tests
        self.topic_flow_map = {t: i for i, t in enumerate(self.all_topics)}
        self.net_flows = [('127.0.0.1', port) for port in self.ports]
        self.assertEqual(len(self.topic_flow_map), len(self.net_flows))

        for port, top in zip(self.ports, self.all_topics):
            pub = ScaleClient(quit_time=self.quit_time, raise_errors=True)
            broker = pub.setup_broker()

            ## Uncomment to add a log sink if there's a problem
            pub.setup_reporter()
            reporter = pub.event_reporter

            self.publishers.append(pub)
            ev_gen_cfg = dict(topic=top, publication_period=self.event_interval, nevents=self.nevents)
            pub_app = RandomVirtualSensor(broker, event_generator=ev_gen_cfg, event_type=top)
            # pub_app = DummyVirtualSensor(broker, sample_interval=self.event_interval, event_type=top)
            self.publishers_stats.append(EventStoringApplication(broker, subscriptions=(top,)))

            # need to run a server and a local sink
            srv_name = "coap_srv_%s" % top
            srv = CoapServer(broker, port=port, server_name=srv_name, events_root="/events/")
            sink = LocalCoapEventSink(broker, server_name=srv_name)
            reporter.add_sink(sink)
            # sub_app = LogEventSink(broker)
            # reporter.add_sink(sub_app)
        self.assertEqual(len(self.publishers_stats), len(self.all_topics))

        self.subscriber = ScaleClient(quit_time=self.quit_time, raise_errors=True)
        broker = self.subscriber.setup_broker()
        self.subscriber_stats = EventStoringApplication(broker, subscriptions=self.all_topics)

    def run_clients(self):
        """Run the ScaleClient instances in separate threads."""

        threads = [threading.Thread(target=self.subscriber.run)]
        for pub in self.publishers:
            threads.append(threading.Thread(target=pub.run))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def assertEventsGathered(self, act_nevents, exp_nevents):
        """
        Helper method for ensuring the subscriber received the expected # events.  This is a bit of a HACK since some
        events don't make it to the subscriber...
        :param act_nevents:
        :param exp_nevents:
        :return:
        """
        if FAST_TESTS:
            # XXX: should get at least half of them, but there's always a few missing that weren't reported in time
            self.assertGreaterEqual(act_nevents, exp_nevents / 1.5)
        else:
            self.assertGreaterEqual(act_nevents, exp_nevents - 1)

    def test_coap_single_topic(self):
        """
        Test the FiredexCoapSubscriber integration configuring the following SCALE apps:
         - DummyVirtualSensor publishes events
         - CoapServer exposes these events on a non-well-known port
         - FiredexCoapSubscriber connects on that server's port to receive the requested topics
        :return:
        """

        topic = self.all_topics[0]
        port = self.ports[0]
        flow = self.net_flows[self.topic_flow_map[topic]]
        fdx_sub = FiredexCoapSubscriber(self.subscriber.broker, subscriptions=(topic,),
                                        net_flows=(flow,), static_topic_flow_map={topic: 0})

        self.run_clients()

        # verify all published/received events are for the correct topic ONLY!
        for ev in self.publishers_stats[0].events + self.subscriber_stats.events:
            self.assertEqual(ev.topic, topic)

        # verify # events generated/received
        # TODO: these tests are not very good.... need to ensure enough time for events to make it thru system!
        exp_nevents = len(self.publishers_stats[0].events)
        self.assertEqual(exp_nevents, self.nevents)
        act_nevents = len(self.subscriber_stats.events)
        self.assertGreater(exp_nevents, 2)
        self.assertGreater(act_nevents, 2)
        self.assertEventsGathered(act_nevents, exp_nevents)

    def test_coap_one_topic_per_flow(self):
        """
        TEST: now we need to check that the received events' source match with the expected port #
        :return:
        """

        fdx_sub = FiredexCoapSubscriber(self.subscriber.broker, subscriptions=self.all_topics,
                                        net_flows=self.net_flows, static_topic_flow_map=self.topic_flow_map)

        self.run_clients()

        act_nevents = len(self.subscriber_stats.events)
        self.assertGreater(act_nevents, 2)

        # verify subscriber received at least one event for every published topic
        events_captured = set(ev.topic for ev in self.subscriber_stats.events)
        self.assertEqual(events_captured, set(self.all_topics))

        for topic, pub_client, pub_stats in zip(self.all_topics, self.publishers, self.publishers_stats):

            # verify all published/received events are for the correct topic ONLY!
            for ev in pub_stats.events:
                self.assertEqual(ev.topic, topic)

            # verify # events generated/received
            exp_nevents = len(pub_stats.events)
            self.assertEqual(exp_nevents, self.nevents)
            self.assertGreater(exp_nevents, 2)
            this_act_nevents = len([ev for ev in self.subscriber_stats.events if ev.topic == topic])
            self.assertGreater(this_act_nevents, 2)
            self.assertEventsGathered(act_nevents, exp_nevents)

    # TODO: figure out how to test this if we have problems during integration... the way we build one Client for each
    # publisher / net flow would have to change so that a client has a single flow but multiple topics

    # def test_coap_topics_overlap_flows(self):
    #     """
    #     TEST: check that we can assign multiple topics to a flow properly
    #     :return:
    #     """
    #
    #     fdx_sub = FiredexCoapSubscriber(self.subscriber.broker, subscriptions=self.all_topics,
    #                                     net_flows=self.net_flows, static_topic_flow_map=self.topic_flow_map)
    #
    #     self.run_clients()
    #
    #     act_nevents = len(self.subscriber_stats.events)
    #     self.assertGreater(act_nevents, 2)
    #
    #     # verify subscriber received at least one event for every published topic
    #     events_captured = set(ev.topic for ev in self.subscriber_stats.events)
    #     self.assertEqual(events_captured, set(self.all_topics))
    #
    #     for topic, pub_client, pub_stats in zip(self.all_topics, self.publishers, self.publishers_stats):
    #
    #         # verify all published/received events are for the correct topic ONLY!
    #         for ev in pub_stats.events:
    #             self.assertEqual(ev.topic, topic)
    #
    #         # verify # events generated/received
    #         exp_nevents = len(pub_stats.events)
    #         self.assertEqual(exp_nevents, self.nevents)
    #         self.assertGreater(exp_nevents, 2)
    #         this_act_nevents = len([ev for ev in self.subscriber_stats.events if ev.topic == topic])
    #         self.assertGreater(this_act_nevents, 2)
    #         self.assertEventsGathered(act_nevents, exp_nevents)


if __name__ == '__main__':
    unittest.main()
