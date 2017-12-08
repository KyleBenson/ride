import unittest
import os
from threading import Thread
from time import sleep

import networkx as nx
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('ride')
log.setLevel(logging.DEBUG)

from ride.ride_d import RideD
from topology_manager.networkx_sdn_topology import NetworkxSdnTopology

ALERT_TOPIC = 'alert'
ALERT_MSG = "warning!"
# timeout for between re-tries
TIMEOUT = 0.2

class TestMdmtSelection(unittest.TestCase):
    """Tests the RideD algorithms but NOT the SDN mechanisms"""

    def setUp(self):

        # Our test topology is a basic campus network topology (constructed with the campus_topo_gen.py script) with:
        # 4 core, 2 buildings per core, 2 hosts/building, and 2 inter-building links;
        # TODO: see example diagram to visualize the relevant parts
        topo_file = os.path.join(os.path.split(__file__)[0], 'test_topo.json')
        self.topology = NetworkxSdnTopology(topo_file)
        self.root = self.topology.get_servers()[0]
        # self.topology.draw()

        # set up some manual MDMTs by just building networkx Graphs using collections of links
        self.ntrees = 4
        self.mdmts = [# tree1
                      nx.Graph(((self.root, 'c0'), ('c0', 'c1'), ('c1', 'b0'), ('b0', 'h0-b0'),
                                ('c0', 'c2'), ('c2', 'b1'), ('b1', 'h0-b1'), ('b1', 'h1-b1'),
                                ('c2', 'b3'), ('b3', 'h0-b3'))),
                      # tree2
                      nx.Graph(((self.root, 'c3'), ('c3', 'c2'), ('c2', 'b1'), ('b1', 'h0-b1'), ('b1', 'h1-b1'),
                                ('b1', 'b0'), ('b0', 'h0-b0'), ('b0', 'c1'), ('c1', 'b5'), ('b5', 'b3'), ('b3', 'h0-b3'))),
                      # tree3
                      nx.Graph(((self.root, 'c0'), ('c0', 'c1'), ('c1', 'b0'), ('b0', 'h0-b0'),
                                ('b0', 'b1'), ('b1', 'h0-b1'), ('b1', 'h1-b1'),
                                (self.root, 'c3'), ('c3', 'c2'), ('c2', 'b3'), ('b3', 'h0-b3'))),
                      # tree4
                      nx.Graph(((self.root, 'c0'), ('c0', 'c1'), ('c1', 'b0'), ('b0', 'h0-b0'),
                                ('c2', 'b1'), ('b1', 'h0-b1'), ('b1', 'h1-b1'),
                                (self.root, 'c3'), ('c3', 'c2'), ('c2', 'b3'), ('b3', 'h0-b3')))
                      ]
        # self.topology.draw_multicast_trees(self.mdmts[2:3])
        mdmt_addresses = ['tree%d' % (d+1) for d in range(self.ntrees)]

        self.rided = RideD(topology_mgr=self.topology, ntrees=self.ntrees, dpid=self.root,
                           addresses=mdmt_addresses, tree_choosing_heuristic=RideD.MAX_LINK_IMPORTANCE,
                           # This test callback notifies us of subscribers reached and ensures the right MDMT was selected
                           alert_sending_callback=self.__send_alert_test_callback)

        # XXX: manually set the MDMTs to avoid calling RideD.update(), which will try to run SDN operations in addition
        # to creating the MDMTs using the construction algorithms
        self.rided.mdmts[ALERT_TOPIC] = self.mdmts
        for mdmt, addr in zip(self.mdmts, mdmt_addresses):
            self.rided.set_address_for_mdmt(mdmt, addr)

        # set up manual publisher routes
        self.publishers = ['h1-b5', 'h1-b1']
        self.publisher_routes = [['h1-b5', 'b5', 'c1', 'c0', self.root],
                                 ['h1-b1', 'b1', 'c2', 'c3', self.root]]
        for pub_route in self.publisher_routes:
            self.rided.set_publisher_route(pub_route[0], pub_route)
        for pub in self.publishers:
            self.rided.notify_publication(pub)

        # register the subscribers
        self.subscribers = ['h0-b0', 'h0-b1', 'h1-b1', 'h0-b3']
        for sub in self.subscribers:
            self.rided.add_subscriber(sub, ALERT_TOPIC)

        # We expect the MDMTs to be selected (via 'importance' policy) in this order for the following tests...
        self.expected_mdmts = [('tree4',), ('tree2',), ('tree3',), ('tree1',), ('tree2',), ('tree1', 'tree3', 'tree4')]
        # ... based on these subscribers being reached during each attempt.
        self.subs_reached_at_attempt = [('h0-b1', 'h1-b1'), #0
                                        tuple(), tuple(), tuple(), # 1-3 no responses...
                                        ('h0-b3',), #4
                                        ('h0-b0',) #5 ; all done!
                                        ]
        # NOTES about the test cases:
        # NOTE: we only do these tests for 'importance' since the others will have a tie between tree3/4
        #  we should choose tree2 second due to update about subs reached...
        #  because AlertContext tracks trees tried we should use tree3 third
        #  furthermore, we should lastly try tree1 even though it had lowest importance!
        #  then, we should try tree2 as the highest current importance after a notification since we've tried all of them
        #  finally, since we have a tie among all the others

        self.attempt_num = 0

        self.alert = self.rided._make_new_alert(ALERT_MSG, ALERT_TOPIC)

    def test_basic_mdmt_selection(self):
        """Tests MDMT-selection (without alerting context) for the default policy by manually assigning
        MDMTs, publisher routes, notifying RideD about a few publications and verifying that the selected MDMT is
        the one expected given this information."""

        mdmt = self.rided.get_best_mdmt(self.alert, heuristic=self.rided.MAX_LINK_IMPORTANCE)
        self.assertIn(self.rided.get_address_for_mdmt(mdmt), self.expected_mdmts[0])

        mdmt = self.rided.get_best_mdmt(self.alert, heuristic=self.rided.MAX_OVERLAPPING_LINKS)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')

        mdmt = self.rided.get_best_mdmt(self.alert, heuristic=self.rided.MIN_MISSING_LINKS)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')

        # TODO: ENHANCE: additional test...
        # Now, if we reset the STT and change the publisher routes we should get different MDMTs
        # self.rided.stt_mgr.reset()
        # self.rided.set_publisher_route('h1-b5', ['h1-b5', 'b5', 'c1', 'c0', self.root])
        #
        # for pub in self.publishers:
        #     self.rided.notify_publication(pub)
        #
        # mdmt = self.rided.get_best_mdmt(ALERT_TOPIC, heuristic=self.rided.MAX_OVERLAPPING_LINKS)
        # self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')
        #
        # mdmt = self.rided.get_best_mdmt(ALERT_TOPIC, heuristic=self.rided.MAX_LINK_IMPORTANCE)
        # self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')
        #
        # mdmt = self.rided.get_best_mdmt(ALERT_TOPIC, heuristic=self.rided.MIN_MISSING_LINKS)
        # self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')

    def test_mdmt_selection_with_context(self):
        """Tests MDMT-selection WITH alerting context in a similar manner to the basic tests.  Here we use an
        AlertContext object to change the MDMT choice based on claiming that some subscribers have already been alerted."""

        # NOTE: we're actually using the _do_send_alert method instead of manually recording and doing notifications.
        # The callback used to actually 'send the alert packet' (no network operations) will handle notifying subs.

        for attempt_num, subs_reached in enumerate(self.subs_reached_at_attempt):
            mdmt = self.rided._do_send_alert(self.alert)
            self.assertIn(self.rided.get_address_for_mdmt(mdmt), self.expected_mdmts[attempt_num])

    ####    TEST ACTUAL send_alert(...) API     ######

    def test_send_alert(self):
        """
        Tests the main send_alert API that exercises everything previously tested along with the retransmit
        capability.  This uses a custom testing callback instead of opening a socket and test servers to receive alerts.
        """

        expected_num_attempts = len(self.subs_reached_at_attempt)

        # Send the alert and ensure it took the right # retries
        alert = self.rided.send_alert(ALERT_MSG, ALERT_TOPIC, timeout=TIMEOUT, max_retries=expected_num_attempts + 1)
        sleep((expected_num_attempts + 1) * TIMEOUT)
        self.assertFalse(alert.active)
        self.assertEqual(self.attempt_num, expected_num_attempts)
        self.assertEqual(len(alert.subscribers_reached), len(self.subscribers))  # not all subs reached????

    def test_cancel_alert(self):
        """Ensure that cancelling alerts works properly by cancelling it before it finishes and verify that some
        subscribers remain unreached."""

        expected_num_attempts = len(self.subs_reached_at_attempt)

        alert = self.rided.send_alert(ALERT_MSG, ALERT_TOPIC, timeout=TIMEOUT, max_retries=expected_num_attempts + 1)

        # instead of waiting for it to finish, cancel the alert right before the last one gets sent
        sleep((expected_num_attempts - 1.5) * TIMEOUT)
        self.rided.cancel_alert(alert)
        sleep(TIMEOUT)

        # Now we should note that the last alert message wasn't sent!
        self.assertFalse(alert.active)
        self.assertEqual(self.attempt_num, expected_num_attempts - 1)
        self.assertEqual(len(alert.subscribers_reached), len(self.subscribers) - 1)

    def test_send_alert_unsuccessfully(self):
        expected_num_attempts = len(self.subs_reached_at_attempt)

        # since we set max_retries to be less than the number required this alert should stop early despite not reaching all subs
        alert = self.rided.send_alert(ALERT_MSG, ALERT_TOPIC, timeout=TIMEOUT, max_retries=expected_num_attempts - 2)
        sleep((expected_num_attempts + 1) * TIMEOUT)
        self.assertFalse(alert.active)
        self.assertEqual(self.attempt_num, expected_num_attempts - 1)
        self.assertEqual(len(alert.subscribers_reached), len(self.subscribers) - 1)  # not all subs reached????

    def __send_alert_test_callback(self, alert, mdmt):
        """
        Custom callback to handle verifying that the expected MDMT was used in between each attempt and
        notifies RideD of which subscribers were reached.
        :param alert:
        :type alert: RideD.AlertContext
        :param mdmt:
        :return:
        """

        self.assertTrue(alert.active, "__send_alert_test_callback should not fire if alert isn't active!")

        expected_mdmt = self.expected_mdmts[self.attempt_num]
        self.assertIn(self.rided.get_address_for_mdmt(mdmt), expected_mdmt,
                         "incorrect MDMT selected for attempt %d: expected one of %s but got %s" % (self.attempt_num, expected_mdmt, mdmt))

        for s in self.subs_reached_at_attempt[self.attempt_num]:
            # XXX: because this callback is fired while the alert's thread_lock is acquired, we have to do this
            # from inside another thread so that it will run after this callback returns.  Otherwise, deadlock!
            # self.rided.notify_alert_response(s, alert, mdmt)
            Thread(target=self.rided.notify_alert_response, args=(s, alert, mdmt)).start()

        self.attempt_num += 1

    # ENHANCE: test_send_alert_multi_threaded????
    # ENHANCE: test_send_alert_network_socket

    ## helper functions
    def _build_mdmts(self, subscribers=None):
        mdmts = self.rided.build_mdmts(subscribers=subscribers)
        self.rided.mdmts = mdmts
        return mdmts


class TestImportanceMetric(unittest.TestCase):
    """Tests the RideD 'max-link-importance' metric/algorithm"""

    def setUp(self):
        self.tree = nx.Graph(((0,1),(1,2),(1,3),(3,4),(3,5),(5,6),(0,7),(7,8),(8,9)))
        self.subscribers = [2, 4, 6, 9]
        self.root = 0

    def test_basic(self):
        """Basic test case where all leaves are subscribers"""
        imp_graph = RideD.get_importance_graph(self.tree, self.subscribers, self.root)
        self.assertEqual(imp_graph[0][1][RideD.IMPORTANCE_ATTRIBUTE_NAME], 3)
        self.assertEqual(imp_graph[1][3][RideD.IMPORTANCE_ATTRIBUTE_NAME], 2)
        for u, v in ((0, 7), (1, 2), (3, 4), (3, 5), (5, 6), (7, 8), (8, 9)):
            self.assertEqual(imp_graph[u][v][RideD.IMPORTANCE_ATTRIBUTE_NAME], 1)

    def test_subscriber_subset(self):
        """Tests case where not all leaf nodes are subscribers"""
        imp_graph = RideD.get_importance_graph(self.tree, [2, 4, 6], self.root)
        self.assertEqual(imp_graph[0][1][RideD.IMPORTANCE_ATTRIBUTE_NAME], 3)
        self.assertEqual(imp_graph[1][3][RideD.IMPORTANCE_ATTRIBUTE_NAME], 2)
        for u, v in ((1, 2), (3, 4), (3, 5), (5, 6)):
            self.assertEqual(imp_graph[u][v][RideD.IMPORTANCE_ATTRIBUTE_NAME], 1)

        for u, v in ((0, 7), (7, 8), (8, 9)):
            self.assertEqual(imp_graph[u][v][RideD.IMPORTANCE_ATTRIBUTE_NAME], 0)

    def test_internal_subscribers(self):
        """Tests case where some subscribers are non-leaf nodes"""
        subs = [2, 4, 6, 3]
        imp_graph = RideD.get_importance_graph(self.tree, subs, self.root)
        # Only these two links should have increased 'importance'
        self.assertEqual(imp_graph[0][1][RideD.IMPORTANCE_ATTRIBUTE_NAME], 4)
        self.assertEqual(imp_graph[1][3][RideD.IMPORTANCE_ATTRIBUTE_NAME], 3)

        for u, v in ((1, 2), (3, 4), (3, 5), (5, 6)):
            self.assertEqual(imp_graph[u][v][RideD.IMPORTANCE_ATTRIBUTE_NAME], 1)

        for u, v in ((0, 7), (7, 8), (8, 9)):
            self.assertEqual(imp_graph[u][v][RideD.IMPORTANCE_ATTRIBUTE_NAME], 0)

    def test_no_subscribers(self):
        """Tests case where we have no subscribers left"""
        imp_graph = RideD.get_importance_graph(self.tree, [], self.root)
        for u, v, imp in imp_graph.edges(data=RideD.IMPORTANCE_ATTRIBUTE_NAME):
            self.assertEqual(imp_graph[u][v][RideD.IMPORTANCE_ATTRIBUTE_NAME], 0)
        self.assertEqual(self.tree.number_of_edges(), imp_graph.number_of_edges())


if __name__ == '__main__':
    unittest.main()
