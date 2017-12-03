import unittest
import os
import networkx as nx
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger('ride')
log.setLevel(logging.DEBUG)

from ride.ride_d import RideD
from topology_manager.networkx_sdn_topology import NetworkxSdnTopology

ALERT_TOPIC = 'alert'


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
                           # we don't actually send any packets so we just need a dummy callback
                           alert_sending_callback=lambda x,y: True)

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

        # register the subscribers
        self.subscribers = ['h0-b0', 'h0-b1', 'h1-b1', 'h0-b3']
        for sub in self.subscribers:
            self.rided.add_subscriber(sub, ALERT_TOPIC)

        self.alert = self.rided._make_new_alert("warning!", ALERT_TOPIC)

    def test_basic_mdmt_selection(self):
        """Tests MDMT-selection (without alerting context) for the default policy by manually assigning
        MDMTs, publisher routes, notifying RideD about a few publications and verifying that the selected MDMT is
        the one expected given this information."""

        for pub in self.publishers:
            self.rided.notify_publication(pub)

        mdmt = self.rided.get_best_mdmt(self.alert, heuristic=self.rided.MAX_LINK_IMPORTANCE)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')

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

        for pub in self.publishers:
            self.rided.notify_publication(pub)

        # NOTE: we only do this test for importance now since the others will have a tie between tree3/4
        mdmt = self.rided._do_send_alert(self.alert)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')

        # first, set context to be aware that both b0 hosts were alerted when this MDMT was used
        self.rided.notify_alert_response('h0-b1', self.alert, mdmt)
        self.rided.notify_alert_response('h1-b1', self.alert, mdmt)

        # now we should choose T2 instead...
        mdmt = self.rided._do_send_alert(self.alert)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree2')

        # because AlertContext tracks trees tried we should now use T3
        mdmt = self.rided._do_send_alert(self.alert)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree3')

        # furthermore, we should lastly try tree1 even though it had lowest importance!
        mdmt = self.rided._do_send_alert(self.alert)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree1')

        # lastly, despite T4 not having the highest metric anymore, we should go back to using it
        # since it's the least-recently selected one
        mdmt = self.rided._do_send_alert(self.alert)
        self.assertEqual(self.rided.get_address_for_mdmt(mdmt), 'tree4')

    ## helper functions
    def _build_mdmts(self, subscribers=None):
        mdmts = self.rided.build_mdmts(subscribers=subscribers)
        self.rided.mdmts = mdmts
        return mdmts

    # TODO: test RideD.send_alert including use of timeout/retries!

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
