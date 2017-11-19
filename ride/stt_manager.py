# Manages the Successfully Traversed Topology (STT) data structure
import time

import networkx as nx


class SttManager(object):
    """
    Manages the Successfully Traversed Topology (STT) data structure.
    This mainly consists of being notified about a route that was
    successfully used at a specific time.
    """

    def __init__(self):
        super(SttManager, self).__init__()
        self.stt = nx.Graph()

    def route_update(self, route, at_time=None, is_up=True):
        """
        Updates the STT with the given route being up (or down) at a possibly-specified time that defaults to now.
        :param route:
        :param is_up:
        :param at_time:
        :return:
        """

        if at_time is None:
            at_time = time.time()

        links = zip(route, route[1:])
        for u, v in links:
            if is_up:
                self.stt.add_edge(u, v, update_time=at_time)
            else:
                self.stt.remove_edge(u, v)

    def get_stt(self):
        return self.stt

    def get_stt_edges(self):
        """
        :return: set of STT edges currently deemed up
        """
        edges = set(self.stt.edges())
        for u, v in list(edges):
            # NOTE: because we're using undirected graphs, we have to worry about
            # whether edge tuples are formatted (nodes ordered) properly, hence
            # we just add edges to the set object in both orders (u,v) and (v,u)
            edges.add((v, u))

        return edges

    def reset(self):
        """Resets the STT by clearing out all edges."""
        self.stt = nx.Graph()