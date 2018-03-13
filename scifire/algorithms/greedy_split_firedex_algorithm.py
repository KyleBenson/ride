
from firedex_algorithm import FiredexAlgorithm
from ..firedex_configuration import FiredexConfiguration

import logging
log = logging.getLogger(__name__)


class GreedySplitFiredexAlgorithm(FiredexAlgorithm):
    """
    Algorithm implementation that assigns priorities by sorting subscription according to their "info/byte" metric,
    which is max utility / (pub rate * pub size).
    """

    def info_per_byte(self, topic, configuration, subscriber):
        """
        Calculate the info/byte metric: max utility / (pub rate * pub size)
        :param topic:
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber: which subscriber we're considering the utility function of for this topic
        :return:
        """

        # TODO: maybe test out different approaches to this method?
        # QUESTION: maybe we shouldn't be actually calculating utility but just use the weight?  it should be same for
        #   any topic since all utility calculations are done with max rate assumed...
        # Similarly, we should maybe not be scaling by pub rate since utility already accounts for that...
        #   maybe instead calculate incremental additional info/byte according to some decision?

        pub_rate = self.publication_rates(configuration)[topic]
        # XXX: use 0 delay, which in the future we may assume is impossible hence making an epsilon value
        delay = 0.000001
        weight = configuration.get_utility_weight(topic, subscriber)
        max_util = self.calculate_utility(pub_rate, pub_rate, delay, weight)

        pub_size = configuration.data_sizes[topic]
        info_per_byte = max_util / (pub_rate * pub_size)
        # info_per_byte = weight / (pub_rate * pub_size)

        return info_per_byte

    def sorted_subscriptions(self, configuration, subscriber):
        """
        Sorts the subscriptions (in reverse order) for this subscriber by its info/byte.
        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscriber:
        :return:
        """

        subs = configuration.get_subscriptions(subscriber)
        ipbs = sorted(((self.info_per_byte(sub, configuration, subscriber), sub) for sub in subs), reverse=True)
        return [sub for ipb, sub in ipbs]

    def _even_split_groups(self, items, ngroups):
        """
        Evenly splits up items into the requested ngroups.  Unevenly split items will result in more items in the
        first group(s)
        :param items:
        :type items: list|tuple
        :param ngroups:
        :type ngroups: int
        :return:
        :rtype: list[list]
        """

        if ngroups <= 0:
            raise ValueError("cannot split items into <= 0 groups: requested %d" % ngroups)

        # account for uneven splits by creating an explicit list of sizes for each group
        group_sizes = [len(items) / ngroups] * ngroups
        # add extras to the first groups until everything accounted for
        for i in range((len(items) % sum(group_sizes)) if sum(group_sizes) > 0 else len(items)):
            group_sizes[i] += 1

        # successively slice each group out of items
        ret = []
        idx = 0
        for gsize in group_sizes:
            ret.append(items[idx:idx + gsize])
            idx += gsize

        return ret

    def _run_algorithm(self, configuration, subscribers=None):
        """
        Sorts the subscriptions for each subscriber by their info/byte and then break them into even groups for
         assigning net flows and consequently priorities.

        :param configuration:
        :type configuration: FiredexConfiguration
        :param subscribers:
        :return:
        """

        if subscribers is None:
            subscribers = configuration.subscribers

        for sub in subscribers:
            flows = configuration.net_flows_for_subscriber(sub)
            reqs = self.sorted_subscriptions(configuration, sub)
            reqs = self._even_split_groups(reqs, len(flows))

            for flow, req_group in zip(flows, reqs):
                for req in req_group:
                    self.set_topic_net_flow(req, flow, configuration, subscriber=sub)

            prios = configuration.prio_classes
            flows_for_prios = self._even_split_groups(flows, len(prios))
            for fs, p in zip(flows_for_prios, prios):
                for f in fs:
                    self.set_net_flow_priority(f, p, configuration, subscriber=sub)
