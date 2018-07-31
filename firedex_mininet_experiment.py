#! /usr/bin/env python

CLASS_DESCRIPTION = '''Experiment that models a smart fire fighting effort and supporting IoT systems in a single
structure fire scenario.  The Incident Command Post (ICP) and Building Management System (BMS) coordinate data flows
by each running a data exchange broker (MQTT).  The BMS runs a SDN controller that the ICP can control through its
REST APIs.  IoT devices in the building, which represent either in-building devices or ones brought by FFs, report data
to the BMS data exchange or directly to the ICP broker.  These data flows are prioritized based on dynamic info
requirements specified by the IC.  The SDN data plane is adapted to prioritize certain 'event topics' over others
using some clever SDN mechanisms.'''

# @author: Kyle Benson
# (c) Kyle Benson 2018

import logging
log = logging.getLogger(__name__)

import requests
import json
import time
import os
import random
import argparse

from config import *
from scifire.defaults import *
from firedex_experiment import FiredexExperiment
from mininet_sdn_experiment import MininetSdnExperiment
from scale_client.core.client import make_scale_config_entry, make_scale_config


class FiredexMininetExperiment(MininetSdnExperiment, FiredexExperiment):

    def __init__(self, experiment_duration=FIRE_EXPERIMENT_DURATION,
                 with_eoc=False, with_black_box=False,
                 # HACK: kwargs just used for construction via argparse since they'll include kwargs for other classes
                 **kwargs):
        super(FiredexMininetExperiment, self).__init__(experiment_duration=experiment_duration, **kwargs)

        # Special hosts in our topology
        self.bms = None  # Building Management System    --  manages IoT devices and SDN
        self.eoc = None  # Emergency Operations Center   --  where regional cloud services run e.g. event processing
        self.with_eoc = with_eoc
        self.black_box = None     # simple subscriber near BMS that should receive all data for forensics
        self.with_black_box = with_black_box
        # Because we treat these 3 as servers, they'll get a switch installed for easy multihoming
        self.icp_sw = None
        self.bms_sw = None
        self.eoc_sw = None

        # Special switches in our topology
        # TODO: expand this into a tree network!  ideally, with some switches having wi-fi nodes
        self.bldg = None    # building's internal network
        self.inet = None    # represents Internet connection that allows ICP, BMS, EOC, and hosts to all communicate
        self.prioq_dummy_sw = None    # Adding a dummy switch between inet and bdlg so that we can add priority queues
        self.prioq_links = None       # List of links that require priority queues
        # ENHANCE: we'll add different specific heterogeneous networks e.g. sat, cell, wimax, wifi
        # ENHANCE: add switches for each of the special hosts?  not needed currently for 2-switch topo....

        # TODO: add other params to results['params'] ???

    @classmethod
    def get_arg_parser(cls, parents=(FiredexExperiment.get_arg_parser(add_help=False),
                                     MininetSdnExperiment.get_arg_parser()), add_help=True):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        arg_parser = argparse.ArgumentParser(parents=parents, add_help=add_help, conflict_handler='resolve')

        # set our own custom defaults for this experiment
        arg_parser.set_defaults(experiment_duration=FIRE_EXPERIMENT_DURATION)

        # experimental configuration parameters
        arg_parser.add_argument('--with-eoc', dest='with_eoc', action="store_true",
                                help='''Build network topology with EOC node.''')
        arg_parser.add_argument('--with-black-box', dest='with_black_box', action="store_true",
                                help='''Build network topology with black box (forensics database) node.''')

        # TODO: --test option to run some sort of integration test...

        # TODO: change description for parent args?  specifically, which links do we apply the default channel stats on?

        return arg_parser

    def setup_topology(self):
        # sets up controller and builds a bare mininet net
        super(FiredexMininetExperiment, self).setup_topology()

        # We use special 'coded addresses' to help debugging/understanding what's going on in log files
        # WARNING: need to have the mac addresses start with 'aa' or something similar so that the OS doesn't reject them!
        base_subnet = '10.128.%s/9'  # see note in config.py about why we use this... don't forget subnet mask!
        icp_subnet = base_subnet % '1.%d'
        ff_subnet = base_subnet % '10.%d'
        fire_mac = 'aa:ff:00:00:%s:%s'
        ff_mac = fire_mac % ('ff', "%.2x")
        bldg_subnet = base_subnet % '2.%d'
        iot_subnet = base_subnet % '20.%d'
        bldg_mac = 'aa:bb:00:00:%s:%s'
        iot_mac = bldg_mac % ('dd', '%.2x')
        eoc_subnet = base_subnet % '3.%d'
        eoc_mac = 'aa:ee:0c:00:00:%.2x'

        # 1. create all our special switches for the network itself
        self.bldg = self.add_switch('bldg', dpid=':'.join(['bb']*8))
        #self.inet = self.add_switch('inet', dpid='11:ee:77:00:00:00:00:00')
        #self.prioq_dummy_sw = self.add_switch('prioq', dpid=':'.join(['dd'] * 8))
        # TODO: icp_switch?

        # 2. create special host nodes
        self.icp, self.icp_sw = self.add_server(self.icp, ip=icp_subnet % 1, mac=fire_mac % ('cc', '01'))
        self.bms, self.bms_sw = self.add_server('bms', ip=bldg_subnet % 1, mac=bldg_mac % ('cc', '01'))
        if self.with_eoc:
            self.eoc, self.eoc_sw = self.add_server('eoc', ip=eoc_subnet % 1, mac=eoc_mac % 1)
        if self.with_black_box:
            self.black_box = self.add_host('bb', ip=bldg_subnet % 200, mac=bldg_mac % ('bb', 'bb'))

        # 3. create FF and IoT host nodes
        ffs = self.ffs
        self.ffs = []
        for i, f in enumerate(ffs):
            ff = self.add_host(f, ip=ff_subnet % i, mac=ff_mac % i)
            self.ffs.append(ff)

        iots = self.iots
        self.iots = []
        for i, d in enumerate(iots):
            iot = self.add_host(d, ip=iot_subnet % i, mac=iot_mac % i)
            self.iots.append(iot)

        # 4. create all the necessary links

        # NOTE: we apply default channel characteristics to inet links only
        # TODO: set up some for other links?  internal building topo should have some at least...
        self.prioq_dummy_sw = self.bms_sw
        # inet_connected_nodes = [self.icp_sw]
        # if self.with_eoc:
        #     inet_connected_nodes.append(self.eoc_sw)
        #
        # for peer in inet_connected_nodes:
        #     self.add_link(self.inet, peer)
        #     # QUESTION: should we save these links so that we can e.g. explicitly vary their b/w during the sim?

        # Connect special hosts in the building
        bldg_connected_nodes = [self.bms_sw]
        if self.with_black_box:
            bldg_connected_nodes.append(self.black_box)

        for bldg_comp in bldg_connected_nodes:
            self.add_link(self.bldg, bldg_comp, use_tc=False)

        # Get all the links to add priority queues
        links = self.get_links_between(self.prioq_dummy_sw, self.bldg) # Returns a list of links between the given two nodes
        self.prioq_links = links

        # Connect IoT and FF hosts
        # For now, we'll just add all FFs and IoT devs directly to the bldg
        # TODO: setup wifi network and/or hierarchical switches
        # TODO: set channel characteristics
        bldg_hosts = self.ffs + self.iots + [self.icp_sw]
        for h in bldg_hosts:
            self.add_link(self.bldg, h)

        # 5. add NAT so that any SdnTopology apps will be able to contact the SDN controller's REST API

        # NOTE: we use only a single NAT attached to the BMS so that we can also study control plane
        #    issues caused by channel contention when the ICP contacts the SDN controller.
        nat_ip = bldg_subnet % 250
        self.add_nat(self.bms_sw, nat_name='sdn_ctrl', nat_ip=nat_ip)

    def setup_experiment(self):
        """
        Set up the experiment and configure it as necessary before run_experiment is called.
        :return:
        """

        super(FiredexMininetExperiment, self).setup_experiment()

        # Setup priority queues on the links
        if self.prioq_links:
            self.setup_priority_queues_links(links=self.prioq_links,
                                             prio_levels=self.num_priority_levels,
                                             # XXX: expects it in bits per sec
                                             bandwidth=self.bandwidth_bytes() * 8)

        # Start the brokers first so that they're running by the time the clients start publishing
        self.run_brokers()
        self.run_scale_clients()

    def run_experiment(self):

        start_time = time.time()
        log.info("*** Starting Experiment at %f..." % start_time)

        log.debug("*** sleeping for %d secs while experiment runs..." % self.experiment_duration)
        time.sleep(self.experiment_duration)

        # ENABLE this to inspect the hosts during the experiment and run commands on them
        # from mininet.cli import CLI
        # CLI(self.net)

        log.info("*** Experiment complete!")

        cooldown_time = 20
        log.debug("sleeping %d seconds for procs to finish..." % cooldown_time)
        time.sleep(cooldown_time)

        results = dict(start_time=start_time)
        results.update(self.get_analytical_model_results())
        results.update(self.get_run_config_for_results_dict())

        return results

    def get_analytical_model_results(self):
        """
        Calculates the expected performance using the analytical model in order to determine its accuracy.
        :return: a dict of resulting expectations to be saved in 'results'
        """
        # XXX: just need to convert subscriber mininet Hosts into their names
        ret = super(FiredexMininetExperiment, self).get_analytical_model_results()
        new_ret = dict()
        for k, v in ret.items():
            new_ret[k] = {subscriber.name: data for subscriber, data in v.items()}
        return new_ret

    def get_run_config_for_results_dict(self):
        """
        :return:  a dict of configuration parameters for this run
        """
        # XXX: just need to convert subscriber mininet Hosts into their names
        ret = super(FiredexMininetExperiment, self).get_run_config_for_results_dict()
        # ret['subscriptions'] = {host.name: subs for host, subs in ret['subscriptions'].items()}
        ret['priorities'] = {host.name: v for host, v in ret['priorities'].items()}
        ret['drop_rates'] = {host.name: v for host, v in ret['drop_rates'].items()}
        return ret

    def run_brokers(self):

        # TODO: run VerneMQ broker instead of coap server
        # self.run_proc(vmq_cmd, self.icp, 'icp broker')
        # self.run_proc(vmq_cmd, self.bms, 'bms broker')

        all_topics = [str(t) for t in self.subscription_topics]
        coap_cfg = make_scale_config_entry(name="CoapServer", events_root="/events/", class_path="coap_server.CoapServer")
        # this will just count up the # events received with this topic
        event_log_cfg = make_scale_config_entry(name="EventsLog", subscriptions=all_topics,
                                                output_file=os.path.join(self.outputs_dir, "events_%s"),
                                                class_path="event_file_logging_application.EventFileLoggingApplication")

        # DYNAMICS: bring SdnApp back in so we can run the algorithm from there and manage the network accordingly
        # This is just for the ICP so that it can interact with the SDN Controller, which in our scenario we assume is near the BMS
        # sdn_cfg = make_scale_config_entry(name="SdnApp", topology_mgr=self._get_topology_manager_config(),
        #                                   class_path="topology_manager.scale_sdn_application.ScaleSdnApplication")

        # FUTURE: add in ICP and EOC too?  not sure what exactly we'll do with it though...
        # broker_nodes = (self.icp, self.bms)
        broker_nodes = (self.bms,)
        for broker_node in broker_nodes:
            hostname = "broker@%s" % broker_node.name
            broker_cfg = make_scale_config(networks=coap_cfg, applications=event_log_cfg)
            # applications=event_log_cfg + (sdn_cfg if broker_node is self.icp else ''))
            cmd = self.make_host_cmd(broker_cfg % hostname, hostname)
            self.run_proc(cmd, broker_node, hostname)

    def run_scale_clients(self):
        """
        Configure host devices with SCALE clients to publish sensor data to the data exchange.
        :return:
        """

        #### boilerplate taken from smart campus exp

        # ENHANCE: move this to make_host_cmd()????

        # HACK: Need to set PYTHONPATH since we don't install our Python modules directly and running Mininet
        # as root strips this variable from our environment.
        env = os.environ.copy()
        ride_dir = os.path.dirname(os.path.abspath(__file__))
        if 'PYTHONPATH' not in env:
            env['PYTHONPATH'] = ride_dir + ':'
        else:
            env['PYTHONPATH'] = env['PYTHONPATH'] + ':' + ride_dir

        outputs_dir, logs_dir = self.build_outputs_logs_dirs()
        #####

        system_state = [ ]

        # Building IoT devices all publish their periodic event data to the BMS broker.
        broker_ip = self.bms.IP()
        for host in self.all_hosts:
            # NOTE: make sure we distinguish the coapthon client instances from each other by incrementing CoAP port for multiple topics!
            hostname = "scale-%s" % host.name

            # start with default values and add all the components we want for this host
            sensor_configs = ""
            sink_configs = ""
            app_configs = ""

            # PUBLISHERS run a VirtualSensor for each advertised topic that publishes events and stores them in a file.
            # Also run an EventSink to report these events to remote broker.
            if host in self.publishers:
                # NOTE: scale expects topics to be strings!
                ads = [str(t) for t in self.get_advertisements(host)]
                # XXX: converting topics back to ints for indexing... this might break if we switch to fully-string topics!
                data_sizes = {t: self.data_sizes[int(t)] for t in ads}
                 # TODO: set bounds so we can guarantee it fits in CoAP packet!  maybe we need to have a 'lightweight' SensedEvent transmitted with fewer fields??
                data_size_bounds = (1,500)

                sensor_configs += "".join([make_scale_config_entry(name="IoTSensor_%s_%s" % (host.name, topic), event_type=topic,
                                    event_generator=dict(topic=topic, publication_period=self.get_publication_period(topic),
                                                         data_size_bounds=data_size_bounds, data_size=data_sizes[topic],
                                                         total_time=self.experiment_duration),
                                    class_path="dummy.random_virtual_sensor.RandomVirtualSensor",
                                    output_events_file=os.path.join(self.outputs_dir,
                                                                    'publisher_%s_%s' % (topic, host.name)),
                                    # give servers a chance to start; spread out their reports too
                                    start_delay=random.uniform(5, 10))
                                  for topic in ads])

                # TODO: selectively enable confirmable messages?  we have a reliable_tx var somewhere...
                # forward these SensedEvents to the broker best-effort (non-CONfirmable) for CoAP
                sink_configs = make_scale_config_entry(class_path="remote_coap_event_sink.RemoteCoapEventSink",
                                        name="IotDataCoapEventSink", hostname=broker_ip,
                                        src_port=COAP_CLIENT_BASE_SRC_PORT,
                                        topics_to_sink=ads, confirmable_messages=False)
                                        # Can optionally enable this to print out each event in its entirety.
                                        # + make_scale_config_entry(class_path="log_event_sink.LogEventSink", name="LogSink")

            # SUBSCRIBERS run a subscribing VirtualSensor and an Application for saving received events
            if host in self.subscribers:
                # NOTE: scale expects topics to be strings!
                subs = [str(t) for t in self.get_subscription_topics(host)]

                req_flow_map = self.algorithm.get_subscription_net_flows(self, subscriber=host)
                flow_prio_map = self.algorithm.get_net_flow_priorities(self, subscriber=host)
                drop_rates = self.algorithm.get_drop_rates(self, subscriber=host)

                # XXX: convert our req_flow_map into a topic-to-flow index map.
                # since net flows are unique per subscriber, we need to subtract the minimum flow # from each for this
                #   subscriber to start the flow indices at 0
                min_sim_flow = min(*req_flow_map.values())
                topic_flow_map = {req.topic: (f - min_sim_flow) for req, f in req_flow_map.items()}

                # XXX: let destination port be the default one
                real_flows = [(None, COAP_CLIENT_BASE_SRC_PORT + i, broker_ip, None) for i in range(self.num_net_flows)]

                for i in range(real_flows.__len__()):
                    real_flow = real_flows[i]
                    system_state.append( { "ip": host.params["ip"][:-2], "port": real_flow[1], "priority": flow_prio_map[i], "drop_rate": drop_rates[i] } )

                # TODO: make sure that the two types of flows match up as expected!
                #    else might assign wrong priority to them... should probably be okay since everything sequential

                sensor_configs += make_scale_config_entry(name="FdxSubscriber", subscriptions=subs,
                                                          net_flows=real_flows, static_topic_flow_map=topic_flow_map,
                                                          # TODO: make this work for all Apps not just DummyVS?
                                                          start_delay=random.uniform(15, 20),
                                    class_path="scifire.scale.firedex_coap_subscriber.FiredexCoapSubscriber")

                app_configs += make_scale_config_entry(name="EventStore", subscriptions=subs,
                                        class_path="event_file_logging_application.EventFileLoggingApplication",
                                        output_file=os.path.join(outputs_dir, 'subscriber_%s' % hostname))

            host_cfg = make_scale_config(
                sensors=sensor_configs if sensor_configs else None,
                sinks=sink_configs if sink_configs else None,
                applications=app_configs if app_configs else None,
            )

            cmd = self.make_host_cmd(host_cfg, hostname)
            self.run_proc(cmd, host, hostname)

        url = "http://127.0.0.1:8080/api/flow/push-configuration/"
        data = json.dumps(system_state)
        response = requests.post(url, data=data)
        content = response.json()
        #print(content)


FiredexMininetExperiment.__doc__ = CLASS_DESCRIPTION


if __name__ == "__main__":
    import sys
    exp = FiredexMininetExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()
