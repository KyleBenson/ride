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

import time
import os
import random
import argparse

from config import *
from scifire.config import *
from firedex_algorithm_experiment import FiredexAlgorithmExperiment
from mininet_sdn_experiment import MininetSdnExperiment
from scale_client.core.client import make_scale_config_entry, make_scale_config

class FiredexMininetExperiment(MininetSdnExperiment, FiredexAlgorithmExperiment):

    def __init__(self, experiment_duration=FIRE_EXPERIMENT_DURATION,
                 # HACK: kwargs just used for construction via argparse since they'll include kwargs for other classes
                 **kwargs):
        super(FiredexMininetExperiment, self).__init__(experiment_duration=experiment_duration, **kwargs)

        # Special hosts in our topology
        self.icp = None  # Incident Command Post         --  where we want to collect data for situational awareness
        self.bms = None  # Building Management System    --  manages IoT devices and SDN
        self.eoc = None  # Emergency Operations Center   --  where regional cloud services run e.g. event processing
        self.black_box = None     # simple subscriber near BMS that should receive all data for forensics
        # Because we treat these 3 as servers, they'll get a switch installed for easy multihoming
        self.icp_sw = None
        self.bms_sw = None
        self.eoc_sw = None

        self.ffs = []
        self.iot_devs = []

        # Special switches in our topology
        # TODO: expand this into a tree network!  ideally, with some switches having wi-fi nodes
        self.bldg = None    # building's internal network
        self.inet = None    # represents Internet connection that allows ICP, BMS, EOC, and hosts to all communicate
        # ENHANCE: we'll add different specific heterogeneous networks e.g. sat, cell, wimax, wifi
        # ENHANCE: add switches for each of the special hosts?  not needed currently for 2-switch topo....

        # TODO: add other params to results['params'] ???

    @classmethod
    def get_arg_parser(cls, parents=(FiredexAlgorithmExperiment.get_arg_parser(add_help=False),
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
        # TODO: --test option to run some sort of integration test...

        # TODO: change description for parent args?  specifically, which links do we apply the default channel stats on?

        return arg_parser

    def setup_topology(self):
        # sets up controller and builds a bare mininet net
        super(FiredexMininetExperiment, self).setup_topology()

        # We use special 'coded addresses' to help debugging/understanding what's going on in log files
        base_subnet = '10.128.%s/9'  # see note in config.py about why we use this... don't forget subnet mask!
        icp_subnet = base_subnet % '1.%d'
        ff_subnet = base_subnet % '10.%d'
        fire_mac = 'ff:00:00:00:%s:%s'
        ff_mac = fire_mac % ('ff', "%.2x")
        bldg_subnet = base_subnet % '2.%d'
        iot_subnet = base_subnet % '20.%d'
        bldg_mac = 'bb:00:00:00:%s:%s'
        iot_mac = bldg_mac % ('dd', '%.2x')
        eoc_subnet = base_subnet % '3.%d'
        eoc_mac = 'ee:0c:00:00:00:%.2x'

        # 1. create all our special switches for the network itself
        self.bldg = self.add_switch('bldg', dpid=':'.join(['bb']*8))
        self.inet = self.add_switch('inet', dpid='11:ee:77:00:00:00:00:00')
        # TODO: icp_switch?

        # 2. create special host nodes
        self.icp, self.icp_sw = self.add_server('icp', ip=icp_subnet % 1, mac=fire_mac % ('cc', '01'))
        self.bms, self.bms_sw = self.add_server('bms', ip=bldg_subnet % 1, mac=bldg_mac % ('cc', '01'))
        self.eoc, self.eoc_sw = self.add_server('eoc', ip=eoc_subnet % 1, mac=eoc_mac % 1)
        self.black_box = self.add_host('bb', ip=bldg_subnet % 200, mac=bldg_mac % ('bb', 'bb'))

        # 3. create FF and IoT host nodes
        for i in range(self.num_ffs):
            ff = self.add_host('ff%d' % i, ip=ff_subnet % i, mac=ff_mac % i)
            self.ffs.append(ff)

        for i in range(self.num_iots):
            iot = self.add_host('iot%d' % i, ip=iot_subnet % i, mac=iot_mac % i)
            self.iot_devs.append(iot)

        # 4. create all the necessary links

        # NOTE: we apply default channel characteristics to inet links only
        # TODO: set up some for other links?  internal building topo should have some at least...
        for peer in (self.bldg, self.eoc_sw, self.icp_sw):
            self.add_link(self.inet, peer)
            # QUESTION: should we save these links so that we can e.g. explicitly vary their b/w during the sim?

        # Connect special hosts
        for bldg_comp in (self.bms_sw, self.black_box):
            self.add_link(self.bldg, bldg_comp, use_tc=False)

        # Connect IoT and FF hosts
        # For now, we'll just add all FFs and IoT devs directly to the bldg
        # TODO: setup wifi network and/or hierarchical switches
        # TODO: set channel characteristics
        for h in self.ffs + self.iot_devs:
            self.add_link(self.bldg, h, use_tc=False)

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

        # Start the brokers first so that they're running by the time the clients start publishing
        self.run_brokers()
        self.run_scale_clients()

        # TODO: choose which devs publish what topics?  which devs are video feeds? where are FFs?

    def run_experiment(self):

        log.info("*** Starting Experiment...")

        # TODO: clean up all this hacky iperf stuff; this is all just a place-holder...
        # run iperf for video feeds: we'll use SDN to selectively fork the video feed to black_box and, when requested, ICP
        video_hosts = (self.ffs[0], self.iot_devs[0])
        iperfs = []  # 4-tuples
        iperf_res = []  # actual parsed results
        srv = self.icp
        mn_iperfs = True  # this blocks
        # so run it in background as a proc directly, which sort of lets us parse the results from files but we've had some problems...
        # set the duration to account for this blocking:
        iperf_duration = self.experiment_duration
        if mn_iperfs:
            iperf_duration /= float(len(video_hosts))

        for i, v in enumerate(video_hosts):
            if mn_iperfs:
                ip = self.iperf(v, srv, port=DEFAULT_VIDEO_PORT+i, bandwidth=DEFAULT_VIDEO_RATE_MBPS, duration=iperf_duration, use_mininet=True)
            else:
                # ip = self.iperf(v, srv, port=DEFAULT_VIDEO_PORT+i, bandwidth=DEFAULT_VIDEO_RATE_MBPS, duration=iperf_duration, pipe_results=True, use_mininet=False)
                fname = "iperf%d" % i
                ip = self.iperf(v, self.icp, port=DEFAULT_VIDEO_PORT+i, bandwidth=DEFAULT_VIDEO_RATE_MBPS, duration=iperf_duration, output_results=fname, use_mininet=False)

            if ip is not None:
                ip = zip((v.name, srv.name), ip)
                iperfs.append(ip)

        if not mn_iperfs:
            time.sleep(iperf_duration)

        log.info("*** Experiment complete!")

        log.debug("sleeping a few seconds for procs to finish...")
        time.sleep(10)

        log.info("iperf results:")
        log.info("raw: %s" % iperfs)
        for (cli_name, cli_res), (srv_name, srv_res) in iperfs:
            if not mn_iperfs:
                with open(cli_res) as f:
                    cli_res = self.parse_iperf(f.read())
                with open(srv_res) as f:
                    srv_res = self.parse_iperf(f.read())

            print "iperf from %s --> %s results: %s/%s (s/c)" % (cli_name, srv_name, srv_res, cli_res)
            iperf_res.append({cli_name: cli_res, srv_name: srv_res})

        # TODO: add more stuff to the results here!
        return {'iperfs': iperf_res}

    def run_brokers(self):

        # TODO: run VerneMQ broker instead of coap server
        # self.run_proc(vmq_cmd, self.icp, 'icp broker')
        # self.run_proc(vmq_cmd, self.bms, 'bms broker')

        coap_cfg = make_scale_config_entry(name="CoapServer", events_root="/events/", class_path="coap_server.CoapServer")
        # this will just count up the # events received with this topic
        event_log_cfg = make_scale_config_entry(name="EventsLog", subscriptions=(IOT_DEV_TOPIC,),
                                                output_file=os.path.join(self.outputs_dir, "events_%s"),
                                                class_path="event_file_logging_application.EventFileLoggingApplication")

        # This is just for the ICP so that it can interact with the SDN Controller, which in our scenario we assume is near the BMS
        sdn_cfg = make_scale_config_entry(name="SdnApp", topology_mgr=self._get_topology_manager_config(),
                                          class_path="topology_manager.scale_sdn_application.ScaleSdnApplication")

        # TODO: add in EOC too?  not sure what exactly we'll do with it though...
        for broker_node in (self.icp, self.bms):
            hostname = "broker@%s" % broker_node.name
            broker_cfg = make_scale_config(networks=coap_cfg,
                                           applications=event_log_cfg + (sdn_cfg if broker_node is self.icp else ''))
            cmd = self.make_host_cmd(broker_cfg % hostname, hostname)
            self.run_proc(cmd, broker_node, hostname)

    def run_scale_clients(self):
        """
        Configure host devices with SCALE clients to publish sensor data to the data exchange.
        :return:
        """

        iot_dev_publish_interval = 1.0

        # Building IoT devices all publish their periodic event data to the BMS broker.
        broker_ip = self.bms.IP()
        for dev in self.iot_devs:
            # NOTE: make sure we distinguish the coapthon client instances from each other by incrementing CoAP port for multiple topics!

            dev_cfg = make_scale_config(
                sensors=make_scale_config_entry(name="IoTSensor", event_type=IOT_DEV_TOPIC,
                                                dynamic_event_data=dict(seq=0),
                                                class_path="dummy.dummy_virtual_sensor.DummyVirtualSensor",
                                                output_events_file=os.path.join(self.outputs_dir,
                                                                                '%s_%s' % (IOT_DEV_TOPIC, dev.name)),
                                                # give servers a chance to start; spread out their reports too
                                                start_delay=random.uniform(5, 10),
                                                sample_interval=iot_dev_publish_interval)
                ,  # forward these SensedEvents to the broker best-effort (non-CONfirmable) for CoAP
                sinks=make_scale_config_entry(class_path="remote_coap_event_sink.RemoteCoapEventSink",
                                              name="IotDataCoapEventSink", hostname=broker_ip,
                                              src_port=COAP_CLIENT_BASE_SRC_PORT,
                                              topics_to_sink=(IOT_DEV_TOPIC,), confirmable_messages=False)
                # Can optionally enable this to print out each event in its entirety.
                + make_scale_config_entry(class_path="log_event_sink.LogEventSink", name="LogSink")
            )

            hostname = "scale@%s" % dev.name
            cmd = self.make_host_cmd(dev_cfg, hostname)
            self.run_proc(cmd, dev, hostname)

        # Fire fighter nodes publish their data to the ICP broker.
        # TODO:

FiredexMininetExperiment.__doc__ = CLASS_DESCRIPTION


if __name__ == "__main__":
    import sys
    exp = FiredexMininetExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()
