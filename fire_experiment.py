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

import time
import argparse
import logging
log = logging.getLogger(__name__)

from scifire.config import *
from mininet_sdn_experiment import MininetSdnExperiment

class FireExperiment(MininetSdnExperiment):

    def __init__(self, num_ffs=DEFAULT_NUM_FFS, num_iots=DEFAULT_NUM_IOTS,
                 # HACK: kwargs just used for construction via argparse since they'll include kwargs for other classes
                 **kwargs):
        super(FireExperiment, self).__init__(**kwargs)

        # Save params
        self.num_ffs = num_ffs
        self.num_iots = num_iots

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

        # TODO: add other params
        self.results['params'].update({'num_fire_fighters': num_ffs,
                                       'num_iot_devices': num_iots,
                                       })

    @classmethod
    def get_arg_parser(cls, parents=(MininetSdnExperiment.get_arg_parser(),), add_help=True):
        """
        Argument parser that can be combined with others when this class is used in a script.
        Need to not add help options to use that feature, though.
        :param tuple[argparse.ArgumentParser] parents:
        :param add_help: if True, adds help command (set to False if using this arg_parser as a parent)
        :return argparse.ArgumentParser arg_parser:
        """

        arg_parser = argparse.ArgumentParser(description=CLASS_DESCRIPTION, parents=parents, add_help=add_help)
        # experimental treatment parameters

        arg_parser.add_argument('--num-ffs', '-nf', dest='num_ffs', type=int, default=DEFAULT_NUM_FFS,
                                help='''The number of fire fighter 'hosts' to create, which represent a FF equipped with
                                IoT devices that relay their data through some wireless smart hub (default=%(default)s).''')
        arg_parser.add_argument('--num-iots', '-nd', '-ni', dest='num_iots', type=int, default=DEFAULT_NUM_IOTS,
                                help='''The number of IoT device hosts to create, which represent various sensors,
                                actuators, or other IoT devices that reside within the building and publish
                                fire event-related data to the BMS (default=%(default)s).''')

        # TODO: change description for parent args?  specifically, which links do we apply the default channel stats on?

        return arg_parser

    @classmethod
    def build_from_args(cls, args):
        """Constructs from command line arguments."""

        args = cls.get_arg_parser().parse_args(args)

        # convert to plain dict
        args = vars(args)

        return cls(**args)

    def record_result(self, result):
        # TODO: is this even needed?  might need to add some custom info...
        # First, add additional parameters used on this run.
        return super(FireExperiment, self).record_result(result)

    def setup_topology(self):
        # sets up controller and builds a bare mininet net
        super(FireExperiment, self).setup_topology()

        # We use special 'coded addresses' to help debugging/understanding what's going on in log files
        base_subnet = '10.128.%s'  # see note in config.py about why we use this
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

    def setup_experiment(self):
        """
        Set up the experiment and configure it as necessary before run_experiment is called.
        :return:
        """

        super(FireExperiment, self).setup_experiment()

        # TODO: run VerneMQ brokers
        # self.run_proc(vmq_cmd, self.icp, 'icp broker')
        # self.run_proc(vmq_cmd, self.bms, 'bms broker')

        # TODO: choose which devs publish what topics?  which devs are video feeds? where are FFs?

    def run_experiment(self):

        log.info("*** Starting Experiment...")

        iperf_duration = 15

        # TODO: clean up all this hacky iperf stuff; this is all just a place-holder...
        # run iperf for video feeds: we'll use SDN to selectively fork the video feed to black_box and, when requested, ICP
        video_hosts = (self.ffs[0], self.iot_devs[0])
        iperfs = []  # 4-tuples
        iperf_res = []  # actual parsed results
        srv = self.icp
        mn_iperfs = True  # this blocks
        # so run it in background as a proc directly, which sort of lets us parse the results from files but we've had some problems...

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

FireExperiment.__doc__ = CLASS_DESCRIPTION


if __name__ == "__main__":
    import sys
    exp = FireExperiment.build_from_args(sys.argv[1:])
    exp.run_all_experiments()
