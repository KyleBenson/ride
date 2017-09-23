import unittest
from threading import Thread
from time import sleep

from ride.data_path_monitor import *
from ride.udp_echo_server import EchoServer, parse_args as parse_echo_server_args

import logging
log_level = logging.DEBUG
logging.basicConfig(level=log_level, format='%(levelname)s:%(module)s:%(asctime)s:\t%(message)s')


class TestParameters(unittest.TestCase):
    """Test basic algorithm calculations given different parameters; verify their legal range."""

    def setUp(self):
        self.monitor = RideCDataPathMonitor(address='127.0.0.1', dst_port=9999)

    def test_set_window(self):
        """Verify all legal values for detection window work"""
        mfpr_vals_to_check = (0.0001, 0.5, 0.9999)
        ll_vals_to_check = (0.0, 0.0001, 0.5, 0.9999, 1.0)
        for mfpr, ll in zip(mfpr_vals_to_check, ll_vals_to_check):
            try:
                sz = self.monitor.set_detection_window_size(mfpr, ll)
                assert 0 < sz, "detection_window_size must be positive!"
            except BaseException:
                self.assertTrue(False, "should not generate error for any values in legal range:"
                                       " max_false_pos_rate=%f, link_loss=%f" % (mfpr, ll))

    def tearDown(self):
        # make sure we close the sockets
        self.monitor.finish()


class TestBasicMonitoring(unittest.TestCase):
    """Verify that a DPMonitor will detect an outage, update its status, detect the path recovery, and again update."""

    # We go through several phases during which the DPMonitor should enter different states (e.g. normal, DP down, etc.)
    # NOTE: phase1 must include enough time for the link estimation phase of RideC! use timeout/init_window...
    # These phases determine when we will adjust the test parameters to start/end a 'challenge': loss of packets,
    # increased latency, or simply cutting the DataPath entirely by stopping the EchoServer...
    PHASE1_DURATION = 15  # 'DP failure' after this by starting 'challenge', but first...
    PHASE2_DURATION = 10   # wait this long for DPMonitor to actually detect failure;
    #  then verify it and start echo server again / remove lossy congestion
    PHASE3_DURATION = 10   # 'DP recover' after this since echo server open again for this long
    PHASE4_DURATION = 10   # for lossy/latency congestion, we verify that a mild challenge doesn't mark the path DOWN

    def dp_status_callback(self, dp_id, status):
        """When called by the DPMonitor, this will verify that the new status is the one expected by the tests."""
        self.assertEqual(self.expected_status, status, self._dp_status_msg)

    def setUp(self):
        """Run the DPMonitor and EchoServer in separate threads"""

        # dummy value to ensure we don't have an unexpected status change...
        self.expected_status = -1
        self._dp_status_msg = "new DataPath status did not match the expected status of %s" % self.expected_status

        self.monitor = RideCDataPathMonitor(address='127.0.0.1', dst_port=9999, init_window=10,
                                            status_change_callback=self.dp_status_callback)
        self._monitor_thread = Thread(target=self.monitor.run)

    def run_echo_server(self, quit_time=0, loss_rate=0.0):
        # NOTE: make sure we have the response_delay at least 1 or the delay might actually be calculated as 0,
        # which is used to represent that a timeout happened during probing!
        _echo_args = parse_echo_server_args(['-p', '9999', '-q', str(quit_time), '-l', str(loss_rate), '-d', '0.001'])
        self.echo_server = EchoServer(_echo_args)
        self._echo_thread = Thread(target=self.echo_server.run)
        self._echo_thread.start()

    def test_basic_failure_detection(self):
        """Verify that the DPMonitor detects a failure after the EchoServer stops responding."""

        # start echo server first so monitor immediately receives responses
        self.run_echo_server(self.PHASE1_DURATION)
        self._monitor_thread.start()

        # wait until echo server quits, which should signal a failure a few seconds later...
        sleep(self.PHASE1_DURATION)
        self.expected_status = DATA_PATH_DOWN
        self._dp_status_msg = "EchoServer quitting should have caused the DP to go DOWN!"
        self._echo_thread.join(3)
        print "FAILURE: echo server quit so expect DP down soon..."

        # give DPMonitor a chance to detect the failure before verifying it
        sleep(self.PHASE2_DURATION)
        link_down = self.monitor.is_data_path_down
        self.assertTrue(link_down, "data path monitor should think the data path is down since the echo server stopped!")

        # start the echo server again and give the monitor a chance to detect the path recovery
        self.expected_status = DATA_PATH_UP
        self._dp_status_msg = "EchoServer restarting should have caused the DP to go UP!"
        self.run_echo_server(self.PHASE3_DURATION)
        print "RECOVERY: echo server started so expect DP UP soon..."
        sleep(self.PHASE3_DURATION)

        # Verify the monitor detected the recovery!
        link_up = not self.monitor.is_data_path_down
        self.assertTrue(link_up, "data path monitor should think the data path is back up since the echo server is running again!")

    def test_congestion_loss_rate(self):
        """Verify that the DPMonitor detects severe congestion events and marks the DataPath as DOWN until it recovers.
        It should also keep the path marked as UP despite slight lossiness."""

        self.assertTrue(False, "RideC DPMonitor currently does not handle lossy DataPaths well:"
                               "it expects multiple successive failed probes rather than some % being lost recently...")
        # TODO: fix this and update this test case to be like the others e.g. make use of the status_change_callback
        # ENHANCE: should put assertions in the callbacks so we can verify the DP doesn't flap between our checks!

        down_loss_rate = 0.5
        up_loss_rate = 0.01

        # start echo server first so monitor immediately receives responses
        # NOTE: we want the echo server to run the whole time of the test; we'll tell it when to become lossy
        self.run_echo_server(quit_time=0)
        self._monitor_thread.start()

        # start congestion after monitor stabilizes; wait for its detection...
        sleep(self.PHASE1_DURATION)
        self.echo_server.loss_rate = down_loss_rate
        print "CONGESTION: expect DP DOWN soon due to loss_rate..."
        sleep(self.PHASE2_DURATION)

        # verify DP is down; remove congestion and wait for monitor to find out
        link_down = self.monitor.is_data_path_down
        self.assertTrue(link_down, "loss=%f should make data path monitor think the data path is DOWN!" % down_loss_rate)
        self.echo_server.loss_rate = 0
        print "RECOVERY: expect DP UP soon due to loss_rate..."
        sleep(self.PHASE3_DURATION)

        # Verify the monitor detected the recovery!
        link_up = not self.monitor.is_data_path_down
        self.assertTrue(link_up, "loss=%f should make data path monitor think the data path is UP!" % 0)

        # Lastly, set slight congestion and verify that the DP remains marked UP
        self.echo_server.loss_rate = up_loss_rate
        sleep(self.PHASE4_DURATION)
        link_up = not self.monitor.is_data_path_down
        self.assertTrue(link_up, "loss=%f should make data path monitor think the data path is UP!" % up_loss_rate)

    def test_congestion_latency(self):
        """Verify that the DPMonitor detects increased latency due to severe congestion events and marks the
         DataPath as DOWN until it recovers. It should also keep the path marked as UP despite slight delay."""

        # ENHANCE: should put assertions in the callbacks so we can verify the DP doesn't flap between our checks!

        # NOTE: we shouldn't need the whole max_detection_time to observe this congestion event since the normal RTT
        # is so low... these are in seconds!
        down_latency = 2.0
        up_latency = 0.001
        okay_latency = up_latency * 2

        # start echo server first so monitor immediately receives responses
        # NOTE: we want the echo server to run the whole time of the test; we'll tell it when to become lossy
        self.run_echo_server(quit_time=0)
        self._monitor_thread.start()

        # start congestion after monitor stabilizes; wait for its detection...
        sleep(self.PHASE1_DURATION)
        self.expected_status = DATA_PATH_DOWN
        self._dp_status_msg = "latency=%fs should have caused the DP to go DOWN!" % down_latency
        self.echo_server.response_delay = down_latency
        print "CONGESTION: expect DP DOWN soon due to response_delay=%fs..." % down_latency
        sleep(self.PHASE2_DURATION)

        # verify DP is down; remove congestion and wait for monitor to find out
        link_down = self.monitor.is_data_path_down
        self.assertTrue(link_down, "latency=%fs should make data path monitor think the data path is DOWN!" % down_latency)
        self.expected_status = DATA_PATH_UP
        self._dp_status_msg = "latency=%fs should have caused the DP to go UP!" % down_latency
        self.echo_server.response_delay = up_latency
        print "RECOVERY: expect DP UP soon due to response_delay=%fs..." % up_latency
        sleep(self.PHASE3_DURATION)

        # Verify the monitor detected the recovery!
        link_up = not self.monitor.is_data_path_down
        self.assertTrue(link_up, "latency=%fs should make data path monitor think the data path is UP!" % up_latency)

        # Lastly, set slight congestion and verify that the DP remains marked UP
        # TODO: move this part to a different test case that will explore multiple delays e.g. keep raising it a bit...
        print "OKAY: the DP should stay up as this is just a slight increase in latency..."
        self.expected_status = DATA_PATH_UP
        self._dp_status_msg = "latency=%fs should have caused the DP to go UP!" % okay_latency
        self.echo_server.response_delay = okay_latency
        sleep(self.PHASE4_DURATION)
        link_up = not self.monitor.is_data_path_down
        self.assertTrue(link_up, "latency=%fs should make data path monitor think the data path is UP!" % okay_latency)

    def tearDown(self):
        # make sure we close the sockets and threads
        self.monitor.finish()
        self.echo_server.finish()
        self._monitor_thread.join(3)
        self._echo_thread.join(3)

if __name__ == '__main__':
    unittest.main()
