import json
import socket
import time
import logging

import math

log = logging.getLogger(__name__)
# ENHANCE: get the logger in __init__ so that we can optionally send it to a file instead?

# constants for use in DataPath management: we could extend these to support more statuses in the future...
DATA_PATH_UP = 1
DATA_PATH_DOWN = 0
def data_path_status_code_to_str(code):
    return 'UP' if code == DATA_PATH_UP else 'DOWN'


class DataPathMonitor(object):
    """
    A DataPathMonitor detects challenges (e.g. complete failure or significant congestion) in the DataPath
    (public Internet-routed path between local gateway to cloud service) it's responsible for.  This class
    represents an abstract state machine that unifies the APIs of the underlying algorithms used in concrete
    base class implementations.  These implementations, like this base class, should not directly manipulate
    sockets, threads, etc. but should instead have wrapper classes (or further base classes) to implement
    these specifics.  This improves code reusability and testability, especially since this is a part of an
    experimental framework for evaluating different algorithms both in simulation and emulated/real deployment
    environments.
    """

    def __init__(self, data_path_id=None, status_change_callback=None):
        """
        :param data_path_id: unique ID for the DataPath this instance is responsible for monitoring; will be included
        in status changed messages / passed as an argument to the status_change_callback function (optional)
        :param status_change_callback: callback function accepting two arguments: data_path_id and its new status;
        by default, it calls self._on_status_change so that your base class can handle it locally
        """

        self.data_path_id = data_path_id
        self.status_change_callback = status_change_callback
        self._link_status = DATA_PATH_UP

    def _on_status_change(self, data_path_id, new_status):
        raise NotImplementedError("base classes must either implement this method or specify the"
                                  " status_change_callback to __init__!")

    @property
    def is_data_path_down(self):
        return self._link_status == DATA_PATH_DOWN

    def update_link_status(self, status):
        """Set the link status and call our callback e.g. to notify interested parties."""
        self._link_status = status
        if self.status_change_callback:
            self.status_change_callback(self.data_path_id, self._link_status)
        else:
            self._on_status_change(self.data_path_id, self._link_status)


class ProbingDataPathMonitor(DataPathMonitor):
    """
    A ProbingDataPathMonitor uses network 'probes' (e.g. like ICMP echo aka 'ping') to detect DataPath status changes.
    """

    class ProbeTimeout(socket.timeout):
        """Raised when a probe is not responded to within the required timeout."""
        pass

    def __init__(self, address, dst_port, src_port=None,
                 # TODO: move these to the actual socket implementation class?
                 buffer_size=4096, **kwargs):
        """
        :param address: address of the remote echo server we send probes to
        :param dst_port: port of the remote echo server
        :param src_port: local port to bind to (default lets operating system assign one)
        :param kwargs: passed to super(...)
        """
        super(ProbingDataPathMonitor, self).__init__(**kwargs)

        # ENHANCE: allow specifying the socket directly rather than the address/ports?
        self._address = address
        self._dst_port = dst_port
        self._src_port = src_port

        # socket/threading implementation-related parameters
        self.buffer_size = buffer_size
        self._probing_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if self._src_port:
            self._probing_socket.bind(('', self._src_port))
        self.echo_server = self._address, self._dst_port

    # ENHANCE: make these abstract and move them to an actual implementation class
    def _do_send(self, raw_data):
        """
        Actually send the given raw_data to the echo server as a probe
        :param raw_data:
        :return:
        """
        self._probing_socket.sendto(raw_data, self.echo_server)

    def set_probe_timeout(self, new_timeout):
        """
        Set the timeout for the probe by setting the corresponding socket's timeout.
        :param new_timeout: in milliseconds
        :return:
        """
        self._probing_socket.settimeout(new_timeout / 1000.0)

    def recv_response(self):
        """
        Receive the probe response.
        :return: raw_response_str
        :raises: socket.error if response came from an unexpected address
        """
        resp, addr = self._probing_socket.recvfrom(self.buffer_size)
        if addr == self.echo_server:
            return resp
        else:
            raise socket.error("received probe response from unexpected address %s" % addr)


class RideCDataPathMonitor(ProbingDataPathMonitor):
    """
    RideC uses an adaptive probing mechanism to monitor a DataPath.  It tunes the frequency of its probing interval
    to conserve resources while ensuring that it will meet the specified requirements:
     - maximum detection time for a failure/congestion event
     - maximum false positive rate
    """

    # non-zero default to avoid math errors esp. due to link loss rate of 0
    DEFAULT_DETECTION_WINDOW_SIZE = 3

    def __init__(self, max_detection_time=3000, max_false_positive=0.01, init_window=100, alpha=0.8, **kwargs):
        """
        :param max_detection_time: maximum time (in ms) it will take to detect a failure/congestion event
        :param max_false_positive: in (exclusive) range (0, 1.0) to determine false positive rate
        :param init_window: number of probes required when first starting up to determine the DataPath's normal status
        :param alpha: exponential weighting factor used in calculating average RTT
        :param kwargs: passed to super(...)
        """
        super(RideCDataPathMonitor, self).__init__(**kwargs)

        # RideC adaptive probing algorithm input parameters
        self.max_detection_time = max_detection_time
        self.max_false_positive = max_false_positive
        self.init_window = init_window
        self._alpha = alpha

        if max_false_positive <= 0:
            raise ValueError("cannot specify a max_false_positive rate <= 0!! Requested: %f" % max_false_positive)
        if max_detection_time <= 0:
            raise ValueError("cannot specify a max_detection_time <= 0!! Requested: %f" % max_false_positive)

        # Initialize the adaptive probing state that will be tuned during the initial monitoring phase
        self._timeout = self.max_detection_time
        self._link_loss = 0
        self._rtt_a = None
        # Number of successive non-responses before deeming a DataPath DOWN
        self._detection_window_size = self.DEFAULT_DETECTION_WINDOW_SIZE
        # milliseconds to wait between probes
        self._sending_interval = 0

        # Used really just for testing/experiment purposes
        self._seq = 0
        self._total_sent = 0
        self._total_received = 0

        # Should exit when this is True
        self._running = False

    ##### These functions actually handle probing

    def do_probing_round(self, count=False):
        """
        Sends a probe to the server, waits for a response, and processes this response.  Note that there can be only
         one outstanding probe at a time; once a timeout occurs while waiting for the response to a probe, we must
         safely ignore that response should it arrive later.
        NOTE: we do not currently fully handle out-of-order responses in the sense that the synchronous structure of
         this algorithm cannot allow us to receive a response to a probe with a future sequence number.
        :param count: whether to increment the internal counters for total sent/rcvd
        :return: the delay from this probe (in ms) or False if it timed out
        """

        seq = self.send_probe(count=count)
        # dummy value to enable this while loop
        resp_seq = seq - 1
        delay = None
        try:
            while resp_seq != seq:
                # If we timed out while waiting for the probe, we should just return immediately anyway
                recv_data_str = self.recv_response()
                delay, resp_seq = self.on_response_received(recv_data_str, count=count)

                if resp_seq != seq:
                    log.debug("DP %s: skipping old probe with seq #%d, expecting #%d" % (self.data_path_id, resp_seq, seq))
                else:
                    log.debug("DP %s: Received Probe Response (seq:%d) delay = %dms" % (self.data_path_id, resp_seq, delay))

                assert resp_seq <= seq, "received probe response with sequence # from the future! HOW????"

        except socket.timeout:
            log.info("DP %s: Timeout Probe (seq:%d)" % (self.data_path_id, self._seq - 1))
            # TODO: this hacky return value caused issues when the measured delay was actually 0ms
            # (e.g. due to running on localhost): we should maybe change this API later, but for now see how we use
            # the 'is' statement to distinguish False from 0ms
            return False
        # TODO: handle other errors? such as receiving from the wrong server...
        # except socket.error:
        #     log.warning("socket error: %s")
        return delay

    def send_probe(self, count=False):
        """
        Send the probe to the echo server with the current time and sequence # in it.  Note that this defers to the
        _do_send_probe function responsible for actually sending the raw data.
        :param count: if True, increments the number of probes sent
        :return: the sequence # of the sent probe
        """
        seq = self._seq
        self.set_probe_timeout(self._timeout)
        current_time_millis = int(time.time() * 1000)
        data = dict(seq=seq, time_sent=current_time_millis)

        self._do_send(json.dumps(data))
        log.debug("DP %s: Sent Probe (seq:%d)" % (self.data_path_id, seq))

        if count:
            self._total_sent += 1

        self._seq += 1
        return seq

    def on_response_received(self, recv_data_str, count=False):
        """
        Handle the response to a probe, which we assume came from the echo server, by determining the RTT based on the
        current time and the time we sent this probe (contained in the payload).
        :param recv_data_str: the raw data payload from the response
        :param count: if True, increment the number of received responses
        :return: 2-tuple: (round-trip-time (in ms), probe sequence number)
        """
        receive_time_millis = int(time.time() * 1000)
        if count:
            self._total_received += 1
        recv_data = json.loads(recv_data_str)
        sent_time_millis = recv_data['time_sent']
        delay = receive_time_millis - sent_time_millis
        receive_seq = recv_data['seq']

        return delay, receive_seq

    #### These functions handle adjusting the internal state

    def set_detection_window_size(self, max_false_positive_rate=None, link_loss=None):
        """
        Adapts the detection window size according to the specified parameters (default to values stored on self).
        :param max_false_positive_rate: defaults to self.max_false_positive
        :param link_loss: defaults to self._link_loss
        """

        if max_false_positive_rate is None:
            max_false_positive_rate = self.max_false_positive
        if link_loss is None:
            link_loss = self._link_loss

        try:
            self._detection_window_size = math.ceil(math.log(max_false_positive_rate, link_loss))
        except ValueError as e:
            if link_loss == 0:
                log.debug("link_loss of 0! Setting default _detection_window_size of %d" % self._detection_window_size)
            else:
                log.error("can't determine detection_window_size due to error (%s): max_false_positive=%f,"
                          " link_loss=%f" % (e, max_false_positive_rate, link_loss))
                raise

        return self._detection_window_size

    def adapt_probing_parameters(self):
        """
        Adapts the detector's parameters according to the RideC resource-conserving adaptive probing algorithm.
        """
        # TODO: should probably do a weighted average of link loss rather than this calculation over the whole lifetime
        self._link_loss = 1.0 - float(self._total_received) / self._total_sent
        self.set_detection_window_size()
        self._timeout = self._rtt_a * 2
        self._sending_interval = self.max_detection_time / self._detection_window_size

    def check_data_path_status(self, successive_fails):
        """
        Returns the DataPath status according to the number of successive failures or perceived latency.
        Currently only returns down(0) or up(1)
        :param successive_fails: the # successive probe failures (timeouts)
        :return: DataPath status
        """
        # TODO: include some tolerance to the thresholds in order to prevent 'flapping'
        if successive_fails > self._detection_window_size:
            if not self.is_data_path_down:
                log.info("DP %s status changed to DOWN due to recent failures/timeouts!" % self.data_path_id)
            return DATA_PATH_DOWN
        elif self._rtt_a > self._sending_interval:
            if not self.is_data_path_down:
                log.info("DP %s status changed to DOWN due to increased latency!" % self.data_path_id)
            return DATA_PATH_DOWN
        else:
            return DATA_PATH_UP

    def estimate_rtt(self, delay, alpha=None):
        """
        Maintains a weighted moving average of the Round-Trip Time (RTT).  Each time you call this function with a new
        delay value, the internal RTT estimate is adjusted to account for this new data point.
        :param delay: the delay from the most recent probe
        :param alpha: weighting factor applied to the previously estimated RTT (default=self._alpha; see __init__)
        :return: the new RTT estimate
        """

        if alpha is None:
            alpha = self._alpha

        if self._rtt_a is None:
            self._rtt_a = delay
        else:
            self._rtt_a = alpha * self._rtt_a + (1.0 - alpha) * delay

        return self._rtt_a

    #### These functions represent operation in the DataPathMonitor's various states.

    def link_characteristic_estimation_phase(self, nprobes=None):
        """
        The initial DataPath monitoring phase determines the characteristics of this DataPath (i.e. RTT, loss) and
        sets the algorithm parameters accordingly.
        :param nprobes: number of initial probes to send in this phase
        :return:
        """
        log.debug("DP %s: Entering initial link characteristic estimation phase..." % self.data_path_id)

        if nprobes is None:
            nprobes = self.init_window

        for i in range(nprobes):
            delay = self.do_probing_round(count=True)
            # just ignore timeouts...
            if delay is not False:
                self.estimate_rtt(delay)
            # TODO: sleep for some time between probes???

        log.debug("Initial phase finished!")
        self._link_loss = 1.0 - float(self._total_received) / self._total_sent
        self._timeout = 2 * self._rtt_a
        self.set_detection_window_size()
        self._sending_interval = self.max_detection_time / self._detection_window_size
        log.info("DP %s links status: link_loss:%f, rtt_a:%dms, Nb:%d, interval:%fms" % (self.data_path_id,
                                                                                         self._link_loss,
                                                                                         self._rtt_a,
                                                                                         self._detection_window_size,
                                                                                         self._sending_interval))

    def data_path_recovery_detection(self, nsuccesses=None):
        """
        Continually probe the echo server until we receive nsuccesses successive responses.
        :param nsuccesses: defaults to DEFAULT_DETECTION_WINDOW_SIZE
        :return:
        """
        log.info("Entering DataPath Recovery Detection phase...")

        if nsuccesses is None:
            nsuccesses = self.DEFAULT_DETECTION_WINDOW_SIZE

        # we aim for detecting a recovery (assuming 0 loss rate) within our specified detection time by doing:
        # ENHANCE: instead of waiting for successive responses, perhaps track the recent loss rate and wait for it to
        # exceed some threshold?
        # TODO: perhaps we should actually account for the expected RTT here?  or maybe use our calculated detection
        # window size instead of a default value?
        self._timeout = self.max_detection_time / float(nsuccesses)

        successive_count = 0
        while self.is_data_path_down:
            delay = self.do_probing_round(count=False)
            if delay is not False:
                successive_count += 1
                # TODO: perhaps we should be doing some sort of adaptive probing during recovery?  Or at least continue
                # estimating the link characteristics whenever we do receive a response (see note below after recovery).
                # self.estimate_rtt(delay)
                # self.adapt_probing_parameters()
            else:
                successive_count = 0
                delay = self._timeout

            # Fast-recovery scheme: only wait to send the next probe if we still KNOW the DataPath is down: once we
            # receive a single response, we should send the next probes faster (only one outstanding at a time though)
            # so that we can quickly assess whether it truly recovered.
            if successive_count == 0:
                self.wait_for_next_probe(delay)

            # immediately notify when recovered, but still wait before returning to regular mode / sending another probe
            elif successive_count >= nsuccesses:
                log.debug("DataPath %s recovered after %d successful probes in a row!" % (self.data_path_id, nsuccesses))
                self.update_link_status(DATA_PATH_UP)
                self.wait_for_next_probe(delay)
                # reset our probing parameters, especially timeout
                # TODO: should probably take into consideration that the recovered path may have different
                # characteristics e.g. higher RTT; maybe re-run the initial phase???  Would have to do it carefully and
                # with modifications in order to potentially detect a failure again...
                # NOTE: going back to our original parameters before the failure can cause flapping if we don't
                # re-estimate these characteristics due to the recovered path possibly having a much higher RTT, which
                # would cause further timeouts!
                self.adapt_probing_parameters()
                return

    def run(self):
        """
        The main loop of the RideC DataPath monitoring algorithm: determines the initial DataPath characteristics and
        then continually uses adaptive probes to monitor it.
        :return:
        """

        log.info("starting monitoring on DataPath %s (remote host=%s)" % (self.data_path_id, self.echo_server))

        self._running = True
        self.link_characteristic_estimation_phase()
        successive_fails = 0

        while self._running:
            if not self.is_data_path_down:
                delay = self.do_probing_round(count=True)
                if delay is False:
                    successive_fails += 1
                    delay = self._timeout
                else:
                    successive_fails = 0
                    self.estimate_rtt(delay)

                state = self.check_data_path_status(successive_fails)
                if state == DATA_PATH_DOWN:
                    self.update_link_status(DATA_PATH_DOWN)
                else:
                    self.adapt_probing_parameters()
                    self.wait_for_next_probe(delay)
            # Continue monitoring DataPath until it recovers
            else:
                self.data_path_recovery_detection()
                successive_fails = 0

    def wait_for_next_probe(self, time_since_last_probe=None):
        """
        Sleeps until the next time we should send a probe, which is our probe interval - the time delta from when
        we last sent a probe.  This is to account for the fact that receiving a response incurs some delay.
        :param time_since_last_probe: in ms; if unspecified or None, simply sleep for the sending_interval
        """
        time_until_next_probe = self._sending_interval / 1000.0
        if time_since_last_probe is not None:
            time_until_next_probe -= time_since_last_probe / 1000.0
        if time_until_next_probe > 0:
            time.sleep(time_until_next_probe)

    def finish(self):
        log.info("closing DataPathMonitor...")
        self._running = False
        self._probing_socket.close()


# Simple test main to run the RideC detector
if __name__ == '__main__':
    # XXX: must call basicConfig or logging.[info|debug|etc...] first as no logger for __main__!
    logging.basicConfig()
    global log
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)

    # Extract parameters with defaults
    import sys
    try:
        address = sys.argv[1]
    except IndexError:
        address = '127.0.0.1'
    try:
        port = sys.argv[2]
    except IndexError:
        port = 9999

    def _on_status_change(data_path_id, new_status):
        log.info("DP STATUS CHANGED to %s" % data_path_status_code_to_str(new_status))

    rc = RideCDataPathMonitor(address=address, dst_port=port, status_change_callback=_on_status_change)
    rc.run()