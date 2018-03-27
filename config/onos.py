from config import DEFAULT_USER

CONTROLLER_IP="10.0.2.15"
CONTROLLER_REST_API_PORT=8181
ONOS_ADMIN_USER=DEFAULT_USER  # user that can run ONOS commands, NOT who is running the ONOS service!
ONOS_USER=ONOS_ADMIN_USER     # the user actually running the ONOS service
ONOS_ADMIN_PORT=8101  # SSH port, NOT REST API port!
ONOS_HOSTNAME='localhost'

# For accessing REST API
ONOS_API_USER='karaf'
ONOS_API_PASSWORD='karaf'

CONTROLLER_SERVICE_RESTART_CMD='service onos restart'

# Since Mininet runs as root, we need a way of invoking ONOS commands as the ONOS user
# to reset the controller in between executions.  This is a HACK as we couldn't get the
# shell to execute with ONOS_ADMIN_USER's proper env variables that let us use the 'onos' command,
# which would let us just call 'tools/onos_reset.sh'
CONTROLLER_RESET_CMD="su -c 'ssh -p %d %s@%s wipe-out please' %s" % (ONOS_ADMIN_PORT, ONOS_USER, ONOS_HOSTNAME, ONOS_ADMIN_USER)