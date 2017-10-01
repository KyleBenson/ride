# Configurations that you might need to change for a specific installation.
# TODO: this file can be sourced in bash to collect these variables too!  BUT we need to do string formatting differently...

# When True, runs host processes in mininet with -OO command for optimized python code
OPTIMISED_PYTHON = False

# can just change this and not the others if everything is running under same user account...
DEFAULT_USER='vagrant'

##########         ONOS     CONFIG      ###########

CONTROLLER_IP="10.0.2.15"
ONOS_ADMIN_USER=DEFAULT_USER  # user that can run ONOS commands, NOT who is running the ONOS service!
ONOS_USER=ONOS_ADMIN_USER     # the user actually running the ONOS service
ONOS_ADMIN_PORT=8101  # SSH port, NOT REST API port!
ONOS_HOSTNAME='localhost'










# Since Mininet runs as root, we need a way of invoking ONOS commands as the ONOS user
# to reset the controller in between executions.  This is a HACK as we couldn't get the
# shell to execute with ONOS_ADMIN_USER's proper env variables that let us use the 'onos' command,
# which would let us just call 'tools/onos_reset.sh'
CONTROLLER_RESET_CMD="su -c 'ssh -p %d %s@%s wipe-out please' %s" % (ONOS_ADMIN_PORT, ONOS_USER, ONOS_HOSTNAME, ONOS_ADMIN_USER)

##########    SCALE CLIENT CONFIG       ###########

### Configurations for actually running SCALE as the test client applications
# Make sure you setup a virtual environment called 'scale_client' for this user!
SCALE_USER=DEFAULT_USER
# XXX: HACK: since Mininet runs as root and we use virtual environments, we have to run the client
# within the venv but at the right location, under the right user, with the right PYTHONPATH,
# all as a large complicated command passed as a string to 'su'...
SCALE_EXTRA_ARGS=\
    "--disable-log-module topology_manager.sdn_topology urllib3.connectionpool " \
    "--raise-errors " \
    # "--format-logging '%%(levelname)-6s : %%(name)-55s (%%(asctime)2s) : %%(message)s'"  # TODO: this doesn't work right now due to coapthon logging bug.... add timestamps; make sure to use '%%' to keep it from doing the formatting yet!
    # " --enable-log-module coapthon " \
# Change this command to match your environment's configuration as necessary
VIRTUAL_ENV_CMD="export WORKON_HOME=~/.venvs; source ~/.local/bin/virtualenvwrapper.sh; workon ride_scale_client;"
# WARNING: this took a long time to get actually working as the quotes are quite finicky... careful modifying!
SCALE_CLIENT_BASE_COMMAND='su -c "pushd .; %s popd; python %s -m scale_client %s %%s" ' % (VIRTUAL_ENV_CMD, "-OO" if OPTIMISED_PYTHON else "", SCALE_EXTRA_ARGS) + SCALE_USER


##### Misc.
IGNORE_OUTPUT = ' > /dev/null 2>&1'
