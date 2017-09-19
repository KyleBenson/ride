# this should actually be in mininet_smart_campus_experiment.py...
# When True, runs host processes with -00 command for optimized python code
OPTIMISED_PYTHON = False

### Configurations for actually running SCALE as the test client applications
# Make sure you setup a virtual environment called 'scale_client' for this user!
SCALE_USER = 'vagrant'
# XXX: HACK: since Mininet runs as root and we use virtual environments, we have to run the client
# within the venv but at the right location, under the right user, with the right PYTHONPATH,
# all as a large complicated command passed as a string to 'su'...
SCALE_EXTRA_ARGS = \
    "--disable-log-module topology_manager.sdn_topology urllib3.connectionpool " \
    "--raise-errors " \
    # "--format-logging '%%(levelname)-6s : %%(name)-55s (%%(asctime)2s) : %%(message)s'"  # TODO: this doesn't work right now due to coapthon logging bug.... add timestamps; make sure to use '%%' to keep it from doing the formatting yet!
    # " --enable-log-module coapthon " \
# Change this command to match your environment's configuration as necessary
VIRTUAL_ENV_CMD = "export WORKON_HOME=~/.venvs; source ~/.local/bin/virtualenvwrapper.sh; workon ride_scale_client;"
# WARNING: this took a long time to get actually working as the quotes are quite finicky... careful modifying!
SCALE_CLIENT_BASE_COMMAND = 'su -c "pushd .; %s popd; python %s -m scale_client %s %%s" ' % (VIRTUAL_ENV_CMD, "-O" if OPTIMISED_PYTHON else "", SCALE_EXTRA_ARGS) + SCALE_USER
