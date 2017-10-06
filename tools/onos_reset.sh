#!/bin/bash
if [ -d /opt/onos/ ]; then
    # when we've downloaded/extracted the ONOS tarball and just configured it as a systemctl service
    ONOS_CMD=/opt/onos/bin/onos
elif [ -d ~/repos/onos ]; then
    # when we have built ONOS from source and used e.g. onos-install to install/run it
    # we'll assume that we have aliases setup...
    #pushd ~/repos/onos
    ONOS_CMD="onos localhost"
else
    echo "ERROR: no onos in /opt/onos or ~/repos/onos: where's your 'onos' command???"
    exit -1
fi

# TODO: check if we're root (if [ `whoami` == root ]) and run su -c $CMD $ONOS_USER
RESET_CMD="wipe-out please"
FINAL_CMD="$ONOS_CMD $RESET_CMD"

# echo "$FINAL_CMD"
$FINAL_CMD