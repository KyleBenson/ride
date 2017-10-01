#!/bin/bash
if [ -d ~/repos/onos ]; then
    # when we have built ONOS from source and used e.g. onos-install to install/run it
    pushd ~/repos/onos
    ONOS_CMD="onos localhost"
else
    # when we've downloaded/extracted the ONOS tarball and just configured it as a systemctl service
    ONOS_CMD=/opt/onos/bin/onos
fi

# TODO: check if we're root (if [ `whoami` == root ]) and run su -c $CMD $ONOS_USER
RESET_CMD="wipe-out please"
FINAL_CMD="$ONOS_CMD $RESET_CMD"

# echo "$FINAL_CMD"
$FINAL_CMD

if [ -d ~/repos/onos ]; then
    popd
fi
