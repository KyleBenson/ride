## This file contains various user-configurable constants used by the RIDE components and experiments.

# BUGFIX: when doing redirection to edge server, the static path to cloud server takes precedence.
# Cause: for multiple flow rules matching a packet, OVS uses the first one added!
# OF says this behavior is undefined, so we need to explicitly set priority values.
STATIC_PATH_FLOW_RULE_PRIORITY = 50000
# We use a weird number here to distinguish it from other flows in order to easily delete redirection upon recovery.
# See the XXX note in ride_c.py
REDIRECTION_FLOW_RULE_PRIORITY = 64321
MULTICAST_FLOW_RULE_PRIORITY = 65000