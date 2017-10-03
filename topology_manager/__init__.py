def build_topology_adapter(topology_adapter_type='onos', controller_ip=None, controller_port=None, *args, **kwargs):
    """
    Builds and configures the requested topology adapter from the specifed parameters.  This is essentially
    a convenience function for the individual implementations' constructors; it lets you simply specify a
    string representing the type and get back an actual instance.  This is helpful for converting string parameters
    into class instances :)
    WARNING: you may not be able to mix *args and **kwargs!

    :param topology_adapter_type: one of: onos (default), floodlight
    :param controller_ip: optional IP address/hostname of controller
    :param controller_port: optional port number of controller's REST API
    :param args: extra positional arguments passed to the respective SdnTopology's constructor
    :param kwargs: extra keyword arguments passed to the respective SdnTopology's constructor
    """

    # By building constructor arguments as kwargs, we can use that constructor's defaults if none specified here
    if controller_ip is not None:
        kwargs['ip'] = controller_ip
    if controller_port is not None:
        kwargs['port'] = controller_port

    if topology_adapter_type == 'onos':
        from onos_sdn_topology import OnosSdnTopology
        topo_mgr = OnosSdnTopology(*args, **kwargs)
    elif topology_adapter_type == 'floodlight':
        from floodlight_sdn_topology import FloodlightSdnTopology
        topo_mgr = FloodlightSdnTopology(*args, **kwargs)
    else:
        raise ValueError("unrecognized SdnTopology type: %s" % topology_adapter_type)

    return topo_mgr
