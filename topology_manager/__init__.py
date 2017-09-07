def build_topology_adapter(topology_adapter_type='onos', controller_ip=None, controller_port=None):
    """
    Builds and configures the requested topology adapter from the specifed parameters.  This is essentially
    a convenience function for the individual implementations' constructors; it lets you simply specify a
    string representing the type and get back an actual instance.  This is helpful for converting string parameters
    into class instances :)
    """

    # By building constructor arguments as kwargs, we can use that constructor's defaults if none specified here
    kwargs = dict()
    if controller_ip is not None:
        kwargs['ip'] = controller_ip
    if controller_port is not None:
        kwargs['port'] = controller_port

    if topology_adapter_type == 'onos':
        from onos_sdn_topology import OnosSdnTopology
        topo_mgr = OnosSdnTopology(**kwargs)
    elif topology_adapter_type == 'floodlight':
        from floodlight_sdn_topology import FloodlightSdnTopology
        topo_mgr = FloodlightSdnTopology(**kwargs)
    else:
        raise ValueError("unrecognized SdnTopology type: %s" % topology_adapter_type)

    return topo_mgr
