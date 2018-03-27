from config import DEFAULT_USER

# Basically these are aliases for restarting OVS between runs, which seems to help solve some issues with larger topos...

# Change these paths depending on your system's installation
OVS_PREFIX_DIR='/usr/local'
OVS_SCHEMA='/home/%s/repos/ovs/vswitchd/vswitch.ovsschema;' % DEFAULT_USER
OVS_KERNEL_FILE='/home/%s/repos/ovs/datapath/linux/openvswitch.ko' % DEFAULT_USER

# These two collections of commands are assumed to be run as root!
RESET_OVS='''sudo pkill ovsdb-server;
sudo pkill ovs-vswitchd;
sudo rm -f %s/var/run/openvswitch/*;
sudo rm -f %s/etc/openvswitch/*;
ovsdb-tool create %s/etc/openvswitch/conf.db %s
sudo modprobe libcrc32c;
sudo modprobe gre;
sudo modprobe nf_conntrack;
sudo modprobe nf_nat_ipv6;
sudo modprobe nf_nat_ipv4;
sudo modprobe nf_nat;
sudo modprobe nf_defrag_ipv4;
sudo modprobe nf_defrag_ipv6;
sudo insmod %s
''' % (OVS_PREFIX_DIR, OVS_PREFIX_DIR, OVS_PREFIX_DIR, OVS_SCHEMA, OVS_KERNEL_FILE)

RUN_OVS='''sudo ovsdb-server --remote=punix:%s/var/run/openvswitch/db.sock \
    --private-key=db:Open_vSwitch,SSL,private_key \
    --certificate=db:Open_vSwitch,SSL,certificate \
    --bootstrap-ca-cert=db:Open_vSwitch,SSL,ca_cert \
    --pidfile --detach --log-file;
ovs-vsctl --db=unix:%s/var/run/openvswitch/db.sock --no-wait init;
sudo ovs-vswitchd --pidfile --detach --log-file''' % (OVS_PREFIX_DIR, OVS_PREFIX_DIR)
