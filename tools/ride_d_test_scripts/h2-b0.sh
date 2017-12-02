#!/bin/bash

su -c "pushd .; export WORKON_HOME=~/.venvs; source ~/.local/bin/virtualenvwrapper.sh; workon ride_scale_client; popd; python -OO -m scale_client --disable-log-module topology_manager.sdn_topology urllib3.connectionpool --raise-errors  --enable-log-module coapthon  -q 30 --log debug --applications '{\"SeismicSubscriber\": {\"output_file\": \"results/outputs_2t_0.10f_2s_2p_steiner_disjoint_tiny_0.00e_importance/run0/subscriber_h2-b0\", \"class\": \"seismic_warning_test.seismic_alert_subscriber.SeismicAlertSubscriber\", \"remote_brokers\": [\"10.255.0.0\", \"10.199.0.0\"]}}'   --networks '{\"CoapServer\": {\"events_root\": \"/events/\", \"class\": \"coap_server.CoapServer\"}}'  " vagrant

