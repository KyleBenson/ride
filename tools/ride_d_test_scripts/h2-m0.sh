#!/bin/bash

su -c "pushd .; export WORKON_HOME=~/.venvs; source ~/.local/bin/virtualenvwrapper.sh; workon ride_scale_client; popd; python -OO -m scale_client --disable-log-module topology_manager.sdn_topology urllib3.connectionpool --raise-errors  --enable-log-module coapthon  -q 30 --log debug --sensors '{\"SeismicSensor\": {\"output_events_file\": \"results/outputs_2t_0.10f_2s_2p_steiner_disjoint_tiny_0.00e_importance/run0/publisher_h2-m0\", \"event_type\": \"seismic\", \"start_delay\": 10,
\"sample_interval\": 5, \"dynamic_event_data\": {\"seq\": 0}, \"class\": \"dummy.dummy_virtual_sensor.DummyVirtualSensor\"}}' \
'{\"IoTSensor\": {\"output_events_file\": \"results/outputs_2t_0.10f_2s_2p_steiner_disjoint_tiny_0.00e_importance/run0/congestor_h2-m0\", \"sample_interval\": 3, \"start_delay\": 20, \"dynamic_event_data\": {\"seq\": 0}, \"class\": \"dummy.dummy_virtual_sensor.DummyVirtualSensor\", \"event_type\": \"generic_iot_data\"}}' \
--event-sinks '{\"SeismicCoapEventSink\": {\"topics_to_sink\": [\"seismic\"], \"hostname\": \"10.255.0.0\", \"src_port\": 7777, \"class\": \"remote_coap_event_sink.RemoteCoapEventSink\"}}' '{\"GenericCoapEventSink\": {\"topics_to_sink\": [\"generic_iot_data\"], \"hostname\": \"10.255.0.0\", \"src_port\": 7778, \"confirmable_messages\": false, \"class\": \"remote_coap_event_sink.RemoteCoapEventSink\"}}'  " vagrant

