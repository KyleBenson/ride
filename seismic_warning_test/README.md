# seismic_warning_test
Components for simulating a simplified seismic awareness scenario.  Clients report "picks", a server aggregates them,
 and clients receive notifications of which devices picked for localized situational awareness.

USAGE
Run the clients all at the same time configured to send their data to a particular location.  This could be a multicast
address or it could be the server.  If the server, it will aggregate data and then send it to its specified IP address.
Recommended that you set the destination of this aggregated data to be a multicast address or else use
e.g. OVS with group tables to forward the message to all the subscribing clients.  You could easily extend the server
to do this manually but we didn't want to add the extra out-of-band configuration.

After running the experiment (probably using Mininet), put the output files into a single directory.  When you have
several experiments that you want to compare, run the statistics.py file on those directories to see the results
displayed using matplotlib.

ASSUMPTIONS
-- Clients repeatedly send picks once the earthquake happens up to a specified number of copies.
-- The aggregation server collects all picks during its buffering period in a dict and sends them in an array.
-- The Aggregator keeps sending all of the picks received to date each buffering period.

JSON schemas for events
-- At start of single event's lifetime:
{'id' : client_id,
'time_sent' : 123333434.3314,
-- When aggregated, the server puts them all in an array and adds some fields:
{'id' : 'aggregator',
'time_aggd' : 232423324.223,
'events' : [{event}, {event}, ....]
}
-- Upon receiving an event, the client adds some additional info to each event:
{'time_rcvd' : 1320290202.232,
'copies_rcvd' : 3}
-- It also indexes them by ID to track which ones it's received so far
 before finally outputting this results dict to a file as JSON:
{'h1' : {'id' : 'h1', ...},
'h2' : {...}
}
