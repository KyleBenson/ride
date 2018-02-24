FireDeX
=======

TODO: high-level description

Architecture Overview
---------------------

TODO: re-work this discussion...

TLDR: scenario --> experimental configuration --> system state (point in time)

* `FiredexScenario`: highest-level configuration that defines probability distributions (e.g. over all topics, publishers, subscribers, etc.) from which we can create an actual configuration for an experiment. Also generates deterministic values e.g. topics, FFs, etc.
* `FiredexExperimentConfiguration`: (currently just the dict inside the `FiredexAlgorithmExperiment` class) mid-level configuration that includes assigned values (be they static or further random distributions) pulled from probability distributions for specific topic publication rates, event sizes, utility functions, subscriptions, etc. This represents the scenario over the entire duration of one simulation run.
* `FiredexSystemState`: lowest-level aggregated information object representing the static state of the data exchange and network in a particular point in time e.g.: subscribers' topic interest, which events published during this time slot, etc. This may be an actual measurement (i.e. known events that occur in a simulation or data gathered in the real system after it happens) or an estimation (i.e. future estimation in the real system based on past ones).  The `FiredexAlgorithm` uses this information (possibly along with the entire `FiredexSystemState`) to configure e.g. priority levels for this (or the next) time slot. 