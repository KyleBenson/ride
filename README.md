# FireDeX

FireDeX, partly inspired by the [Research Roadmap for Smart Fire Fighting](https://www.nist.gov/publications/research-roadmap-smart-fire-fighting) that we co-authored, explores resilient IoT data exchange in a smart fire fighting scenario.  It studies appropriate mechanisms for ensuring reliable communications in leveraging an IoT-enabled buildings' infrastructure (e.g. sensors, alarms, occupancy assessment) to improve fire fighting.  The use of prioritization ensures these responders receive pertinent data in a timely manner despite resource constraints, hostile environments, and heterogeneous protocols and data models.

We use a queueing theoretic model of the pub/sub data exchange and underlying network infrastructure layer to analytically derive expected performance.
This analytical model enables our various algorithms for assigning priorities and drop rates in order to improve the overall *utility*  (i.e. value of information or situational awareness) captured by the data exchange process.
The prioritization is enforced at the SDN layer; we are currently working on a prototype system that uses real SDN APIs (OpenFlow).

This repository includes the algorithms and various scripts for generating various pub/sub scenarios, running simulations, and analyzing results.

Refer to [this link for the formal research paper](https://www.ics.uci.edu/~dsm/papers/2018/firedex-middleware.pdf) that presents FireDeX and please cite it as:

Kyle E. Benson, Georgios Bouloukakis, Casey Grant, Valérie Issarny, Sharad Mehrotra, Ioannis Moscholios, Nalini Venkatasubramanian. "FireDeX: a Prioritized IoT Data Exchange Middleware for Emergency Response". *in Proceedings of 19th International Middleware Conference (Middleware ’18)*. 2018.

## Getting Started

FireDeX is mostly implemented in Python so you'll need to set up the dependencies.
We recommend using a virtual environment.
Use your favorite package manager (e.g. pip) to install the various `requirements.txt` files found in: the top-level directory, the [scifire](scifire) directory, and the [statistics](statistics) directory if you plan to analyze the results too.

### Running experiments

See [run.py](run.py) for a script that defines multiple configurations and automatically runs each of them.  It parallelizes the experiments (runs multiple instances to make use of multi-core systems) if not using the multi-threaded queueing simulator.

To directly define and run a single experiment via command line, see the `firedex_[algorithm_]experiment.py` files.  Run one of them with the `-h` flag for a list of configurable options.

Alternatively, you can also manage these configurations through the Python files located in the [config](config) directory.

### Visualizing Results

Run `./stats` to try out the analysis script.  It uses pandas DataFrames that you'll need to manipulate through the code at the very bottom of the [fire_statistics.py](statistics/fire_statistics.py) script.
You can either plot the graphs directly or (our preferred method for generating finalized results) export them to .csv and use another plotting tool.

