import numpy as np
import cvxpy as cvx

from firedex_algorithm import FiredexAlgorithm
from ..firedex_configuration import FiredexConfiguration  # for type hinting

import logging
log = logging.getLogger(__name__)


### NOTE: we don't actually implement the class here since the class is designed for priorities only...
###   future refactor of base class may change this or we may try setting prios through optimization too...
#
# class OptFiredexAlgorithm(FiredexAlgorithm):
#     """
#     Algorithm for assigning drop rates according to an optimization-based approach that maximizes utility.
#     """


def run_opt_alg(configuration, algorithm, sub_flow_map):
    """
    Runs the optimization algorithm by calling out to the other functions below that do the actual work;
    returns a mapping of network flows to drop rates
    :param configuration:
    :type configuration: FiredexConfiguration
    :param algorithm:
    :type algorithm: FiredexAlgorithm
    :param sub_flow_map:
    :type sub_flow_map: dict[Subscription,NetworkFlow]
    :return:
    :rtype: dict[NetworkFlow,float]
    """

    # XXX: need to zero out drop rates in case they haven't been set yet since we need to access the
    # analytical model, which in turn needs to know the current drop rates
    algorithm.zero_drop_rates(configuration)

    inputs = get_opt_alg_inputs(configuration, algorithm, sub_flow_map)
    drop_rates = do_run_opt_alg(*inputs)
    drop_rates = np.squeeze(np.asarray(drop_rates))

    assert len(drop_rates) == len(configuration.net_flows)

    # XXX: strangely the opt formulation can lead to slightly negative values so round them if close to bounds:
    final_drop_rates = []
    for dr in drop_rates:
        if np.isclose(dr, [0.0]):
            dr = 0.0
        # WARNING: this could potentially cause a ro violation if we actually needed to set the drop rate to this just-under-one value!
        elif np.isclose(dr, [1.0]):
            dr = 1.0
        final_drop_rates.append(dr)

    res = {flow: drop_rate for flow, drop_rate in zip(configuration.net_flows, final_drop_rates)}

    return res


def get_opt_alg_inputs(configuration, algorithm, sub_flow_map):
    """
    Extracts the relevant input parameters for the optimization drop rate policy algorithm from the given config and
    converts them to the appropriate matrix-form representations.
    :param configuration:
    :type configuration: FiredexConfiguration
    :param algorithm:
    :type algorithm: FiredexAlgorithm
    :param sub_flow_map:
    :type sub_flow_map: dict[Subscription,NetworkFlow]
    :return:
    """

    alphas = np.array(configuration.subscription_utility_weights)
    lambdas = np.array(algorithm.arrival_rates(configuration).switch_in)
    mus = np.array(algorithm.service_rates(configuration).switch_out)
    error = configuration.error_rate
    ro_tolerance = algorithm.ro_tolerance

    # convert to binary-valued matrix format where rows are subscriptions and the column corresponding to its net flow is 1
    flow_idx_map = {f: i for i, f in enumerate(configuration.net_flows)}
    mat = []
    nsubs = len(sub_flow_map)
    nflows = len(flow_idx_map)
    for sub in configuration.subscriptions:
        flow = sub_flow_map[sub]
        row = [0]*nflows
        row[flow_idx_map[flow]] = 1
        mat.append(row)
    sub_flow_map = np.matrix(mat)

    # check that everything looks good:

    # one flow per subscription
    assert np.all(np.sum(sub_flow_map, axis=1) == np.ones((nsubs, 1))),\
        "sub_flow_map should have exactly one 1-value per row i.e. one flow per subscription!"
    assert sub_flow_map.shape == (nsubs, nflows)

    assert len(lambdas) == len(mus), "lambdas and mus must be same length!"
    assert len(lambdas) == nsubs, "lambdas should be per-subscription!"

    return alphas, lambdas, mus, error, sub_flow_map, ro_tolerance


def do_run_opt_alg(alphas, lambdas, mus, error, sub_flow_map, ro_tolerance):
    """
    Note the repetition between lambdas/mus
    :param alphas: utility weights of each subscription
    :param lambdas: arrival rates (at SDN switch) of events for each subscription
    :param mus: service rates of events for each subscription
    :param error: error rate of channel
    :param sub_flow_map: binary-valued matrix mapping subscriptions (rows) to network flows (cols)
    :param ro_tolerance:
    :return: drop_rates to be applied for each net flow
    """

    nsubs = len(alphas)
    nflows = sub_flow_map.shape[1]

    # ENHANCE: figure out how to also set the mappings through opt approach!
    # subscription-to-flow mapping
    # sfm = cvx.Bool(nsubs, nflows, name="sub_flow_map")

    drop_rates = cvx.Variable(nflows, name="drop_rates")

    # NOTE: making these parameters isn't helpful since we can't make ALL of them params...
    #     need to rebuild problem whenever configuration changes anyway!
    a = cvx.Parameter(nsubs, name="alphas", sign="positive", value=alphas)
    e = cvx.Parameter(name="error_rate", sign="positive", value=error)
    # l = cvx.Parameter(n, name="lambdas", sign="positive", value=lambdas)
    # mus = cvx.Parameter(n, name="mus", sign="positive", value=mus)

    log_exp = cvx.log1p(cvx.mul_elemwise((1.0 - e), sub_flow_map * (1.0 - drop_rates)))
    objective = cvx.Maximize(cvx.sum_entries(cvx.mul_elemwise(a, log_exp)))
    # NOTE: we don't include the lambdas in objective function at all since we're really interested in success rate

    ros = np.array([lam/mu for lam, mu in zip(lambdas, mus)])
    ro_condition = cvx.sum_entries(cvx.mul_elemwise(ros, sub_flow_map * (1.0 - drop_rates)))  # ro condition / Bandwidth constraint

    # ENHANCE: do this properly instead of a hack?  but how to make mus a scalar constant?  probably using kl_div...
    # ro_condition = cvx.sum_entries(cvx.mul_elemwise(lambdas, P) / mus)  # ro condition / Bandwidth constraint
    # ro_condition = cvx.sum_entries(cvx.exp(cvx.log(cvx.mul_elemwise(lambdas, P)) - cvx.log(mus)))  # ro condition / Bandwidth constraint
    # constraints = [cvx.sum_entries(np.array(l/p for l, p in zip(cvx.mul_elemwise(lambdas, P), mus))) <= 1.0],  # ro condition / Bandwidth constraint

    constraints = [ro_condition <= 1.0 - ro_tolerance,
                   drop_rates >= 0.0, drop_rates <= 1.0]  # proper drop rates range

    prob = cvx.Problem(objective, constraints)

    # log.debug("opt problem is solveable? (DCP): %s" % prob.is_dcp())

    prob.solve()

    log.debug("opt finished! solution has utility: %s" % prob.value)
    # print "RO SUM:", ro_condition.value

    return drop_rates.value
