# Copyright 2008-2019 University of Washington
#
# This file is part of DisMod-MR.
#
# DisMod-MR is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DisMod-MR is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with DisMod-MR.  If not, see <http://www.gnu.org/licenses/>.
""" Expert prior models"""
import numpy as np
import pymc as mc

'''
CH 4
level bound priors
level value priors
monotonicity priors
'''

def similar(data_type, mu_child, mu_parent, sigma_parent, sigma_difference, offset=1.e-9):
    """ Generate PyMC objects encoding a simliarity prior on mu_child to mu_parent

    Parameters
    ----------
    data_type : str
    mu_child : array or pymc.Node, the predicted values for the child node in the hierarchy
    mu_parent : array, the predicted values for the parent node in the hierarchy
    sigma_parent : array, the predicted standard devation for the parent node in the hierarchy
    sigma_difference : float, the prior on how similar between child is to parent
    offest : float, an offset to deal with log of zero

    Results
    -------
    Returns dict of PyMC objects, including parent_mu_age and parent_sim the similarity potential
    """
    if isinstance(mu_parent, mc.Node):
        tau = 1. / (sigma_parent**2 + sigma_difference**2)
    else:
        tau = 1. / (((sigma_parent+offset)/(mu_parent+offset))**2 + sigma_difference**2)

    @mc.potential(name='parent_similarity_%s'%data_type)
    def parent_similarity(mu_child=mu_child, mu_parent=mu_parent,
                          tau=tau):
        log_mu_child = np.log(mu_child.clip(offset, np.inf))
        log_mu_parent = np.log(mu_parent.clip(offset, np.inf))
        return mc.normal_like(log_mu_child, log_mu_parent, tau)

    return dict(parent_similarity=parent_similarity)


def level_constraints(data_type, parameters, unconstrained_mu_age, ages):
    """ Generate PyMC objects implementing priors on the value of the rate function

    :Parameters:
      - `data_type` : str
      - `parameters` : dict of dicts, with keys level_value and level_bounds
           level_value with keys value, age_before, and age_after
           level_bounds with keys lower and upper
      - `unconstrained_mu_age` : pymc.Node with values of PCGP
      - `ages` : array

    :Results:
      - Returns dict of PyMC objects, including 'unconstrained_mu_age' and 'mu_age'

    """
    if 'level_value' not in parameters or 'level_bounds' not in parameters:
        return {}

    @mc.deterministic(name='value_constrained_mu_age_%s'%data_type)
    def mu_age(unconstrained_mu_age=unconstrained_mu_age,
               value=parameters['level_value']['value'],
               age_before=np.clip(parameters['level_value']['age_before']-ages[0], 0, len(ages)),
               age_after=np.clip(parameters['level_value']['age_after']-ages[0], 0, len(ages)),
               lower=parameters['level_bounds']['lower'],
               upper=parameters['level_bounds']['upper']):
        mu_age = unconstrained_mu_age.copy()
        mu_age[:int(age_before)] = value
        if age_after < len(mu_age)-1:
            mu_age[(age_after+1):] = value
        return mu_age.clip(lower, upper)
    mu_sim = similar('value_constrained_mu_age_%s'%data_type, mu_age, unconstrained_mu_age, 0., .01, 1.e-6)

    return dict(mu_age=mu_age, unconstrained_mu_age=unconstrained_mu_age, mu_sim=mu_sim)


def covariate_level_constraints(name, model, vars, ages):
    """ Generate PyMC objects implementing priors on the value of the covariate adjusted rate function

    :Parameters:
      - `name` : str
      - `parameters` : dict
      - `unconstrained_mu_age` : pymc.Node with values of PCGP
      - `ages` : array

    :Results:
      - Returns dict of PyMC objects, including 'unconstrained_mu_age' and 'mu_age'

    """
    if name not in model.parameters or 'level_value' not in model.parameters[name] or 'level_bounds' not in model.parameters[name]:
        return {}

    X_sex_max = .5 - vars['X_shift']['x_sex']
    X_sex_min = -.5 - vars['X_shift']['x_sex']  # make sure that the range of sex covariates is included
    index_map = dict([[key, i] for i,key in enumerate(vars['X_shift'].index)])
    sex_index = index_map['x_sex']

    U_all = []
    nodes = ['all']
    for l in range(1,4):
        nodes = [item for n in nodes for item in model.hierarchy.successors(n)]
        U_i = np.array([col in nodes for col in vars['U'].columns])
        if U_i.sum() > 0:
            U_all.append(U_i)

    @mc.potential(name='covariate_constraint_%s'%name)
    def covariate_constraint(mu=vars['mu_age'], alpha=vars['alpha'], beta=vars['beta'],
                             U_all=U_all,
                             X_sex_max=X_sex_max,
                             X_sex_min=X_sex_min,
                             lower=np.log(model.parameters[name]['level_bounds']['lower']),
                             upper=np.log(model.parameters[name]['level_bounds']['upper'])):
        log_mu_max = np.log(mu.max())
        log_mu_min = np.log(mu.min())

        alpha = np.array([float(x) for x in alpha])
        if len(alpha) > 0:
            for U_i in U_all:
                log_mu_max += max(0, alpha[U_i].max())
                log_mu_min += min(0, alpha[U_i].min())

        # this estimate is too crude, and is causing problems
        #if len(beta) > 0:
        #    log_mu_max += np.sum(np.maximum(X_max*beta, X_min*beta))
        #    log_mu_min += np.sum(np.minimum(X_max*beta, X_min*beta))

        # but leaving out the sex effect results in strange problems, too
        log_mu_max += X_sex_max*float(beta[sex_index])
        log_mu_min += X_sex_min*float(beta[sex_index])

        lower_violation = min(0., log_mu_min - lower)
        upper_violation = max(0., log_mu_max - upper)
        return mc.normal_like([lower_violation, upper_violation], 0., 1.e-6**-2)

    return dict(covariate_constraint=covariate_constraint)


def derivative_constraints(name, parameters, mu_age, ages):
    """ Generate PyMC objects implementing priors on the value of the rate function

    :Parameters:
      - `name` : str
      - `parameters` : dict of dicts, with keys increasing and decreasing
           each with keys age_start and age_end
      - `mu_age` : pymc.Node with values of PCGP
      - `ages` : array

    :Results:
      - Returns dict of PyMC objects, including 'mu_age_derivative_potential'

    """
    if 'increasing' not in parameters or 'decreasing' not in parameters:
        return {}

    @mc.potential(name='mu_age_derivative_potential_%s'%name)
    def mu_age_derivative_potential(mu_age=mu_age,
                                    increasing_a0=np.clip(parameters['increasing']['age_start']-ages[0], 0, len(ages)),
                                    increasing_a1=np.clip(parameters['increasing']['age_end']-ages[0], 0, len(ages)),
                                    decreasing_a0=np.clip(parameters['decreasing']['age_start']-ages[0], 0, len(ages)),
                                    decreasing_a1=np.clip(parameters['decreasing']['age_end']-ages[0], 0, len(ages))):
        mu_prime = np.diff(mu_age)
        inc_violation = mu_prime[increasing_a0:increasing_a1].clip(-np.inf, 0.).sum()
        dec_violation = mu_prime[decreasing_a0:decreasing_a1].clip(0., np.inf).sum()
        return -1.e12 * (inc_violation**2 + dec_violation**2)

    return dict(mu_age_derivative_potential=mu_age_derivative_potential)
