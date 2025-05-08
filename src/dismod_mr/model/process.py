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
""" Dismod-MR model creation methods"""
import numpy as np
import pymc as mc
import scipy.interpolate

import dismod_mr


def age_specific_rate(model, 
                      data_type, 
                      reference_area='all', 
                      reference_sex='total', 
                      reference_year='all',
                      mu_age=None, 
                      mu_age_parent=None, 
                      sigma_age_parent=None,
                      rate_type='neg_binom', 
                      lower_bound=None, 
                      interpolation_method='linear',
                      include_covariates=True, 
                      zero_re=False):
    # TODO: alternative rate_type 인터페이스 문서화 및 모델에 참조값 기록
    '''
    * γ : 스플라인 knot 파라미터들 (예: vars['gamma'])
    * α : 랜덤 효과 (지역·성별·연도 등)
    * β : 고정 효과 (공변량 회귀계수)
    * η, ζ : 분산 관련 파라미터
    * 기타 노드(p_obs, pi_sim, smooth_gamma, parent_similarity 등)는 로그우도/로그사전분포 구성용
    '''
    """Generate PyMC objects for epidemiological age-interval data model"""

    _data_type = data_type
    result = dismod_mr.data.ModelVars()  # 반환용 컨테이너 초기화

    # 부모 mu_age_parent나 sigma_age_parent에 NaN이 있으면 계층적 prior 무시
    if (isinstance(mu_age_parent, np.ndarray) and np.any(np.isnan(mu_age_parent))) or \
       (isinstance(sigma_age_parent, np.ndarray) and np.any(np.isnan(sigma_age_parent))):
        mu_age_parent = None
        sigma_age_parent = None
        print('WARNING: nan found in parent mu/sigma.  Ignoring')

    # 연령 목록, 데이터, 파라미터, 계층(hierarchy) 참조
    ages = np.array(model.parameters['ages'])
    data = model.get_data(data_type)
    if lower_bound:
        lb_data = model.get_data(lower_bound)
    parameters = model.parameters.get(data_type, {})
    area_hierarchy = model.hierarchy

    vars = dismod_mr.data.ModelVars()
    vars += dict(data=data)  # 원시 입력 데이터 저장

    # 스플라인 knot 설정: 직접 mesh 지정 없으면 5년 간격
    if 'parameter_age_mesh' in parameters:
        knots = np.array(parameters['parameter_age_mesh'])
    else:
        knots = np.arange(ages[0], ages[-1]+1, 5)

    # smoothing 강도: 문자열 매핑 또는 직접 숫자
    smoothing_dict = {'No Prior': np.inf, 'Slightly': .5, 'Moderately': .05, 'Very': .005}
    if 'smoothness' in parameters:
        try:
            smoothing = float(parameters['smoothness']['amount'])
        except ValueError:
            smoothing = smoothing_dict[parameters['smoothness']['amount']]
    else:
        smoothing = 0.

    # mu_age 지정 없으면 spline 작성, 있으면 재활용
    if mu_age is None:
        vars.update(
            dismod_mr.model.spline.spline(_data_type,
                                          ages=ages,
                                          knots=knots,
                                          smoothing=smoothing,
                                          interpolation_method=interpolation_method)
        )
    else:
        vars.update(dict(mu_age=mu_age, ages=ages))

    # spline level 및 derivative 제약 추가
    vars.update(dismod_mr.model.priors.level_constraints(_data_type, parameters, vars['mu_age'], ages))
    vars.update(dismod_mr.model.priors.derivative_constraints(_data_type, parameters, vars['mu_age'], ages))

    # 부모 패턴과의 유사성 prior 적용 (계층 모델)
    if mu_age_parent is not None:
        vars.update(
            dismod_mr.model.priors.similar('parent_similarity_%s' % _data_type,
                                           vars['mu_age'],
                                           mu_age_parent,
                                           sigma_age_parent,
                                           0.)
        )
        # 초기 gamma 값 부모 mu 기반 설정
        if mu_age is None:
            initial_mu = mu_age_parent.value if isinstance(mu_age_parent, mc.Node) else mu_age_parent
            for i, k_i in enumerate(knots):
                vars['gamma'][i].value = np.log(initial_mu[k_i - ages[0]]).clip(-12, 6)

    age_weights = np.ones_like(vars['mu_age'].value)  # 기본 가중치

    # 데이터가 존재할 때
    if len(data) > 0:
        # interval average 근사
        vars.update(
            dismod_mr.model.age_groups.age_standardize_approx(_data_type,
                                                              age_weights,
                                                              vars['mu_age'],
                                                              data['age_start'],
                                                              data['age_end'],
                                                              ages)
        )
        # covariate 모델링 or interval mu 직접 사용
        if include_covariates:
            vars.update(
                dismod_mr.model.covariates.mean_covariate_model(_data_type,
                                                                 vars['mu_interval'],
                                                                 data,
                                                                 parameters,
                                                                 model,
                                                                 reference_area,
                                                                 reference_sex,
                                                                 reference_year,
                                                                 zero_re=zero_re)
            )
        else:
            vars.update({'pi': vars['mu_interval']})

        # 결측 standard_error 보정: CI 이용
        missing_se = np.isnan(data['standard_error']) | (data['standard_error'] < 0)
        if missing_se.any():

            data.loc[missing_se, 'standard_error'] = (
                data.loc[missing_se, 'upper_ci'] - data.loc[missing_se, 'lower_ci']
            ) / (2 * 1.96)

        # 결측 effective_sample_size 보정
        missing_ess = data['effective_sample_size'].isna()
        if missing_ess.any():
            data.loc[missing_ess, 'effective_sample_size'] = (
                data.loc[missing_ess, 'value'] * (1 - data.loc[missing_ess, 'value'])
                / data.loc[missing_ess, 'standard_error'] ** 2
            )

        # 각 rate_type별 likelihood 설정
        if rate_type == 'neg_binom':
            # non-positive or missing ESS 처리: 음수, 0, NaN 모두 0으로 설정
            bad_ess = (data['effective_sample_size'] <= 0) | data['effective_sample_size'].isna()
            if bad_ess.any():
                print(f'WARNING: {bad_ess.sum()} rows of {_data_type} have non-positive or missing ESS.')
                data.loc[bad_ess, 'effective_sample_size'] = 0.0
            # 과도한 ess 처리
            big_ess = data['effective_sample_size'] >= 1e10
            if big_ess.any():
                print(f'WARNING: {big_ess.sum()} rows of {_data_type} ess >1e10.')
                data.loc[big_ess, 'effective_sample_size'] = 1e10

            # heterogeneity에 따른 dispersion lower bound
            lower = {'Slightly': 9., 'Moderately': 3., 'Very': 1.}.get(
                parameters.get('heterogeneity', None), 1.)
            if data_type == 'pf':
                lower = 1e12

            # dispersion covariate prior
            vars.update(
                dismod_mr.model.covariates.dispersion_covariate_model(_data_type,
                                                                      data,
                                                                      lower,
                                                                      lower * 9.)
            )

            # neg binomial likelihood
            vars.update(
                dismod_mr.model.likelihood.neg_binom(_data_type,
                                                      vars['pi'],
                                                      vars['delta'],
                                                      data['value'],
                                                      data['effective_sample_size'])
            )

        elif rate_type == 'log_normal':
            # missing SE 처리
            missing = data['standard_error'] < 0
            if missing.any():
                print(f'WARNING: {missing.sum()} rows of {_data_type} no SE.')
                data.loc[missing, 'standard_error'] = 1e6
            # sigma prior
            vars['sigma'] = mc.Uniform(f'sigma_{_data_type}', lower=.0001, upper=1., value=.01)
            # log-normal likelihood
            vars.update(
                dismod_mr.model.likelihood.log_normal(_data_type,
                                                       vars['pi'],
                                                       vars['sigma'],
                                                       data['value'],
                                                       data['standard_error'])
            )

        elif rate_type == 'normal':
            # missing SE 처리
            missing = data['standard_error'] < 0
            if missing.any():
                print(f'WARNING: {missing.sum()} rows of {_data_type} no SE.')
                data.loc[missing, 'standard_error'] = 1e6
            # sigma prior
            vars['sigma'] = mc.Uniform(f'sigma_{_data_type}', lower=.0001, upper=.1, value=.01)
            # normal likelihood
            vars.update(
                dismod_mr.model.likelihood.normal(_data_type,
                                                  vars['pi'],
                                                  vars['sigma'],
                                                  data['value'],
                                                  data['standard_error'])
            )

        elif rate_type == 'binom':
            # invalid ess 처리
            bad_ess = data['effective_sample_size'] < 0
            if bad_ess.any():
                print(f'WARNING: {bad_ess.sum()} rows of {_data_type} invalid ess.')
                data.loc[bad_ess, 'effective_sample_size'] = 0.0
            # binomial likelihood
            vars += dismod_mr.model.likelihood.binom(_data_type,
                                                      vars['pi'],
                                                      data['value'],
                                                      data['effective_sample_size'])

        elif rate_type in ['beta_binom', 'beta_binom_2']:
            # beta-binomial variants
            fn = dismod_mr.model.likelihood.beta_binom if rate_type == 'beta_binom' else dismod_mr.model.likelihood.beta_binom_2
            vars += fn(_data_type,
                       vars['pi'],
                       data['value'],
                       data['effective_sample_size'])

        elif rate_type == 'poisson':
            # invalid ess 처리
            bad_ess = data['effective_sample_size'] < 0
            if bad_ess.any():
                print(f'WARNING: {bad_ess.sum()} rows of {_data_type} invalid ess.')
                data.loc[bad_ess, 'effective_sample_size'] = 0.0
            # poisson likelihood
            vars += dismod_mr.model.likelihood.poisson(_data_type,
                                                        vars['pi'],
                                                        data['value'],
                                                        data['effective_sample_size'])

        elif rate_type == 'offset_log_normal':
            # offset log-normal likelihood
            vars['sigma'] = mc.Uniform(f'sigma_{_data_type}', lower=.0001, upper=10., value=.01)
            vars += dismod_mr.model.likelihood.offset_log_normal(_data_type,
                                                                  vars['pi'],
                                                                  vars['sigma'],
                                                                  data['value'],
                                                                  data['standard_error'])
            
        else:
            raise Exception(f'rate_model "{rate_type}" not implemented')

    else:
        # 데이터 없으면 covariate 만 모델링
        if include_covariates:
            vars.update(
                dismod_mr.model.covariates.mean_covariate_model(_data_type,
                                                                 [],
                                                                 data,
                                                                 parameters,
                                                                 model,
                                                                 reference_area,
                                                                 reference_sex,
                                                                 reference_year,
                                                                 zero_re=zero_re)
            )

    # covariate-level 제약 추가
    if include_covariates:
        vars.update(dismod_mr.model.priors.covariate_level_constraints(_data_type, model, vars, ages))

    # lower_bound 데이터 별도 처리
    if lower_bound and len(lb_data) > 0:
        # 하위 경계 interval approx
        vars['lb'] = dismod_mr.model.age_groups.age_standardize_approx(
            f'lb_{_data_type}',
            age_weights,
            vars['mu_age'],
            lb_data['age_start'],
            lb_data['age_end'],
            ages
        )
        # covariate or direct pi
        if include_covariates:
            vars['lb'].update(
                dismod_mr.model.covariates.mean_covariate_model(
                    f'lb_{_data_type}',
                    vars['lb']['mu_interval'],
                    lb_data,
                    parameters,
                    model,
                    reference_area,
                    reference_sex,
                    reference_year,
                    zero_re=zero_re
                )
            )
        else:
            vars['lb'].update({'pi': vars['lb']['mu_interval']})
        # dispersion covariate
        vars['lb'].update(
            dismod_mr.model.covariates.dispersion_covariate_model(
                f'lb_{_data_type}',
                lb_data,
                1e12,
                1e13
            )
        )
        # missing SE, ESS 처리 동일 로직
        missing_se_lb = (lb_data['standard_error'] <= 0) | lb_data['standard_error'].isna()
        lb_data.loc[missing_se_lb, 'standard_error'] = (
            lb_data.loc[missing_se_lb, 'upper_ci'] - lb_data.loc[missing_se_lb, 'lower_ci']
        ) / (2 * 1.96)
        missing_ess_lb = lb_data['effective_sample_size'].isna()
        lb_data.loc[missing_ess_lb, 'effective_sample_size'] = (
            lb_data.loc[missing_ess_lb, 'value'] * (
                1 - lb_data.loc[missing_ess_lb, 'value']
            ) / lb_data.loc[missing_ess_lb, 'standard_error'] ** 2
        )
        bad_ess_lb = lb_data['effective_sample_size'] <= 0
        if bad_ess_lb.any():
            print(f'WARNING: {bad_ess_lb.sum()} rows of {_data_type} lower bound invalid ess.')
            lb_data.loc[bad_ess_lb, 'effective_sample_size'] = 1.0
        # neg_binom lower-bound likelihood
        vars['lb'].update(
            dismod_mr.model.likelihood.neg_binom_lower_bound(
                f'lb_{_data_type}',
                vars['lb']['pi'],
                vars['lb']['delta'],
                lb_data['value'],
                lb_data['effective_sample_size']
            )
        )

    result[data_type] = vars
    return result




# def age_specific_rate(model, 
#                       data_type, 
#                       reference_area='all', 
#                       reference_sex='total', 
#                       reference_year='all',
#                       mu_age=None, 
#                       mu_age_parent=None, 
#                       sigma_age_parent=None,
#                       rate_type='neg_binom', 
#                       lower_bound=None, 
#                       interpolation_method='linear',
#                       include_covariates=True, 
#                       zero_re=False):
#     # TODO: expose (and document) interface for alternative rate_type as well as other options,
#     # record reference values in the model
#     '''
#     * $\gamma$ : **스플라인 knot** 파라미터들 (예: `vars['gamma']`)
#     * $\alpha$ : **랜덤 효과(R.E.)** (예: 지역·성별·연도 등)
#     * $\beta$ : **고정 효과(F.E.)** (예: 공변량 회귀계수)
#     * $\eta, \zeta$ : **분산(이질성) 관련 파라미터**
#     * 기타 함께 들어가는 항들(예: `p_obs`, `pi_sim`, `smooth_gamma`, `parent_similarity` 등)은 모델의 로그우도/로그사전분포를 구성하는 데 필요하지만, 
#     직접적으로 최적화 대상이 아닌 이미 정해진(?) Observed Stochastic이거나 Potential 노드일 수도 있습니다.
#     '''
#     """ Generate PyMC objects for model of epidemological age-interval data

#     :Parameters:
#       - `model` : data.ModelData
#       - `data_type` : str, one of 'i', 'r', 'f', 'p', or 'pf'
#       - `reference_area, reference_sex, reference_year` : the node of the model to fit consistently
#       - `mu_age` : pymc.Node, will be used as the age pattern, set to None if not needed
#       - `mu_age_parent` : pymc.Node, will be used as the age pattern of the parent of the root area, set to None if not needed
#       - `sigma_age_parent` : pymc.Node, will be used as the standard deviation of the age pattern, set to None if not needed
#       - `rate_type` : str, optional. One of 'beta_binom', 'beta_binom_2', 'binom', 'log_normal_model', 'neg_binom', 'neg_binom_lower_bound_model', 'neg_binom_model', 'normal_model', 'offest_log_normal', or 'poisson'
#       - `lower_bound` :
#       - `interpolation_method` : str, optional, one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic, or 'cubic'
#       - `include_covariates` : boolean
#       - `zero_re` : boolean, change one stoch from each set of siblings in area hierarchy to a 'sum to zero' deterministic

#     :Results:
#       - Returns dict of PyMC objects, including 'pi', the covariate adjusted predicted values for each row of data

#     """

#     name = data_type
#     result = dismod_mr.data.ModelVars()

#     if (isinstance(mu_age_parent, np.ndarray) and np.any(np.isnan(mu_age_parent))) \
#            or (isinstance(sigma_age_parent, np.ndarray) and np.any(np.isnan(sigma_age_parent))):
#         mu_age_parent = None
#         sigma_age_parent = None
#         print('WARNING: nan found in parent mu/sigma.  Ignoring')

#     ages = np.array(model.parameters['ages'])
#     data = model.get_data(data_type)
#     if lower_bound:
#         lb_data = model.get_data(lower_bound)
#     parameters = model.parameters.get(data_type, {})
#     area_hierarchy = model.hierarchy

#     vars = dismod_mr.data.ModelVars()
#     vars += dict(data=data)

#     if 'parameter_age_mesh' in parameters:
#         knots = np.array(parameters['parameter_age_mesh'])
#     else:
#         knots = np.arange(ages[0], ages[-1]+1, 5)

#     smoothing_dict = {'No Prior':np.inf, 'Slightly':.5, 'Moderately': .05, 'Very': .005}
#     if 'smoothness' in parameters:
#         try:
#             smoothing = float(parameters['smoothness']['amount'])
#         except ValueError:
#             smoothing = smoothing_dict[parameters['smoothness']['amount']]
#     else:
#         smoothing = 0.

#     if mu_age == None:
#         vars.update(
#             dismod_mr.model.spline.spline(name, ages=ages, knots=knots, smoothing=smoothing, interpolation_method=interpolation_method)
#             )
#     else:
#         vars.update(dict(mu_age=mu_age, ages=ages))

#     vars.update(dismod_mr.model.priors.level_constraints(name, parameters, vars['mu_age'], ages))
#     vars.update(dismod_mr.model.priors.derivative_constraints(name, parameters, vars['mu_age'], ages))


#     if type(mu_age_parent) != type(None):
#         # setup a hierarchical prior on the simliarity between the
#         # consistent estimate here and (inconsistent) estimate for its
#         # parent in the areas hierarchy
#         #weight_dict = {'Unusable': 10., 'Slightly': 10., 'Moderately': 1., 'Very': .1}
#         #weight = weight_dict[parameters['heterogeneity']]

#         vars.update(
#             dismod_mr.model.priors.similar('parent_similarity_%s' % name, vars['mu_age'], mu_age_parent, sigma_age_parent, 0.)
#             )
#         # also use this as the initial value for the age pattern, if it is not already specified
#         if mu_age == None:
#             if isinstance(mu_age_parent, mc.Node):  # TODO: test this code
#                 initial_mu = mu_age_parent.value
#             else:
#                 initial_mu = mu_age_parent

#             for i, k_i in enumerate(knots):
#                 vars['gamma'][i].value = (np.log(initial_mu[k_i-ages[0]])).clip(-12,6)



#     age_weights = np.ones_like(vars['mu_age'].value) # TODO: use age pattern appropriate to the rate type
#     if len(data) > 0:
#         vars.update(
#             dismod_mr.model.age_groups.age_standardize_approx(name, age_weights, vars['mu_age'], data['age_start'], data['age_end'], ages)
#             )

#         # uncomment the following to effectively remove alleffects
#         #if 'random_effects' in parameters:
#         #    for i in range(5):
#         #        effect = 'sigma_alpha_%s_%d' % (name, i)
#         #        parameters['random_effects'][effect] = dict(dist='TruncatedNormal', mu=.0001, sigma=.00001, lower=.00009, upper=.00011)
#         #if 'fixed_effects' in parameters:
#         #    for effect in ['x_sex', 'x_LDI_id_Updated_7July2011']:
#         #        parameters['fixed_effects'][effect] = dict(dist='normal', mu=.0001, sigma=.00001)

#         if include_covariates:
#             vars.update(
#                 dismod_mr.model.covariates.mean_covariate_model(name, vars['mu_interval'], data, parameters, model, reference_area, reference_sex, reference_year, zero_re=zero_re)
#                 )
#         else:
#             vars.update({'pi': vars['mu_interval']})

#         ## ensure that all data has uncertainty quantified appropriately
#         # first replace all missing se from ci
#         missing_se = np.isnan(data['standard_error']) | (data['standard_error'] < 0)
#         # missing_se is your boolean mask indicating where standard_error is missing
#         if missing_se.any():
#             # only print & fill if there’s at least one True
#             print(data[missing_se])
#             data.loc[missing_se, 'standard_error'] = (
#                 data.loc[missing_se, 'upper_ci'] - data.loc[missing_se, 'lower_ci']
#             ) / (2 * 1.96)

#         # then replace all missing ess with se
#         missing_ess = data[np.isnan(data['effective_sample_size'])].index
#         data.loc[missing_ess, 'effective_sample_size'] = \
#                 data.loc[missing_ess, 'value']*(1-data.loc[missing_ess, 'value']) \
#                     / data.loc[missing_ess, 'standard_error']**2

#         if rate_type == 'neg_binom':

#             # warn and drop data that doesn't have effective sample size quantified, or is is non-positive
#             missing_ess = np.isnan(data['effective_sample_size']) | (data['effective_sample_size'] < 0)
#             if sum(missing_ess) > 0:
#                 print('WARNING: %d rows of %s data has invalid quantification of uncertainty.' % (sum(missing_ess), name))
#                 missing_ess = data[missing_ess].index
#                 data.loc[missing_ess, 'effective_sample_size'] = 0.0

#             # warn and change data where ess is unreasonably huge
#             large_ess = data['effective_sample_size'] >= 1.e10
#             if sum(large_ess) > 0:
#                 print('WARNING: %d rows of %s data have effective sample size exceeding 10 billion.' % (sum(large_ess), name))
#                 data['effective_sample_size'][large_ess] = 1.e10


#             if 'heterogeneity' in parameters:
#                 lower_dict = {'Slightly': 9., 'Moderately': 3., 'Very': 1.}
#                 lower = lower_dict[parameters['heterogeneity']]
#             else:
#                 lower = 1.

#             # special case, treat pf data as poisson
#             if data_type == 'pf':
#                 lower = 1.e12

#             vars.update(
#                 dismod_mr.model.covariates.dispersion_covariate_model(name, data, lower, lower * 9.)
#                 )

#             vars.update(
#                 dismod_mr.model.likelihood.neg_binom(name, vars['pi'], vars['delta'], data['value'], data['effective_sample_size'])
#                 )
#         elif rate_type == 'log_normal':

#             # warn and drop data that doesn't have effective sample size quantified
#             missing = np.isnan(data['standard_error']) | (data['standard_error'] < 0)
#             if sum(missing) > 0:
#                 print('WARNING: %d rows of %s data has no quantification of uncertainty.' % (sum(missing), name))
#                 data['standard_error'][missing] = 1.e6

#             # TODO: allow options for alternative priors for sigma
#             vars['sigma'] = mc.Uniform('sigma_%s'%name, lower=.0001, upper=1., value=.01)
#             #vars['sigma'] = mc.Exponential('sigma_%s'%name, beta=100., value=.01)
#             vars.update(
#                 dismod_mr.model.likelihood.log_normal(name, vars['pi'], vars['sigma'], data['value'], data['standard_error'])
#                 )
#         elif rate_type == 'normal':

#             # warn and drop data that doesn't have standard error quantified
#             missing = np.isnan(data['standard_error']) | (data['standard_error'] < 0)
#             if sum(missing) > 0:
#                 print('WARNING: %d rows of %s data has no quantification of uncertainty.' % (sum(missing), name))
#                 data['standard_error'][missing] = 1.e6

#             vars['sigma'] = mc.Uniform('sigma_%s'%name, lower=.0001, upper=.1, value=.01)
#             vars.update(
#                 dismod_mr.model.likelihood.normal(name, vars['pi'], vars['sigma'], data['value'], data['standard_error'])
#                 )
#         elif rate_type == 'binom':
#             missing_ess = np.isnan(data['effective_sample_size']) | (data['effective_sample_size'] < 0)
#             if sum(missing_ess) > 0:
#                 print('WARNING: %d rows of %s data has invalid quantification of uncertainty.' % (sum(missing_ess), name))
#                 data['effective_sample_size'][missing_ess] = 0.0
#             vars += dismod_mr.model.likelihood.binom(name, vars['pi'], data['value'], data['effective_sample_size'])
#         elif rate_type == 'beta_binom':
#             vars += dismod_mr.model.likelihood.beta_binom(name, vars['pi'], data['value'], data['effective_sample_size'])
#         elif rate_type == 'beta_binom_2':
#             vars += dismod_mr.model.likelihood.beta_binom_2(name, vars['pi'], data['value'], data['effective_sample_size'])
#         elif rate_type == 'poisson':
#             missing_ess = np.isnan(data['effective_sample_size']) | (data['effective_sample_size'] < 0)
#             if sum(missing_ess) > 0:
#                 print('WARNING: %d rows of %s data has invalid quantification of uncertainty.' % (sum(missing_ess), name))
#                 data['effective_sample_size'][missing_ess] = 0.0

#             vars += dismod_mr.model.likelihood.poisson(name, vars['pi'], data['value'], data['effective_sample_size'])
#         elif rate_type == 'offset_log_normal':
#             vars['sigma'] = mc.Uniform('sigma_%s'%name, lower=.0001, upper=10., value=.01)
#             vars += dismod_mr.model.likelihood.offset_log_normal(name, vars['pi'], vars['sigma'], data['value'], data['standard_error'])
#         else:
#             raise Exception('rate_model "%s" not implemented' % rate_type)
    
#     else:
#         if include_covariates:
#             vars.update(
#                 dismod_mr.model.covariates.mean_covariate_model(name, [], data, parameters, model, reference_area, reference_sex, reference_year, zero_re=zero_re)
#                 )
#     if include_covariates:
#         vars.update(dismod_mr.model.priors.covariate_level_constraints(name, model, vars, ages))

    

#     if lower_bound and len(lb_data) > 0:
#         vars['lb'] = dismod_mr.model.age_groups.age_standardize_approx('lb_%s' % name, age_weights, vars['mu_age'], lb_data['age_start'], lb_data['age_end'], ages)

#         if include_covariates:

#             vars['lb'].update(
#                 dismod_mr.model.covariates.mean_covariate_model('lb_%s' % name, vars['lb']['mu_interval'], lb_data, parameters, model, reference_area, reference_sex, reference_year, zero_re=zero_re)
#                 )
#         else:
#             vars['lb'].update({'pi': vars['lb']['mu_interval']})

#         vars['lb'].update(
#             dismod_mr.model.covariates.dispersion_covariate_model('lb_%s' % name, lb_data, 1e12, 1e13)  # treat like poisson
#             )

#         ## ensure that all data has uncertainty quantified appropriately
#         # first replace all missing se from ci
#         missing_se = np.isnan(lb_data['standard_error']) | (lb_data['standard_error'] <= 0)
#         lb_data.loc[lb_data[missing_se].index, 'standard_error'] = (lb_data['upper_ci'][missing_se] - lb_data['lower_ci'][missing_se]) / (2*1.96)

#         # then replace all missing ess with se
#         missing_ess = np.isnan(lb_data['effective_sample_size'])
#         lb_data.loc[lb_data[missing_ess].index, 'effective_sample_size'] = lb_data['value'][missing_ess]*(1-lb_data['value'][missing_ess])/lb_data['standard_error'][missing_ess]**2

#         # warn and drop lb_data that doesn't have effective sample size quantified
#         missing_ess = np.isnan(lb_data['effective_sample_size']) | (lb_data['effective_sample_size'] <= 0)
#         if sum(missing_ess) > 0:
#             print('WARNING: %d rows of %s lower bound data has no quantification of uncertainty.' % (sum(missing_ess), name))
#             lb_data.loc[lb_data[missing_ess].index, 'effective_sample_size'] = 1.0

#         vars['lb'].update(
#             dismod_mr.model.likelihood.neg_binom_lower_bound('lb_%s' % name, vars['lb']['pi'], vars['lb']['delta'], lb_data['value'], lb_data['effective_sample_size'])
#             )

#     result[data_type] = vars
#     return result






### ==============================================
## Not used for now
def consistent(model, reference_area='all', reference_sex='total', reference_year='all', priors={}, zero_re=True, rate_type='neg_binom'):
    '''
    * **`consistent`**: 
    여러 레이트(`i,r,f,p,pf,m_with` 등)를 **동시에** 연결해, “질병 자연사(disease natural history) 모델” 같은 **전체(consistent) 모델**을 구성하는 **상위 레벨(종합) 함수**.
    * 내부에서 `age_specific_rate`를 여러 번 호출하고, 그 결과를 서로 연계(ODE, logit, etc.)하여 일관적 관계를 유지하는 하나의 큰 모델을 만든다.
    '''
    """ Generate PyMC objects for consistent model of epidemological data

    :Parameters:
      - `model` : data.ModelData
      - `data_type` : str, one of 'i', 'r', 'f', 'p', or 'pf'
      - `root_area, root_sex, root_year` : the node of the model to
        fit consistently
      - `priors` : dictionary, with keys for data types for lists of
        priors on age patterns
      - `zero_re` : boolean, change one stoch from each set of
        siblings in area hierarchy to a 'sum to zero' deterministic
      - `rate_type` : str or dict, optional. One of 'beta_binom', 'beta_binom_2',
        'binom', 'log_normal_model', 'neg_binom',
        'neg_binom_lower_bound_model', 'neg_binom_model',
        'normal_model', 'offest_log_normal', or 'poisson', optionally
        as a dict, with keys i, r, f, p, m_with

    :Results:
      - Returns dict of dicts of PyMC objects, including 'i', 'p',
        'r', 'f', the covariate adjusted predicted values for each row
        of data

    .. note::
      - dict priors can contain keys (t, 'mu') and (t, 'sigma') to
        tell the consistent model about the priors on levels for the
        age-specific rate of type t (these are arrays for mean and
        standard deviation a priori for mu_age[t]
      - it can also contain dicts keyed by t alone to insert empirical
        priors on the fixed effects and random effects

    """
    # TODO: refactor the way priors are handled
    # current approach is much more complicated than necessary
    for t in 'i r pf p rr f'.split():
        if t in priors:
            model.parameters[t]['random_effects'].update(priors[t]['random_effects'])
            model.parameters[t]['fixed_effects'].update(priors[t]['fixed_effects'])

    # if rate_type is a string, make it into a dict
    if type(rate_type) == str:
        rate_type = dict(i=rate_type, r=rate_type, f=rate_type, p=rate_type, m_with=rate_type)

    rate = {}
    ages = model.parameters['ages']

    for t in 'irf':
        rate[t] = age_specific_rate(model, t, reference_area, reference_sex, reference_year,
                                    mu_age=None, mu_age_parent=priors.get((t, 'mu')), sigma_age_parent=priors.get((t, 'sigma')),
                                    zero_re=zero_re, rate_type=rate_type[t])[t] # age_specific_rate()[t] is to create proper nesting of dict

        # set initial values from data
        if t in priors:
            if isinstance(priors[t], mc.Node):
                initial = priors[t].value
            else:
                initial = np.array(priors[t])
        else:
            initial = rate[t]['mu_age'].value.copy()
            df = model.get_data(t)
            if len(df.index) > 0:
                mean_data = df.groupby(['age_start', 'age_end']).mean().reset_index()
                for i, row in mean_data.T.iteritems():
                    start = row['age_start'] - rate[t]['ages'][0]
                    end = row['age_end'] - rate[t]['ages'][0]
                    initial[int(start):int(end)] = row['value']

        for i,k in enumerate(rate[t]['knots']):
            rate[t]['gamma'][int(i)].value = np.log(initial[int(k - rate[t]['ages'][0])]+1.e-9)


    # TODO: re-engineer this m_all interpolation section
    df = model.get_data('m_all')
    if len(df.index) == 0:
        print('WARNING: all-cause mortality data not found, using m_all = .01')
        m_all = .01*np.ones_like(ages)
    else:
        mean_mortality = df.groupby(['age_start', 'age_end']).mean().reset_index()

        knots = []
        vals = []
        for i, row in mean_mortality.T.iteritems():
            knots.append((row['age_start'] + row['age_end'] + 1.) / 2.)  # FIXME: change m_all data to half-open intervals, and then remove +1 here

            vals.append(row['value'])

        # extend knots as constant beyond endpoints
        knots.insert(0, ages[0])
        vals.insert(0, vals[0])

        knots.append(ages[-1])
        vals.append(vals[-1])


        m_all = scipy.interpolate.interp1d(knots, vals, kind='linear')(ages)

    logit_C0 = mc.Uniform('logit_C0', -15, 15, value=-10.)


    N = len(m_all)
    num_step = 2  # double until it works
    ages = np.array(ages, dtype=float)
    @mc.deterministic
    def mu_age_p(logit_C0=logit_C0,
                 i=rate['i']['mu_age'],
                 r=rate['r']['mu_age'],
                 f=rate['f']['mu_age']):

        # for acute conditions, it is silly to use ODE solver to
        # derive prevalence, and it can be approximated with a simple
        # transformation of incidence
        if r.min() > 5.99:
            return i / (r + m_all + f)

        C0 = float(mc.invlogit(logit_C0))

        susceptible = np.zeros(len(ages))
        condition = np.zeros(len(ages))
        dismod_mr.model.ode.ode_function(susceptible, condition, num_step, ages, m_all, i, r, f, 1 - C0, C0)

        p = condition / (susceptible + condition)
        p[np.isnan(p)] = 0.
        return p

    p = age_specific_rate(model, 'p',
                          reference_area, reference_sex, reference_year,
                          mu_age_p,
                          mu_age_parent=priors.get(('p', 'mu')),
                          sigma_age_parent=priors.get(('p', 'sigma')),
                          zero_re=zero_re, rate_type=rate_type['p'])['p']

    @mc.deterministic
    def mu_age_pf(p=p['mu_age'], f=rate['f']['mu_age']):
        return p*f
    pf = age_specific_rate(model, 'pf',
                           reference_area, reference_sex, reference_year,
                           mu_age_pf,
                           mu_age_parent=priors.get(('pf', 'mu')),
                           sigma_age_parent=priors.get(('pf', 'sigma')),
                           lower_bound='csmr',
                           include_covariates=False,
                           zero_re=zero_re)['pf']

    @mc.deterministic
    def mu_age_m(pf=pf['mu_age'], m_all=m_all):
        return (m_all - pf).clip(1.e-6, 1.e6)
    rate['m'] = age_specific_rate(model, 'm_wo',
                                  reference_area, reference_sex, reference_year,
                                  mu_age_m,
                                  None, None,
                                  include_covariates=False,
                                  zero_re=zero_re)['m_wo']

    @mc.deterministic
    def mu_age_rr(m=rate['m']['mu_age'], f=rate['f']['mu_age']):
        return (m+f) / m
    rr = age_specific_rate(model, 'rr',
                           reference_area, reference_sex, reference_year,
                           mu_age_rr,
                           mu_age_parent=priors.get(('rr', 'mu')),
                           sigma_age_parent=priors.get(('rr', 'sigma')),
                           rate_type='log_normal',
                           include_covariates=False,
                           zero_re=zero_re)['rr']

    @mc.deterministic
    def mu_age_smr(m=rate['m']['mu_age'], f=rate['f']['mu_age'], m_all=m_all):
        return (m+f) / m_all
    smr = age_specific_rate(model, 'smr',
                            reference_area, reference_sex, reference_year,
                            mu_age_smr,
                            mu_age_parent=priors.get(('smr', 'mu')),
                            sigma_age_parent=priors.get(('smr', 'sigma')),
                            rate_type='log_normal',
                            include_covariates=False,
                            zero_re=zero_re)['smr']

    @mc.deterministic
    def mu_age_m_with(m=rate['m']['mu_age'], f=rate['f']['mu_age']):
        return m+f
    m_with = age_specific_rate(model, 'm_with',
                               reference_area, reference_sex, reference_year,
                               mu_age_m_with,
                               mu_age_parent=priors.get(('m_with', 'mu')),
                               sigma_age_parent=priors.get(('m_with', 'sigma')),
                               include_covariates=False,
                               zero_re=zero_re, rate_type=rate_type['m_with'])['m_with']

    # duration = E[time in bin C]
    @mc.deterministic
    def mu_age_X(r=rate['r']['mu_age'], m=rate['m']['mu_age'], f=rate['f']['mu_age']):
        hazard = r + m + f
        pr_not_exit = np.exp(-hazard)
        X = np.empty(len(hazard))
        X[-1] = 1 / hazard[-1]
        for i in reversed(range(len(X)-1)):
            X[i] = pr_not_exit[i] * (X[i+1] + 1) + 1 / hazard[i] * (1 - pr_not_exit[i]) - pr_not_exit[i]
        return X
    X = age_specific_rate(model, 'X',
                          reference_area, reference_sex, reference_year,
                          mu_age_X,
                          mu_age_parent=priors.get(('X', 'mu')),
                          sigma_age_parent=priors.get(('X', 'sigma')),
                          rate_type='normal',
                          include_covariates=True,
                          zero_re=zero_re)['X']

    vars = rate
    vars.update(logit_C0=logit_C0, p=p, pf=pf, rr=rr, smr=smr, m_with=m_with, X=X)
    return vars

