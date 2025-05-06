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
""" Covariate models"""
import networkx as nx
import numpy as np
import pandas as pd
import pymc as mc


SEX_VALUE = {'male': .5, 'total': 0., 'female': -.5}


def MyTruncatedNormal(name, mu, tau, lb, ub, value):
    """ Need to make my own, because PyMC has underflow error when
    truncation is not doing anything

    :Parameters:
      - `name` : str
      - `mu` : float, the unadjusted mean parameter for this node
      - `tau` : float, standard error?
      - `lb` : float, lower bound
      - `ub` : float, upper bound
      - `value` : float

    """
    @mc.stochastic(name=name)
    def my_trunc_norm(value=value, mu=mu, tau=tau, lb=lb, ub=ub):
        if lb <= value <= ub:
            return mc.normal_like(value, mu, tau)
        else:
            return -np.inf
    return my_trunc_norm

def mean_covariate_model(name, mu, input_data, parameters, model, root_area, root_sex, root_year, zero_re=True):
    """ Generate PyMC objects covariate adjusted version of mu

    :Parameters:
      - `name` : str
      - `mu` : the unadjusted mean parameter for this node
      - `model` : ModelData to use for covariates
      - `root_area, root_sex, root_year` : str, str, int
      - `zero_re` : boolean, change one stoch from each set of siblings in area hierarchy to a 'sum to zero' deterministic

    :Results:
      - Returns dict of PyMC objects, including 'pi', the covariate adjusted predicted values for the mu and X provided

    """
    # 관측치 개수 구하기
    n = len(input_data.index)

    # === 랜덤 효과 행렬 U 생성 ===
    # 모델 계층(hierarchy)의 노드 수를 p_U로 설정
    p_U = model.hierarchy.number_of_nodes()  # random effects for area
    # U: (n x p_U) 크기의 0으로 초기화된 DataFrame
    U = pd.DataFrame(np.zeros((n, p_U)), columns=model.hierarchy.nodes(), index=input_data.index)

    # 각 관측치별로 해당 지역까지의 경로(path)를 따라 1을 설정
    for i, row in input_data.T.iteritems():
        if row['area'] not in model.hierarchy:
            # 계층에 없는 지역 경고 출력 후 건너뛰기
            print(row['area'])
            print('WARNING: "%s" not in model hierarchy, skipping random effects for this observation' % row['area'])
            continue

        # 루트 'all' 노드에서 관측치 지역까지 최단 경로를 탐색하여 level과 U 값을 설정
        for level, node in enumerate(nx.shortest_path(model.hierarchy, 'all', input_data.loc[i, 'area'])):
            model.hierarchy.node[node]['level'] = level
            U.loc[i, node] = 1.

    # 모든 노드에 대해 level 속성 재설정 (재확인)
    for n2 in model.hierarchy.nodes():
        for level, node in enumerate(nx.shortest_path(model.hierarchy, 'all', n2)):
            model.hierarchy.node[node]['level'] = level

    # 관측치가 없으면 빈 DataFrame으로 설정
    if len(U.index) == 0:
        U = pd.DataFrame()
    else:
        # 단일 관측이 없거나 상위 레벨인 열 제거를 위한 후보 필터링
        keep_cols = [col for col in U.columns if (
            (U[col].max() > 0) and  # 하나라도 1인 랜덤 효과
            (model.hierarchy.node[col].get('level') > model.hierarchy.node[root_area]['level']))]
        U_filtered = U.filter(keep_cols)

        # 최소 1개 이상 관측 또는 상수 prior인 랜덤 효과만 유지
        keep = []
        if 'random_effects' in parameters:
            for re in parameters['random_effects']:
                if parameters['random_effects'][re].get('dist') == 'Constant':
                    keep.append(re)

        # 1 <= count < 전체 관측수 또는 상수 효과
        keep_cols = [col for col in U_filtered.columns if 1 <= U_filtered[col].sum() < len(U_filtered[col]) or col in keep]
        U = U_filtered.filter(keep_cols).copy()

    # ===== U_shift: root_area 기준 shift 벡터 생성 =====
    # root_area까지 경로에 포함된 노드 위치에 1 설정
    U_shift = pd.Series(0., index=U.columns)
    for level, node in enumerate(nx.shortest_path(model.hierarchy, 'all', root_area)):
        if node in U_shift:
            U_shift[node] = 1.
    # U에서 U_shift 빼서 기준 레벨 중앙정렬
    U -= U_shift

    # === 랜덤 효과 분산(sigma_alpha) 초기화 ===
    sigma_alpha = []
    # 최대 계층 깊이 5까지 순환
    for i in range(5):
        effect = 'sigma_alpha_%s_%d'%(name,i)
        if 'random_effects' in parameters and effect in parameters['random_effects']:
            # 저장된 하이퍼 prior 사용
            prior = parameters['random_effects'][effect]
            print('using stored RE hyperprior for', effect, prior)
            sigma_alpha.append(MyTruncatedNormal(name=effect, 
                                                mu=prior['mu'], 
                                                tau=np.maximum(prior['sigma'], .001)**-2,
                                                lb=min(prior['mu'], prior['lower']),
                                                ub=max(prior['mu'], prior['upper']),
                                                value=prior['mu']))
        else:
            # 기본 prior 설정
            sigma_alpha.append(MyTruncatedNormal(name=effect, 
                                                mu=.05, 
                                                tau=.03**-2, 
                                                lb=.05, 
                                                ub=.5, 
                                                value=.1))

    # === alpha (랜덤 효과 계수) 정의 및 prior 적용 ===
    alpha = np.array([])
    const_alpha_sigma = np.array([])
    alpha_potentials = []
    if len(U.columns) > 0:
        # 각 alpha가 속할 계층 레벨 인덱스 생성
        tau_alpha_index = []
        for alpha_name in U.columns:
            tau_alpha_index.append(model.hierarchy.node[alpha_name]['level'])
        tau_alpha_index = np.array(tau_alpha_index, dtype=int)

        # 각 alpha별 tau (precision) 계산
        tau_alpha_for_alpha = [sigma_alpha[i]**-2 for i in tau_alpha_index]

        # 개별 alpha 변수 생성
        alpha = []
        for i, tau_alpha_i in enumerate(tau_alpha_for_alpha):
            effect = 'alpha_%s_%s'%(name, U.columns[i])
            if 'random_effects' in parameters and U.columns[i] in parameters['random_effects']:
                prior = parameters['random_effects'][U.columns[i]]
                print('using stored RE for', effect, prior)
                if prior['dist'] == 'Normal':
                    alpha.append(mc.Normal(effect, prior['mu'], np.maximum(prior['sigma'], .001)**-2,
                                           value=0.))
                elif prior['dist'] == 'TruncatedNormal':
                    alpha.append(MyTruncatedNormal(effect, prior['mu'], np.maximum(prior['sigma'], .001)**-2,
                                                   prior['lower'], prior['upper'], value=0.))
                elif prior['dist'] == 'Constant':
                    alpha.append(float(prior['mu']))
                else:
                    assert 0, 'ERROR: prior distribution "%s" is not implemented' % prior['dist']
            else:
                # 기본 Normal prior
                alpha.append(mc.Normal(effect, 0, tau=tau_alpha_i, value=0))

        # 상수 alpha의 sigma 값 기록
        const_alpha_sigma = []
        for i, tau_alpha_i in enumerate(tau_alpha_for_alpha):
            if parameters.get('random_effects', {}).get(U.columns[i], {}).get('dist') == 'Constant':
                const_alpha_sigma.append(float(parameters['random_effects'][U.columns[i]]['sigma']))
            else:
                const_alpha_sigma.append(np.nan)

        # sum-to-zero 제약 적용 (zero_re=True 경우)
        if zero_re:
            column_map = dict([(n,i) for i,n in enumerate(U.columns)])
            # 각 부모별 자식 노드 집합에서 첫 번째 alpha를 deterministic으로 변환
            for parent in model.hierarchy:
                node_names = model.hierarchy.successors(parent)
                nodes = [column_map[n] for n in node_names if n in U]
                if len(nodes) > 0:
                    i = nodes[0]
                    old_alpha_i = alpha[i]

                    # 상수 prior이면 패스
                    if parameters.get('random_effects', {}).get(U.columns[i], {}).get('dist') == 'Constant':
                        continue

                    # 나머지 siblings alpha 합이 0이 되도록 deterministic 정의
                    alpha[i] = mc.Lambda('alpha_det_%s_%d'%(name, i),
                                         lambda other_alphas_at_this_level=[alpha[n] for n in nodes[1:]]: -sum(other_alphas_at_this_level))

                    # 원래 stochastic alpha prior 잠재력(potential) 추가
                    if isinstance(old_alpha_i, mc.Stochastic):
                        @mc.potential(name='alpha_pot_%s_%s'%(name, U.columns[i]))
                        def alpha_potential(alpha=alpha[i], mu=old_alpha_i.parents['mu'], tau=old_alpha_i.parents['tau']):
                            return mc.normal_like(alpha, mu, tau)
                        alpha_potentials.append(alpha_potential)

    # === 고정 효과 행렬 X와 beta 생성 ===
    keep_cols = [col for col in input_data.columns if col.startswith('x_')]
    X = input_data.filter(keep_cols).copy()

    # 성별을 더미 변수로 변환하여 고정 효과에 추가
    X['x_sex'] = [SEX_VALUE[row['sex']] for i, row in input_data.T.iteritems()]

    beta = np.array([])
    const_beta_sigma = np.array([])
    X_shift = pd.Series(0., index=X.columns)
    if len(X.columns) > 0:
        # 기준값(root) 수준의 covariate 평균으로 shift 벡터 계산
        try:
            output_template = model.output_template.groupby(['area', 'sex', 'year']).mean()
        except pd.core.groupby.groupby.DataError:
            output_template = model.output_template.groupby(['area', 'sex', 'year']).first()
        covs = output_template.filter(list(X.columns) + ['pop'])
        if len(covs.columns) > 1:
            leaves = [n for n in nx.traversal.bfs_tree(model.hierarchy, root_area)
                      if len(list(model.hierarchy.successors(n))) == 0]
            if len(leaves) == 0:
                leaves = [root_area]

            # 전체 성별, 연도인 경우 area별 평균으로 계산
            if root_sex == 'total' and root_year == 'all':
                covs = covs.reset_index().drop(['year', 'sex'], axis=1).groupby('area').mean()
                leaf_covs = covs.loc[leaves]
            elif root_sex == 'total':
                raise Exception('root_sex == total, root_year != all is Not Yet Implemented')
            elif root_year == 'all':
                raise Exception('root_year == all, root_sex != total is Not Yet Implemented')
            else:
                leaf_covs = covs.loc[[(l, root_sex, root_year) for l in leaves]]

            for cov in covs:
                if cov != 'pop':
                    # 각 leaf에 대한 population 가중 평균
                    X_shift[cov] = (leaf_covs[cov] * leaf_covs['pop']).sum() / leaf_covs['pop'].sum()

        # 성별 shift 값 지정
        if 'x_sex' in X.columns:
            X_shift['x_sex'] = SEX_VALUE[root_sex]

        # covariate 중심화
        X = X - X_shift

        assert not np.any(np.isnan(np.array(X, dtype=float))), 'matrix should have no missing values'

        # beta (고정 효과 계수) prior 정의
        beta = []
        for i, effect in enumerate(X.columns):
            name_i = 'beta_%s_%s'%(name, effect)
            if 'fixed_effects' in parameters and effect in parameters['fixed_effects']:
                prior = parameters['fixed_effects'][effect]
                print('using stored FE for', name_i, effect, prior)
                if prior['dist'] == 'TruncatedNormal':
                    beta.append(MyTruncatedNormal(name_i, mu=float(prior['mu']), tau=np.maximum(prior['sigma'], .001)**-2, a=prior['lower'], b=prior['upper'], value=.5*(prior['lower']+prior['upper'])))
                elif prior['dist'] == 'Normal':
                    beta.append(mc.Normal(name_i, mu=float(prior['mu']), tau=np.maximum(prior['sigma'], .001)**-2, value=float(prior['mu'])))
                elif prior['dist'] == 'Constant':
                    beta.append(float(prior['mu']))
                else:
                    assert 0, 'ERROR: prior distribution "%s" is not implemented' % prior['dist']
            else:
                beta.append(mc.Normal(name_i, mu=0., tau=1.**-2, value=0))

        # 상수 beta의 sigma 값 기록
        const_beta_sigma = []
        for i, effect in enumerate(X.columns):
            name_i = 'beta_%s_%s'%(name, effect)
            if 'fixed_effects' in parameters and effect in parameters['fixed_effects']:
                prior = parameters['fixed_effects'][effect]
                if prior['dist'] == 'Constant':
                    const_beta_sigma.append(float(prior.get('sigma', 1.e-6)))
                else:
                    const_beta_sigma.append(np.nan)
            else:
                const_beta_sigma.append(np.nan)

    # === 최종 예측값 pi 정의 (deterministic node) ===
    @mc.deterministic(name='pi_%s'%name)
    def pi(mu=mu, U=np.array(U, dtype=float), alpha=alpha, X=np.array(X, dtype=float), beta=beta):
        # mu * exp(U*alpha + X*beta)
        return mu * np.exp(np.dot(U, [float(x) for x in alpha]) + np.dot(X, [float(x) for x in beta]))

    # 결과로 필요한 객체들을 딕셔너리 형태로 반환
    return dict(pi=pi, U=U, U_shift=U_shift, sigma_alpha=sigma_alpha, alpha=alpha, alpha_potentials=alpha_potentials, X=X, X_shift=X_shift, beta=beta, hierarchy=model.hierarchy, const_alpha_sigma=const_alpha_sigma, const_beta_sigma=const_beta_sigma)


def dispersion_covariate_model(name, input_data, delta_lb, delta_ub):
    lower = np.log(delta_lb)
    upper = np.log(delta_ub)
    eta = mc.Uniform('eta_%s'%name, lower=lower, upper=upper, value=.5*(lower+upper))

    keep_cols = [col for col in input_data.columns if col.startswith('z_') and input_data[col].std() > 0]
    Z = input_data.filter(keep_cols)
    if len(Z.columns) > 0:
        zeta = mc.Normal('zeta_%s'%name, 0, .25**-2, value=np.zeros(len(Z.columns)))

        @mc.deterministic(name='delta_%s'%name)
        def delta(eta=eta, zeta=zeta, Z=Z.__array__()):
            return np.exp(eta + np.dot(Z, zeta))

        return dict(eta=eta, Z=Z, zeta=zeta, delta=delta)

    else:
        @mc.deterministic(name='delta_%s'%name)
        def delta(eta=eta):
            return np.exp(eta) * np.ones_like(input_data.index)
        return dict(eta=eta, delta=delta)


def predict_for(model, 
                parameters,
                root_area, root_sex, root_year,
                area, sex, year,
                population_weighted,
                vars,
                lower, upper):
    """ Generate draws from posterior predicted distribution for a
    specific (area, sex, year)

    :Parameters:
      - `model` : data.DataModel
      - `root_area` : str, area for which this model was fit consistently
      - `root_sex` : str, area for which this model was fit consistently
      - `root_year` : str, area for which this model was fit consistently
      - `area` : str, area to predict for
      - `sex` : str, sex to predict for
      - `year` : str, year to predict for
      - `population_weighted` : bool, should prediction be population weighted if it is the aggregation of units area RE hierarchy?
      - `vars` : dict, including entries for alpha, beta, mu_age, U, and X
      - `lower, upper` : float, bounds on predictions from expert priors

    :Results:
      - Returns array of draws from posterior predicted distribution

    """
    area_hierarchy = model.hierarchy
    output_template = model.output_template.copy()

    # find number of samples from posterior
    len_trace = len(vars['mu_age'].trace())

    # compile array of draws from posterior distribution of alpha (random effect covariate values)
    # a row for each draw from the posterior distribution
    # a column for each random effect (e.g. countries with data, regions with countries with data, etc)
    #
    # there are several cases to handle, or at least at one time there were:
    #   vars['alpha'] is a pymc Stochastic with an array for its value (no longer used?)
    #   vars['alpha'] is a list of pymc Nodes
    #   vars['alpha'] is a list of floats
    #   vars['alpha'] is a list of some floats and some pymc Nodes
    #   'alpha' is not in vars
    #
    # when vars['alpha'][i] is a float, there is also information on the uncertainty in this value, stored in
    # vars['const_alpha_sigma'][i], which is not used when fitting the model, but should be incorporated in
    # the prediction

    if 'alpha' in vars and isinstance(vars['alpha'], mc.Node):
        assert 0, 'No longer used'
        alpha_trace = vars['alpha'].trace()
    elif 'alpha' in vars and isinstance(vars['alpha'], list):
        alpha_trace = []
        for n, sigma in zip(vars['alpha'], vars['const_alpha_sigma']):
            if isinstance(n, mc.Node):
                alpha_trace.append(n.trace())
            else:
                # uncertainty of constant alpha incorporated here
                sigma = max(sigma, 1.e-9) # make sure sigma is non-zero
                assert not np.isnan(sigma)
                alpha_trace.append(mc.rnormal(float(n), sigma**-2, size=len_trace))
        alpha_trace = np.vstack(alpha_trace).T
    else:
        alpha_trace = np.array([])

    # compile array of draws from posterior distribution of beta (fixed effect covariate values)
    # a row for each draw from the posterior distribution
    # a column for each fixed effect
    #
    # there are several cases to handle, or at least at one time there were:
    #   vars['beta'] is a pymc Stochastic with an array for its value (no longer used?)
    #   vars['beta'] is a list of pymc Nodes
    #   vars['beta'] is a list of floats
    #   vars['beta'] is a list of some floats and some pymc Nodes
    #   'beta' is not in vars
    #
    # when vars['beta'][i] is a float, there is also information on the uncertainty in this value, stored in
    # vars['const_beta_sigma'][i], which is not used when fitting the model, but should be incorporated in
    # the prediction
    #
    # TODO: refactor to reduce duplicate code (this is very similar to code for alpha above)

    if 'beta' in vars and isinstance(vars['beta'], mc.Node):
        assert 0, 'No longer used'
        beta_trace = vars['beta'].trace()
    elif 'beta' in vars and isinstance(vars['beta'], list):
        beta_trace = []
        for n, sigma in zip(vars['beta'], vars['const_beta_sigma']):
            if isinstance(n, mc.Node):
                beta_trace.append(n.trace())
            else:
                # uncertainty of constant beta incorporated here
                sigma = max(sigma, 1.e-9) # make sure sigma is non-zero
                assert not np.isnan(sigma)
                beta_trace.append(mc.rnormal(float(n), sigma**-2., size=len_trace))
        beta_trace = np.vstack(beta_trace).T
    else:
        beta_trace = np.array([])

    # the prediction for the requested area is produced by aggregating predictions for all of the childred
    # of that area in the area_hierarchy (a networkx.DiGraph)

    leaves = [n for n in nx.traversal.bfs_tree(area_hierarchy, area) if area_hierarchy.successors(n) == []]
    if len(leaves) == 0:
        # networkx returns an empty list when the bfs tree is a single node
        leaves = [area]


    # initialize covariate_shift and total_population
    covariate_shift = np.zeros(len_trace)
    total_population = 0.

    # group output_template for easy access
    output_template = output_template.groupby(['area', 'sex', 'year']).mean()

    # if there are fixed effects, the effect coefficients are stored as an array in vars['X']
    # use this to put together a covariate matrix for the predictions, according to the output_template
    # covariate values
    #
    # the resulting array is covs
    if 'X' in vars:
        covs = output_template.filter(vars['X'].columns).copy()
        if 'x_sex' in vars['X'].columns:
            covs['x_sex'] = SEX_VALUE[sex]
        assert np.all(covs.columns == vars['X_shift'].index), 'covariate columns and unshift index should match up'
        for x_i in vars['X_shift'].index:
            covs[x_i] -= vars['X_shift'][x_i] # shift covariates so that the root node has X_ar,sr,yr == 0
    else:
        covs = pd.DataFrame(index=output_template.index)

    # if there are random effects, put together an indicator based on
    # their hierarchical relationships
    #
    if 'U' in vars:
        p_U = area_hierarchy.number_of_nodes()  # random effects for area
        U_l = pd.DataFrame(np.zeros((1, p_U)), columns=area_hierarchy.nodes())
        U_l = U_l.filter(vars['U'].columns).copy()
    else:
        U_l = pd.DataFrame(index=[0])

    # loop through leaves of area_hierarchy subtree rooted at 'area',
    # make prediction for each using appropriate random
    # effects and appropriate fixed effect covariates
    #
    for l in leaves:
        log_shift_l = np.zeros(len_trace)
        if len(U_l.columns) > 0:
            U_l.loc[0,:] = 0.

        root_to_leaf = nx.shortest_path(area_hierarchy, root_area, l)
        print(root_to_leaf)
        
        for node in root_to_leaf[1:]:
            if node not in U_l.columns:
                ## Add a columns U_l[node] = rnormal(0, appropriate_tau)
                level = len(nx.shortest_path(area_hierarchy, 'all', node))-1
                if 'sigma_alpha' in vars:
                    tau_l = vars['sigma_alpha'][level].trace()**-2

                U_l[node] = 0.

                # if this node was not already included in the alpha_trace array, add it
                # there are several cases for adding:
                #  if the random effect has a distribution of Constant
                #    add it, using a sigma as well
                #  otherwise, sample from a normal with mean zero and standard deviation tau_l
                if parameters.get('random_effects', {}).get(node, {}).get('dist') == 'Constant':
                    mu = parameters['random_effects'][node]['mu']
                    sigma = parameters['random_effects'][node]['sigma']
                    sigma = max(sigma, 1.e-9) # make sure sigma is non-zero

                    alpha_node = mc.rnormal(mu,
                                            sigma**-2,
                                            size=len_trace)
                else:
                    if 'sigma_alpha' in vars:
                        alpha_node = mc.rnormal(0., tau_l)
                    else:
                        alpha_node = np.zeros(len_trace)

                if len(alpha_trace) > 0:
                    alpha_trace = np.vstack((alpha_trace.T, alpha_node)).T
                else:
                    alpha_trace = np.atleast_2d(alpha_node).T

            # TODO: implement a more robust way to align alpha_trace and U_l
            U_l.loc[0, node] = 1.

        # 'shift' the random effects matrix to have the intended
        # level of the hierarchy as the reference value
        if 'U_shift' in vars:
            for node in U_l.columns:
                if node in vars['U_shift'].index:
                    U_l[node] -= vars['U_shift'][node]

        # add the random effect intercept shift (len_trace draws)
        log_shift_l += np.dot(alpha_trace, U_l.T).flatten()

        # make X_l
        if len(beta_trace) > 0:
            if (l,sex,year) in covs.index:
                X_l = covs.loc[(l,sex,year)]
            else:
                # HACK: if this location/sex/year is not in index, return zeros
                X_l = np.zeros_like(covs.iloc[0])

            log_shift_l += np.dot(beta_trace, X_l.T).flatten()

        if population_weighted:
            # combine in linear-space with population weights
            shift_l = np.exp(log_shift_l)
            covariate_shift += shift_l * output_template['pop'][l,sex,year]
            total_population += output_template['pop'][l,sex,year]
        else:
            # combine in log-space without weights
            covariate_shift += log_shift_l
            total_population += 1.

    if population_weighted:
        covariate_shift /= total_population
    else:
        covariate_shift = np.exp(covariate_shift / total_population)

    parameter_prediction = (vars['mu_age'].trace().T * covariate_shift).T

    # clip predictions to bounds from expert priors
    parameter_prediction = parameter_prediction.clip(lower, upper)

    return parameter_prediction

