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
""" Spline model used for age-specific rates"""
import pylab as pl
import pymc as mc
import scipy.interpolate


def spline(name, ages, knots, smoothing, interpolation_method='linear'):
    """ Generate PyMC objects for a spline model of age-specific rate

    Parameters
    ----------
    name : str
    knots : array
    ages : array, points to interpolate to
    smoothing : pymc.Node, smoothness parameter for smoothing spline
    interpolation_method : str, optional, one of 'linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'

    Results
    -------
    Returns dict of PyMC objects, including 
    'gamma' (log of rate at knots) and 
    'mu_age' (age-specific rate interpolated at all age points)
    """
    assert pl.all(pl.diff(knots) > 0), 'Spline knots must be strictly increasing'

    # TODO: consider changing this prior distribution to be something more familiar in linear space
    gamma = [mc.Normal('gamma_%s_%d'%(name,k), 0., 10.**-2, value=-10.) for k in knots]
    #gamma = [mc.Uniform('gamma_%s_%d'%(name,k), -20., 20., value=-10.) for k in knots]

    # TODO: fix AdaptiveMetropolis so that this is not necessary
    flat_gamma = mc.Lambda('flat_gamma_%s'%name, lambda gamma=gamma: pl.array([x for x in pl.flatten(gamma)]))

    @mc.deterministic(name='mu_age_%s'%name)
    def mu_age(gamma=flat_gamma, knots=knots, ages=ages):
        mu = scipy.interpolate.interp1d(knots, pl.exp(gamma), kind=interpolation_method, bounds_error=False, fill_value=0.)
        return mu(ages)

    vars = dict(gamma=gamma, mu_age=mu_age, ages=ages, knots=knots)

    if (smoothing > 0) and (not pl.isinf(smoothing)):
        #print 'adding smoothing of', smoothing
        @mc.potential(name='smooth_mu_%s'%name)
        def smooth_gamma(gamma=flat_gamma, knots=knots, tau=smoothing**-2):
            # the following is to include a "noise floor" so that level value
            # zero prior does not exert undue influence on age pattern
            # smoothing
            # TODO: consider changing this to an offset log normal
            gamma = gamma.clip(pl.log(pl.exp(gamma).mean()/10.), pl.inf)  # only include smoothing on values within 10x of mean

            return mc.normal_like(pl.sqrt(pl.sum(pl.diff(gamma)**2 / pl.diff(knots))), 0, tau)
        vars['smooth_gamma'] = smooth_gamma

    return vars

'''
Below is a step‐by‐step **mathematical** (rather than purely code) description of what this spline code does.

---

## 1. Knot Definition and Log-Scale Parameters

1. **Knots**:
   Let

   $$
     k_1 < k_2 < \cdots < k_m
   $$

   be an increasing sequence of **knot points** (the array `knots`). These are special ages at which we define explicit parameters.

2. **Log‐scale parameters ($\gamma$)**:
   For each knot $k_i$, we define a PyMC **Normal** random variable (stochastic)

   $$
     \gamma_i \;\sim\; \mathcal{N}(0,\; 10^{-2}), 
   $$

   whose initial value is set to $-10$.
   Collect them into a vector:

   $$
     \boldsymbol{\gamma} \;=\;(\gamma_1,\,\gamma_2,\,\dots,\gamma_m).
   $$

   In code: `gamma = [mc.Normal(...), ..., mc.Normal(...)]`.

> Interpretation: $\exp(\gamma_i)$ is the **rate** (or value) at the $i$-th knot, so we store it in a log‐space parameter $\gamma_i$.

---

## 2. 1D Interpolation to All Ages

We want a function $\mu(\text{age})$ that smoothly connects these knot points. Suppose you have a (possibly large) array of ages,

$$
  a_1,\,a_2,\,\dots,\,a_n.
$$

We define

$$
  \mu(\text{age}) 
  \;=\; \text{interp1d}\!\Bigl(
       \{k_i\},\;\exp(\gamma_i),
       \text{kind}=\textit{interpolation\_method}\Bigr)(\text{age}).
$$

* **interp1d** with `'linear'` means **piecewise linear** interpolation in the $(\mathrm{age},\,\mathrm{value})$ plane.
* If `'cubic'`, `'quadratic'`, etc., then a higher‐order spline or polynomial interpolation is used.
* `fill_value=0.0` and `bounds_error=False` in the code means if $\text{age}$ is outside $[k_1,\,k_m]$, the spline returns $0$ instead of an error.

Hence for each age $a_j$, we get:

$$
  \mu(a_j) 
  \;=\; s(a_j),
$$

where $s(\cdot)$ is the interpolating spline through points $\{(k_i,\;\exp(\gamma_i))\}$.

> In code:
>
> ```python
> mu = scipy.interpolate.interp1d(knots, exp(gamma), kind=...)
> return mu(ages)
> ```
>
> This is the **Deterministic** node `mu_age`.

---

## 3. Optional Smoothing Penalty

If the user sets a **smoothing** parameter $>0$ (and not infinite), the function adds a **Potential** node that penalizes rough changes in $\gamma_i$.

```python
@mc.potential
def smooth_gamma(gamma=flat_gamma, knots=knots, tau=smoothing**-2):
    # ...
    return mc.normal_like(
        sqrt(sum( diff(gamma)**2 / diff(knots) )),
        0,
        tau
    )
```

1. **Clipping**

   $$
     \gamma_i^\prime 
     \;=\; \max\!\Bigl(\,\gamma_i,\; \log(\overline{y}/10)\Bigr),
     \quad 
     \overline{y} = \exp(\gamma)\text{.mean()}
   $$

   i.e., forcibly ensure that $\gamma_i$ can’t drop below $\log(\overline{y}/10)$. This step is a bit hacky but tries to avoid “negative infinite penalty” from extremely low values.

2. **Derivative penalty**
   The code calculates

   $$
     R(\boldsymbol{\gamma})
     \;=\;
     \sqrt{\sum_{i=1}^{m-1} \frac{\bigl(\gamma_{i+1}^\prime - \gamma_i^\prime\bigr)^2}{k_{i+1}-k_i}}
     \quad
     \text{(a measure of roughness in the log-scale spline)}
   $$

   and then imposes

   $$
     \log p(\boldsymbol{\gamma})
     \;\propto\;
     \mathcal{N}\bigl(R(\boldsymbol{\gamma});\,0,\, \tau\bigr),
   $$

   which is the same as

   $$
     -\tfrac{1}{2}\,\tau\;\bigl(R(\boldsymbol{\gamma})\bigr)^2
     \quad
     \text{in the log posterior.}
   $$

   This encourages $R(\boldsymbol{\gamma}) \approx 0$ ⇒ the $\gamma_i$ are *smooth* in a sense.

Thus, smaller $R(\gamma)$ means **less slope** between adjacent knots in log scale, and the parameter $\tau = \tfrac{1}{(\text{smoothing})^2}$ controls how strongly we penalize that slope.

---

## 4. Final Return

The function returns a **dictionary** of PyMC nodes:

1. `gamma`: list of $\gamma_i$ **Stochastic** variables
2. `mu_age`: a **Deterministic** array of interpolated values $\mu(a_j)$
3. `smooth_gamma` (optional): a **Potential** node imposing smoothing if $smoothing>0$.
4. `ages, knots`: just stored for reference

This dictionary is typically merged into a larger PyMC model, so that subsequent MAP or MCMC sampling can estimate the $\gamma_i$ values (i.e., how the rate changes with age).

---

### Summary (One‐Liner)

> **The code defines a spline** for “age vs. rate” by:
>
> * placing **log‐scale Normal priors** on the function’s values at certain **knots**,
> * **interpolating** those exponentiated values across all ages,
> * optionally adding a **smoothness penalty** (difference across adjacent knots),
> * and returning the corresponding **PyMC** nodes that form this random spline.

'''