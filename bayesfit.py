# -*-me as above coding: utf-8 -*-
"""
Created on Mon Jan 11 12:47:10 2016
i
@author: nmondrik
"""
##IN TESTING...

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as T
import emcee
import scipy.optimize as op

def select_sampler(sampler,start=None):
"""
Choose the sampler to use (for pymc3 only).  So far only "Metropolis" and "NUTS" are the only
samplers enabled.
"""
    if sampler == "NUTS":
        step = pm.NUTS(scaling=start)
    elif sampler == "Metropolis":
        step = pm.Metropolis()
    else:
        raise ValueError("Valid samplers are \"Metropolis\" and \"NUTS\".  You picked \"%s\"." % sampler)
    return step


def lin_fit(t, y, yerr=None, samples=10000, sampler="NUTS", alphalims=[-100,100]):
    """
    Bayesian linear fitting function.
    See Jake Vanderplas' blog post on how to be a
    bayesian in python for more details

    uses pymc3 MCMC sampling

    inputs:
        t    ::    Vector of values at which the function is evaluated ("x" values)
        y    ::    Vector of dependent values (observed y(t))
        yerr (optional = None) :: Errors on y values.  If not provided, errors are taken to be the same for each dta point,
            with a 1/sigma (jefferys) prior.
        samples (optional = 1000)  :: Number of samples to draw from MCMC
        sampler (optional = "NUTS")  :: Type of MCMC sampler to use.  "NUTS" or "Metropolis"
        alphalims (optional = [-100,100])  ::  Length 2 vector of endpoints for uniform prior on intercept of the line
    """
    with pm.Model() as model:
            #Use uninformative priors on slope/intercept of line
            alpha = pm.Uniform('alpha',-100,100)
            #this defines an uninformative prior on slope.  See Jake's blog post
            beta = pm.DensityDist('beta',lambda value: -1.5 * T.log(1 + value**2.),testval=0)
            #if yerr not given, assume all values have same errorbar
            if yerr is None:
                sigma = pm.DensityDist('sigma', lambda value: -T.log(T.abs_(value)),testval=1)
            else:
                sigma = yerr
            like = pm.Normal('likelihood',mu=alpha+beta*t, sd=sigma, observed=y)
            #start the sampler at the maximum a-posteriori value
            start = pm.find_MAP(disp=True)
            step = select_sampler(sampler,start)
            trace = pm.sample(draws=samples,start=start,step=step)
    return trace

def sin_fit(t, y, yerr=None, samples=10000, sampler="NUTS",
            Alims=[0,10], omegalims=[0.1,100], philims=[0,2.*np.pi], siglims=[0,1], offsetlims=[0.,20.]):
    """
    Bayesian sinusoidal fitting function.  Beware highly multimodal posteriors...
    Uses pymc3 implementation of MCMC fitting.
    inputs:
        t                                  ::  Vector of values at which the function is evaluated ("x" values)
        y                                  ::  Vector of dependent values (observed y(t))
        yerr (optional = None)             ::  Errors on y values.  If not provided, errors are taken to be the same for
            each dta point, with a 1/sigma (jefferys) prior.
        samples (optional = 1000)          ::  Number of samples to draw from MCMC
        sampler (optional = "NUTS")        ::  Type of MCMC sampler to use.  "NUTS" or "Metropolis"
        Alims (optional = [0,10])          ::  Length 2 vector of endpoints for uniform prior on amplitude of sin fn
        omegalims (optional = [0.1,100])   ::  Length 2 vector of endpoints for uniform prior on frequency
        philims (optional = [0,2*np.pi])   ::  Length 2 vector of endpoints for uniform prior on phase
        siglims (optional = [0,1])         ::  Length 2 vector of endpoints for unifrom distribution of sigma for error dist
        offsetlims (optional = [0,20])     ::  Length 2 vector of endpoints for DC offset of sin function
   """
    with pm.Model() as model:
        #priors on parameters.   maybe put in adjustable distributions later?
        #i think maybe I should change the priors on A,omega,and sigma to be Jefferys priors?
        A = pm.Uniform('A', Alims[0], Alims[1])
        omega = pm.Uniform('omega', omegalims[0], omegalims[1])
        phi = pm.Uniform('phi', philims[0], philims[1])
        offset = pm.Uniform('offset', offsetlims[0], offsetlims[1])
        if yerr is None:
            sig = pm.Uniform('sig', siglims[0], siglims[1])
        else:
            sigma = yerr

        y_est = A*pm.sin(omega*t+phi) + offset
        like = pm.Normal('likelihood', mu=y_est, sd=sig, observed = y)

        start = pm.find_MAP(disp=True)
        step = select_sampler(sampler,start)
        trace = pm.sample(draws=samples, start=start, step=step)
        return trace

def plot_lin_model(ax,trace):
"""
    Convienience function for plotting results from lin_fit.  Plots most probable parameters (not marginalized) and the 2-sigma fit region.
        inputs:
            ax  ::  Axes object from matplotlib
            trace  ::  trace from lin_fit
"""
    alpha = trace['alpha']
    beta = trace['beta']
    xf = np.linspace(ax.get_xbound()[0],ax.get_xbound()[1],10)
    yf = alpha[:,None] + beta[:,None]*xf
    mu = yf.mean(0)
    sig = 2* yf.std(0)

    ax.plot(xf,mu,'--k',lw=2,label='Posterior mean')
    ax.fill_between(xf, mu-sig, mu+sig, color='lightgray')

def emcee_test(t, y, yerr, samples=10000, burn=0):
    def lnprior(theta):
        m, b = theta
        if -5.0 < m < 0.5 and 0.0 < b < 10.0:
            return 0.0
        return -np.inf
    def lnlike(theta, t, y, yerr):
        m, b = theta
        model = m * t + b
        ivar = 1.0/(yerr**2)
        return -0.5*(np.sum((y-model)**2*ivar))
    def lnprob(theta, t, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, t, y, yerr)
    nll = lambda *args: -lnlike(*args)
    result = op.minimize(nll, [0.42,0.1], args=(t,y,yerr))
    mml, bml = result["x"]
    ndim, nwalkers = 2, 100
    print result["x"]
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, y, yerr))
    sampler.run_mcmc(pos, samples)
    sam = sampler.chain[:, burn:, :].reshape((-1, ndim))
    return sam

def emcee_test_sin(t, y, yerr, samples=10000, burn=0, nwalkers=100):
    def lnprior(theta):
        A, omega, phi, offset = theta
        if 0 < A < 10 and 0.01 < omega < 100. and 0 < phi < 2*np.pi and 0 < offset < 0.5:
            return 0.0
        return -np.inf
    def lnlike(theta, t, y, yerr):
        A, omega, phi, offset = theta
        model = A*np.sin(omega*t+phi) + offset
        ivar = 1.0/(yerr**2)
        return -0.5*(np.sum((y-model)**2*ivar))
    def lnprob(theta, t, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, t, y, yerr)
    nll = lambda *args: -lnlike(*args)
#    ini = [[np.random.uniform(low=0,high=10),np.random.uniform(low=0.01,high=100),np.random.uniform(low=0,high=2*np.pi),np.random.uniform(low=0,high=0.5)] for i in range(nwalkers)]
    ini = [0.02,0.12566,np.pi/2.,0.]
    result = op.minimize(nll, ini, args=(t,y, yerr))
    ndim = 4
    print result["x"]
    pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]
#    pos = ini
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t, y, yerr))
    sampler.run_mcmc(pos, samples)
    sam = sampler.chain[:, burn:, :].reshape((-1, ndim))
    return sam

def emcee_gaussian(t, y, yerr=None, samples=10000, burn=1000, nwalkers=100):
    def lnprior(theta):
        A, mu, sig, off = theta
        if 0 < A < 66000 and np.min(t) < mu < np.max(t) and -1000. < off < 1000. and 0 < sig < 10:
            return -np.log(sig)
        else:
            return -np.inf
    def lnlike(theta, t, y, yerr):
        A, mu, sig, off = theta
        model = A * np.exp(-0.5*((t-mu)/sig)**2.) + off
        return -0.5*np.sum(((y-model)/yerr)**2.)
    def lnprob(theta, t, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + lnlike(theta, t, y, yerr)
    nll = lambda *args: -lnlike(*args)
    ini = [np.max(y),np.mean(t),3.,0.]
    #initial guess for test_emcee.py
#    ini = [np.max(y), 17. , 2., 0.]
    result = op.minimize(nll,ini,args=(t,y,yerr))
    #print result
    ndim = 4
    pos = [ini + np.random.normal(loc=0,scale=0.5,size=ndim)  for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(t,y,yerr))
    sampler.run_mcmc(pos,samples )
    sam = sampler.chain[:, burn:, :].reshape((-1,ndim))
    print sam
    return sam
