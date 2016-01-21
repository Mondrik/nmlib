# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 12:47:10 2016

@author: nmondrik
"""
##IN TESTING...

import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as T

def select_sampler(sampler,start=None):
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
    """
    with pm.Model() as model:
            #Use uninformative priors on slope/intercept of line
            alpha = pm.Uniform('alpha',-100,100)
            #we don't need priors on beta because this distribution is already minimally informative
            #and allows all values of a slope
            beta = pm.DensityDist('beta',lambda value: -1.5 * T.log(1 + value**2.),testval=0)
            #if yerr not given, assume all values have same errorbar            
            if yerr is None:
                sigma = pm.DensityDist('sigma', lambda value: -T.log(T.abs_(value)),testval=1)
            else:
                sigma = yerr
            like = pm.Normal('likelihood',mu=alpha+beta*t, sd=sigma, observed=y)
            start = pm.find_MAP(disp=True)
            step = select_sampler(sampler,start)
            trace = pm.sample(draws=samples,start=start,step=step)
    return trace

def sin_fit(t, y, yerr=None, samples=10000, sampler="NUTS",
            Alims=[0,10], omegalims=[0.1,100], philims=[0,2.*np.pi], siglims=[0,01,5], offsetlims=[0.,20.]):
    """
    Bayesian sinusoidal fitting function.  Beware highly multimodal posteriors...
    """
    with pm.Model() as model:
        #priors on parameters.   maybe put in adjustable distributions later?
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
    alpha = trace['alpha']
    beta = trace['beta']
    xf = np.linspace(ax.get_xbound()[0],ax.get_xbound()[1],10)
    yf = alpha[:,None] + beta[:,None]*xf
    mu = yf.mean(0)
    sig = 2* yf.std(0)
    
    ax.plot(xf,mu,'--k',lw=2,label='Posterior mean')
    ax.fill_between(xf, mu-sig, mu+sig, color='lightgray')