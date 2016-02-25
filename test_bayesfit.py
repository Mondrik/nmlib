import numpy as np
import matplotlib.pyplot as plt
from nmlib import bayesfit as bf

do_lin_fit = False
do_gauss_fit = True

if do_lin_fit:
    #create some test data from a line:
    mtrue = 0.42
    btrue = 0.314159
    x = np.random.uniform(low=0,high=10,size=15)
    ytrue = mtrue * x + btrue

    #add error to line:
    scale = 0.4
    yerr = np.random.normal(loc=0,scale=scale,size=x.size)
    yobs = ytrue + yerr

    #assume we have some error in our error estimates:
    err_scale = 0.05
    yerr_obs = np.abs(yerr + np.random.normal(loc=0,scale=err_scale,size=yerr.size))

    #plot line
    plt.figure(facecolor='white')
    plt.plot(x,ytrue,'-r',lw=2,label='True')
    plt.plot(x,yobs,'ko',label='Obs')
    plt.errorbar(x,yobs,yerr=yerr_obs,fmt='none',ecolor='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower right')
    plt.show()

    #now lets use pymc3 to find a fit:
    alphalims = [-3,3]
    trace = bf.lin_fit(t=x,y=yobs,yerr=yerr_obs,alphalims=alphalims)
    plt.figure(facecolor='white')
    #plot the best fit + 2-sigma region
    plt.plot(x,ytrue,'-r',lw=2,label='True')
    bf.plot_lin_model(plt.gca(),trace)
    plt.plot(x,yobs,'ko',lw=2)
    plt.errorbar(x,yobs,yerr=yerr_obs,fmt='none',ecolor='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


###
#now lets do some gaussian fitting with emcee
#------------------------------------------------------------------------------
###


if do_gauss_fit:
    #generate some data
    #you should also experiment with the number of x points you sample to get an idea
    #for how bayesian fitting with MCMC handles regions without a lot of data/data with large errors
    x = np.random.uniform(low=-10,high=10,size=30)
    mutrue = 0.
    Atrue = 10.
    sigtrue = 1.5
    ytrue = Atrue * np.exp(-0.5 * ((x-mutrue)/sigtrue)**2.)

    #make a model, since the gaussian is kind of hard to see
    xmodel = np.linspace(x.min(),x.max(),1000)
    ymodel = Atrue * np.exp(-0.5 * ((xmodel-mutrue)/sigtrue)**2.)

    #add error to line:
    scale = 1.
    yerr = np.random.normal(loc=0,scale=scale,size=x.size)
    yobs = ytrue + yerr

    #assume we have some error in our error estimates:
    err_scale = 0.5
    yerr_obs = np.abs(yerr + np.random.normal(loc=0,scale=err_scale,size=yerr.size))

    #plot line
    plt.figure(facecolor='white')
    plt.plot(xmodel,ymodel,'-r',lw=2,label='True')
    plt.plot(x,yobs,'ko',label='Obs')
    plt.errorbar(x,yobs,yerr=yerr_obs,fmt='none',ecolor='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='lower right')
    plt.show()

    #do fit
    Alims = [0,20]
    siglims = [0.1,5]
    mulims = [-3,5]
    trace = bf.emcee_gaussian(t=x,y=yobs, yerr=yerr_obs, samples=1000, burn=200, Alims=Alims, siglims=siglims, mulims=mulims)
    #don't yet have a fancy plotter for the emcee stuff yet...
    #trace is returned as [nwalkers*samples x N_params] numpy array with order
    #[A, mu, sig, offset] so
    Aobs = np.mean(trace[:,0])
    muobs = np.mean(trace[:,1])
    sigobs = np.mean(trace[:,2])
    offobs = np.mean(trace[:,3])

    #construct CR
    A = trace[:,0]
    mu = trace[:,1]
    sig = trace[:,2]
    off = trace[:,3]
    yvals = A[:,None] * np.exp(-0.5 * ((xmodel-mu[:,None])/sig[:,None])**2.) + off[:,None]
    yvals = np.array(yvals)
    fitmean = yvals.mean(0)
    fitstdev = yvals.std(0)

    #plots plots plots
    plt.plot(xmodel,ymodel,'-r',lw=2,label='True')
    plt.plot(x,yobs,'ko',label='Obs')
    plt.errorbar(x,yobs,yerr=yerr_obs,fmt='none',ecolor='k')
    plt.plot(xmodel,fitmean,'--k',lw=2,label='Fit')
    plt.fill_between(xmodel,fitmean-fitstdev,fitmean+fitstdev,color='lightgray')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
