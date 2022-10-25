# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3.0 of the License, or (at your option)
# any later version.
#
# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
#

"""
Created on: 06/10/22
Author: Andres Navarro
"""

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings(
        "ignore", category=DeprecationWarning,
        message=r"tostring\(\) is deprecated\. Use tobytes\(\) instead\.")

import sys, os
import numpy as np
import random
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot
import calc
   
import logging
logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Main script for sampling a distribution using mcmc')
    parser.add_argument('--workdir',
                        default='out/mcmc_posterior', 
                        help='diractory of work')
   
    args = parser.parse_args()

    return args
   
def make_dir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except OSError:
        if not os.path.exists(dirname): raise

def log_prior(pars):
    assert len(pars)==2
    mu,c =pars

    # Flat prior
    mu_min, mu_max=-1,1
    c_min, c_max=-1,1
    if (mu_min < mu < mu_max)&(c_min < c < c_max): return 0.0
    return -np.inf

def log_likelihood(pars, x, y, yerr):
    assert len(pars)==2
    mu,c=pars
    model=(1+mu)*x+c
    sigma2= yerr**2
    return -0.5*np.sum((y-model)**2/sigma2 +np.log(2*np.pi*sigma2))

def log_probability(pars,  x, y, yerr):
    lp = log_prior(pars)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(pars, x, y, yerr)
    

def make_data(npts):
    mutrue=0.5
    btrue=-0.2
    x=np.random.uniform(-0.8,0.8, size=npts)
    x=np.sort(x)
    y=(1+mutrue)*x+btrue
    ynoise=0.08*np.random.uniform(-1,1,npts)
    sortflag=np.argsort(-np.abs(ynoise))
    y+=ynoise[sortflag]
    yerr = 1e-2*np.array(range(1,npts+1))
    logger.info("Data was done")
    return np.array([x,y,yerr]).T

def make_plots(data, fitdata=None, logy=False, filename=None):

    l=4 #inch
    plt.clf()
    data=data.T
    x,y,yerr=data[0], data[1], data[2]
    ebarskwargs = {"fmt":'.', "color":"black", "ls":"",'elinewidth':0.5}
    plt.errorbar(x,y, yerr=yerr, label="data", capsize=5, **ebarskwargs)
    #plt.scatter(x, y)
    ret=calc.linreg(x,y)
    m,merr,c, cerr=(ret["m"]+1),ret["merr"],ret["c"],ret["cerr"]
    xplot=np.linspace(min(x),max(x))
    plt.plot(xplot,m*xplot+c, ls='-',linewidth=2, color='red', label='Simple lr: $m:%.2f \pm %.2f, c:%.2f \pm %.2f$ '%(m,merr, c,cerr))

    ret=calc.linregw(x,y,1./(yerr**2))
    m,merr,c, cerr=(ret["m"]+1),ret["merr"],ret["c"],ret["cerr"]
    plt.plot(xplot,m*xplot+c, ls='-',linewidth=2, color='red', label='Weighted lr: $m:%.2f \pm %.2f, c:%.2f \pm %.2f$ '%(m,merr, c,cerr))

    plt.plot(xplot,(1+0.5)*xplot-0.2, ls='-',linewidth=2, color='blue', label='true')
    print(fitdata.shape)
  
    m_mcmc,m_mcmc_err=np.median(fitdata.T[0]+1),np.std(fitdata.T[0]+1)
    c_mcmc,c_mcmc_err=np.median(fitdata.T[1]),np.std(fitdata.T[1])
    plt.plot(xplot,m_mcmc*xplot+c_mcmc, ls='-',linewidth=2, color='green', label='mcmc: $m:%.2f \pm %.2f, c:%.2f \pm %.2f$ '%(m_mcmc,m_mcmc_err, c_mcmc,c_mcmc_err))

    '''
    inds = np.random.randint(len(fitdata), size=100)
    for ind in inds:
        sample = fitdata[ind]
        sample[0]+=1
        plt.plot(xplot, np.dot(np.vander(xplot, 2), sample[:2]), "C1", alpha=0.1)
    '''
    
    
    plt.legend(loc='best')        
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


    #contour plots
    plt.clf()
    colors=["green", "blue", "red", "black"]
    nvars=2
    names=["p%i"%(i) for i in range(nvars)]
    plot.getdist_plots(fitdata,names,names, filename.replace(".png", "_contourns.png"), weights=None, title=None)
        
def get_ipos_walkers(log_dist, nwalkers, ndim, args=None):
    import scipy.optimize as optimize
    i_guess= [0.4, -0.3]
    nll = lambda *args: -log_dist(*args)
    result = optimize.minimize(nll, i_guess, method='Nelder-Mead', tol=1e-6, args=args )
    if result.success:
        i_guess = result.x
    else:
        raise ValueError(result.message)
    logger.info("Used initial guess is")
    print(i_guess)
    epsilon=1e-4
    pos = [i_guess + epsilon*np.random.randn(ndim) for i in range(nwalkers)]
    print(np.array(pos).shape)
    return pos
    
def mcmc(nwalkers, ndim, log_dist, nsteps, args=None):
    import emcee
    from multiprocessing import Pool
    with Pool() as pool:
        sampler= emcee.EnsembleSampler(nwalkers, ndim, log_dist, pool=pool, args=args )
        #starting point the minimum (just for making this quicker
        pos=get_ipos_walkers(log_dist, nwalkers, ndim, args=args)
        logger.info("Running MCMC")
        sampler.run_mcmc(pos, nsteps)
    # (nwalker, steps, nvars)
    chain=sampler.chain
    nstep=chain.shape[1]
    burning=0.1
    chain=chain[:,int(nstep*burning):,:]
    # (nvars, nwalker, steps)
    chains=np.transpose(chain,[2,0,1])
    logger.info("Run finished")
    return chains

def main():
    
    args = parse_args()
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    #logging.basicConfig(format=loggerformat, level=logging.DEBUG)
    logging.basicConfig(format=loggerformat, level=logging.INFO)
    #logging.basicConfig(format=loggerformat, level=logging.NOTSET)

    
    outpath = os.path.expanduser(args.workdir)
    make_dir(outpath)
    logger.info("work path done")

    npts=50
    idata=make_data(npts)
    logdist_args=tuple(idata.T)
    
    nwalkers=200
    ndim=2
    nsteps=1000
    
    chains=mcmc(nwalkers, ndim, log_probability, nsteps, args=logdist_args)
    filename=os.path.join(outpath,"mcmc_walkers.png")
    plot.plot_walkers(chains, names=None, filename=filename)

    fitdata=np.vstack([arr.flatten() for arr in chains]).T
  
    filename=os.path.join(outpath,"mcmc_vs_realdistribution.png")
    make_plots(idata, fitdata, filename=filename )
    logger.info("plotting done")
    
  
    
    
      
if __name__ == "__main__":
    main()
