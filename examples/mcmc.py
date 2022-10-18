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
   
import logging
logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Main script for sampling a distribution using mcmc')
    parser.add_argument('--workdir',
                        default='out/mcmc', 
                        help='diractory of work')
   
    args = parser.parse_args()

    return args
   
def make_dir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except OSError:
        if not os.path.exists(dirname): raise


def gaussian(x, mu, sig):
    return (1./(np.sqrt(2.*np.pi)*sig))*np.exp(-0.5*np.power((x - mu)/sig, 2.))

def dist(pars):
    x,y,z=pars
    fx=gaussian(x, 0, 2.5)
    fy=gaussian(y, 3.2, 1.5)
    #fz=np.sqrt(np.abs(x*y))
    fz=gaussian(z, x, np.abs(y))
    return fx*fy*fz

def log_dist(pars):
    return np.log(dist(pars))
#this is just for scipy minimization initial guess
def nlog_dist(pars):
    return -np.log(dist(pars))


def make_grid_sample(data, npts=1000):
    ndims=len(data.T)
    outdata=[]
    for i in range(ndims):
        mi,ma=np.min(data.T[i]),np.max(data.T[i])
        #outdata.append(np.linspace(mi,ma, npts))
        outdata.append(np.random.uniform(mi,ma, npts))
    return np.array(outdata).T  

def make_plots(data, valdata=None, out=None, bins=200, density=True, logy=False, filename=None):
    data=data.T
    l=4 #inch
    plt.clf()
    fig, ax= plt.subplots( 1, len(data), sharey=True, figsize=(len(data)*l,l), squeeze=0)
   
    
    for i, f in enumerate(data):
        ax[0,i].hist(f, bins=bins, log=logy, density=density,  histtype="step", color='blue', label="input dist")
        ax[0,i].set_xlabel("")
        med=np.median(f)
        ax[0,i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
        ax[0,i].legend(loc='best')
    
    if valdata is not None:
        valdata=valdata.T
        for i, f in enumerate(valdata):
            #ax[0,i].plot(f, out,color='red', label="kde dist") # only work ins 1d
            ax[0,i].hist(f, weights=out, bins=bins, log=logy, density=density,histtype="step", color='red', label="kde dist")
            ax[0,i].set_xlabel("")
            med=np.median(f)
            ax[0,i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
            ax[0,i].legend(loc='best')
            
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


    l=4
    data=data.T; 
    nvars=data.shape[1]
    plt.clf()
    fig, ax= plt.subplots( 1, nvars, sharey=True, figsize=(nvars*l,l), squeeze=0)
    plotkwargs={"color":'blue', "label":"input dist"}
    plot.make_marginal_percentiles_plot(ax, data, weights=None, nbins=bins, logy=logy, plotkwargs=plotkwargs)
    if valdata is not None:
        valdata=valdata.T
        plotkwargs={"color":'red', "label":"val dist"}
        plot.make_marginal_percentiles_plot(ax, valdata, weights=out, nbins=bins, logy=logy, plotkwargs=plotkwargs)
    fig.tight_layout()
    fig.savefig(filename.replace(".png", "_percentiles.png"), dpi=200)
    plt.close(fig)

    #contour plots
    plt.clf()
    colors=["green", "blue", "red", "black"]
    if valdata is not None:
        names=["p%i"%(i) for i in range(nvars)]
        trianglekwargs={"filled_compare":[True,True],"contour_colors":colors[:2], "line_args":[{'ls':'solid', 'lw':2, 'color':colors[i]} for i in range(nvars)], "title_limit":1, }
        plot.getdist_plots_list([data, valdata], [names]*2, [names]*2, filename.replace(".png", "_contourns.png"),trianglekwargs,  weights_list=[None, out], title=None)
    else:
        names=["p%i"%(i) for i in range(nvars)]
        plot.getdist_plots(data,names,names, filename.replace(".png", "_contourns.png"), weights=None, title=None)
        
def get_ipos_walkers(nwalkers, ndim):
    import scipy.optimize as optimize
    i_guess= [0.1, 3.2, 1.0]
    result = optimize.minimize(nlog_dist, i_guess, method='Nelder-Mead', tol=1e-6 )
    if result.success:
        i_guess = result.x
    else:
        raise ValueError(result.message)
    print(i_guess)
    epsilon=1e-4
    pos = [i_guess + epsilon*np.random.randn(ndim) for i in range(nwalkers)]
    print(np.array(pos).shape)
    return pos
    
def mcmc(nwalkers, ndim, log_dist, nsteps):
    import emcee
    from multiprocessing import Pool
    with Pool() as pool:
        sampler= emcee.EnsembleSampler(nwalkers, ndim, log_dist, pool=pool)
        #starting point the minimum (just for making this quicker
        pos=get_ipos_walkers(nwalkers, ndim)
        logger.info("Running MCMC")
        sampler.run_mcmc(pos, nsteps)
    # (nwalker, steps, nvars)
    chain=sampler.chain
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

    nwalkers=200
    ndim=3
    nsteps=10000
    chains=mcmc(nwalkers, ndim, log_dist, nsteps)
    filename=os.path.join(outpath,"mcmc_walkers.png")
    plot.plot_walkers(chains, names=None, filename=filename)

    data=np.vstack([arr.flatten() for arr in chains]).T
    valdata=make_grid_sample(data, 100000)
    logger.info("data done")
    out=np.array([dist(pars) for pars in valdata])
  
    filename=os.path.join(outpath,"mcmc_vs_realdistribution.png")
    make_plots(data, valdata, out, filename=filename )
    logger.info("plotting done")
    
  
    
    
      
if __name__ == "__main__":
    main()
