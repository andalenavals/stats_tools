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
   
import logging
logger = logging.getLogger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Main script for training and validating fiducial simulations')
    parser.add_argument('--workdir',
                        default='out/kde', 
                        help='diractory of work')
   
    args = parser.parse_args()

    return args
   
def make_dir(dirname):
    try:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
    except OSError:
        if not os.path.exists(dirname): raise

def make_sample(npts=1000):
    x=np.random.normal(0,2.5, npts)
    y=np.random.normal(3.2,1.5, npts)
    z=np.random.uniform(2,2.5, npts)
    ##each row must correspond to a single data point
    data=np.array([x,y,np.sqrt(np.abs(x*y))])
    #data=np.array([x,y])
    return data.T
def make_grid_sample(data, npts=1000):
    ndims=len(data.T)
    outdata=[]
    for i in range(ndims):
        mi,ma=np.min(data.T[i]),np.max(data.T[i])
        outdata.append(np.linspace(mi,ma, npts))
    return np.array(outdata).T  

def make_plots(data, valdata=None, out=None, density=True, logy=False, filename=None):
    data=data.T
    l=4 #inch
    fig, ax= plt.subplots( 1, len(data), sharey=True, figsize=(len(data)*l,l))
    
    for i, f in enumerate(data):
        ax[i].hist(f, bins=200, log=logy, density=density,  histtype="step", color='blue', label="input dist")
        ax[i].set_xlabel("")
        med=np.median(f)
        ax[i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
        ax[i].legend(loc='best')
    
    if valdata is not None:
        valdata=valdata.T
        for i, f in enumerate(valdata):
            ax[i].plot(f, out,color='red', label="kde dist")
            #ax[i].hist(f, weights=out, bins=200, log=log, density=density,histtype="step", color='red', label="kde dist")
            ax[i].set_xlabel("")
            med=np.median(f)
            ax[i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
            ax[i].legend(loc='best')
            
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close()


    l=4
    data=data.T; 
    nvars=data.shape[1]
    fig, ax= plt.subplots( 1, nvars, sharey=True, figsize=(nvars*l,l))
    plotkwargs={"color":'blue', "label":"input dist"}
    make_marginal_plot(ax, data, weights=None, nbins=200, logy=logy, plotkwargs=plotkwargs)
    if valdata is not None:
        valdata=valdata.T
        plotkwargs={"color":'red', "label":"val dist"}
        make_marginal_plot(ax, valdata, weights=out, nbins=200, logy=logy, plotkwargs=plotkwargs)
    fig.tight_layout()
    fig.savefig(filename.replace(".png", "_percentiles.png"), dpi=200)
    plt.close()
    
def make_marginal_plot(ax, data, weights=None, nbins=200, mode='marginal', logy=False, showbins=False,  plotkwargs={}):
    pers=np.linspace(0.0, 100.0, nbins+1)
    binlim= np.percentile(data, pers, axis=0)
    binlows = binlim[0:-1,:]
    binhighs = binlim[1:,:]
    binlows=binlows[:,np.newaxis,:]
    binhighs=binhighs[:,np.newaxis,:]
    grid=(data>=binlows)&(data<=binhighs)
    binsizes=binhighs-binlows

    #marginal distribution
    #(nbins, npoints, nvars)
    if mode=='marginal':
        if weights is None:
            q=np.sum(grid, axis=1, keepdims=True)
            Q=np.sum(q,axis=0,keepdims=True)
            pdf=q/(Q*binsizes)
        else:
            q=np.sum(grid*weights[:,np.newaxis], axis=1, keepdims=True)
            Q=np.sum(q,axis=0,keepdims=True)
            pdf=q/(Q*binsizes)

    nvars=pdf.shape[2]
    for i in range(nvars):
        bincenter=binlows[:,0,i]+np.median(data*grid, axis=1, keepdims=True)[:,0,i]
        ax[i].plot(bincenter ,pdf[:,0,i], **plotkwargs)
        ax[i].set_xlabel("")
        if logy: ax[i].set_yscale('log')
        ax[i].legend(loc='best')
        if showbins:
                for x in binlim[:,i]:
                    ax[i].axvline(x, color='gray', lw=0.5)
    
    
    
    

def kde_scipy(data, valdata, bandwidth=0.2, **kwargs):
    import scipy.stats
    kde = scipy.stats.gaussian_kde(data,**kwargs)
    return kde.evaluate(valdata)

def kde_sklearn(data, valdata, **kwargs):
    from sklearn.neighbors import KernelDensity
    kde_skl = KernelDensity(**kwargs)
    kde_skl.fit(data)
    #out=kde_skl.sample(valdata)
    out=np.exp(kde_skl.score_samples(valdata))
    return out

def kde_statsmodels_m(data, valdata, vartype='c', **kwargs):
    from statsmodels.nonparametric.kernel_density import KDEMultivariate
    #bandwidth=0.2
    #bw=bandwidth * np.ones_like(x)
    kde = KDEMultivariate(data,var_type=vartype, **kwargs)
    return kde.pdf(valdata)

def get_hist_pdf(data, weights=None, nbins=200, mode='marginal'):
    pers=np.linspace(0.0, 100.0, nbins+1)
    binlim= np.percentile(data, pers, axis=0)
    binlows = binlim[0:-1,:]
    binhighs = binlim[1:,:]
    binlows=binlows[:,np.newaxis,:]
    binhighs=binhighs[:,np.newaxis,:]
    grid=(data>=binlows)&(data<=binhighs)
    binsizes=binhighs-binlows

    #marginal distribution
    #(nbins, npoints, nvars)
    if mode=='marginal':
        q=np.sum(grid, axis=1, keepdims=True)
        Q=np.sum(q,axis=0,keepdims=True)
        pdf=q/(Q*binsizes)

    #total distribution
    #if mode=='full':
    #    q=np.all(grid,axis=2,keepdims=True)
    #    hyperarea=np.product(binsizes, axis=2, keepdims=True)

    #print(pdf.shape)
    #print(pdf)
    return pdf


        
    

def custom_scoring( model, X, y,  npoints=200,):
    ## marginal distribution for a validation ser should be the same that
    ## the training set
    assert valdata.ndim==2
    assert preds.shape.ndim==1

    preds=model.predict(X)
    vec_preds=get_hist_pdf(X, weights=preds, nbins=npoints, mode='marginal')
    mse=np.mean(np.square(vec_preds-y))
    return mse
    
    
def find_best_bandwidth(data, valdata, njobs=200):
    from sklearn.metrics import fbeta_score, make_scorer
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    npoints=200
    target=get_hist_pdf(data, weights=None, nbins=npoints, mode='marginal')
    funckwargs={"npoints":npoints} 
    score = make_scorer(custom_scoring, greater_is_better=False, **funckwargs)
    
    grid = GridSearchCV(KernelDensity(kernel="gaussian"), 
                    {'bandwidth': np.linspace(0.01, 100, njobs)}, scoring=score,
                        n_jobs=1) # 20-fold cross-validation
    print(target.shape)
    print(data.shape)
    grid.fit(valdata, target)
    print (grid.best_params_)
    return grid.best_params_
       

def main():    
    args = parse_args()
    loggerformat='PID %(process)06d | %(asctime)s | %(levelname)s: %(name)s(%(funcName)s): %(message)s'
    #logging.basicConfig(format=loggerformat, level=logging.DEBUG)
    logging.basicConfig(format=loggerformat, level=logging.INFO)
    #logging.basicConfig(format=loggerformat, level=logging.NOTSET)

    
    outpath = os.path.expanduser(args.workdir)
    make_dir(outpath)
    logger.info("work path done")
    

    data=make_sample(10000)
    valdata=make_grid_sample(data, 10000)
    #valdata=make_sample(100000)
    #valdata=data
    logger.info("data done")

    filename=os.path.join(outpath,"input_distribution.png")
    make_plots(data, filename=filename )
    logger.info("plotting done")
    
    di=find_best_bandwidth(data, valdata)
    #di={}
    
    kwargs_sklearn={"kernel":"gaussian", "bandwidth":0.5}
    kwargs_sklearn.update(di)
    out=kde_sklearn(data, valdata,**kwargs_sklearn)
    filename=os.path.join(outpath,"kde_distribution_sklearn.png")
    logger.info("sklearn fitting done")
    make_plots(data, valdata, out, filename=filename )
    logger.info("plotting done")

    kwargs={}
    #out=kde_scipy(data, valdata,**kwargs)
    out=kde_statsmodels_m(data, valdata,vartype="ccc", **kwargs)
    logger.info("statsmodels fitting done")
    filename=os.path.join(outpath,"kde_distribution_statsmodel.png")
    make_plots(data, valdata, out, filename=filename )
    logger.info("plotting done")
    
    
      
if __name__ == "__main__":
    main()
