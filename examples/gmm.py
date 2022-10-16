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
    parser = argparse.ArgumentParser(description='Gauxian mixture mode for multidimensional distribution')
    parser.add_argument('--workdir',
                        default='out/gmm', 
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
    #data=np.array([x])
    return data.T
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
        
def gmm_sklearn(data, valdata, **kwargs):
    from sklearn.mixture import GMM
    gmm = GMM(**kwargs)
    gmm.fit(data)

    out=gmm.predict(valdata)
    return out

   
def find_best_bandwidth(data,njobs=200):
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KernelDensity

    grid = GridSearchCV(KernelDensity(kernel="gaussian"), 
                    {'bandwidth': np.linspace(0.01, 100, njobs)}, 
                        cv=20, n_jobs=njobs) # 20-fold cross-validation
    grid.fit(data)
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
    

    data=make_sample(1000)
    valdata=make_grid_sample(data, 100000)
    #valdata=make_sample(100000)
    #valdata=data
    logger.info("data done")

    filename=os.path.join(outpath,"input_distribution.png")
    make_plots(data, filename=filename )
    logger.info("plotting done")
    
    #di=find_best_bandwidth(data)
    di={}


    kwargs_sklearn={"ncomp":1}
    kwargs_sklearn.update(di)
    out=gmm_sklearn(data, valdata,**kwargs_sklearn)
    print(out)
    filename=os.path.join(outpath,"gmm_distribution_sklearn.png")
    logger.info("sklearn fitting done")
    make_plots(data, valdata, out, filename=filename )
    logger.info("plotting done")
    

    
    
      
if __name__ == "__main__":
    main()