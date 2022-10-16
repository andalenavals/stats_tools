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

     
def gmm_sklearn(data, valdata, **kwargs):
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(**kwargs)
    h=gmm.fit(data)
    #print(type(h))
    out=np.exp(h.score_samples(valdata))
    #out=h.predict_proba(valdata)
    #out=gmm.fit_predict(data,valdata)
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
    

    data=make_sample(100000)
    valdata=make_grid_sample(data, 100000)
    #valdata=make_sample(100000)
    #valdata=data
    logger.info("data done")

    filename=os.path.join(outpath,"input_distribution.png")
    plot.make_plots(data, filename=filename )
    logger.info("plotting done")
    
    #di=find_best_bandwidth(data)
    di={}


    kwargs_sklearn={"n_components":10, "covariance_type":"tied"}
    kwargs_sklearn.update(di)
    out=gmm_sklearn(data, valdata,**kwargs_sklearn)
    print(type(out))
    print(out.shape)
    print(out)
    filename=os.path.join(outpath,"gmm_distribution_sklearn.png")
    logger.info("sklearn fitting done")
    plot.make_plots(data, valdata, out, filename=filename )
    logger.info("plotting done")
    

    
    
      
if __name__ == "__main__":
    main()
