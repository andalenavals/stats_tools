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
    data=np.array([x,y,z])
    return data.T
def kde_fit(data, valdata, filename, **kwargs):
    from sklearn.neighbors import KernelDensity
    kde_skl = KernelDensity(**kwargs)
    kde_skl.fit(data)
    #out=kde_skl.sample(valdata)
    out=np.exp(kde_skl.score_samples(valdata))

    density=True
    log=False
    
    
    fig, ax= plt.subplots( 1, len(data.T), sharey=False, figsize=(8,4))
    
    data=data.T
    for i, f in enumerate(data):
        ax[i].hist(f, bins=200, log=log, density=density,histtype="step", color='blue', label="input dist")
        ax[i].set_xlabel("")
        med=np.median(f)
        ax[i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
        ax[i].legend(loc='best')

    
    valdata=valdata.T
    for i, f in enumerate(valdata):
        ax[i].hist(f, weights=out, bins=200, log=log, density=density,histtype="step", color='red', label="kde dist")
        ax[i].set_xlabel("")
        med=np.median(f)
        ax[i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
        ax[i].legend(loc='best')
    
    


    fig.tight_layout()
    
    fig.savefig(filename, dpi=200)
    plt.close()
    
    
def plot_data(data, filename):
    plt.clf()
    data=data.T
    fig, ax= plt.subplots( 1, len(data), sharey=False, figsize=(8,4))
    for i, f in enumerate(data):
        ax[i].hist(f, bins=200, log=False, density=True, histtype="step")
        ax[i].set_xlabel("")
        med=np.median(f)
        ax[i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
        ax[i].legend(loc='best')
    fig.tight_layout()
    
    fig.savefig(filename, dpi=200)
    plt.close()
    

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
    valdata=make_sample(100000)
    #valdata=data
    logger.info("data done")

    filename=os.path.join(outpath,"input_distribution.png")
    plot_data(data, filename )
    logger.info("plotting done")

    kwargs={"kernel":"gaussian", "bandwidth":10.0}
    filename=os.path.join(outpath,"kde_distribution.png")
    kde_fit(data, valdata,filename, **kwargs)
    logger.info("fitting done")
      
if __name__ == "__main__":
    main()
