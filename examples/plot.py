import numpy as np
import matplotlib.pyplot as plt

def make_marginal_percentiles_plot(ax, data, weights=None, names= None, nbins=200,
                                   mode='marginal', logy=False, showbins=False, plotkwargs={}):
    
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
    if names is None: names=["p%i"%(i) for i in range(nvars)]
    for i in range(nvars):
        bincenter=binlows[:,0,i]+np.median(data*grid, axis=1, keepdims=True)[:,0,i]
        ax[0,i].plot(bincenter ,pdf[:,0,i], **plotkwargs)
        ax[0,i].set_xlabel(names[i])
        if logy: ax[0,i].set_yscale('log')
        ax[0,i].legend(loc='best')
        if showbins:
                for x in binlim[:,i]:
                    ax[0,i].axvline(x, color='gray', lw=0.5)

def corner_plot(samples, labels, filename, title=None):
    import corner
    import matplotlib.ticker as ticker
    #burn = 5000
    plt.clf()
    #in samples each row is a point and each column is a variable
    fig = corner.corner(samples, labels=labels, 
                        quantiles=[0.16, 0.5, 0.84],  #-1sigma,0sigma,1sigma
                        levels=(1-np.exp(-0.5), 1-np.exp(-2), 1-np.exp(-9./2)), #1sigma, 2sigma and 3sigma contours
                        show_titles=True, title_kwargs={"fontsize": 16}, title_fmt= '.4f', 
                        smooth1d=None, plot_contours=True, 
                        no_fill_contours=False, plot_density=True, use_math_text=True)
    for i in range(len(fig.axes)):
        fig.axes[i].locator_params(axis='x', nbins=2)
        fig.axes[i].locator_params(axis='y', nbins=2)
        fig.axes[i].tick_params(axis='x', rotation =0, labelsize=16)
        fig.axes[i].tick_params(axis='y', rotation =90, labelsize=16)
        fig.axes[i].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
        fig.axes[i].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
  
    if title is not None:
        plt.suptitle(title,  fontsize=24,  color='blue', x=0.8  )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(filename,  dpi=150)
    plt.close(fig)
    print(filename, "Printed")

def getdist_plots(samples, labels, names, filename, trianglekwargs={"filled_compare":True,"contour_colors":["red"], "line_args":[{'ls':'solid', 'lw':2, 'color':"red"}], "title_limit":1, }, weights=None, title=None):
    from getdist import plots, MCSamples
    samples1 = MCSamples(samples=samples, weights=weights, names=names, labels=labels)
    
    g = plots.getSubplotPlotter()
    g.settings.plot_meanlikes = False
    g.settings.alpha_factor_contour_lines = True
    #g.settings.axis_marker_lw = 5
    g.settings.figure_legend_frame = True
    g.settings.alpha_filled_add=0.35
    g.settings.title_limit_fontsize = 16
    g.settings.figure_legend_loc = 'best'
    g.settings.rcSizes(axes_fontsize = 12, lab_fontsize=20, legend_fontsize =40)
    g.triangle_plot([samples1],**trianglekwargs)
    #g.add_legend(legend_labels=[legend_name], fontsize=36, legend_loc=(-3.5,7))
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print('Printed', filename)
    
def getdist_plots_list(samples_list, labels_list, names_list, filename,trianglekwargs,  weights_list=None, title=None):
    #from getdist import plots, MCSamples
    import getdist
    samples=[]
    for sample, weight,name,label in zip(samples_list,weights_list,names_list, labels_list):
        samples.append(getdist.MCSamples(samples=sample, weights=weight, names=name, labels=label))
    
    g = getdist.plots.getSubplotPlotter()
    g.settings.plot_meanlikes = False
    g.settings.alpha_factor_contour_lines = True
    #g.settings.axis_marker_lw = 5
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.35
    g.settings.title_limit_fontsize = 16
    #g.settings.figure_legend_loc = 'best'
    g.settings.rcSizes(axes_fontsize = 12, lab_fontsize=20, legend_fontsize =20)
    g.triangle_plot(samples, **trianglekwargs)
    #g.add_legend(legend_labels=[legend_name], fontsize=36, legend_loc=(-3.5,7))
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print('Printed', filename)


def make_plots(data, valdata=None, out=None, names=None, bins=200, density=True, logy=False, filename=None):
    data=data.T
    l=4 #inch
    plt.clf()
    nvars=data.shape[0]
    fig, ax= plt.subplots( 1, nvars, sharey=True, figsize=(len(data)*l,l), squeeze=0)
    
    if names is None: names=["p%i"%(i) for i in range(nvars)]
    
    for i, f in enumerate(data):
        ax[0,i].hist(f, bins=bins, log=logy, density=density,  histtype="step", color='blue', label="input dist")
        ax[0,i].set_xlabel(names[i])
        med=np.median(f)
        ax[0,i].axvline(med, color='gray', lw=1.5,ls=":", label="Median: %.2f"%(med))
        ax[0,i].legend(loc='best')
    
    if valdata is not None:
        valdata=valdata.T
        for i, f in enumerate(valdata):
            #ax[0,i].plot(f, out,color='red', label="pred dist") # only work ins 1d
            ax[0,i].hist(f, weights=out, bins=bins, log=logy, density=density,histtype="step", color='red', label="pred dist")
            ax[0,i].set_xlabel(names[i])
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
    make_marginal_percentiles_plot(ax, data, weights=None,names=names, nbins=bins, logy=logy, plotkwargs=plotkwargs)
    if valdata is not None:
        valdata=valdata.T
        plotkwargs={"color":'red', "label":"val dist"}
        make_marginal_percentiles_plot(ax, valdata, weights=out,names=names, nbins=bins, logy=logy, plotkwargs=plotkwargs)
    fig.tight_layout()
    fig.savefig(filename.replace(".png", "_percentiles.png"), dpi=200)
    plt.close(fig)

    #contour plots
    plt.clf()
    colors=["green", "blue", "red", "black"]
    if valdata is not None:
        trianglekwargs={"filled_compare":[True,True],"contour_colors":colors[:2], "line_args":[{'ls':'solid', 'lw':2, 'color':colors[i]} for i in range(nvars)], "title_limit":1, "legend_labels":["Fit","Val"]}
        getdist_plots_list([data, valdata], [names]*2, [names]*2, filename.replace(".png", "_contourns.png"),trianglekwargs,  weights_list=[None, out], title=None)
    else:
        getdist_plots(data,names,names, filename.replace(".png", "_contourns.png"), weights=None, title=None)
 

def plot_walkers(chains, names=None, filename=None):
    import emcee
    # (nvars, nwalker, steps) chains
    l=3
    nvars=chains.shape[0]
    nsteps=chains.shape[2]
    nwalkers=chains.shape[1]
    if names is None: names=['p%i'%(i) for i in range(nvars)]
    fig, axs= plt.subplots( nvars, 3, sharey=False, figsize=(3*l,nvars*l), squeeze=0)
    samples=np.vstack([arr.flatten() for arr in chains])

    for i, chain in enumerate(chains):
        par=chain.flatten()
        axs[i][0].set_ylabel(names[i])
        idx = np.arange(len(par))
        axs[i][0].scatter(idx, par[idx], marker='o', c='k', s=10.0, alpha=0.1, linewidth=0)
        # Get selfcorrelation using emcee
        ac = emcee.autocorr.function_1d(par)
        idx = np.arange(len(ac),step=1)
        axs[i][1].scatter(idx, ac[idx], marker='o', c='k', s=10.0, alpha=0.1, linewidth=0)
        axs[i][1].axhline(alpha=1., lw=1., color='red')
        
        par_mean = np.mean(chain, axis=0)
        par_err = np.std(chain, axis=0) / np.sqrt(nwalkers)
        idx = np.arange(len(par_mean))
        axs[i][2].errorbar(x=idx, y=par_mean,
                           yerr=par_err, errorevery=50,
                           ecolor='red', lw=0.5, elinewidth=2.,
                           color='k')
    axs[nvars-1][0].set_xlabel("Ensemble step")
    axs[nvars-1][1].set_xlabel("Ensemble step")
    axs[nvars-1][2].set_xlabel("Walker Step")
    axs[0][0].set_title("Ensemble dispersion")
    axs[0][1].set_title("Ensemble autocorrelation")
    axs[0][2].set_title("Walker mean and stdev")
        
    print("Printing file: %s"%(filename))
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(samples.shape)
    #corner_plot(samples.T, names, filename.replace(".png", "_contour.png"))
    getdist_plots(samples.T, names, names, filename.replace(".png", "_contour.png"))
    
