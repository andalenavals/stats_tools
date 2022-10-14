import numpy as np
import matplotlib.pyplot as plt

def make_marginal_percentiles_plot(ax, data, weights=None, nbins=200,
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
    for i in range(nvars):
        bincenter=binlows[:,0,i]+np.median(data*grid, axis=1, keepdims=True)[:,0,i]
        ax[0,i].plot(bincenter ,pdf[:,0,i], **plotkwargs)
        ax[0,i].set_xlabel("")
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
