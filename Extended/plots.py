from bilby.core.result import read_in_result 
import matplotlib.pyplot as plt
import corner
import json
import matplotlib
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.lines as mlines

def plot_corner(fig,plot_data,color,hist_alpha=0.7,bins=20,mirror=True,kdewidth=1.):
    
    # Input variable plot_data should be a dictionary whose keys are the individual variables to plot.
    # Each key should, in turn, link to another nested dictionary with the following keys:
    # - "data" : Actual data values
    # - "plot_bounds" : Tuple of min/max values to display on plot
    # - "label" : Latex string for figure labeling
    # - "priors" : Tuple of min/max prior bounds

    # Define a linear color map
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white",color])
    
    # Loop across dimensions that we want to plot
    keys = list(plot_data)    
    ndim = len(keys)
    for i,key in enumerate(keys):
       
        # Plot the marginal 1D posterior (i.e. top of a corner plot column)
        ax = fig.add_subplot(ndim,ndim,int(1+(ndim+1)*i))
        ax.set_rasterization_zorder(1)
        
        ax.hist(plot_data[key]['data'],bins=np.linspace(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds']  [1],bins),\
               rasterized=True,color=color,alpha=hist_alpha,density=True,zorder=0)
        ax.hist(plot_data[key]['data'],bins=np.linspace(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1],bins),\
                histtype='step',color='black',density=True,zorder=2)
        ax.grid(True,dashes=(1,3))
        ax.set_xlim(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1])
        ax.set_title(r"${0:.2f}^{{+{1:.2f}}}_{{-{2:.2f}}}$".format(*getBounds(plot_data[key]['data'])),fontsize=14)

        # Turn off tick labels if this isn't the first dimension
        if i!=0:
            ax.set_yticklabels([])

        # If this is the last dimension add an x-axis label
        if i==ndim-1:
            ax.set_xlabel(plot_data[key]['label'])
            
        # If not the last dimension, loop across other variables and fill in the rest of the column with 2D plots
        else:
            
            ax.set_xticklabels([])
            for j,k in enumerate(keys[i+1:]):
                
                # Make a 2D density plot
                ax = fig.add_subplot(ndim,ndim,int(1+(ndim+1)*i + (j+1)*ndim))
                ax.set_rasterization_zorder(1)
                
                ax.hexbin(plot_data[key]['data'],plot_data[k]['data'],cmap=cmap,mincnt=1,gridsize=bins,\
                         extent=(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1],plot_data[k]['plot_bounds'][0],plot_data[k]['plot_bounds'][1]),
                         linewidths=(0,),zorder=0)

                # The rest of this "for" loop involves code to additionally add contours.
                # First, reflect data across all boundaries to have reasonable edge conditions
                if mirror:
                    mirrored_x,mirrored_y = mirrorData(plot_data[key]['data'],plot_data[k]['data'],plot_data[key]['priors'],plot_data[k]['priors'])            
                else:
                    mirrored_x = plot_data[key]['data']
                    mirrored_y = plot_data[k]['data']                
                
                # In drawing contours, we'll also need to KDE the data
                # Different dimensions will naturally want different KDE bandwidths.
                # To get reasonable-looking contours, reach a "compromise" between the two dimensions in question
                kde_width1 = np.std(plot_data[key]['data'])*np.power(float(len(plot_data[key]['data'])),-1./6.)/np.std(mirrored_x)
                kde_width2 = np.std(plot_data[k]['data'])*np.power(float(len(plot_data[k]['data'])),-1./6.)/np.std(mirrored_y)
                kde_width = kdewidth*np.sqrt(kde_width1*kde_width2)                
                print(key,k,kde_width)
                kde = gaussian_kde([mirrored_x,mirrored_y],bw_method=kde_width)

                # Make a regular grid and evaluate the KDE
                x_gridpoints = np.linspace(plot_data[key]['priors'][0],plot_data[key]['priors'][1],60)
                y_gridpoints = np.linspace(plot_data[k]['priors'][0],plot_data[k]['priors'][1],59)
                x_grid,y_grid = np.meshgrid(x_gridpoints,y_gridpoints)
                z_grid = kde([x_grid.reshape(-1),y_grid.reshape(-1)]).reshape(y_gridpoints.size,x_gridpoints.size)

                # Find the probabilities corresponding to central 50% and 90% probabilities
                sortedVals = np.sort(z_grid.flatten())[::-1]
                cdfVals = np.cumsum(sortedVals)/np.sum(sortedVals)
                i50 = np.argmin(np.abs(cdfVals - 0.50))
                i90 = np.argmin(np.abs(cdfVals - 0.90))
                val50 = sortedVals[i50]
                val90 = sortedVals[i90]
                
                # Draw contours
                CS = ax.contour(x_gridpoints,y_gridpoints,z_grid,levels=(val90,val50),linestyles=     ('dashed','solid'),colors='k',linewidths=1,zorder=2)
                
                # Set plot bounds
                ax.set_xlim(plot_data[key]['plot_bounds'][0],plot_data[key]['plot_bounds'][1])
                ax.set_ylim(plot_data[k]['plot_bounds'][0],plot_data[k]['plot_bounds'][1])
                ax.grid(True,dashes=(1,3))
                
                # If still in the first column, add a y-axis label
                if i==0:
                    ax.set_ylabel(plot_data[k]['label'])
                else:
                    ax.set_yticklabels([])
               
                # If on the last row, add an x-axis label
                if j==ndim-i-2:
                    ax.set_xlabel(plot_data[key]['label'])
                else:
                    ax.set_xticklabels([])
                    
    plt.tight_layout()    
    return fig

def getBounds(data):
    
    # Transform to a numpy arry
    data = np.array(data)

    # Get median, 5% and 95% quantiles
    med = np.median(data)
    upperLim = np.sort(data)[int(0.95*data.size)]
    lowerLim = np.sort(data)[int(0.05*data.size)]
 
    # Turn quantiles into upper and lower uncertainties
    upperError = upperLim-med
    lowerError = med-lowerLim
    
    return med,upperError,lowerError

def mirrorData(ref_data_x,ref_data_y,x_priors,y_priors):
    
    low_x,high_x = x_priors
    low_y,high_y = y_priors
        
    # Original data
    data_x = np.copy(ref_data_x)
    data_y = np.copy(ref_data_y)
    data_x_mirrored = np.copy(data_x)
    data_y_mirrored = np.copy(data_y)
    
    # Left
    data_x_mirrored = np.append(data_x_mirrored,low_x-(data_x-low_x))
    data_y_mirrored = np.append(data_y_mirrored,data_y)
    
    # Right
    data_x_mirrored = np.append(data_x_mirrored,high_x+(high_x-data_x))
    data_y_mirrored = np.append(data_y_mirrored,data_y)   
    
    # Top
    data_x_mirrored = np.append(data_x_mirrored,data_x)
    data_y_mirrored = np.append(data_y_mirrored,high_y+(high_y-data_y))
    
    # Bottom
    data_x_mirrored = np.append(data_x_mirrored,data_x)
    data_y_mirrored = np.append(data_y_mirrored,low_y-(data_y-low_y))
    
    # Upper left
    data_x_mirrored = np.append(data_x_mirrored,low_x-(data_x-low_x))
    data_y_mirrored = np.append(data_y_mirrored,high_y+(high_y-data_y))

    # Lower left
    data_x_mirrored = np.append(data_x_mirrored,low_x-(data_x-low_x))
    data_y_mirrored = np.append(data_y_mirrored,low_y-(data_y-low_y))

    # Upper right
    data_x_mirrored = np.append(data_x_mirrored,high_x+(high_x-data_x))
    data_y_mirrored = np.append(data_y_mirrored,high_y+(high_y-data_y))

    # Lower right
    data_x_mirrored = np.append(data_x_mirrored,high_x+(high_x-data_x))
    data_y_mirrored = np.append(data_y_mirrored,low_y-(data_y-low_y))
    
    return data_x_mirrored,data_y_mirrored


def overlayed_plot_extended(result_O3a_file, result1_file, filename, label1, label2, save = False):
    with open(result_O3a_file,'r') as ff1:
        data1 = json.load(ff1)
        
    plot_data1 = {
        'mu_chi':{'data':data1['posterior']['content']['mu_chi'],'plot_bounds':(0.,0.8),'priors':(0.,1.),'label':r'$\mu_\chi$'},
        'sigma_chi':{'data':data1['posterior']['content']['sigma_chi'],'plot_bounds':(0.,0.3),'priors':(0.,.25),'label':r'$\sigma^2_\chi$'},
        'xi_spin':{'data':data1['posterior']['content']['xi_spin'],'plot_bounds':(0.,1.),'priors':(0.,1.),'label':r'$\zeta$'},
        'sigma_spin':{'data':data1['posterior']['content']['sigma_spin'],'plot_bounds':(0.,4.),'priors':(0.,4.),'label':r'$\sigma_t$'},
        'lambda_chi_peak':{'data':data1['posterior']['content']['lambda_chi_peak'],'plot_bounds':(0.,1.),'priors':(0.,1.),'label':r'$\lambda_\mathrm{0}$'},
        'zmin':{'data':data1['posterior']['content']['zmin'],'plot_bounds':(-1.,1.),'priors':(-1.,1.),'label':'$z_{\\min}$'},}
    
    with open(result1_file,'r') as ff2:
        data2 = json.load(ff2)
        
    plot_data2 = {
        'mu_chi':{'data':data2['posterior']['content']['mu_chi'],'plot_bounds':(0.,0.8),'priors':(0.,1.),'label':r'$\mu_\chi$'},
        'sigma_chi':{'data':data2['posterior']['content']['sigma_chi'],'plot_bounds':(0.,0.3),'priors':(0.,.25),'label':r'$\sigma^2_\chi$'},
        'xi_spin':{'data':data2['posterior']['content']['xi_spin'],'plot_bounds':(0.,1.),'priors':(0.,1.),'label':r'$\zeta$'},
        'sigma_spin':{'data':data2['posterior']['content']['sigma_spin'],'plot_bounds':(0.,4.),'priors':(0.,4.),'label':r'$\sigma_t$'},
        'lambda_chi_peak':{'data':data2['posterior']['content']['lambda_chi_peak'],'plot_bounds':(0.,1.),'priors':(0.,1.),'label':r'$\lambda_\mathrm{0}$'},
        'zmin':{'data':data2['posterior']['content']['zmin'],'plot_bounds':(-1.,1.),'priors':(-1.,1.),'label':'$z_{\\min}$'},}
    
    colors = ["#01275B", "#FF0000"]
    labels = [label1, label2]
    
    fig = plt.figure(figsize=(8,8))
    plot_corner(fig, plot_data1, colors[0], bins=25, hist_alpha=0.75)
    plot_corner(fig, plot_data2, colors[1], bins=25, hist_alpha=0.75)
                                             
    line1 = mlines.Line2D([],[], color = colors[0], label = label1, lw = 2)
    line2 = mlines.Line2D([],[], color = colors[1], label = label2, lw = 2)
    fig.legend(handles=[line1, line2], frameon = False, fontsize=12, handlelength = 5)

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.18, hspace=0.16)
    if save == True:
        filename = "./Results/Overlayed_Plots_Extended/" + filename
        fig.savefig(filename, dpi=400)
    plt.show()