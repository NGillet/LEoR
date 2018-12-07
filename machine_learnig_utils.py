
# coding: utf-8

# In[1]:

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as plticker
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter, AutoLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import gaussian_kde

from astropy.modeling import models, fitting

import numpy as np

from time import time

import sys
sys.path.insert( 1, "/astro/home/nicolas.gillet/myPythonLibrary" )

import pyemma.utils_functions as utils


# In[ ]:




# In[ ]:

paramName = [ 'ZETA', 'Tvir', 'LX', 'E0', 'all4' ]


# In[ ]:




# In[ ]:

def load_model( fileName ):
    """
    load a keras model from fileName
    return the model
    """
    from keras.models import model_from_json
    json_file = open(  fileName+'.json', 'r' )
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json( loaded_model_json )
    ### the weights
    loaded_model.load_weights( fileName+'_weights.h5' )

    return loaded_model #.summary()


# In[ ]:




# In[1]:

def plot_CNN_out( out, param, paramNum, paramMins, paramMaxs, color='b', Nbin=20, Brad_pt=None ):
    
    pmin = paramMins[paramNum] ### minimum of the selected parameter
    pmax = paramMaxs[paramNum] ### maximun of the selected parameter
    print(pmax)
    d    = pmax - pmin         ### range of the selected parameter
    VALUE_TRUE = param[:,paramNum]*d+pmin ### recovert 'physical value' of input
    VALUE_PRED = out*d+pmin ### recovert 'physical value' of output
    #RESIDUAL = 100* (VALUE_PRED - VALUE_TRUE) / VALUE_TRUE ### epsilon express in fraction of the true value
    RESIDUAL = (VALUE_PRED - VALUE_TRUE)
    #RESIDUAL =  np.log10(VALUE_PRED/VALUE_TRUE)
    
    
    pmin = VALUE_TRUE.min()
    pmax = VALUE_TRUE.max()
    
    print( 'number of points: ', RESIDUAL.size )
    #print( 'X2: ', (RESIDUAL**2).sum() )
    ######################################
    ######################################
    #bin_VALUE_TRUE = np.linspace( VALUE_TRUE.min(), VALUE_TRUE.max(), Nbin )
    bin_VALUE_TRUE = np.linspace( pmin, pmax, Nbin )
    ######################################
    ### error of the network
    h2, yh2, xh2 = np.histogram2d( VALUE_PRED, VALUE_TRUE, bins=[bin_VALUE_TRUE,bin_VALUE_TRUE] )
    h1, xh1 = np.histogram( VALUE_TRUE, bins=bin_VALUE_TRUE )
    h_weight, xh_weight = np.histogram( VALUE_TRUE, bins=bin_VALUE_TRUE, weights=RESIDUAL )
    h_id = np.digitize( VALUE_TRUE, xh1 )
    std = np.zeros(Nbin-1)
    mean = np.zeros(Nbin-1)
    median = np.zeros(Nbin-1)
    for i in range(Nbin-1):
        id_ = np.where( h_id==i+1 )[0]
        mean[i] = RESIDUAL[id_].sum() / id_.size
        median[i] = np.median( RESIDUAL[id_] )
        std[i] =  np.sqrt( (( RESIDUAL[id_] - mean[i] )**2).sum() / (id_.size-1) )
    eps_mean = mean
    eps_std  = std
    
    ######################################
    ######################################
    ### error on the prediction
    h3, xh3 = np.histogram( VALUE_PRED, bins=bin_VALUE_TRUE )
    h_weight, xh_weight = np.histogram( VALUE_PRED, bins=bin_VALUE_TRUE, weights=RESIDUAL )
    h_id = np.digitize( VALUE_PRED, bin_VALUE_TRUE )
    std_2 = np.zeros(Nbin-1)
    mean_2 = np.zeros(Nbin-1)
    median_2 = np.zeros(Nbin-1)
    for i in range(Nbin-1):
        id_ = np.where( h_id==i+1 )[0]
        mean_2[i] = RESIDUAL[id_].sum() / id_.size
        median_2[i] = np.median( RESIDUAL[id_] )
        std_2[i] =  np.sqrt( (( RESIDUAL[id_] - mean_2[i] )**2).sum() / (id_.size-1) )
    u_mean = mean_2
    u_std  = std_2
    # paramName = ['Zeta', 'R_mfp', 'Tvir', 'Lx', 'E0', 'ax']
    #paramName = ['Zeta', 'Tvir', 'Lx', 'E0']
    paramName = [r'$\rm{\zeta}$', 
                 r'$\rm{log_{10}(T_{vir})}$', 
                 r'$\rm{log_{10}(L_X/SFR) }$', 
                 r'$\rm{E_0}$' ]
    paramUnit = ['', 
                 r'$\rm{ [K] }$', 
                 r'$\rm{ [erg\ s^{-1}\ M^{-1}_{\odot}\ yr] }$', 
                 r'$\rm{ [keV] }$' ]
    ######################################
    ######################################
    RAND_TRUE = np.random.rand( 4000 )*d+pmin ### 
    RAND_PRED = np.random.rand( 4000 )*d+pmin ### 
    RAND_RESIDUAL = RAND_PRED - RAND_TRUE
    
    h4, xh4 = np.histogram( RAND_TRUE, bins=bin_VALUE_TRUE )
    h_weight, xh_weight = np.histogram( RAND_TRUE, bins=bin_VALUE_TRUE, weights=RAND_RESIDUAL )
    h_id = np.digitize( RAND_TRUE, bin_VALUE_TRUE )
    std_3 = np.zeros(Nbin-1)
    mean_3 = np.zeros(Nbin-1)
    median_3 = np.zeros(Nbin-1)
    for i in range(Nbin-1):
        id_ = np.where( h_id==i+1 )[0]
        mean_3[i] = RAND_RESIDUAL[id_].sum() / id_.size
        median_3[i] = np.median( RAND_RESIDUAL[id_] )
        std_3[i] =  np.sqrt( (( RAND_RESIDUAL[id_] - mean_3[i] )**2).sum() / (id_.size-1) )
    ######################################
    ######################################
    fig = plt.figure(  )
    ######################################
    ######################################
    ### TRUE-PREDICTED plot
    ######################################
    ax1 = plt.subplot( 111 )
#     plt.plot( VALUE_TRUE, VALUE_PRED, 'k.', alpha=0.15)
    im = plt.imshow( np.log10(h2 +1), interpolation='gaussian', origin='lower', 
                     extent=[bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1],bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1]],
                     cmap= cm.hot_r )#cm.Greys) 

#     data = np.vstack( [VALUE_TRUE, VALUE_PRED] )
#     kde = gaussian_kde(data)
#     X_grid, Y_grid = np.meshgrid( bin_VALUE_TRUE, bin_VALUE_TRUE )
#     Z = kde.evaluate( np.vstack( [X_grid.ravel(), Y_grid.ravel()] ) )
#     im = plt.imshow( (Z.reshape(X_grid.shape)), interpolation='nearest', origin='lower', 
#                      extent=[bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1],bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1]],
#                      cmap=cm.Greys) 
    
    plt.plot( [bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1]], [bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1]], 'k:', alpha=0.5 ) ### diagonal
    plt.xlim( bin_VALUE_TRUE[0], bin_VALUE_TRUE[-1] )
    plt.ylim( bin_VALUE_TRUE[0], bin_VALUE_TRUE[-1] )
    ax1.tick_params( axis='x', which='both', bottom='on', top='on', labelbottom='off' )
    ax1.tick_params( axis='y', which='both', bottom='on', top='on', labelbottom='off' )
    ax1.set_xticklabels([])
    ax1.xaxis.set_major_locator( NullLocator() )
    ax1.set_yticklabels([])
    ax1.yaxis.set_major_locator( NullLocator() )
    
    ax1.plot( Brad_pt[0,0], Brad_pt[0,1], '^c', label='Faint' )
    ax1.plot( Brad_pt[1,0], Brad_pt[1,1], 'vc', label='Bright' )
    
    plt.legend(loc='best', fontsize=10)
    
    ax1.errorbar( Brad_pt[0,0], Brad_pt[0,1], xerr=[Brad_pt[0,2:]], ecolor='c', fmt='.', color='c', ms=0, errorevery=2 )
    ax1.errorbar( Brad_pt[1,0], Brad_pt[1,1], xerr=[Brad_pt[1,2:]], ecolor='c', fmt='.', color='c', ms=0, errorevery=2 )
    
#     ax1.set_aspect( (bin_VALUE_TRUE[-1]-bin_VALUE_TRUE[0]) / (bin_VALUE_TRUE[-1]-bin_VALUE_TRUE[0]) )
    
    #plt.ylabel( '%s : Predicted'%paramName[paramNum], fontsize=10 )
    ######################################
    ######################################
    divider = make_axes_locatable(ax1)
    #ax2 = divider.append_axes('bottom', size='50%', pad=0.08)
    ax2 = divider.append_axes('bottom', size='50%', pad=0.25)
    cax = divider.append_axes('right' , size='5%', pad=0.08)
    cb = plt.colorbar( im, ax=ax1, cax=cax )
    cb.set_label( r'$\rm{ log_{10}( Number+1 ) }$', fontsize=10 ) 
    ax3 = divider.append_axes('left', size='50%', pad=0.25)
    ######################################
    ######################################
    ### MEAN-MEDIANE plot
    #ax2.errorbar( utils.zcen(xh1), mean*100, yerr=std*100, ecolor='b', fmt='.', color='b', ms=0, errorevery=2 )
    #ax2.step( xh1, np.append( mean[0], mean )*100, where='pre', lw=2, color='b', label='mean' )
    
    #ax2.fill_between( utils.zcen(bin_VALUE_TRUE), mean_3+std_3, mean_3-std_3, color='k', alpha=0.3 )
                
    ax2.errorbar( utils.zcen(bin_VALUE_TRUE), mean, yerr=std, ecolor='royalblue', fmt='.', color='royalblue', ms=0, errorevery=2 )
    ax2.step( bin_VALUE_TRUE, np.append( mean[0], mean ), where='pre', lw=2, color='royalblue' )#, label='mean' )
    
    ax2.plot( [bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1]], [0,0], 'k-' )
    ax2.set_xlim( bin_VALUE_TRUE[0], bin_VALUE_TRUE[-1] )
    
    #ylim = ax2.get_ylim()
    #Ntick = np.diff( ylim ) / 4
    #ax2.yaxis.set_major_locator( plticker.MultipleLocator(base=Ntick) )
    
    #[l.set_rotation(45) for l in ax2.get_xticklabels()]
    ax2.xaxis.set_major_locator( MaxNLocator( 5, prune='lower' ) )
    
    [l.set_rotation(45) for l in ax2.get_yticklabels()]
    ax2.yaxis.set_major_locator( MaxNLocator( 5, prune='upper' ) )
    #ax2.yaxis.set_major_locator( AutoLocator() )
        
    #ax2.legend(loc='best')
    ax2.tick_params( axis='x', which='both', bottom='on', top='on', labelbottom='on' )
    ax2.set_xlabel( '%s %s, True'%(paramName[paramNum], paramUnit[paramNum]), fontsize=10 )
    ax2.set_ylabel( r'$ \rm{ \epsilon } $', fontsize=10 )
    
    ######################################
    ######################################          
    ax3.plot( [0,0], [bin_VALUE_TRUE[0],bin_VALUE_TRUE[-1]] , 'k-' )
    
    ax3.errorbar( mean_2, utils.zcen(bin_VALUE_TRUE), xerr=std_2, ecolor='royalblue', fmt='.', color='royalblue', ms=0, errorevery=2 )
    
    #plt.barh( bin_VALUE_TRUE[:-1], mean_2, height=np.diff(bin_VALUE_TRUE)[0], 
    #          align='edge', color='w', edgecolor='b', lw=2 )
    
    plt.plot(  np.append( mean_2, mean_2[-1] ), bin_VALUE_TRUE, color='royalblue', lw=2, drawstyle='steps-pre' )
    
    ax3.set_ylim( bin_VALUE_TRUE[0], bin_VALUE_TRUE[-1] )
    ax3.set_xlim( ax3.get_xlim()[::-1] )
    
    [l.set_rotation(45) for l in ax3.get_xticklabels()]  
    ax3.xaxis.set_major_locator( MaxNLocator( 6, prune='lower', symmetric=True ) )
     
    fig.canvas.draw()    
    labels = [ item.get_text() for item in ax3.get_xticklabels() ]
    #print(labels)
    labels[0] = ''
    ax3.set_xticklabels( labels )
    
    ax3.plot( Brad_pt[0,1]-Brad_pt[0,0], Brad_pt[0,1], '^c' )
    ax3.plot( Brad_pt[1,1]-Brad_pt[1,0], Brad_pt[1,1], 'vc' )
    
    ax3.errorbar( Brad_pt[0,1]-Brad_pt[0,0], Brad_pt[0,1], xerr=[Brad_pt[0,2:]], ecolor='c', fmt='.', color='c', ms=0, errorevery=2 )
    ax3.errorbar( Brad_pt[1,1]-Brad_pt[1,0], Brad_pt[1,1], xerr=[Brad_pt[1,2:]], ecolor='c', fmt='.', color='c', ms=0, errorevery=2 )
    
    ax3.set_ylabel( '%s, Predicted'%paramName[paramNum], fontsize=10 )
    ax3.set_xlabel( r'$ \rm{ u } $', fontsize=10 )
    ######################################
    ######################################
    fig.tight_layout()
    return fig, bin_VALUE_TRUE, eps_mean,eps_std, u_mean, u_std


# In[ ]:




# In[ ]:

def save_pred( pred, prediction_file ):
    np.save( prediction_file, pred )
def load_pred( prediction_file ):
    return np.load( prediction_file )


# In[ ]:




# In[ ]:

def R2( out_model, Param):
    return 1 - ( (out_model - Param)**2).sum(axis=0) / ((out_model - out_model.mean(axis=0) )**2).sum(axis=0) 

###############################################################
def print_R2( prediction_file, param_num, Param, sub_select=None  ):
    
    out_model = load_pred( prediction_file )
    
    if not(sub_select is None):
        out_model = out_model[sub_select]
        Param     = Param[sub_select]        
    
    if np.isscalar(param_num):
        num = param_num
        print( 'R2 %s: '%(paramName[num]),  R2( out_model, Param[:,num] ) )
    else:
        for num in param_num:
            print( 'R2 %s: '%(paramName[num]),  R2( out_model[:,num], Param[:,num] ) )


# In[ ]:




# In[ ]:

def plot_history( history_file, model_file, fig=None, save=False ):
    
    history = np.load( history_file )
    
    hist_Nepoch = len( history.all()['loss'] )
    epoch = np.arange(hist_Nepoch,dtype=np.int)+1
    loss     = np.log10( history.all()[    'loss'] )
    val_loss = np.log10( history.all()['val_loss'] )
    
    if fig is None:
        fig, ax = plt.subplots()
        
    line = plt.plot( epoch, loss, label='Training loss' )
    if 'val_loss' in history.all():
        plt.plot( epoch, val_loss, 
                  '--', color=line[0].get_color(), label='Validation loss' )
        
    plt.legend(loc='best')
    #plt.semilogy()
    #plt.ylim(1e-3, 1e0)
    
    #plt.axhline( np.log10(0.01), color='k', alpha=0.5, linewidth=0.5 )
    #plt.axhline( np.log10(0.02) , color='k', alpha=0.5, linewidth=0.5 )
        
    plt.xlabel( 'Epochs' )
    plt.ylabel( 'Loss: MSE [log]' )
    
    YLIM = plt.ylim()
    XLIM = plt.xlim()
    ax.set_aspect( np.abs( np.diff(XLIM) ) / np.abs( np.diff(YLIM) ) )
    
    if save:
        plot_file = 'plots/%s'%model_file+'_loss.pdf'
        print( plot_file )
        utils.saveFig( fig, plot_file )
    return fig


# In[ ]:




# In[ ]:

def plot_result( prediction_file, param_num, Param, Param_raw, name, Nbin=100, save=False, save_name='',
                 Brad_pt=None, sub_select=None ):
    
    ### GET THE PREDICTED VALUES
    out_model = load_pred( prediction_file )
    out_shape = out_model.shape
    
    ### MIN AND MAX OF PARAM 
    paramMins = Param_raw.min(axis=0)
    paramMaxs = Param_raw.max(axis=0)
        
    if not(sub_select is None):
        out_model = out_model[sub_select]
        Param     = Param[sub_select]
        
    print(paramMaxs)
    
    if np.isscalar(param_num):
        num = param_num
        print( 'R2 %s: '%(paramName[num]),  R2( out_model, Param[:,num] ) )
        print( 'plots/%s'%paramName[num]+name+'.pdf' )
        fT, eps_mean,eps_std, u_mean, u_std = plot_CNN_out( out_model, Param, num, Nbin=Nbin, 
                                                            paramMins=paramMins, 
                                                            paramMaxs=paramMaxs )
        if save:
            print( 'plots/%s'%paramName[num]+name+'.pdf' )
            utils.saveFig( fT, 'plots/%s'%paramName[num]+name+'.pdf' )
    else:
        bin_true = np.zeros([len(param_num),Nbin])
        eps_mean = np.zeros([len(param_num),Nbin-1])
        eps_std  = np.zeros([len(param_num),Nbin-1])
        u_mean   = np.zeros([len(param_num),Nbin-1])
        u_std    = np.zeros([len(param_num),Nbin-1])
        for num in param_num:
            print( 'R2 %s: '%(paramName[num]),  R2( out_model[:,num], Param[:,num] ) )
            print( 'plots/%s'%paramName[num]+name+'.pdf' )
                        
            fT, bin_true[num,:], eps_mean[num,:], eps_std[num,:], u_mean[num,:], u_std[num,:] = plot_CNN_out( out_model[:,num],  
                                                                                                              Param, num, 
                                                                                                              Nbin=Nbin, 
                                                                                                              paramMins=paramMins, 
                                                                                                              paramMaxs=paramMaxs,
                                                                                                              Brad_pt=Brad_pt[:,num,:] )
            if save:
                print( 'plots/%s'%paramName[num]+name+save_name )
                utils.saveFig( fT, 'plots/%s'%paramName[num]+name+save_name+'.pdf' )
                utils.saveFig( fT, 'plots/%s'%paramName[num]+name+save_name )
                
                
    return bin_true, eps_mean, eps_std, u_mean, u_std


# In[ ]:




# In[2]:


def plot_CNN_error( out, param, paramNum, paramMins, paramMaxs, color='b', Nbin=20 ):
    
    pmin = paramMins[paramNum] ### minimum of the selected parameter
    d    = paramMaxs[paramNum] - paramMins[paramNum] ### range of the selected parameter
    VALUE_TRUE = param[:,paramNum]*d+pmin ### recovert 'physical value' of input
    VALUE_PRED = out*d+pmin ### recovert 'physical value' of output
    
    #VALUE_PRED[ np.where(VALUE_PRED<0) ] = VALUE_TRUE.min()
    bin_VALUE_TRUE = np.linspace( VALUE_TRUE.min(), VALUE_TRUE.max(), Nbin )
    
    #coef = 100 * (VALUE_PRED - VALUE_TRUE) / VALUE_TRUE
    residual = (VALUE_PRED - VALUE_TRUE)
    
    residual_avg = np.average( residual )
    residual_med = np.median( residual )
    residual_std = np.std( residual ) 
    
    #paramName = [r'$\rm{\zeta}$', r'$\rm{T_{vir}}$', r'$\rm{L_X}$', r'$\rm{E_o}$' ]
    paramName = [ r'$\rm{ u_{\zeta}} $', 
                  r'$\rm{ u_{T_{vir}}} $', 
                  r'$\rm{ u_{L_X}} $', 
                  r'$\rm{ u_{E_o}} $' ]
    
    ######################################
    ######################################
    fig = plt.figure(  )
    ######################################
    ######################################
    ### TRUE-PREDICTED plot
    ax = plt.subplot( 111 )
        
    MIN = residual.min()
    MAX = residual.max()
    residual_bins = np.linspace( MIN, MAX, Nbin )
    
    h_residual, x_h = np.histogram( residual, bins=residual_bins )
    SAVED_NORMALIZATION = h_residual.sum()
    residual_pdf = h_residual / SAVED_NORMALIZATION
    
    ### FIND ERROR SIGMA
    s1m, s1p = utils.quantile( utils.zcen(residual_bins), [0.16,0.84], weights=residual_pdf )
    s2m, s2p = utils.quantile( utils.zcen(residual_bins), [0.025,1.-0.025], weights=residual_pdf )
    s3m, s3p = utils.quantile( utils.zcen(residual_bins), [0.005,1.-0.005], weights=residual_pdf )
    
    ### gauss fit
#     gx = np.linspace( MIN, MAX, 200 )
#     g_init = models.Gaussian1D( amplitude=residual_pdf.max(), mean=residual_avg, stddev=np.min( [-s1m, s1p] ) )
#     fit_g = fitting.LevMarLSQFitter()
#     g = fit_g( g_init, utils.zcen(residual_bins), residual_pdf )
    
#     l_init = models.Lorentz1D( amplitude=residual_pdf.max(), x_0=residual_avg, fwhm=np.min( [-s1m, s1p] ) )
#     fit_l = fitting.LevMarLSQFitter()
#     l = fit_l( l_init, utils.zcen(residual_bins), residual_pdf )
    
#     v_init = models.Voigt1D( amplitude_L=residual_pdf.max(), x_0=residual_avg, fwhm_L=np.min( [-s1m, s1p] ), fwhm_G=np.min( [-s1m, s1p] ) )
#     fit_v = fitting.LevMarLSQFitter()
#     v = fit_v( v_init, utils.zcen(residual_bins), residual_pdf )

    ### NEW HISTOGRAM JUST FOR NICE PLOT
#     MIN = np.max( [residual.min(), -200] )
#     MAX = np.min( [residual.max(),  200] )
#     plt.xlim( MIN, MAX )
#     residual_bins = np.linspace( MIN, MAX, Nbin )
#     h_residual, x_h = np.histogram( residual, bins=residual_bins )
#     h_residual = h_residual / h_residual.max() * residual_pdf.max()
    h_residual = residual_pdf
    
    ### plot 1, 2, 3 sigma
    x_s3 = np.where( (utils.zcen(residual_bins)>=s3m) * (utils.zcen(residual_bins)<=s3p) )[0]
    x_s3 = np.append( x_s3[0]-1, x_s3 )
    x_s3 = np.append( x_s3, x_s3[-1]+1,  )
    plt.fill_between( utils.zcen(residual_bins)[x_s3], h_residual[x_s3], step='mid', 
                      alpha=0.2, color='royalblue' )
    
    x_s2 = np.where( (utils.zcen(residual_bins)>=s2m) * (utils.zcen(residual_bins)<=s2p) )[0]
    x_s2 = np.append( x_s2[0]-1, x_s2 )
    x_s2 = np.append( x_s2, x_s2[-1]+1,  )
    plt.fill_between( utils.zcen(residual_bins)[x_s2], h_residual[x_s2], step='mid', 
                      alpha=0.5, color='royalblue' )
    
    x_s1 = np.where( (utils.zcen(residual_bins)>=s1m) * (utils.zcen(residual_bins)<=s1p) )[0]
    x_s1 = np.append( x_s1[0]-1, x_s1 )
    x_s1 = np.append( x_s1, x_s1[-1]+1,  )
    plt.fill_between( utils.zcen(residual_bins)[x_s1], h_residual[x_s1], step='mid', 
                      color='royalblue' )
    
    plt.grid( color='k', linestyle=':', alpha=0.5 )
    
    ax.plot( utils.zcen(residual_bins), h_residual, 'b-', lw=2, drawstyle='steps-mid' )
    
    YLIM = plt.ylim()

    #ax.axvline( residual_avg, color='g' )
    #ax.axvline( residual_med, color='r' ) #, ls="dashed" )
    
    #ax.axvline( s1m, color='k', ls="dashed", lw=0.5 )
    #ax.axvline( s2m, color='k', ls="dashed", lw=0.5 )
    #ax.axvline( s3m, color='k', ls="dashed", lw=0.5 )
    
    #ax.axvline( s1p, color='k', ls="dashed", lw=0.5 )
    #ax.axvline( s2p, color='k', ls="dashed", lw=0.5 )
    #ax.axvline( s3p, color='k', ls="dashed", lw=0.5 )
    
#     ax.plot( gx, g(gx), 'c--', lw=1 )
#     ax.plot( gx, l(gx), 'y--', lw=1 )
#     ax.plot( gx, v(gx), 'r', lw=1 )
#     print(g)
#     print(l)
#     print(v)
    
    print( 'residual mean:', residual_avg )
    print( 'residual std :', residual_std )
    print( '16-84 percentil :', s1m, s1p )
    print( '2.5-97.5 percentil :', s2m, s2p )
    print( '0.5-99.5 percentil :', s3m, s3p )
    
    plt.ylim( YLIM )
    
    XLIM = plt.xlim()
    ax.set_aspect( np.abs(np.diff(XLIM)) /  np.abs(np.diff(YLIM)) )
    
    if paramNum==0:
        text = 'residual mean: %.2f\n'%(residual_avg)
        text += ' 16-84   %s: %.1f, %.1f\n'%('%',s1m,s1p)
        text += '2.5-97.5 %s: %.1f, %.1f\n'%('%',s2m,s2p)
        text += '0.5-99.5 %s: %.1f, %.1f'%(  '%',s3m,s3p)
        xBox = -170        
        yBox = 0.2
        size=10
        alpha=1
    if paramNum==1:
        text = 'residual mean: %.2e\n'%(residual_avg)
        text += ' 16-84   %s: %.1e, %.1e\n'%('%',s1m,s1p)
        text += '2.5-97.5 %s: %.1e, %.1e\n'%('%',s2m,s2p)
        text += '0.5-99.5 %s: %.1e, %.1e'%(  '%',s3m,s3p)
        xBox = 0.025
        yBox = 0.12
        size=10
        alpha=1
    if paramNum==2:
        text = 'residual mean: %.2f\n'%(residual_avg)
        text += ' 16-84   %s: %.2f, %.2f\n'%('%',s1m,s1p)
        text += '2.5-97.5 %s: %.2f, %.2f\n'%('%',s2m,s2p)
        text += '0.5-99.5 %s: %.2f, %.2f'%(  '%',s3m,s3p)
        xBox = 0.25
        yBox = 0.08
        size=10
        alpha=1
    if paramNum==3:
        text = 'residual mean: %.2f\n'%(residual_avg)
        text += ' 16-84   %s: %.2f, %.2f\n'%('%',s1m,s1p)
        text += '2.5-97.5 %s: %.2f, %.2f\n'%('%',s2m,s2p)
        text += '0.5-99.5 %s: %.2f, %.2f'%(  '%',s3m,s3p)
        xBox = -1.150
        yBox = 0.12
        size=10
        alpha=1.
    plt.text(xBox, yBox, text,
             bbox={'facecolor':'white', 'alpha':alpha}, size=size,
             #horizontalalignment='center',
             #verticalalignment='top',
             #multialignment='center'
            )
    
    plt.xlabel( r' %s $\rm{ [Pred-True] }$ '%paramName[paramNum], fontsize=10 )
    plt.ylabel( 'pdf' )
    ######################################
    ######################################
    fig.tight_layout()
    return fig


# In[ ]:




# In[ ]:

def plot_result_2( prediction_file, param_num, Param, Param_raw, name, Nbin=100, save=False ):
    
    out_model = load_pred( prediction_file )
    out_shape = out_model.shape
    
    #model.predict( LC_test, verbose=True )
    
    if np.isscalar(param_num):
        num = param_num

        fT = plot_CNN_error( out_model,  Param, num, Nbin=Nbin, 
                             paramMins=Param_raw.min(axis=0), 
                             paramMaxs=Param_raw.max(axis=0) )
        print( 'R2 %s: '%(paramName[num]),  R2( out_model, Param_test[:,num] ) )
        print( 'plots/%s'%paramName[num]+name+'_error.pdf' )
        if(save):
            utils.saveFig( fT, 'plots/%s'%paramName[num]+name+'_error.pdf' )
    
    ### ALL4 
    else:
        for num in param_num:
            
            fT = plot_CNN_error( out_model[:,num],  Param, num, Nbin=Nbin, 
                                 paramMins=Param_raw.min(axis=0), 
                                 paramMaxs=Param_raw.max(axis=0) )
            print( 'R2 %s: '%(paramName[num]),  R2( out_model[:,num], Param[:,num] ) )
            print( 'plots/%s'%paramName[num]+name+'_error.pdf' )
                
            if save:
                print( 'plots/%s'%paramName[num]+name+'_error' )
                utils.saveFig( fT, 'plots/%s'%paramName[num]+name+'_error'+'.pdf' )
                utils.saveFig( fT, 'plots/%s'%paramName[num]+name+'_error' )


# In[ ]:




# In[4]:

def plot_outConv( inputLC, inputLayer, model, extent=None, freq_label=None, redshift_label=None ):
    
    
    from keras import backend as K
    
    ### create a function that will mimic the CNN
    inputs = [K.learning_phase()] + model.inputs
    _convout1_f = K.function(inputs, [inputLayer.output])
    def convout1_f(X):
        return _convout1_f( [0] + [X] )
    ### 
    convolutions = np.squeeze( convout1_f( inputLC ) )
            
    N_filters = np.array( convolutions ).shape[0]
    ###
    print(convolutions.shape)
    
    R = convolutions.shape[2] / convolutions.shape[1]

    N = 1
    H =  N_filters//N + 1

    factor = 15
    lrdim = 0.1*factor
    tbdim = 0.1*factor
    whspace = 0.03

    plot_Largeur = factor*(N) + factor*(N-1) * whspace
    dim_Largeur = lrdim + plot_Largeur + tbdim 

    #plot_Hauteur = factor*(H) + factor*(H-1) * whspace
    plot_Hauteur = plot_Largeur * H / R
    dim_Hauteur = lrdim + plot_Hauteur + tbdim 

    #cbpad = 0.01
    #cbfraction = 0.05
    #cbspace = plot_Hauteur * ( cbfraction + cbpad )
    #dim_Hauteur += cbspace

    fig, axArray = plt.subplots( H, N, figsize=(dim_Largeur,dim_Hauteur) )
    #fig, axArray = plt.subplots( H, N, figsize=(10,16) )

    l = lrdim / dim_Largeur
    r = 1 - l
    b = tbdim / dim_Hauteur
    t = 1 - b
    fig.subplots_adjust( left=l, bottom=b, right=r, top=t, wspace=whspace, hspace=whspace )
    
    ax = axArray[ 0 ]
    ax.imshow( np.squeeze(inputLC), cmap='EoR_colour', vmin=0, vmax=1, extent=extent ) # cmap=cm.seismic, interpolation=None )
    
    ax.set_yticks([0,75,150])
    ax.set_ylabel( 'L [Mpc]' )
    
    if not(freq_label is None):
        f_to_D, freq = freq_label
        ax.tick_params( labelbottom='off', labeltop='on', bottom='off', top='on' )
        ax.set_xticks( f_to_D.value )
        ax.set_xticklabels( freq.astype( np.int ) )
        ax.set_xlabel( 'Frequency [MHz]' )
        ax.xaxis.set_label_position('top')

    for i in range( N_filters ):
        ax = axArray[ i+1 ]
        im = ax.imshow( convolutions[i], cmap='nipy_spectral', extent=extent ) # cmap=cm.seismic, interpolation=None )
        if i < (N_filters-1):
            ax.set_xticks([])
        else:
            if not(redshift_label is None) :
                #plt.xlabel( 'Mpc' )
                z_to_D, redshift = redshift_label
                plt.xticks( z_to_D.value, redshift )
                ax.set_xlabel( 'Redshift, z' )
                
        ax.set_yticks([])
        
    return fig  


# In[ ]:



