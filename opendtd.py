"""
opendtd.py

Functions to open observational/theoretical DTD data from Maoz+14 review.
"""

#Backend for python3 on stravinsky
from pickle import BINSTRING
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Other packages
import numpy as np
from scipy.io import loadmat

# Do some formatting stuff with matplotlib
from matplotlib import rc
from matplotlib.ticker import MultipleLocator, AutoMinorLocator, MaxNLocator
rc('font', family='serif')
rc('axes', labelsize=14) 
rc('xtick', labelsize=12)
rc('ytick', labelsize=12)
rc('xtick.major', size=10)
rc('ytick.major', size=10)
rc('legend', fontsize=12, frameon=False)
rc('text',usetex=True)
rc('xtick',direction='in')
rc('ytick',direction='in')

# Get some cool colors
import cmasher as cmr
#colors = cmr.take_cmap_colors('cmr.neon', 6, cmap_range=(0.1, 0.9), return_fmt='hex')
colors = plt.cm.plasma(np.linspace(0,1,7))

# Open matlab file from Maoz+14 review
m = loadmat('data/maoz14_dtd.mat', mat_dtype=True, chars_as_strings=True)
#print(m)

# Time array
t = np.array([30., 100., 300., 1000., 3000., 10000., 13700.])/1000. # Gyr
#t = np.array([30., 60., 200., 600., 2000., 6000., 13700.])/1000. # Gyr

# Double-degenerate models
dd = {'Maoz et al. (2010)': 1e-3*t**(-1.1)/1e9,
        'Yungelson (2010)':[0.0000000e+00, 2.8160000e-13, 2.3040000e-13, 8.8000000e-14, 2.5680000e-14, 7.5040000e-15, 3.3360000e-15],
        'Wang \& Han (2012)':[0.0000000e+00, 3.4028777e-14, 9.3646407e-13, 1.0675539e-13, 2.9692086e-14, 8.7999997e-15, 5.2401322e-15],
        'Ruiter et al. (2009)':[0.0000000e+00, 1.9347582e-13, 2.2860892e-13, 1.7559055e-13, 3.9514436e-14, 1.0228721e-14, 2.6459530e-15],
        'Mennekens et al. (2010)':np.array([0.0000000e+00, 2.5420000e-13, 1.2152000e-13, 6.0388000e-14, 1.0726000e-14, 4.5942000e-15, 4.2532000e-15]),
        'Bours et al. (2013)':[0.0000000e+00, 6.4028571e-15, 1.2076500e-13, 3.4255286e-14, 1.5263700e-14, 4.5638143e-15, 2.3477143e-15],
        'Claeys et al. (2014)':[0.0000000e+00, 0.0000000e+00, 3.6818252e-14, 1.7991092e-13, 3.7722534e-14, 8.7764985e-15, 3.2027281e-15]}

dd['Maoz et al. (2010)'][0] = 0.

# Single-degenerate models
sd = {'Yungelson (2010)':[0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 3.8160000e-17, 1.8240000e-16, 0.0000000e+00, 0.0000000e+00],
        'Wang \& Han (2012)':[0.0000000e+00, 0.0000000e+00, 4.1546763e-15, 2.8386331e-13, 3.8139928e-14, 7.0497338e-16, 5.9887225e-17],
        'Ruiter et al. (2009)':[0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.6497938e-15, 1.3385827e-15, 8.1364829e-16, 1.5606157e-16],
        'Mennekens et al. (2010)':[0.0000000e+00, 6.7580000e-13, 9.6100000e-13, 1.4818000e-13, 7.6260000e-15, 2.9574000e-16, 0.0000000e+00],
        'Bours et al. (2013)':[0.0000000e+00, 4.0000000e-14, 7.0000000e-14, 4.4000000e-14, 9.0000000e-15, 5.4000000e-17, 0.0000000e+00],
        'Claeys et al. (2014)':[0.0000000e+00, 7.9306748e-14, 3.9005567e-14, 3.0061347e-15, 0.0000000e+00, 0.0000000e+00, 0.0000000e+00]}

def plotdtd():
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,10)) #, sharex=True, sharey=True)
    axs = axs.ravel()

    # Plot double-degenerate models
    for i, modelname in enumerate(dd.keys()):
        axs[0].plot(t, np.asarray(dd[modelname])*1e9, ls='-', color=colors[i], lw=2, label=modelname)
    axs[0].set_title('Double-degenerate', fontsize=18)

    # Plot single-degenerate models
    for i, modelname in enumerate(sd.keys()):
        axs[1].plot(t, np.asarray(sd[modelname])*1e9, ls='-', color=colors[i+1], lw=2, label=modelname)
    axs[1].set_title('Single-degenerate', fontsize=18)

    # Plot Maoz+10 power law used in GCE
    '''
    t_ia = 1e-1 #Gyr
    maoz10 = (1e-3)*t**(-1.1)
    w = np.where(t < t_ia)[0]
    if len(w) > 0: maoz10[w] = 0.0
    axs[0].plot(t, maoz10, color='C0', ls='--', lw=1.5, label='Maoz et al. (2010)')
    axs[1].plot(t, maoz10, color='C0', ls='--', lw=1.5, label='Maoz et al. (2010)')
    '''

    # Formatting stuff
    for axnum in range(2):
        axs[axnum].set_xlim(0.05,13.7)
        #axs[axnum].set_ylim(1e-8,1e-1)
        axs[axnum].set_xscale('log')
        axs[axnum].set_yscale('log')
        axs[axnum].set_ylabel(r'DTD ($M_{\odot}^{-1}~\mathrm{Gyr}^{-1}$)', fontsize=16)

    axs[0].set_xlabel('')
    axs[1].set_xlabel(r'Delay time $\tau$ (Gyr)', fontsize=16)

    legend = axs[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=16)

    plt.savefig('figures/DTDmodels.pdf', bbox_inches='tight')
    plt.show()

    return

if __name__=="__main__":
    plotdtd()