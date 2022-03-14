"""
amr.py

Functions to compute the age-metallicity relation for each galaxy.
"""

#Backend for python3 on stravinsky
from pickle import BINSTRING
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Import other modules
from astropy.io import ascii  # only needed for SFH test stuff
import numpy as np
from astropy.cosmology import FlatLambdaCDM  # needed to compute redshifts
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)  # using Planck (2018) params
import astropy.units as u
from astropy.cosmology import z_at_value
from astropy.table import Table

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

# Some other stuff to make cool colors
import cycler
import cmasher as cmr

def computeamr(galaxy, plot=False):
    '''Compute age-metallicity relation from SFH and MDF.'''

    galaxynames = {'Scl':'Sculptor', 'Dra':'Draco', 'LeoII':'Leo II'}
    #logMstar = {'LeoII':6.16, 'Scl':6.08, 'Dra':5.96, 'Sex':5.93, 'UMi':5.75}  # Stellar masses from Woo et al. (2008)

    # Open SFH (from Weisz+14)
    if galaxy=='Scl':
        weisz = ascii.read('data/sfh/weisz14_multfields.dat')
    else:
        weisz = ascii.read('data/sfh/weisz14_singlefield.dat')

    gal_idx = np.where(weisz['ID']==galaxynames[galaxy])
    colnames = [name for name in weisz.colnames if name.startswith('f')]
    t = [(10.**10.15)/1e9]+[10.**float(name[1:])/1e9 for name in colnames] # Lookback in Gyr
    cumsfh = np.asarray([0]+[float(weisz[name][gal_idx]) for name in colnames])
    cumsfh_uplim = np.asarray([0]+[float(weisz['Ut'+name][gal_idx]) for name in colnames])
    cumsfh_lolim = np.asarray([0]+[float(weisz['Lt'+name][gal_idx]) for name in colnames])

    # Get cumulative MDF from Kirby+09
    kirby09_new = Table.read('data/mdf/kirby_dsph_catalog.dat', format='ascii.cds')
    idx = np.where(kirby09_new['dSph']==galaxy)
    mdf_feh = kirby09_new['eps(Fe)'][idx] - 7.50
    mdf_feh_err = kirby09_new['e_eps(Fe)'][idx]

    # Remove stars with large error
    goodidx = np.where(mdf_feh_err < 0.3)
    mdf_feh = mdf_feh[goodidx]
    mdf_feh_err = mdf_feh_err[goodidx]

    cdf_feh, counts = np.unique(mdf_feh, return_counts=True)
    cummdf = np.cumsum(counts)
    cummdf = cummdf / cummdf[-1]

    # Interpolate to the same array of fraction of stars formed
    cumsfh_lo = np.interp(cummdf, cumsfh-cumsfh_lolim, t)
    cumsfh_hi = np.interp(cummdf, cumsfh+cumsfh_uplim, t)
    t = np.interp(cummdf, cumsfh, t)

    if plot:
        # Test: compare Sculptor AMR against GCE model
        if galaxy=='Scl':
            amr_delosreyes = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/plots/amr_test.npy')
            feh = amr_delosreyes[:,0]  # [Fe/H]
            age = 13.791 - amr_delosreyes[:,1]  # Gyr
            goodidx = np.where(np.isfinite(feh))[0]
            feh = feh[goodidx]
            age = age[goodidx]
            plt.plot(feh, age, 'r--', label='GCE')

        # Try plotting lookback time and metallicity against each other now?
        plt.plot(cdf_feh, t, 'k-', label='Weisz+14')
        plt.xlim(-3.2,-0.5)
        plt.xlabel('[Fe/H]')
        plt.ylabel('Lookback time (Gyr)')
        plt.legend(loc='best')
        if galaxy=='Scl':
            plt.savefig('figures/Scl_amr_test.png', bbox_inches='tight')
        plt.show()

    return cdf_feh, t, cummdf, cumsfh_lo, cumsfh_hi

def makeplots():
    '''Plot the SFH, MDF, and AMR for multiple galaxies.'''

    galaxynames = {'Scl':'Sculptor', 'Dra':'Draco', 'LeoII':'Leo II'}

    # Some plotting things
    handles, labels = [], []
    handles2, labels2 = [], []
    #fig = plt.figure(figsize=(8,4))
    #ax = fig.add_subplot(rasterized=True)

    fig, axs = plt.subplots(3, figsize=(6,9))
    fig.subplots_adjust(hspace=0.4) 
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune=None)) for a in [fig.axes[0]]])
    plt.setp([a.yaxis.set_major_locator(MaxNLocator(nbins=5,prune='upper')) for a in fig.axes[1:]])
    #plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    plt.setp([a.minorticks_on() for a in fig.axes[:]])
    axs = axs.ravel()

    # Add redshift axis on top
    '''
    ages = np.array([10, 5, 3, 2, 1, 0.5])*u.Gyr
    ageticks = [13.791 - z_at_value(cosmo.age, age) for age in ages]
    ax2 = axs[0].twiny()
    ax2.set_xticks(ageticks)
    ax2.set_xticklabels(['{:g}'.format(age) for age in ages.value])
    ax2.set_xlabel('Redshift')
    zmin, zmax = 14.0, 0
    axs[0].set_xlim(zmin, zmax)
    ax2.set_xlim(zmin, zmax)
    '''

    # Loop over all galaxies
    for i, galaxy in enumerate(galaxynames.keys()):

        feh, lookback, cumfrac, cumsfh_lolim, cumsfh_uplim = computeamr(galaxy)

        # Find [Fe/H] limit for when galaxy formed 95% of its mass
        lim95 = np.where(cumfrac <= 0.95)[0]
        badlim95 = np.where(cumfrac >= 0.95)[0]
        badlim95 = np.concatenate(([lim95[-1]],badlim95))

        # Plot SFH
        p3, = axs[0].plot(lookback, cumfrac, ls='-', lw=2, color=plt.cm.Dark2(i))
        p4 = axs[0].fill_betweenx(cumfrac, cumsfh_lolim, cumsfh_uplim, color=plt.cm.Dark2(i), alpha=0.3)
        handles.append((p3,p4))
        labels.append(galaxynames[galaxy])
        axs[0].set_xlabel('Lookback time (Gyr)')
        axs[0].set_ylabel('Cumulative SFH')
        axs[0].set_xlim(14,0)
        axs[0].set_ylim(0,1)
        axs[0].legend(handles=handles, labels=labels, loc='best')

        # Plot MDF
        axs[1].plot(feh, cumfrac, ls='-', lw=2, color=plt.cm.Dark2(i))
        axs[1].set_xlabel('[Fe/H]')
        axs[1].set_ylabel('Cumulative MDF')
        axs[1].set_xlim(-3.2,-0.7)
        axs[1].set_ylim(0,1)

        # Plot AMR
        axs[2].plot(feh[lim95], lookback[lim95], ls='-', lw=2, color=plt.cm.Dark2(i))
        axs[2].plot(feh[badlim95], lookback[badlim95], ls='-', lw=2, color=plt.cm.Pastel2(i))
        #axs[2].axvline(feh[lim95][-1], ls='--', lw=1, color=plt.cm.Dark2(i))
        axs[2].set_xlabel('[Fe/H]')
        axs[2].set_ylabel('Lookback time (Gyr)')
        axs[2].set_xlim(-3.2,-0.7)
        axs[2].set_ylim(14,0)

        # Plot linear fits to AMR
        age, feh, p, fehlimidx = amr(galaxy)
        linearfit = np.polyval(p,feh)
        p5, = axs[2].plot(feh, linearfit, ls=':', lw=2, color=plt.cm.Dark2(i))
        handles2.append(p5)
        labels2.append(galaxynames[galaxy]+': $y='+'{:.2f}'.format(p[0])+'x'+'{:+.2f}'.format(p[1])+'$')
        print(labels2)
        axs[2].legend(handles=handles2, labels=labels2, loc='best')

    plt.savefig('figures/params_time.pdf', bbox_inches='tight')
    plt.show()

    '''
    # Plot MDF
    plt.figure(figsize=(8,4))
    plt.plot(cdf_feh, cummdf, ls='--', lw=2, color=plt.cm.Dark2(4))
    plt.xlabel('[Fe/H]')
    plt.ylabel('Cumulative MDF')
    plt.show()
    '''

    return

def amr(name, gcetest=False, plot=False):
    '''Returns age, metallicity, best-fit poly1d, indices (over which poly1d is fit)'''

    feh, age, cumfrac, _, _ = computeamr(name)
    fehlimidx = np.where(cumfrac < 0.95)[0]  # [Fe/H] limit for when galaxy formed 95% of its mass
    feh = np.asarray(feh)
    age = np.asarray(age)

    if name=='Scl' and gcetest==True:
        # Get age-metallicity relation from my GCE model
        amr_delosreyes = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/plots/amr_test.npy')
        feh = amr_delosreyes[:,0]  # [Fe/H]
        age = 13.791 - amr_delosreyes[:,1]  # Gyr
        goodidx = np.where(np.isfinite(feh))[0]
        feh = feh[goodidx]
        age = age[goodidx]

        fehlimidx = np.where((feh > -2) & (feh < -1.5))[0]

    '''
    elif name=='deBoer':
        # Get age-metallicity relation from de Boer et al. (2012)
        amr_deBoer = np.loadtxt('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/data/sfhtest/deboer12-ageZ.dat', delimiter=',')
        age_deBoer_idx = np.argsort(amr_deBoer[:,1])  # Sort by metallicity
        age = amr_deBoer[age_deBoer_idx,0]  # Gyr
        feh = amr_deBoer[age_deBoer_idx,1]  # [Fe/H]

    elif name=='Weisz':
        feh, age, _, _, _ = computeamr('Scl')
        feh = np.asarray(feh)
    '''

    # Fit polynomial to age-Z relation
    #fehlimidx = np.where((feh > fehlim[0]) & (feh < fehlim[1]))[0]
    p = np.polyfit(feh[fehlimidx], age[fehlimidx], 1)

    if plot:
        print(p)

        plt.plot(feh[fehlimidx], age[fehlimidx], 'ko', label='Observed AMR')
        plt.plot(feh[fehlimidx], np.polyval(p, feh[fehlimidx]), 'r-', label='Best-fit line')
        #plt.axvline(feh[fehlimidx][-1], color='b', ls=':', lw=2)
        plt.legend(title=name, fontsize=12, title_fontsize=14)
        plt.xlabel('[Fe/H]', fontsize=16)
        plt.ylabel('Time (Gyr)', fontsize=16)
        plt.savefig('figures/amr_'+name+'.png', bbox_inches='tight')
        plt.show()

    return age, feh, p, fehlimidx

if __name__=="__main__":
    #computeamr('Scl', plot=True)
    makeplots()
    #amr('Scl', plot=True, gcetest=False)