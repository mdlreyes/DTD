"""
DTD_fit.py

Computes and fits Type Ia DTD from abundance data.
Doesn't use age-metallicity relation! (DTD is function of [Fe/H])
"""

#Backend for python3 on stravinsky
from pickle import BINSTRING
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Do some formatting stuff with matplotlib
from matplotlib import rc
rc('font', family='serif')
rc('text',usetex=True)

# Import other packages
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from astropy.table import Table
from scipy.interpolate import interp1d
from scipy.stats import norm
from scipy import integrate
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter
from tqdm import tqdm

# Some constants
MScl = 1.2e6  # Stellar mass of Sculptor (Msun)

# Get Fe_Ia/Fe_CC info from Kirby+19
rfile = fits.open('/Users/miadelosreyes/Documents/Research/MnDwarfs/data/dsph_data/Evandata/chemev_scl.fits')
R = rfile[1].data['R'].T[:,0]
Rerrlo = rfile[1].data['RERRL'].T[:,0]
Rerrhi = rfile[1].data['RERRH'].T[:,0]
feh = np.linspace(-2.97,0, num=100)

# Denote metallicity limits
#fehlim = (-2.05,-1.0)  # Metallicity limits
#fehidx = np.where((feh > fehlim[0]) & (feh < fehlim[1]))[0]

# Define number of bins to treat as free parameters
bins = 50
#if len(feh[fehidx]) % bins != 0:
#    print(bins, len(feh[fehidx]))
#    raise ValueError('Bin size and [Fe/H] limits not compatible')

# Get Fe_Ia/Fe_CC info from the output from getdata.py
'''
rfile = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/scl_r.npy')
R = rfile[0]
Rerrlo = R-rfile[2]
Rerrhi = rfile[1]-R
feh = np.linspace(-2.97,0, num=100)
'''

# Get MDF data from Kirby+09
kirby09 = Table.read('data/kirby_scl_mdf.txt', format='ascii.cds')
scl_FeH = kirby09['[Fe/H]']
scl_FeH_err = kirby09['e_[Fe/H]']

# Normalization from Kroupa IMF
class kroupaIMF(object):
    def __init__(self, A1=1.):
        self.A1 = A1
        self.A2 = self.A1*0.08**(-0.3)/(0.08**(-1.3))
        self.A3 = self.A2*0.5**(-1.3)/(0.5**(-2.3))
    def __call__(self, x):
        if np.size(x) == 1:
            x = [x]
        gamma = np.zeros(len(x))
        A = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] <= 0.08:
                gamma[i] = -0.3
                A[i] = self.A1
            elif x[i] <= 0.5:
                gamma[i] = -1.3
                A[i] = self.A2
            else:
                gamma[i] = -2.3
                A[i] = self.A3
            
        return np.power(x,gamma) * A
    
# Renormalize Kroupa IMF across reasonable range of stellar masses
kroupa = kroupaIMF()
total, _ = integrate.quad(kroupa, a=0.01, b=100.)
kroupa.A1 = kroupa.A1/total
kroupa.A2 = kroupa.A2/total
kroupa.A3 = kroupa.A3/total

# Compute (# stars formed with M > 8 Msun)/(total # stars)
highmass, _ = integrate.quad(kroupa, a=10., b=40.)
allmass, _ = integrate.quad(kroupa, a=0.01, b=100.)
CCfrac = highmass/allmass

def expandDTD(DTD, length, bins):
    nreps = int((length+1)/bins)
    newDTD = np.asarray([[DTD[i]] * nreps for i in range(len(DTD))])
    return newDTD.flatten()[:-1]

def fitdtd(Niter, plot=False):
    """Fit DTD for a galaxy using MC-like method 
    (perturbing MDF, SFH to get realizations of Type Ia rates, 
    and fitting those realizations)."""

    # Let's do MC estimation of errors
    exp_rates = np.zeros((Niter,len(feh)-1)) #[fehidx])))
    obs_rates = np.zeros((Niter,len(feh)-1)) #[fehidx])))
    DTD_array = np.zeros((Niter, bins))
    sfh_array = np.zeros((Niter, len(feh)-1)) #[fehidx])))

    for iteration in tqdm(range(Niter)):

        # Get single realization of R
        percentile = np.random.uniform()
        new_r = np.zeros(len(R))
        for i in range(len(R)):
            if percentile > 0.5:
                scale=Rerrhi[i]
            else:
                scale=Rerrlo[i]
            new_r[i] = norm(loc=R[i],scale=scale).ppf(percentile)

        # Convert R to cumulative fraction of Ia/CC SNe
        fe_CC = 0.074  # Core-collapse yield from Maoz&Graur17
        fe_Ia = np.sum([7.80e-3,6.10e-1,2.12e-2,4.39e-4])  # Type Ia yields from Leung+19 (1.1 Msun, solar Z=0.02)
        N_Ia_CC = new_r/fe_Ia * fe_CC
        
        # Get single realization of MDF
        perturb = np.random.normal(scl_FeH, scl_FeH_err)
        mdf = np.histogram(perturb, bins=feh)

        # Normalize MDF -> SFH of Sculptor
        sfh = mdf[0]/np.nansum(mdf[0]) * MScl
        
        # Convert total mass formed to cumulative # of CCSNe
        N_CC = sfh*CCfrac
        N_CC_cumulative = np.zeros(len(N_CC))
        for i in range(len(N_CC)):
            N_CC_cumulative[i] = np.sum(N_CC[:(i+1)])

        # Compute rate of IaSNe as function of [Fe/H]
        N_Ia_cumulative = N_CC_cumulative * N_Ia_CC[:-1]
        N_Ia = np.diff(N_Ia_cumulative)
        N_Ia = np.insert(N_Ia, 0, 0)
        N_Ia[~np.isfinite(N_Ia)] = 0.
        N_Ia[N_Ia < 0.] = 0.

        # Save indices where rate of IaSNe is zero
        noIa_idx = np.where(np.isclose(N_Ia, 0.))[0]

        # Crop the observed properties to the [Fe/H] limits
        #sfh = sfh[fehidx]
        #N_Ia = N_Ia[fehidx]

        # Save SFH (for use in test plotting later)
        sfh_array[iteration, :] = sfh

        # Compute likelihood
        def log_likelihood(params):

            # Compute DTD by expanding parameters into full DTD
            DTD = expandDTD(params, len(sfh), bins)

            # Convolve DTD with SFH
            test = np.convolve(DTD, sfh)
            test[~np.isfinite(test)] = 0.
            N_Ia_exp = test[:len(N_Ia)]    

            # Compute log likelihood within [Fe/H] limits
            resid = N_Ia - N_Ia_exp
            likelihood = -0.5 * np.sum(np.power(resid,2.))
            #print('test', likelihood)
            return likelihood, N_Ia_exp

        def log_probability(params):

            likelihood, N_Ia_exp = log_likelihood(params)

            # Add in prior (make sure DTD is always positive)
            if np.any(params < 0.) or np.any(N_Ia_exp[noIa_idx] > 0.):
                return -np.inf
            else:
                return likelihood

        # Maximize likelihood (minimize negative likelihood)
        nll = lambda *args: -log_probability(*args)
        initial = np.zeros(bins)
        soln = minimize(nll, initial, method='powell', options={'ftol':1e-6, 'maxiter':100000})#, 'direc':np.diag([-0.01, 0.01, 0.01])})
        #print(soln.x, log_probability(soln.x))

        # Store solution in array
        DTD_array[iteration, :] = soln.x

        # Store observed and expected rates of IaSNe in arrays
        DTD = expandDTD(soln.x, len(sfh), bins)
        exprate = np.convolve(DTD, sfh)
        exprate[~np.isfinite(exprate)] = 0.

        exp_rates[iteration, :] = exprate[:len(N_Ia)]
        obs_rates[iteration, :] = N_Ia

    # Compute percentiles
    DTD_median = expandDTD(np.percentile(DTD_array, 50, axis=0), len(sfh), bins)
    DTD_lo = expandDTD(np.percentile(DTD_array, 16, axis=0), len(sfh), bins)
    DTD_hi = expandDTD(np.percentile(DTD_array, 84, axis=0), len(sfh), bins)

    if plot:

        # Make plot
        plt.plot(feh[:-1], DTD_median, 'k-')
        plt.fill_between(feh[:-1], DTD_lo, DTD_hi, color='gray', alpha=0.5)
        plt.xlabel('[Fe/H]', fontsize=16)
        plt.ylabel('DTD', fontsize=16)
        plt.ylim(-0.005,0.2)
        plt.savefig('figures/noageZ_DTD.png', bbox_inches='tight')
        plt.show()

        # Try plotting rates?
        plt.fill_between(feh[:-1], np.percentile(obs_rates, 16, axis=0), np.percentile(obs_rates, 84, axis=0), color=plt.cm.Pastel2(0), alpha=0.5)
        plt.plot(feh[:-1], np.percentile(obs_rates, 50, axis=0), linestyle='-', color=plt.cm.Dark2(0), label='Observed rate')
        plt.fill_between(feh[:-1], np.percentile(exp_rates, 16, axis=0), np.percentile(exp_rates, 84, axis=0), color=plt.cm.Pastel2(1), alpha=0.5)
        plt.plot(feh[:-1], np.percentile(exp_rates, 50, axis=0), linestyle='-', color=plt.cm.Dark2(1), label='Expected rate')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.xlabel('[Fe/H]', fontsize=16)
        plt.ylabel(r'Type Ia rate $(\mathrm{Gyr}^{-1})$', fontsize=16)
        plt.savefig('figures/noageZ_rates.png', bbox_inches='tight')
        plt.show()

        # Make test plot of SFH?
        plt.fill_between(feh[:-1], np.percentile(sfh_array, 16, axis=0), np.percentile(sfh_array, 84, axis=0), color='gray', alpha=0.5)
        plt.plot(feh[:-1], np.percentile(sfh_array, 50, axis=0), 'k-')
        #plt.legend(loc='best')
        plt.xlabel('[Fe/H]', fontsize=16)
        plt.ylabel(r'SFR', fontsize=16)
        plt.savefig('figures/noageZ_sfh.png', bbox_inches='tight')
        plt.show()

        # Plot residuals between rates?
        percenterrors = (obs_rates - exp_rates)/exp_rates * 100.
        plt.plot(feh[:-1], np.percentile(percenterrors, 50, axis=0), 'k-')
        #plt.fill_between(feh[:-1], np.percentile(percenterrors, 16, axis=0), np.percentile(percenterrors, 84, axis=0), color='gray', alpha=0.5)
        plt.xlabel('[Fe/H]', fontsize=16)
        plt.ylabel(r'Percent error in Type Ia rates (\%)', fontsize=16)
        plt.ylim(-300,300)
        plt.savefig('figures/noageZ_rates_percenterrors.png', bbox_inches='tight')
        plt.show()

    return DTD_median, DTD_lo, DTD_hi

def convertDTD(DTD, dtd_lo=None, dtd_hi=None, plotageZ=False):
    """Convert DTD from function of [Fe/H] to function of time."""

    # Set limits for computing the age-Z relation
    fehlim = (-2,-1.5)

    # Get age-metallicity relation from my GCE model
    amr_delosreyes = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/plots/amr_test.npy')
    feh_dlr = amr_delosreyes[:,0]  # [Fe/H]
    age_dlr = 13.791 - amr_delosreyes[:,1]  # Gyr
    goodidx = np.where(np.isfinite(feh_dlr))[0]
    feh_dlr = feh_dlr[goodidx]
    age_dlr = age_dlr[goodidx]

    # Get age-metallicity relation from de Boer et al. (2012)
    amr_deBoer = np.loadtxt('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/data/sfhtest/deboer12-ageZ.dat', delimiter=',')
    age_deBoer_idx = np.argsort(amr_deBoer[:,1])  # Sort by metallicity
    age_deBoer = amr_deBoer[age_deBoer_idx,0]  # Gyr
    feh_deBoer = amr_deBoer[age_deBoer_idx,1]  # [Fe/H]

    ages = [age_dlr, age_deBoer]
    fehs = [feh_dlr, feh_deBoer]
    amrs = ['GCE model', 'de Boer et al. (2012)']

    # Prep plot
    if plotageZ==False:
        plt.figure(figsize=(8,6))

    # Loop over all age-Z relations
    for i, amr in enumerate(amrs):

        # Linearly extrapolate to [Fe/H] ~ 0.
        feh0 = 0.0
        age0 = ages[i][-1]+(feh0-fehs[i][-1])*(ages[i][-1]-ages[i][-2])/(fehs[i][-1]-fehs[i][-2])
        fehs[i] = np.concatenate((fehs[i],[feh0]))
        ages[i] = np.concatenate((ages[i],[age0]))
        
        # Fit polynomial to age-Z relation
        fehlimidx = np.where((fehs[i] > fehlim[0]) & (fehs[i] < fehlim[1]))[0]
        p = np.polyfit(fehs[i][fehlimidx], ages[i][fehlimidx], 1)
        #print(p)

        if plotageZ:
            plt.plot(fehs[i][fehlimidx], ages[i][fehlimidx], 'ko', label='Observed AMR')
            plt.plot(fehs[i][fehlimidx], np.polyval(p, fehs[i][fehlimidx]), 'r-', label='Best-fit line')
            plt.legend(title=amr, fontsize=12, title_fontsize=14)
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel('Time (Gyr)', fontsize=16)
            plt.savefig('figures/amr_'+amr+'.png', bbox_inches='tight')
            plt.show()

        # Convert from metallicity to age
        print(p)
        age = p[0] * feh[:-1] # * np.polyval(p, feh[fehidx]) # 
        #print(feh[:-1], age)

        # Plot resulting DTD
        if plotageZ==False:
            #plt.ylim(1e-4,1.5)
            plt.plot(age, DTD, linestyle='-', color=plt.cm.Dark2(i), label=amr)
            if dtd_lo is not None and dtd_hi is not None:
                plt.fill_between(age, dtd_lo, dtd_hi, color=plt.cm.Pastel2(i))
            #print(age, DTD)

            if amr=='GCE model':
                plt.plot(age, (DTD[-1]/(age[-1]**-1.1))*age**(-1.1), 'r--')

    # Finish plot
    if plotageZ==False:

        # Plot formatting
        #plt.xlabel('Time (Gyr)', fontsize=16)
        plt.ylim(1e-5,1)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Log(t) [Gyr]', fontsize=16)
        plt.ylabel(r'DTD ($10^{-3}\mathrm{Gyr}~M_{\odot}^{-1}$)', fontsize=16)
        plt.legend(loc='best', title='Age-Z relations', fontsize=12, title_fontsize=14)
        plt.savefig('figures/noageZ_DTD_testageZ.png', bbox_inches='tight')
        plt.show()

    return

if __name__ == "__main__":
    dtd, dtdlo, dtdhi = fitdtd(Niter=100, plot=True)
    convertDTD(dtd, dtd_lo=dtdlo, dtd_hi=dtdhi, plotageZ=False)