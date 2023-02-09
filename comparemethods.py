# Test code to compare two methods:
# Approach 1: Assume DTD, convert everything to time using AMR, then compare Ia rates
# Approach 2: Find best-fit DMD, then convert to DTD

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
from scipy.io.idl import readsav
from astropy.table import Table
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm
from scipy.interpolate import interp1d
import sys
sys.path.append('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/dtd')
from amr import amr

# Some constants
MScl = 1.2e6  # Stellar mass of Sculptor (Msun)
from ccfrac import CCfrac  # Fraction of CCSNe

# Define an age-Z relation
fehlim = (-2,-1.5)
age, feh, feh2age_conversion, fehlimidx = amr('Scl', gcetest=True, plot=False)

def feh2age(a):
    return np.polyval(feh2age_conversion,a)

def ecdf(a):
    '''Function to compute empirical CDF.'''
    x, counts = np.unique(a, return_counts=True)
    cusum = np.cumsum(counts)
    return x, cusum / cusum[-1] * MScl #* CCfrac

# First let's start with Approach 1:
def approach1(index, norm, mindelay):
    '''Convolve DTD with SFH (from MDF converted to time) -> get Ia rate as function of time.'''

    # Prep some important packages
    from numpy.random import default_rng
    rng = default_rng()

    # Get MDF from Kirby+09
    kirby09 = Table.read('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/dtd/data/mdf/kirby_scl_mdf.txt', format='ascii.cds')
    scl_FeH = kirby09['[Fe/H]']
    scl_FeH_err = kirby09['e_[Fe/H]']

    # Remove stars with large error
    goodidx = np.where(scl_FeH_err < 0.3)
    scl_FeH = scl_FeH[goodidx]
    scl_FeH_err = scl_FeH_err[goodidx]

    # Perturb each star by given error to get new realization        
    # Assume each star has [Fe/H] distributed normally (mean = given [Fe/H], sigma = error in [Fe/H])
    perturb = np.random.normal(scl_FeH, scl_FeH_err)

    # Compute empirical cumulative MDF 
    x, y = ecdf(scl_FeH)
    #x, y = ecdf(perturb)
    ecdf_feh_old = np.insert(x, 0, x[0])
    ecdf_N = np.insert(y, 0, 0.) 

    # Interpolate to fill in the zeros in the CDF
    ecdf_feh = np.linspace(-2.97,0, num=100) #np.arange(-3.8 ,-0.85, step=0.01)
    sfh_cumulative = np.interp(ecdf_feh, ecdf_feh_old, ecdf_N)

    # Get ages (in Gyr)
    ia_age = 13.791 - feh2age(ecdf_feh)
    ia_agebin = np.diff(ia_age)[0]  # Age bin edges (Gyr)
    ia_fehbin = np.diff(ecdf_feh)[0]  # Metallicity bin edges (dex)

    # Define "reasonable" age range
    #agelim = (13.791-feh2age(fehlim[0]), 13.791-feh2age(fehlim[1]))
    agelim = (0,0.7)
    crop_idx = np.where((ia_age > agelim[0]) & (ia_age < agelim[1]))[0]    
    age_crop = ia_age[crop_idx]

    # Get SFH from MDF
    sfh = np.diff(sfh_cumulative)  # Mass of stars formed in each bin
    scaled_sfh = (sfh/ia_agebin)[crop_idx]  # units: Msun/Gyr
    print(ia_agebin/ia_fehbin, feh2age_conversion[0])

    # Compute Ia rate by convolving model DTD with SFH
    def testdtd(index, norm, mindelay):
        DTD = (norm/1000.) * age_crop**index
        DTD[np.where(age_crop < mindelay)[0]] = 0.  # units: SNe/Gyr/Msun

        # Convolve DTD with SFH
        test = np.convolve(DTD, scaled_sfh) * ia_agebin  # units: (SNe/Gyr/Msun) * (Msun/Gyr) * Gyr = SNe/Gyr
        test[~np.isfinite(test)] = 0.
        test = test[:len(age_crop)]
        
        return test

    return age_crop, testdtd(index, norm, mindelay) #[uniform_idx]

def approach2(index, norm, mindelay):
    '''Convert DTD to DMD, convolve with MDF -> get Ia rate as function of metallicity, then convert back.'''

    # Get MDF from Kirby+09
    kirby09 = Table.read('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/dtd/data/mdf/kirby_scl_mdf.txt', format='ascii.cds')
    scl_FeH = kirby09['[Fe/H]']
    scl_FeH_err = kirby09['e_[Fe/H]']

    # Remove stars with large error
    goodidx = np.where(scl_FeH_err < 0.3)
    scl_FeH = scl_FeH[goodidx]
    scl_FeH_err = scl_FeH_err[goodidx]

    # Perturb each star by given error to get new realization        
    # Assume each star has [Fe/H] distributed normally (mean = given [Fe/H], sigma = error in [Fe/H])
    perturb = np.random.normal(scl_FeH, scl_FeH_err)

    # Compute empirical cumulative MDF and convert to cumulative # of CCSNe
    x, y = ecdf(scl_FeH)
    #x, y = ecdf(perturb)
    ecdf_feh_old = np.insert(x, 0, x[0])
    ecdf_N = np.insert(y, 0, 0.) 

    # Get SFH as function of [Fe/H]
    ecdf_feh = np.linspace(-2.97,0, num=100) #np.arange(-3.8 ,-0.2, step=0.01)
    N_CC_cumulative = np.interp(ecdf_feh, ecdf_feh_old, ecdf_N)

    # Get ages (in Gyr)
    ia_age = 13.791 - feh2age(ecdf_feh)
    ia_agebin = np.diff(ia_age)[0]  # Age bin edges (Gyr)
    ia_fehbin = np.diff(ecdf_feh)[0]  # Metallicity bin edges (dex)

    # Define reasonable age range
    agelim = (0,0.7)
    crop_idx = np.where((ia_age > agelim[0]) & (ia_age < agelim[1]))[0]    
    age_crop = ia_age[crop_idx]

    # Get SFH
    sfh = np.diff(N_CC_cumulative) # Mass of stars formed at each bin
    sfh = sfh[crop_idx]

    # Compute Ia rate by convolving model DMD with SFH
    def testdtd(index, norm, mindelay):

        DTD = (norm/1000.) * age_crop**index
        DTD[np.where(age_crop < mindelay)[0]] = 0.
        DTD = DTD[:-1]  # units: SNe/Gyr/Msun

        # Convert DTD to DMD using AMR
        DMD = DTD * ia_agebin/ia_fehbin  # units: (SNe/Gyr/Msun) * Gyr / dex = SNe/Msun/dex

        #plt.plot(ecdf_feh[crop_idx][:-1], DMD)
        #plt.plot((age_crop/feh2age_conversion[0])[:-1], DMD)
        #plt.show()

        # Convolve DMD with SFH to get Ia rate as function of metallicity
        test = np.convolve(DMD, sfh) * ia_fehbin  # units: (SNe/Msun/dex) * Msun * dex = SNe
        test[~np.isfinite(test)] = 0.
        test = test[:len(age_crop)]

        # Convert this Type Ia rate to a function of time, not metallicity
        test = test/ia_agebin  # units: SNe / Gyr = SNe/Gyr

        return test

    #plt.plot(ecdf_feh, testdtd(-1.1,1,0.1, age), 'k-')
    #plt.show()

    #plt.plot(ecdf_feh, testdtd(-1.1,1,0.1, age))
    #plt.show()

    # Final age
    #age_final = 13.791-feh2age(ecdf_feh)
    #plt.plot(age_final, testdtd(-1.1,1,0.1, age))
    #plt.show()

    # Define uniform age array
    #agelim = (13.791-feh2age(fehlim[0]), 13.791-feh2age(fehlim[1]))
    #uniform_idx = np.where((age_final > agelim[0]) & (age_final < agelim[1]))[0]    
    #age_crop = age_final[uniform_idx]

    return age_crop, testdtd(index, norm, mindelay)

def comparemethods():
    # Prep plot
    plt.figure(figsize=(8,6))

    # Approach 1
    age1, iarate1 = approach1(-1.1, 1, 0.1)
    plt.plot(age1, iarate1/1e3, color='k', ls='-', lw=1, label='Approach 1: apply AMR before')

    # Approach 2
    age2, iarate2 = approach2(-1.1, 1, 0.1)
    plt.plot(age2, iarate2/1e3, color='r', ls='--', lw=1, label='Approach 2: apply AMR after')

    # Plot formatting
    plt.xlabel('Time (Gyr)')
    plt.ylabel('Type Ia rate~$(10^{3}~\mathrm{Gyr}^{-1})$', fontsize=20)
    plt.legend()

    plt.show()

    return

if __name__=="__main__":
    comparemethods()
    #approach1(-1.1, 1, 0.1)
    #approach2(-1.1, 1, 0.1)