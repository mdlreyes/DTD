"""
DTD_fit_binned.py

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
from scipy.io.idl import readsav
from astropy.table import Table
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from tqdm import tqdm
from ccfrac import CCfrac
from amr import amr

'''
# Test AMR
amr_delosreyes = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/output/amr_test.npy')
feh_dlr = amr_delosreyes[:,0]  # [Fe/H]
age_dlr = amr_delosreyes[:,1]  # Gyr
goodidx = np.where(np.isfinite(feh_dlr))[0]
feh_dlr = feh_dlr[goodidx]
age_dlr = age_dlr[goodidx]

# Extrapolate to [Fe/H] ~ 0.
feh0 = 0.0
age0 = age_dlr[0]+(feh0-feh_dlr[0])*(age_dlr[1]-age_dlr[0])/(feh_dlr[1]-feh_dlr[0])
age0 = age_dlr[-1]+(feh0-feh_dlr[-1])*(age_dlr[-1]-age_dlr[-2])/(feh_dlr[-1]-feh_dlr[-2])
feh_dlr = np.concatenate((feh_dlr,[feh0]))
age_dlr = np.concatenate((age_dlr,[age0]))
age_feh_conversion = interp1d(age_dlr, feh_dlr)

# Get Ia rate from GCE
iarate_data = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/output/iarate_test.npy')
iarate = iarate_data[0][10:]/1e3  # per Gyr
t = iarate_data[1][10:]  # Gyr
# Convert to [Fe/H] scale
gcefeh = age_feh_conversion(t)  # dex
iarate = iarate[:-1] * np.diff(t)/np.diff(gcefeh)  # units: SNe/Gyr * Gyr/dex = SNe/dex
gcefeh = gcefeh[:-1]
'''

def feh2age(a, feh2age_conversion):
    return np.polyval(feh2age_conversion,a)

def expandDTD(DTD, length, Nbins):
    nreps = int((length+1)/Nbins)
    newDTD = np.asarray([[DTD[i]] * nreps for i in range(len(DTD))])
    return newDTD.flatten()[:-1]

# Class to wrap each galaxy
class galaxy:
    def __init__(self, name, Mstar, bins):

        self.name = name  # ID string for galaxy
        self.Mstar = Mstar  # Stellar mass (Msun)
        self.bins = bins  # Number of bins

        # Define an age-Z relation
        _, _, self.feh2age_conversion, _ = amr(self.name, gcetest=False)

        # Get Fe_Ia/Fe_CC info from Kirby+19
        if name=='Scl':
            rfile = fits.open('data/rfrac/chemev_scl.fits')
            self.R = rfile[1].data['R'].T[:,0]
            self.R_feh = np.linspace(-2.97,0, num=100)
            self.Rerrlo = rfile[1].data['RERRL'].T[:,0]
            self.Rerrhi = rfile[1].data['RERRH'].T[:,0]
        else:
            data = readsav('data/rfrac/sn_decomposition_'+self.name.lower()+'.sav')
            self.R = data.trend.R[0]
            self.R_feh = data.trendfeh

        # Define metallicity and age ranges
        self.feh0 = -3.2
        self.ecdf_feh = np.linspace(self.feh0,-0.5, num=120) #np.arange(-4, -0.2, step=0.01) #self.R_feh # 
        self.ia_age = 13.791 - feh2age(self.ecdf_feh, self.feh2age_conversion)
        self.age0 = 13.791 - feh2age(self.feh0, self.feh2age_conversion)
        self.ia_fehbin = np.diff(self.ecdf_feh)[0]  # Metallicity bin edges (dex)
        self.ia_agebin = np.diff(self.ia_age)[0]  # Age bin edges (Gyr)

        # Define reasonable age range
        agelim = (0,0.7)
        self.crop_idx = np.where((self.ia_age > agelim[0]) & (self.ia_age < agelim[1]))[0]    

        # Get MDF data from Kirby+09
        '''
        if name=='Scl':
            kirby09 = Table.read('data/mdf/kirby_scl_mdf.txt', format='ascii.cds')
            self.mdf_feh = kirby09['[Fe/H]']
            self.mdf_feh_err = kirby09['e_[Fe/H]']
        '''
        kirby09_new = Table.read('data/mdf/kirby_dsph_catalog.dat', format='ascii.cds')
        idx = np.where(kirby09_new['dSph']==name)
        self.mdf_feh = kirby09_new['eps(Fe)'][idx] - 7.50
        self.mdf_feh_err = kirby09_new['e_eps(Fe)'][idx]

        # Remove stars with large error
        goodidx = np.where(self.mdf_feh_err < 0.3)
        self.mdf_feh = self.mdf_feh[goodidx]
        self.mdf_feh_err = self.mdf_feh_err[goodidx]

    def ecdf(self, a):
        """Function to compute empirical cumulative sum, scaled to stellar mass of galaxy."""
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1] * self.Mstar #* CCfrac

    def fitdtd(self, Niter, plot=False):
        """Fit DTD for a galaxy using MC-like method 
        (perturbing MDF, SFH to get realizations of Type Ia rates, 
        and fitting those realizations)."""

        # Let's do MC estimation of errors
        exp_rates = np.zeros((Niter,len(self.ecdf_feh))) #[fehidx])))
        obs_rates = np.zeros((Niter,len(self.ecdf_feh))) #[fehidx])))
        DTD_array = np.zeros((Niter, self.bins))
        sfh_array = np.zeros((Niter, len(self.ecdf_feh))) #[fehidx])))

        # List of bad iterations to throw out
        baditer = []

        for iteration in tqdm(range(Niter)):

            # Get single realization of R
            #percentile = np.random.uniform()
            #new_r = np.zeros(len(self.R))
            #for i in range(len(self.R)):
            #    if percentile > 0.5:
            #        scale=self.Rerrhi[i]
            #    else:
            #        scale=self.Rerrlo[i]
            #    new_r[i] = norm(loc=self.R[i],scale=scale).ppf(percentile)

            # Convert R to cumulative fraction of Ia/CC SNe
            fe_CC = 0.074  # Core-collapse yield from Maoz&Graur17
            fe_Ia = np.sum([7.80e-3,6.10e-1,2.12e-2,4.39e-4])  # Type Ia yields from Leung+19 (1.1 Msun, solar Z=0.02)
            N_Ia_CC = self.R/fe_Ia * fe_CC
            N_Ia_CC = np.interp(self.ecdf_feh, self.R_feh, N_Ia_CC)
            
            # Get single realization of MDF
            perturb = np.random.normal(self.mdf_feh, self.mdf_feh_err)
            
            # Compute empirical cumulative MDF and convert to cumulative # of stars formed per [Fe/H] bin
            x, y = self.ecdf(perturb)
            ecdf_feh_old = np.insert(x, 0, x[0])
            ecdf_N = np.insert(y, 0, 0.) 

            # Get (cumulative) SFH and N(CCSNe) as function of [Fe/H]
            sfh_cumulative = np.interp(self.ecdf_feh, ecdf_feh_old, ecdf_N)  # Total mass of stars formed per bin
            N_CC_cumulative = sfh_cumulative*CCfrac

            # Compute rate of IaSNe as function of [Fe/H]
            N_Ia_cumulative = N_CC_cumulative * N_Ia_CC  
            N_Ia = np.diff(N_Ia_cumulative)  # SNe in each bin (units: SNe/delta[Z])
            N_Ia = np.insert(N_Ia, 0, 0)
            N_Ia[~np.isfinite(N_Ia)] = 0.
            N_Ia[N_Ia < 0.] = 0.

            # Save indices where rate of IaSNe is zero
            noIa_idx = np.where(np.isclose(N_Ia, 0.))[0]
            #noIa_idx = np.where(self.ecdf_feh < -2.1)[0]

            # Save SFH (for use in test plotting later)
            sfh = np.diff(sfh_cumulative)  # Mass of stars formed in each bin (units: Msun/delta[Z])
            sfh = np.insert(sfh,0,0)
            sfh_array[iteration, :] = sfh

            # Compute likelihood
            def log_likelihood(params):

                # Compute DMD by expanding parameters into full DMD
                DMD = expandDTD(params, len(sfh), self.bins)  # DMD units: SNe/Msun/delta[Z]

                # Convolve DMD with SFH
                test = np.convolve(DMD, sfh) * self.ia_fehbin  # units: DMD (SNe/Msun/delta[Z]) * SFH (Msun/delta[Z]) * delta[Z] = SNe/delta[Z]
                test[~np.isfinite(test)] = 0.
                N_Ia_exp = test[:len(N_Ia)]    

                # Convert this Type Ia rate to a function of time, not metallicity
                #N_Ia_exp = N_Ia_exp/self.ia_agebin

                # Compute log likelihood within [Fe/H] limits
                resid = (N_Ia - N_Ia_exp) #[self.crop_idx]
                likelihood = -0.5 * np.sum(np.power(resid,2.))
                return likelihood, N_Ia_exp

            def log_probability(params):

                likelihood, N_Ia_exp = log_likelihood(params)

                # Add in prior (make sure DTD and N_Ia is always positive)
                if np.any(params < 0.) or np.any(N_Ia_exp[noIa_idx] > 0.):
                    return -np.inf
                else:
                    return likelihood

            # Maximize likelihood (minimize negative likelihood)
            nll = lambda *args: -log_probability(*args)
            initial = np.zeros(self.bins)
            soln = minimize(nll, initial, method='powell') #, options={'ftol':1e-6, 'maxiter':100000})#, 'direc':np.diag([-0.01, 0.01, 0.01])})

            # Get DMD
            finalsoln = soln.x

            # Try cutting out later parts of DMD
            fehidx = np.where((self.ecdf_feh - self.feh0) > 2.0)[0]
            maxint = int((fehidx[0]+1)/(len(self.ecdf_feh)/self.bins))
            finalsoln[maxint:] = 0.

            # Store solution in array
            DTD_array[iteration, :] = finalsoln

            # Store observed and expected rates of IaSNe in arrays
            DTD = expandDTD(finalsoln, len(sfh), self.bins)

            exprate = np.convolve(DTD, sfh) * self.ia_fehbin
            exprate[~np.isfinite(exprate)] = 0.

            # Convert to 10^3 SNe/Gyr
            exp_rates[iteration, :] = exprate[:len(N_Ia)]/self.ia_fehbin/1e3
            obs_rates[iteration, :] = N_Ia/self.ia_fehbin/1e3

        # Compute percentiles
        print(np.percentile(DTD_array, 50, axis=0))
        DTD_median = expandDTD(np.percentile(DTD_array, 50, axis=0), len(sfh), self.bins)
        DTD_lo = expandDTD(np.percentile(DTD_array, 16, axis=0), len(sfh), self.bins)
        DTD_hi = expandDTD(np.percentile(DTD_array, 84, axis=0), len(sfh), self.bins)

        if plot:

            # Make plot
            plt.plot(self.ecdf_feh[:-1]-self.feh0, DTD_median, 'k-')
            plt.fill_between(self.ecdf_feh[:-1]-self.feh0, DTD_lo, DTD_hi, color='gray', alpha=0.5)
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel('DTD', fontsize=16)
            plt.ylim(-0.005,0.2)
            plt.savefig('figures/'+self.name+'_binned_DTD.png', bbox_inches='tight')
            plt.show()

            # Try plotting rates?
            #plt.plot(gcefeh, iarate, 'r--', label='GCE')  # Test

            plt.fill_between(self.ecdf_feh, np.percentile(obs_rates, 16, axis=0), np.percentile(obs_rates, 84, axis=0), color=plt.cm.Pastel2(0), alpha=0.5)
            plt.plot(self.ecdf_feh, np.percentile(obs_rates, 50, axis=0), linestyle='-', color=plt.cm.Dark2(0), label='Observed rate')
            plt.fill_between(self.ecdf_feh, np.percentile(exp_rates, 16, axis=0), np.percentile(exp_rates, 84, axis=0), color=plt.cm.Pastel2(1), alpha=0.5)
            plt.plot(self.ecdf_feh, np.percentile(exp_rates, 50, axis=0), linestyle='-', color=plt.cm.Dark2(1), label='Expected rate')
            #plt.axvspan(self.ecdf_feh[self.crop_idx][0], self.ecdf_feh[self.crop_idx][-1], color='gray', alpha=0.2)
            #plt.axvspan(self.ecdf_feh[noIa_idx][0], self.ecdf_feh[noIa_idx][-1], color='blue', alpha=0.2)
            plt.legend(loc='best')
            plt.yscale('log')
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel(r'Type Ia rate $(\mathrm{Gyr}^{-1})$', fontsize=16)
            plt.savefig('figures/'+self.name+'_binned_rates.png', bbox_inches='tight')
            plt.show()

            # Make test plot of SFH?
            plt.fill_between(self.ecdf_feh, np.percentile(sfh_array, 16, axis=0), np.percentile(sfh_array, 84, axis=0), color='gray', alpha=0.5)
            plt.plot(self.ecdf_feh, np.percentile(sfh_array, 50, axis=0), 'k-')
            #plt.legend(loc='best')
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel(r'SFR', fontsize=16)
            plt.savefig('figures/'+self.name+'_binned_sfh.png', bbox_inches='tight')
            plt.show()

            # Plot residuals between rates?
            percenterrors = (obs_rates - exp_rates)/exp_rates * 100.
            plt.plot(self.ecdf_feh, np.percentile(percenterrors, 50, axis=0), 'k-')
            #plt.fill_between(feh[:-1], np.percentile(percenterrors, 16, axis=0), np.percentile(percenterrors, 84, axis=0), color='gray', alpha=0.5)
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel(r'Percent error in Type Ia rates (\%)', fontsize=16)
            plt.ylim(-300,300)
            plt.savefig('figures/'+self.name+'_binned_rates_percenterrors.png', bbox_inches='tight')
            plt.show()

        return DTD_median, DTD_lo, DTD_hi

    def convertDTD(self, Niter, testplot=False):
        """Convert DTD from function of [Fe/H] to function of time."""

        # Get DTD from fitting
        DMD, dmd_lo, dmd_hi = self.fitdtd(Niter, plot=testplot) 

        # Prep plot
        plt.figure(figsize=(8,6))

        # Convert from metallicity to age
        print(self.ia_agebin/self.ia_fehbin, self.feh2age_conversion)
        slope = self.ia_agebin/self.ia_fehbin
        DTD = DMD * slope  # units: (SNe/Msun/bin) * delta[Z]/Gyr = SNe/Msun/Gyr
        dtd_lo = dmd_lo * slope
        dtd_hi = dmd_hi * slope

        # Plot resulting DTD
        plt.plot(self.ia_age[:-1]-self.age0, DTD, linestyle='-', color=plt.cm.Dark2(0))
        if dtd_lo is not None and dtd_hi is not None:
            plt.fill_between(self.ia_age[:-1] - self.age0, dtd_lo, dtd_hi, color=plt.cm.Pastel2(0))

        # Plot t^-1 line
        testage = np.linspace(0, self.ia_age[-1], 100)
        t_ia = 1e-1 #Gyr
        rate = (1e-3)*testage**(-1.1)
        w = np.where(testage <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        plt.plot(testage, rate, 'r--')

        # Plot formatting
        #plt.xlabel('Time (Gyr)', fontsize=16)
        #plt.ylim(1e-6,1)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.xlabel('Time (Gyr)', fontsize=16)
        plt.ylabel(r'DTD ($\mathrm{Gyr}^{-1}~M_{\odot}^{-1}$)', fontsize=16)
        #plt.legend(loc='best', title='Age-Z relations', fontsize=12, title_fontsize=14)
        plt.savefig('figures/'+self.name+'_binned_DTD_final.png', bbox_inches='tight')
        plt.show()

        return

def plotgalaxies(keyword, Nbins, Niter):
    """Plots DTDs from several galaxies."""

    galaxynames = ['Scl','Dra','LeoII'] #,'Sex','UMi']
    logMstar = np.asarray([6.08,5.96,6.16]) #,5.93,5.75])  # Stellar masses from Woo et al. (2008)
    galaxyfullnames = ['Sculptor','Draco','Leo II']

    # Prepare plot
    plt.figure(figsize=(8,6))

    handles = []
    labels = []

    for i, galaxyname in enumerate(galaxynames):

        # Instantiate galaxy object
        galaxyobj = galaxy(galaxyname, Mstar=10**logMstar[i], bins=Nbins)

        # Get DMD
        DMD, dmd_lo, dmd_hi = galaxyobj.fitdtd(Niter=Niter)

        if keyword=='dmd':
            # Plot DMD
            p1, = plt.plot(galaxyobj.ecdf_feh[:-1]-galaxyobj.feh0, DMD, linestyle='-', color=plt.cm.Dark2(i))
            #p2, = plt.plot(galaxyobj.ecdf_feh[:-1]-galaxyobj.feh0, dmd_lo, linestyle='-', color=plt.cm.Pastel2(i))
            #plt.plot(galaxyobj.ecdf_feh[:-1]-galaxyobj.feh0, dmd_hi, linestyle='-', color=plt.cm.Pastel2(i))
            p2 = plt.fill_between(galaxyobj.ecdf_feh[:-1]-galaxyobj.feh0, dmd_lo, dmd_hi, alpha=0.5, color=plt.cm.Pastel2(i))
            handles.append((p1,p2))
            labels.append(galaxyfullnames[i])
        
        elif keyword=='dtd':
            # Convert DMD to DTD
            slope = galaxyobj.ia_agebin/galaxyobj.ia_fehbin
            DTD = DMD * slope  # units: (SNe/Msun/bin) * delta[Z]/Gyr = SNe/Msun/Gyr
            dtd_lo = dmd_lo * slope
            dtd_hi = dmd_hi * slope

            # Plot resulting DTD
            p1, = plt.plot(galaxyobj.ia_age[:-1]-galaxyobj.age0, DTD, linestyle='-', color=plt.cm.Dark2(i))
            p2 = plt.fill_between(galaxyobj.ia_age[:-1]-galaxyobj.age0, dtd_lo, dtd_hi, alpha=0.5, color=plt.cm.Pastel2(i))
            handles.append((p1,p2))
            labels.append(galaxyfullnames[i])

    if keyword=='dmd':
        plt.axvline(2.0, color='r', linestyle='--')
        plt.xlabel(r'Delay metallicity $\zeta$', fontsize=16)
        plt.ylabel(r'$\Psi$ ($M_{\odot}^{-1}~\mathrm{dex}^{-1}$)', fontsize=16)
        plt.ylim(-0.005,0.2)
        outputname = 'noageZ_DMD.png'

    elif keyword=='dtd':
        # Plot t^-1 line
        testage = np.linspace(0, galaxyobj.ia_age[-1], 100)
        t_ia = 1e-1 #Gyr
        rate = (1e-3)*testage**(-1.1)
        w = np.where(testage <= t_ia)[0]
        if len(w) > 0: rate[w] = 0.0
        plt.plot(testage, rate, 'r--')

        plt.xlabel('Time (Gyr)', fontsize=16)
        plt.ylabel(r'$\Psi(\tau)$ (M_{\odot}^{-1}~\mathrm{Gyr}^{-1}$)', fontsize=16)
        outputname = 'noageZ_DTD.png'

    # Additional plot formatting
    plt.legend(handles=handles, labels=labels, loc='best', fontsize=12)
    plt.savefig('figures/'+outputname, bbox_inches='tight')
    plt.show()

    return

if __name__ == "__main__":
    # DMD calculation
    #galaxy('Scl', Mstar=10**6.08, bins=40).fitdtd(20, plot=True) #
    #galaxy('Dra', Mstar=10**5.96, bins=40).fitdtd(30, plot=True)
    #galaxy('LeoII', Mstar=10**6.16, bins=40).fitdtd(20, plot=True)
    #galaxy('Sex', Mstar=10**5.93, bins=40).fitdtd(20, plot=True)

    # DTD conversion
    #galaxy('Scl', Mstar=1.2e6, bins=40).convertDTD(Niter=20, testplot=False)  # Sculptor dSph
    #galaxy('Dra', Mstar=10**5.96, bins=40).convertDTD(Niter=20, testplot=True)  # Draco dSph

    # Plotting multiple galaxies
    plotgalaxies('dmd', Nbins=40, Niter=20)