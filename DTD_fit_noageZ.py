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
from scipy.io.idl import readsav
from astropy.table import Table
from scipy.stats import norm
from scipy.optimize import minimize
from tqdm import tqdm
from ccfrac import CCfrac

# Class to wrap each galaxy
class galaxy:
    def __init__(self, name, Mstar, bins):

        self.Mstar = Mstar  # Stellar mass (Msun)
        self.bins = bins  # Number of bins

        # Get Fe_Ia/Fe_CC info from Kirby+19
        if name=='scl':
            rfile = fits.open('data/rfrac/chemev_scl.fits')
            self.R = rfile[1].data['R'].T[:,0]
            self.Rerrlo = rfile[1].data['RERRL'].T[:,0]
            self.Rerrhi = rfile[1].data['RERRH'].T[:,0]
            self.feh = np.linspace(-2.97,0, num=100)
        else:
            data = readsav('data/rfrac/sn_decomposition_leoii.sav')
            self.R = data.data.R
            self.Rerrlo = data.data.RERR
            self.Rerrhi = data.data.RERR
            self.feh = data.data.FEH

        # Get MDF data from Kirby+09
        if name=='scl':
            kirby09 = Table.read('data/mdf/kirby_scl_mdf.txt', format='ascii.cds')
            self.mdf_feh = kirby09['[Fe/H]']
            self.mdf_feh_err = kirby09['e_[Fe/H]']
        else:
            # FINISH THIS
            pass

    def expandDTD(self, DTD, length, Nbins):
        nreps = int((length+1)/Nbins)
        newDTD = np.asarray([[DTD[i]] * nreps for i in range(len(DTD))])
        return newDTD.flatten()[:-1]

    def fitdtd(self, Niter, plot=False):
        """Fit DTD for a galaxy using MC-like method 
        (perturbing MDF, SFH to get realizations of Type Ia rates, 
        and fitting those realizations)."""

        # Let's do MC estimation of errors
        exp_rates = np.zeros((Niter,len(self.feh)-1)) #[fehidx])))
        obs_rates = np.zeros((Niter,len(self.feh)-1)) #[fehidx])))
        DTD_array = np.zeros((Niter, self.bins))
        sfh_array = np.zeros((Niter, len(self.feh)-1)) #[fehidx])))

        for iteration in tqdm(range(Niter)):

            # Get single realization of R
            percentile = np.random.uniform()
            new_r = np.zeros(len(self.R))
            for i in range(len(self.R)):
                if percentile > 0.5:
                    scale=self.Rerrhi[i]
                else:
                    scale=self.Rerrlo[i]
                new_r[i] = norm(loc=self.R[i],scale=scale).ppf(percentile)

            # Convert R to cumulative fraction of Ia/CC SNe
            fe_CC = 0.074  # Core-collapse yield from Maoz&Graur17
            fe_Ia = np.sum([7.80e-3,6.10e-1,2.12e-2,4.39e-4])  # Type Ia yields from Leung+19 (1.1 Msun, solar Z=0.02)
            N_Ia_CC = new_r/fe_Ia * fe_CC
            
            # Get single realization of MDF
            perturb = np.random.normal(self.mdf_feh, self.mdf_feh_err)
            mdf = np.histogram(perturb, bins=self.feh)

            # Normalize MDF -> SFH of Sculptor
            sfh = mdf[0]/np.nansum(mdf[0]) * self.Mstar
            
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

            # Save SFH (for use in test plotting later)
            sfh_array[iteration, :] = sfh

            # Compute likelihood
            def log_likelihood(params):

                # Compute DTD by expanding parameters into full DTD
                DTD = self.expandDTD(params, len(sfh), self.bins)

                # Convolve DTD with SFH
                test = np.convolve(DTD, sfh)
                test[~np.isfinite(test)] = 0.
                N_Ia_exp = test[:len(N_Ia)]    

                # Compute log likelihood within [Fe/H] limits
                resid = N_Ia - N_Ia_exp
                likelihood = -0.5 * np.sum(np.power(resid,2.))
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
            initial = np.zeros(self.bins)
            soln = minimize(nll, initial, method='powell', options={'ftol':1e-6, 'maxiter':100000})#, 'direc':np.diag([-0.01, 0.01, 0.01])})

            # Store solution in array
            DTD_array[iteration, :] = soln.x

            # Store observed and expected rates of IaSNe in arrays
            DTD = self.expandDTD(soln.x, len(sfh), self.bins)
            exprate = np.convolve(DTD, sfh)
            exprate[~np.isfinite(exprate)] = 0.

            exp_rates[iteration, :] = exprate[:len(N_Ia)]
            obs_rates[iteration, :] = N_Ia

        # Compute percentiles
        DTD_median = self.expandDTD(np.percentile(DTD_array, 50, axis=0), len(sfh), self.bins)
        DTD_lo = self.expandDTD(np.percentile(DTD_array, 16, axis=0), len(sfh), self.bins)
        DTD_hi = self.expandDTD(np.percentile(DTD_array, 84, axis=0), len(sfh), self.bins)

        if plot:

            # Make plot
            plt.plot(self.feh[:-1], DTD_median, 'k-')
            plt.fill_between(self.feh[:-1], DTD_lo, DTD_hi, color='gray', alpha=0.5)
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel('DTD', fontsize=16)
            plt.ylim(-0.005,0.2)
            plt.savefig('figures/noageZ_DTD.png', bbox_inches='tight')
            plt.show()

            # Try plotting rates?
            plt.fill_between(self.feh[:-1], np.percentile(obs_rates, 16, axis=0), np.percentile(obs_rates, 84, axis=0), color=plt.cm.Pastel2(0), alpha=0.5)
            plt.plot(self.feh[:-1], np.percentile(obs_rates, 50, axis=0), linestyle='-', color=plt.cm.Dark2(0), label='Observed rate')
            plt.fill_between(self.feh[:-1], np.percentile(exp_rates, 16, axis=0), np.percentile(exp_rates, 84, axis=0), color=plt.cm.Pastel2(1), alpha=0.5)
            plt.plot(self.feh[:-1], np.percentile(exp_rates, 50, axis=0), linestyle='-', color=plt.cm.Dark2(1), label='Expected rate')
            plt.legend(loc='best')
            plt.yscale('log')
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel(r'Type Ia rate $(\mathrm{Gyr}^{-1})$', fontsize=16)
            plt.savefig('figures/noageZ_rates.png', bbox_inches='tight')
            plt.show()

            # Make test plot of SFH?
            plt.fill_between(self.feh[:-1], np.percentile(sfh_array, 16, axis=0), np.percentile(sfh_array, 84, axis=0), color='gray', alpha=0.5)
            plt.plot(self.feh[:-1], np.percentile(sfh_array, 50, axis=0), 'k-')
            #plt.legend(loc='best')
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel(r'SFR', fontsize=16)
            plt.savefig('figures/noageZ_sfh.png', bbox_inches='tight')
            plt.show()

            # Plot residuals between rates?
            percenterrors = (obs_rates - exp_rates)/exp_rates * 100.
            plt.plot(self.feh[:-1], np.percentile(percenterrors, 50, axis=0), 'k-')
            #plt.fill_between(feh[:-1], np.percentile(percenterrors, 16, axis=0), np.percentile(percenterrors, 84, axis=0), color='gray', alpha=0.5)
            plt.xlabel('[Fe/H]', fontsize=16)
            plt.ylabel(r'Percent error in Type Ia rates (\%)', fontsize=16)
            plt.ylim(-300,300)
            plt.savefig('figures/noageZ_rates_percenterrors.png', bbox_inches='tight')
            plt.show()

        return DTD_median, DTD_lo, DTD_hi

    def convertDTD(self, Niter=100, testplot=False, plotageZ=False):
        """Convert DTD from function of [Fe/H] to function of time."""

        # Get DTD from fitting
        DTD, dtd_lo, dtd_hi = self.fitdtd(Niter, plot=testplot)

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

            if plotageZ:
                plt.plot(fehs[i][fehlimidx], ages[i][fehlimidx], 'ko', label='Observed AMR')
                plt.plot(fehs[i][fehlimidx], np.polyval(p, fehs[i][fehlimidx]), 'r-', label='Best-fit line')
                plt.legend(title=amr, fontsize=12, title_fontsize=14)
                plt.xlabel('[Fe/H]', fontsize=16)
                plt.ylabel('Time (Gyr)', fontsize=16)
                plt.savefig('figures/amr_'+amr+'.png', bbox_inches='tight')
                plt.show()

            # Convert from metallicity to age
            age = p[0] * self.feh[:-1] # * np.polyval(p, feh[fehidx]) # 

            # Plot resulting DTD
            if plotageZ==False:
                plt.plot(age, DTD, linestyle='-', color=plt.cm.Dark2(i), label=amr)
                if dtd_lo is not None and dtd_hi is not None:
                    plt.fill_between(age, dtd_lo, dtd_hi, color=plt.cm.Pastel2(i))

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
    scl = galaxy('scl', Mstar=1.2e6, bins=10)
    scl.convertDTD(Niter=100, testplot=False, plotageZ=False)