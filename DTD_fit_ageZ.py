"""
DTD_fit_binned.py

Computes and fits Type Ia DTD from abundance data.
Assumes age-Z relations from SFH.
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
from amr import amr
from opendtd import t as dtd_t
from opendtd import dd, sd

# Get some cool colors
import cmasher as cmr
#colors = cmr.take_cmap_colors('cmr.tropical', 6, cmap_range=(0.1, 0.9), return_fmt='hex')
colors = plt.cm.plasma(np.linspace(0,1,7))

# Get fraction of CCSNe
from ccfrac import CCfrac
#CCfrac = 0.008

galaxynames = {'Scl':'Sculptor dSph', 'Dra':'Draco dSph', 'LeoII':'Leo II dSph'}

# Class to wrap each galaxy
class galaxy:
    def __init__(self, name, Mstar, gce=False, num=50):

        self.name = name  # ID string for galaxy
        self.Mstar = Mstar  # Stellar mass (Msun)
        self.gce = gce  # Bool for use if GCE AMR is being used

        # Define an age-Z relation
        self.amr_age, self.amr_feh, _, fehlimidx, self.sfr, self.t = amr(self.name, gcetest=gce)
        self.feh2age = interp1d(self.amr_feh, self.amr_age)
        self.age2feh = interp1d(self.amr_age, self.amr_feh)

        # Set initial time (Gyr)
        if gce:
            self.t0 = 13.791  # for GCE only
        else:
            self.t0 = 10**10.15/1e9  # for Weisz+14 AMR

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

        # Get MDF data from Kirby+09
        kirby09_new = Table.read('data/mdf/kirby_dsph_catalog.dat', format='ascii.cds')
        idx = np.where(kirby09_new['dSph']==name)
        self.mdf_feh = kirby09_new['eps(Fe)'][idx] - 7.50
        self.mdf_feh_err = kirby09_new['e_eps(Fe)'][idx]

        # Remove stars with large error
        goodidx = np.where((self.mdf_feh_err < 0.3)) # & (self.mdf_feh > -3.5))
        self.mdf_feh = self.mdf_feh[goodidx]
        self.mdf_feh_err = self.mdf_feh_err[goodidx]

        # Define uniform age array on which to put all observed quantities
        self.ia_age = np.linspace(min(self.t0-self.amr_age), (self.t0 - self.amr_age[fehlimidx])[-1], num=num)

        # Convert uniform age array to [Fe/H] array
        self.ia_feh = self.age2feh(self.t0-self.ia_age)

        # Extrapolate to t=0 (needed to get the convolution right)
        feh0 = self.ia_feh[0] + (0. - self.ia_age[0])*(self.ia_feh[1]-self.ia_feh[0])/(self.ia_age[1]-self.ia_age[0])
        self.ia_age = np.insert(self.ia_age, 0, 0.)
        self.ia_feh = np.insert(self.ia_feh, 0, feh0)

        # Get bin edges
        self.ia_fehbin = np.diff(self.ia_feh)  # [Fe/H] bin edges (dex)
        self.ia_agebin = np.diff(self.ia_age)  # Age bin edges (Gyr)

    def ecdf(self, a):
        """Function to compute empirical cumulative sum, scaled to stellar mass of galaxy."""
        x, counts = np.unique(a, return_counts=True)
        cusum = np.cumsum(counts)
        return x, cusum / cusum[-1] 

    def fitdtd(self, Niter, plot=False, ploterrors=False, outputrmse=False):
        """Fit DTD for a galaxy using MC-like method 
        (perturbing MDF, SFH to get realizations of Type Ia rates, 
        and fitting those realizations)."""

        # Let's do MC estimation of errors
        exp_rates_sd = np.zeros((len(sd.keys()),Niter,len(self.ia_age)-2))
        exp_rates_dd = np.zeros((len(dd.keys()),Niter,len(self.ia_age)-2))
        obs_rates = np.zeros((Niter,len(self.ia_age)-2))

        # Also make some arrays to hold chi2 values
        chisq_sd = np.zeros((len(sd.keys()),Niter))
        chisq_dd = np.zeros((len(dd.keys()),Niter))

        for iteration in tqdm(range(Niter)):

            # Get single realization of R
            '''
            percentile = np.random.uniform()
            new_r = np.zeros(len(self.R))
            for i in range(len(self.R)):
                if percentile > 0.5:
                    scale=self.Rerrhi[i]
                else:
                    scale=self.Rerrlo[i]
                new_r[i] = norm(loc=self.R[i],scale=scale).ppf(percentile)
            '''

            # Convert R to cumulative fraction of Ia/CC SNe
            fe_CC = 0.074  # Core-collapse yield from Maoz&Graur17
            fe_Ia = np.sum([7.80e-3,6.10e-1,2.12e-2,4.39e-4])  # Type Ia yields from Leung+19 (1.1 Msun, solar Z=0.02)
            N_Ia_CC = self.R/fe_Ia * fe_CC
            
            # Interpolate N_Ia/N_CC to [Fe/H] array
            N_Ia_CC = np.interp(self.ia_feh, self.R_feh, N_Ia_CC)
            
            # Get single realization of MDF
            perturb = np.random.normal(self.mdf_feh, self.mdf_feh_err)

            # Compute MDF
            hist, _ = np.histogram(perturb, bins=self.ia_feh)
            sfh = hist/np.sum(hist) * self.Mstar  # mass of stars formed in each bin
            sfh[np.where(self.ia_feh < min(self.R_feh))] = 0.

            # Compute cumulative number of CCSNe
            N_CC = sfh * CCfrac
            N_CC_cumulative = np.cumsum(N_CC)

            # Compute rate of IaSNe as function of [Fe/H]
            N_Ia_cumulative = N_CC_cumulative * N_Ia_CC[:-1]
            N_Ia_obs = np.diff(N_Ia_cumulative)  # SNe in each bin (units: SNe/bin)
            N_Ia_obs[~np.isfinite(N_Ia_obs)] = 0.
            N_Ia_obs[N_Ia_obs < 0.] = 0.

            # Convert SFH and N_Ia to functions of time
            sfh *= 1./self.ia_agebin  # units: Msun/bin * bin/Gyr = Msun/Gyr
            N_Ia_obs *= 1./(self.ia_agebin[:-1])**2   # units: SNe/bin * bin/Gyr = SNe/Gyr
            #plt.plot(self.ia_age[:-1], sfh)

            if not self.gce:
                self.sfr = self.sfr * self.Mstar

            # Can use this instead of MDF
            #sfh = np.interp(self.ia_age[:-1], self.t[:-1], self.sfr)
            #plt.plot(self.ia_age[:-1], sfh)
            #plt.show()

            # Test if SFH integrates to total Mstar
            #sfhtest = np.trapz(sfh, self.ia_age[:-1])
            #NIatest = np.trapz(N_Ia_obs, self.ia_age[:-2])
            #print(sfhtest, NIatest, self.Mstar)
            #print(NIatest/self.Mstar)

            # Save observed rate in array
            obs_rates[iteration, :] = N_Ia_obs

            # Get DTD from DD models
            for i, modelname in enumerate(dd.keys()):

                # Interpolate DTD to match the age array
                DTD = np.asarray(dd[modelname])*1e9  # SNe/Msun/Gyr
                mint = dtd_t[np.where(DTD > 0.)][0]
                DTD = np.interp(self.ia_age, dtd_t, DTD)
                DTD[np.where(self.ia_age < mint)] = 0.

                # Convolve DTD with SFH
                test = np.convolve(DTD[:-1], sfh)[:len(N_Ia_obs)]  # units: (SNe/Gyr/Msun) * (Msun/Gyr) * Gyr = SNe/Gyr
                test[~np.isfinite(test)] = 0.
                N_Ia_exp = test

                # Save expected rate in array
                exp_rates_dd[i, iteration, :] = N_Ia_exp  # Units: 10^3 SNe/Gyr

            # Test a delta function
            '''
            dtd = np.zeros_like(self.ia_age)
            dtd[5] = 1.
            test = np.convolve(dtd[:-1], sfh)[:len(N_Ia_obs)]  # units: (SNe/Gyr/Msun) * (Msun/Gyr) * Gyr = SNe/Gyr
            test[~np.isfinite(test)] = 0.           
            plt.plot(self.ia_age[:-2], test, 'k-')
            plt.plot(self.ia_age[:-1], sfh, 'r--')
            plt.show()
            '''

            # Get DTD from SD models
            for i, modelname in enumerate(sd.keys()):
                # Interpolate DTD to match the age array
                DTD = np.asarray(sd[modelname])*1e9  # SNe/Msun/Gyr
                mint = dtd_t[np.where(DTD > 0.)][0]
                DTD = np.interp(self.ia_age, dtd_t, DTD)
                DTD[np.where(self.ia_age < mint)] = 0.

                # Convolve DTD with SFH
                test = np.convolve(DTD, sfh)[:len(N_Ia_obs)]  # units: (SNe/Gyr/Msun) * (Msun/Gyr) * Gyr = SNe/Gyr
                test[~np.isfinite(test)] = 0.
                N_Ia_exp = test

                # Save expected rate in array
                exp_rates_sd[i, iteration, :] = N_Ia_exp  # Units: 10^3 SNe/Gyr

        if plot:

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,10), sharex=False, sharey=True)
            axs = axs.ravel()
            handles = []

            # Plot expected rates from Maoz+10 model
            '''
            axs[0].fill_between(self.ia_age[:-2], np.percentile(exp_rates_dd[-1,:,:], 16, axis=0), np.percentile(exp_rates_dd[-1,:,:], 84, axis=0), color='mediumblue', alpha=0.4)
            p, = axs[0].plot(self.ia_age[:-2], np.percentile(exp_rates_dd[-1,:,:], 50, axis=0), linestyle='-', color='mediumblue', label='Maoz+10')
            handles.append(p) 
            '''

            # Plot expected rates from double-degenerate models
            axs[0].set_title('Double-degenerate', fontsize=18)
            for i, modelname in enumerate(dd.keys()):
                axs[0].fill_between(self.ia_age[:-2], np.percentile(exp_rates_dd[i,:,:], 16, axis=0), np.percentile(exp_rates_dd[i,:,:], 84, axis=0), color=colors[i], alpha=0.4)
                p, = axs[0].plot(self.ia_age[:-2], np.percentile(exp_rates_dd[i,:,:], 50, axis=0), linestyle='-', color=colors[i], label=modelname)
                handles.append(p)

            # Plot expected rates from single-degenerate models
            axs[1].set_title('Single-degenerate', fontsize=18)
            for i, modelname in enumerate(sd.keys()):
                axs[1].fill_between(self.ia_age[:-2], np.percentile(exp_rates_sd[i,:,:], 16, axis=0), np.percentile(exp_rates_sd[i,:,:], 84, axis=0), color=colors[i+1], alpha=0.4)
                axs[1].plot(self.ia_age[:-2], np.percentile(exp_rates_sd[i,:,:], 50, axis=0), linestyle='-', color=colors[i+1], label=modelname)

            handles2 = []
            for axnum in range(len(axs)):
                # Plot observed rates
                axs[axnum].fill_between(self.ia_age[:-2], np.percentile(obs_rates, 16, axis=0), np.percentile(obs_rates, 84, axis=0), color='C0', alpha=0.4)
                p, = axs[axnum].plot(self.ia_age[:-2], np.percentile(obs_rates, 50, axis=0), linestyle='--', lw=2, color='C0', label='Observed rate')

                if self.name=='Scl':
                    # Plot GCE rates
                    iarate_gce = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/output/iarate_test.npy')
                    gcerate = np.interp(self.ia_age, iarate_gce[1], iarate_gce[0])
                    gcerate = gcerate[:-1]/np.diff(self.ia_age)
                    p1, = axs[axnum].plot(self.ia_age[:-1], gcerate, linestyle=':', lw=2, color='k', label='GCE model rate')

                    axs[axnum].set_xlim(0,0.7)

                # Some formatting stuff
                axs[axnum].set_yscale('log')
                axs[axnum].set_ylabel(r'$R_{\mathrm{IaSNe}}$ ($\mathrm{Gyr}^{-1}$)', fontsize=16)

            # Some formatting stuff
            handles2.append(p)      
            axs[0].text(0.05, 0.9, galaxynames[self.name], transform=axs[0].transAxes, fontsize=16)
            axs[1].set_xlabel('Time (Gyr)', fontsize=16)      
            if self.name=='Scl':
                handles2.append(p1)
                axs[0].set_ylim(1,1e6)
            
            # Legends
            legend1 = axs[0].legend(handles=handles2, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=16)
            axs[0].legend(handles=handles, bbox_to_anchor=(1.05, 0.8), loc='upper left', fontsize=16)
            axs[0].add_artist(legend1)

            # Save plot
            plottitle = 'figures/'+self.name+'_dtdtest_rates'
            if self.gce:
                plottitle += '_gce'

            plt.savefig(plottitle+'.pdf', bbox_inches='tight')
            plt.show()

        # Plot residuals between rates
        if ploterrors:

            fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,10), sharex=False, sharey=True)
            axs = axs.ravel()
            handles = []

            idx = np.where(np.all(np.isclose(obs_rates, 0.), axis=0))
            print(idx)

            # Plot expected rates from double-degenerate models
            axs[0].set_title('Double-degenerate', fontsize=18)
            for i, modelname in enumerate(dd.keys()):
                rmse = np.nanmean(np.abs(obs_rates - exp_rates_dd[i,:,:]), axis=0)
                rmse[idx] = np.nan
                #percenterrors = np.abs((obs_rates - exp_rates_dd[i,:,:])/obs_rates)
                #axs[0].fill_between(self.ia_age[:-2], np.percentile(percenterrors, 16, axis=0), np.percentile(percenterrors, 84, axis=0), color=colors[i], alpha=0.4)
                p, = axs[0].plot(self.ia_age[:-2], rmse, linestyle='-', color=colors[i], label=modelname)
                handles.append(p)

            # Plot percent errors from single-degenerate models
            axs[1].set_title('Single-degenerate', fontsize=18)
            for i, modelname in enumerate(sd.keys()):
                rmse = np.nanmean(np.abs(obs_rates - exp_rates_sd[i,:,:]), axis=0)
                rmse[idx] = np.nan
                percenterrors = np.abs((obs_rates - exp_rates_sd[i,:,:])/obs_rates)
                #axs[1].fill_between(self.ia_age[:-2], np.percentile(percenterrors, 16, axis=0), np.percentile(percenterrors, 84, axis=0), color=colors[i+1], alpha=0.4)
                axs[1].plot(self.ia_age[:-2], rmse, linestyle='-', color=colors[i+1], label=modelname)

            if self.name=='Scl':
                for axnum in range(len(axs)):
                    # Plot GCE errors
                    gcerate = np.interp(self.ia_age, iarate_gce[1], iarate_gce[0])
                    gcerate = gcerate[:-1]/np.diff(self.ia_age)
                    rmse = np.mean(np.abs(obs_rates - gcerate[:-1]), axis=0)
                    #percenterrors = np.abs((obs_rates - gcerate)/gcerate)
                    #percenterrors[idx] = np.nan
                    rmse[idx] = np.nan
                    p, = axs[axnum].plot(self.ia_age[:-2], rmse, linestyle=':', lw=2, color='k', label='GCE model rate')
                    #axs[axnum].fill_between(self.ia_age[:-2], np.percentile(percenterrors, 16, axis=0), np.percentile(percenterrors, 84, axis=0), color='k', alpha=0.4)
                handles2 = [p]

            # Some formatting stuff
            axs[1].set_xlabel('Time (Gyr)', fontsize=16)
            for axnum in range(len(axs)):
                axs[axnum].set_ylabel(r'RMSE', fontsize=16)
                #axs[axnum].set_ylim(-100,15000)

            if self.name=='Scl':
                legend1 = axs[0].legend(handles=handles2, bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=16)
            axs[0].legend(handles=handles, bbox_to_anchor=(1.05, 0.8), loc='upper left', fontsize=16)
            if self.name=='Scl':
                axs[0].add_artist(legend1)

            axs[0].set_yscale('log')
            axs[1].set_yscale('log')

            # Save plot
            plottitle = 'figures/'+self.name+'_dtdtest_rates_errors'
            if self.gce:
                plottitle += '_gce'

            plt.savefig(plottitle+'.pdf', bbox_inches='tight')
            plt.show()

        # Compute RMSE?
        if outputrmse:
            for i, modelname in enumerate(dd.keys()):
                rmse = np.mean(np.abs(obs_rates - exp_rates_dd[i,:,:]))
                #rmse = np.sqrt(np.mean((obs_rates[:,:-4] - exp_rates_dd[i,:,:-4])**2.))
                print(modelname, rmse)

            # Plot percent errors from single-degenerate models
            for i, modelname in enumerate(sd.keys()):
                rmse = np.mean(np.abs(obs_rates - exp_rates_sd[i,:,:]))
                #rmse = np.sqrt(np.mean((obs_rates[:,:-4] - exp_rates_sd[i,:,:-4])**2.))
                print(modelname, rmse)

        print(self.ia_agebin*2)

        return

if __name__ == "__main__":
    # DMD calculation
    #galaxy('Scl', Mstar=10**6.08, gce=True, num=20).fitdtd(100, plot=False, ploterrors=False, outputrmse=True) #
    galaxy('Dra', Mstar=10**5.96, num=50).fitdtd(100000, plot=True, ploterrors=True, outputrmse=True)
    #galaxy('LeoII', Mstar=10**6.16, num=50).fitdtd(100000, plot=False, ploterrors=False, outputrmse=True)
    #galaxy('Sex', Mstar=10**5.93, bins=40).fitdtd(20, plot=True)