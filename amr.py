"""
amr.py

Linearly fits the age-metallicity relation for each galaxy.
"""

#Backend for python3 on stravinsky
from pickle import BINSTRING
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

def amr(name, fehlim=(-2,-1.5)):
    '''Returns age, metallicity, best-fit poly1d, indices (over which poly1d is fit)'''

    if name=='gce':
        # Get age-metallicity relation from my GCE model
        amr_delosreyes = np.load('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/plots/amr_test.npy')
        feh = amr_delosreyes[:,0]  # [Fe/H]
        age = 13.791 - amr_delosreyes[:,1]  # Gyr
        goodidx = np.where(np.isfinite(feh))[0]
        feh = feh[goodidx]
        age = age[goodidx]

    elif name=='deBoer':
        # Get age-metallicity relation from de Boer et al. (2012)
        amr_deBoer = np.loadtxt('/Users/miadelosreyes/Documents/Research/MnDwarfs_DTD/code/gce/data/sfhtest/deboer12-ageZ.dat', delimiter=',')
        age_deBoer_idx = np.argsort(amr_deBoer[:,1])  # Sort by metallicity
        age = amr_deBoer[age_deBoer_idx,0]  # Gyr
        feh = amr_deBoer[age_deBoer_idx,1]  # [Fe/H]

    # Linearly extrapolate to [Fe/H] ~ 0.
    feh0 = 0.0
    age0 = age[-1]+(feh0-feh[-1])*(age[-1]-age[-2])/(feh[-1]-feh[-2])
    feh = np.concatenate((feh,[feh0]))
    age = np.concatenate((age,[age0]))

    # Fit polynomial to age-Z relation
    fehlimidx = np.where((feh > fehlim[0]) & (feh < fehlim[1]))[0]
    p = np.polyfit(feh[fehlimidx], age[fehlimidx], 1)

    return age, feh, p, fehlimidx

if __name__=="__main__":
    names = ['gce','deBoer']
    amrs = ['GCE model', 'de Boer et al. (2012)']

    # Loop over all age-Z relations
    for i, amr in enumerate(amrs):

        age, feh, p, fehlimidx = amr(names[i])
        plt.plot(feh[fehlimidx], age[fehlimidx], 'ko', label='Observed AMR')
        plt.plot(feh[fehlimidx], np.polyval(p, feh[fehlimidx]), 'r-', label='Best-fit line')
        plt.legend(title=amr, fontsize=12, title_fontsize=14)
        plt.xlabel('[Fe/H]', fontsize=16)
        plt.ylabel('Time (Gyr)', fontsize=16)
        plt.savefig('figures/amr_'+amr+'.png', bbox_inches='tight')
        plt.show()