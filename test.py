import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.io.idl import readsav
from astropy.io import fits

data = readsav('data/rfrac/sn_decomposition_leoii.sav')
print(data.data.dtype.names)
plt.plot(data.data['FEH'], data.data['R'], 'ko')
plt.title('Leo II', fontsize=14)
plt.xlabel('[Fe/H]', fontsize=14)
plt.ylabel('R', fontsize=14)
plt.savefig('testR.png', bbox_inches='tight')
plt.show()

# Get Fe_Ia/Fe_CC info from Kirby+19
rfile = fits.open('data/rfrac/chemev_scl.fits')
R = rfile[1].data['R'].T[:,0]
Rerrlo = rfile[1].data['RERRL'].T[:,0]
Rerrhi = rfile[1].data['RERRH'].T[:,0]
feh = np.linspace(-2.97,0, num=100)
#plt.plot(feh, R, 'ko')
#plt.show()