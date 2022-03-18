import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from polarization_fit import pol_fit

#*********************** Parameters for simulating data ************************
U_0, Q_0, BigOmega, tau_star, inc = 0.11, 0.65, 220, 0.29, 58*np.pi/180
P, T0 = 1.90753233, 2459107.55
#*******************************************************************************

phi_l = np.arange(0,1, 0.05)
phi_l = np.array([(0.1*np.random.rand()+phi) for phi in phi_l])
MJD = phi_l*P+T0-2440000 # Getting MJD from phase

U,Q = pol_fit(phi_l, U_0, Q_0, BigOmega, tau_star, inc)+(0.05*np.random.rand(len(phi_l)))

data = pd.DataFrame(data=MJD[:,np.newaxis], columns=['MJD'])

data['U'] = U
data['Q'] = Q
data['sig(P)'] = 0.025

plt.errorbar(MJD, U, xerr=0.025, yerr=0.025, fmt='.')
plt.errorbar(MJD, Q, xerr=0.025, yerr=0.025, fmt='.')
plt.show()

pol_path = os.path.join('Data', 'polarimetry_data.csv')
data.to_csv(pol_path)