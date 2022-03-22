import numpy as np
import pandas as pd
import os
import lmfit
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from astropy.io import ascii
from LC_fit import LC_func, plot_chi2
from polarization_fit import pol_fit


# Constants
euler = np.exp(1)
sigt = 6.652E-29 # m^2
mp = 1.67E-27 # kg
Rsun = 6.96E8 # m
Msun = 2E30 # kg

#********************* Define initial guesses for system ***********************
# Star-dependent constants
#! Numberless vars are O-star
alpha  = 0.5 # free electrons per baryon mass
alpha2 = 0.5
v_inf  = 2600*1000 # m/s Following values are from PoWR model fitting
v_inf2 = 1200*1000 # Shenar+2019 use 2000
R  = 9*Rsun
R2 = 4*Rsun
# Orbit-dependent parameters
Pini  = 1.90753233
T0ini = 2459107.55

#************************** Load in Light Curve data ***************************
LC_path = os.path.join('NormBAT99-32', 'light_curve.csv')
light_curve = pd.read_csv(LC_path, header=0, usecols=[0,3,4])
LC_HJDs = light_curve['hjd'].values
LC_mags = np.array(light_curve['mag'].values)
LC_errs = np.array(light_curve['mag err'].values)

#************************** Load in Polarimetry data ***************************
pols = np.genfromtxt('BAT99-32_polarimetry_data.txt')
pol_HJDs = pols[:,0] + 2440000
pol_Qs   = pols[:,1]
pol_Us   = pols[:,2]
pol_sig  = pols[:,3]

#************************ Load in Radial Velocity data *************************
line_name = 'NV-4604'
# line_name = 'NV-4951'

RVs_path = os.path.join('CCF_results', 'BAT99-32_rel', line_name, 'RVs_CCF.txt')
velos = ascii.read(RVs_path)

RV_HJDs   = np.array(velos['MJD']+2400000.5)
RV_v1s    = np.array(velos['v1'])
RV_errv1s = np.array(velos['sig'])

#********************************** Functions **********************************
def kepler_eq(E, M, ecc):
    """Convert mean anomaly to eccentric anomaly according to Kepler's equation."""
    E2 = (M - ecc*(E*np.cos(E) - np.sin(E))) / (1. - ecc*np.cos(E))
    eps = np.abs(E2 - E) 
    if np.all(eps < 1E-10):
        return E2
    else:
        return kepler_eq(E2, M, ecc)

def nus(P, T0, ecc):
    """Gets true anomaly for each observation given the phase information."""
    phis = (RV_HJDs - T0)/P - ((RV_HJDs - T0)/P).astype(int) + 0.25
    phis[phis < 0] = phis[phis<0] + 1.
    Ms = 2 * np.pi * phis
    Es =  kepler_eq(0.5, Ms, ecc)
    eccfac = np.sqrt((1 + ecc) / (1 - ecc))
    nusdata = 2 * np.arctan(eccfac * np.tan(0.5 * Es))
    # nusdata[nusdata < 0] += 2*np.pi
    return nusdata

def v1only(nu, Gamma, K1, Omega, ecc):
    """
    Given true anomaly nu and parameters, function returns radial velocity v1.
    """
    Omega *= np.pi/180
    v1 = Gamma + K1*(np.cos(Omega + nu) + ecc* np.cos(Omega))
    return v1

def chisqr(params):
    P      = params['P'].value
    T0     = params['T0'].value
    Gamma  = params['Gamma'].value    
    K1     = params['K1'].value
    K2     = params['K2'].value      
    Omega  = params['Omega'].value * np.pi/180
    ecc    = params['ecc'].value

    dotM   = 10**(params['dotM'].value)* Msun / (365*24*3600.) #WR
    dotM2  = 10**(params['dotM2'].value)* Msun /  (365*24*3600.) #O
    inc    = params['inc'].value * np.pi/180
    fWR2O  = params['fWR2O'].value
    offset = params['offset'].value

    BigOmega = params['BigOmega'].value*np.pi/180
    tau_star = params['tau_star'].value
    Q_0      = params['Q_0'].value
    U_0      = params['U_0'].value
    asini    = 0.0198 * (1-ecc**2)**(1./2.) * (K1+K2) * P *Rsun
    phasesLC = (LC_HJDs-T0)/P - ((LC_HJDs-T0)/P).astype(int) + 1
    lamLC = phasesLC # lambda = nu + omega + 90 (shenar+2021 Tarantula)
    phases_pol = (pol_HJDs-T0)/P - ((pol_HJDs-T0)/P).astype(int)+1
    lampol = phases_pol # lambda = nu + omega + 90 (shenar+2021 Tarantula)


    lightmod = LC_func(lamLC, offset, inc, dotM, dotM2, asini, fWR2O)
    v1       = v1only(nus(P, T0, ecc), Gamma, K1, Omega, ecc)
    U, Q     = pol_fit(lampol, U_0, Q_0, BigOmega, tau_star, inc)
    return np.concatenate(((lightmod-LC_mags)/LC_errs, (v1 - RV_v1s)/RV_errv1s,
                            (U-pol_Us)/pol_sig, (Q-pol_Qs)/pol_sig))

def minimizer(mini_func, MiniMethod='differential_evolution', set_inc=None):
    """
    Minimize and give final results. If given a specific set_inc,
    does not fit for inclination.
    """
    params = lmfit.Parameters()
    params.add('P', value=Pini, min=Pini - 0.1*Pini, max=Pini + 0.1*Pini, vary=False)
    params.add('T0', value=T0ini, min=T0ini-2*Pini, max=T0ini+0.1*Pini, vary=False) # in HJD
    params.add('Gamma', value=130, min=0, max=400,  vary=False)
    params.add('K1', value=266.4921, min=110, max=300,  vary=False)
    params.add('K2', value=141.8765, min=110, max=150, vary=False)
    params.add('Omega', value= 90, min=0, max=360, vary=False)
    params.add('ecc', value=0, min=0, max=1e-4, vary=False)

    params.add('dotM', value=-5.895, min=-6.3, max=-4.5, vary=True)
    params.add('dotM2', value=-4.695, min=-5.5, max=-3, vary=True)
    params.add('fWR2O', value=1.5, min=0.2, max=2, vary=False)
    params.add('offset', value=12.524, min=12.4, max=12.8, vary=True) # Delta mag in (13)

    params.add('BigOmega', value=218.614, min=0, max=360, vary=False)
    params.add('tau_star', value=0.2429, min=0, max=1, vary=False)
    params.add('Q_0', value=0.592, min=0.5, max=0.75, vary=False)
    params.add('U_0', value=0.061, min=-0.1, max=0.2, vary=False)
    if set_inc is None:
        params.add('inc', value=54.87, min=20, max=70, vary=True)
    else:
        params.add('inc', value=set_inc, min=0, max=90, vary=False)

    mini = lmfit.Minimizer(mini_func, params, nan_policy='omit')
    result = mini.minimize(method=MiniMethod)
    ci = None
    if set_inc is None:
        print(lmfit.fit_report(result))
        ci = lmfit.conf_interval(mini, result, sigmas=[1,2])
    return result, ci

def plot_result(result, ci, plot_phase=True, interv_flag=True, plot_ellipse=True):
    """
    Plot minimization results. Two available plots:
    Plot phase: Phase vs. LC, Polarization and RVs. Additional option for 
    ellipse plot, Q vs. U.
    """
    P        = result.params['P'].value
    T0       = result.params['T0'].value
    Gamma    = result.params['Gamma'].value
    K1       = result.params['K1'].value
    K2       = result.params['K2'].value
    Omega    = result.params['Omega'].value
    ecc      = result.params['ecc'].value
    dotM     = result.params['dotM'].value
    dotM2    = result.params['dotM2'].value
    inc      = result.params['inc'].value * np.pi/180
    fWR2O    = result.params['fWR2O'].value
    offset   = result.params['offset'].value
    BigOmega = result.params['BigOmega'].value*np.pi/180
    tau_star = result.params['tau_star'].value
    Q_0      = result.params['Q_0'].value
    U_0      = result.params['U_0'].value
    asini = 0.0198 * (1-ecc**2)**(1./2.) * (K1+K2) * P  *Rsun
    dotM  = 10**(dotM)* Msun / (365*24*3600.)
    dotM2 = 10**(dotM2)* Msun /  (365*24*3600.)

    # Create phase array for plotting simulated lines
    phi_arr = np.linspace(0, 1, 100)
    # Calculating phases for data points
    RV_phis  = (RV_HJDs-T0)/P - ((RV_HJDs-T0)/P).astype(int)
    LC_phis  = (LC_HJDs-T0)/P - ((LC_HJDs-T0)/P).astype(int) + 1
    pol_phis = (pol_HJDs-T0)/P - ((pol_HJDs-T0)/P).astype(int) + 1

    # Fit result for LC
    LC_sim  = LC_func(phi_arr, offset, inc, dotM, dotM2, asini, fWR2O)
    if interv_flag:
        # for plotting 1 and 2 sigma intervals
        LC_sim_1sigP = LC_func(phi_arr, offset, ci['inc'][1][1]*np.pi/180, dotM, dotM2, asini, fWR2O)
        LC_sim_1sigN = LC_func(phi_arr, offset, ci['inc'][3][1]*np.pi/180, dotM, dotM2, asini, fWR2O)
        LC_sim_2sigP = LC_func(phi_arr, offset, ci['inc'][0][1]*np.pi/180, dotM, dotM2, asini, fWR2O)
        LC_sim_2sigN = LC_func(phi_arr, offset, ci['inc'][4][1]*np.pi/180, dotM, dotM2, asini, fWR2O)
    
    # Fit results for RVs
    Ms = 2 * np.pi * phi_arr
    Es = kepler_eq(np.pi, Ms, ecc)
    eccfac = np.sqrt((1 + ecc) / (1 - ecc))
    l_nus = 2. * np.arctan(eccfac * np.tan(0.5 * Es))
    v1_sim =  v1only(l_nus, Gamma, K1, Omega, ecc)

    # Fit result for pol
    U_fit, Q_fit = pol_fit(phi_arr, U_0, Q_0, BigOmega, tau_star, inc)
    if interv_flag:
        U_fit_1sigP, Q_fit_1sigP = pol_fit(phi_arr, U_0, Q_0, BigOmega, tau_star, 
                                            ci['inc'][1][1]*np.pi/180)
        U_fit_1sigN, Q_fit_1sigN = pol_fit(phi_arr, U_0, Q_0, BigOmega, tau_star, 
                                            ci['inc'][3][1]*np.pi/180)
        U_fit_2sigP, Q_fit_2sigP = pol_fit(phi_arr, U_0, Q_0, BigOmega, tau_star, 
                                            ci['inc'][0][1]*np.pi/180)
        U_fit_2sigN, Q_fit_2sigN = pol_fit(phi_arr, U_0, Q_0, BigOmega, tau_star, 
                                            ci['inc'][4][1]*np.pi/180)
    if plot_phase:
        # Plotting the three fits together
        fig, ax = plt.subplots(nrows=3, figsize=(7,8),  gridspec_kw={'height_ratios': [1,1,0.7]},
                                sharex=True)
        ax[0].set_title('Simultaneous fit on LC, polarimetry and radial velocities of NV-4604')
        ax[0].errorbar(LC_phis, LC_mags, yerr=LC_errs, fmt='.', color='Green', alpha=0.4)
        ax[0].plot(phi_arr, LC_sim, color='red', label='Fit')
        if interv_flag:
            # For 1- and 2-sigma region
            ax[0].plot(phi_arr, LC_sim_1sigP, color='red', linestyle=':', label='1-$\sigma$ conf. level')
            ax[0].plot(phi_arr, LC_sim_1sigN, color='red', linestyle=':')
            ax[0].plot(phi_arr, LC_sim_2sigP, color='red', linestyle='-.', label='3-$\sigma$ conf. level')
            ax[0].plot(phi_arr, LC_sim_2sigN, color='red', linestyle='-.')
            ax[0].fill_between(phi_arr, LC_sim_1sigP, LC_sim_1sigN, color='red', alpha=0.4)
            ax[0].fill_between(phi_arr, LC_sim_2sigP, LC_sim_2sigN, color='red', alpha=0.2)
        ax[0].grid(axis='x', linestyle='--')
        ax[0].set_ylabel('$M_V$')
        ax[0].set_xlim(0,1)
        ax[0].invert_yaxis()
        U_p = ax[1].errorbar(pol_phis, pol_Us, yerr=pol_sig, fmt='.', color='darkgreen',
                            label='U')
        Q_p = ax[1].errorbar(pol_phis, pol_Qs, yerr=pol_sig, fmt='.', color='forestgreen', 
                            label='Q')
        U_l, = ax[1].plot(phi_arr, U_fit, color='red', label='U fit')
        Q_l, = ax[1].plot(phi_arr, Q_fit, color='coral', label='Q fit')
        if interv_flag:
            # For 1- and 2-sigma region
            ax[1].plot(phi_arr, U_fit_1sigP, color='red', linestyle=':')
            ax[1].plot(phi_arr, U_fit_1sigN, color='red', linestyle=':')
            ax[1].plot(phi_arr, U_fit_2sigP, color='red', linestyle='-.')
            ax[1].plot(phi_arr, U_fit_2sigN, color='red', linestyle='-.')
            ax[1].plot(phi_arr, Q_fit_1sigP, color='coral', linestyle=':')
            ax[1].plot(phi_arr, Q_fit_1sigN, color='coral', linestyle=':')
            ax[1].plot(phi_arr, Q_fit_2sigP, color='coral', linestyle='-.')
            ax[1].plot(phi_arr, Q_fit_2sigN, color='coral', linestyle='-.')
            ax[1].fill_between(phi_arr, U_fit_1sigP, U_fit_1sigN, color='red', alpha=0.4)
            ax[1].fill_between(phi_arr, U_fit_2sigP, U_fit_2sigN, color='red', alpha=0.2)
            ax[1].fill_between(phi_arr, Q_fit_1sigP, Q_fit_1sigN, color='coral', alpha=0.4)
            ax[1].fill_between(phi_arr, Q_fit_2sigP, Q_fit_2sigN, color='coral', alpha=0.2)

        ax[1].grid(axis='x', linestyle='--')
        ax[1].set_ylabel('Stokes Param.')
        ax[1].legend([(Q_p, Q_l), (U_p, U_l)], ['Q', 'U'], numpoints=1,
                        handler_map={tuple: HandlerTuple(ndivide=None)})
        ax[2].errorbar(RV_phis, RV_v1s, yerr=RV_errv1s, fmt='.', color='Green')
        ax[2].plot(phi_arr, v1_sim, color='Red')
        ax[2].grid(axis='x', linestyle='--')
        ax[2].set_ylabel(r'RV $[{\rm km}\,{\rm s}^{-1}]$')
        ax[2].grid(axis='x', linestyle='--')
        ax[2].set_xlabel('Phase')
        plt.tight_layout()
        plt.subplots_adjust(hspace=0)
        plt.show()

    if plot_ellipse:
        # Plotting elliptic fit of polarization
        plt.errorbar(pol_Qs, pol_Us, yerr=pol_sig, xerr=pol_sig, fmt='o', )
        plt.plot(Q_fit, U_fit, color='r', label=round(inc*180/np.pi, 2))
        plt.xlabel('Q')
        plt.ylabel('U')
        plt.legend()
        plt.tight_layout()
        plt.show()

def main(plot_best=False, inc_grid=False, step=5):
    """
    When plot_best, fits for the parameters specified in minimizer. When inc_grid,
    greates a grid of inclinations at a resolution of step and plots reduced chi^2
    obtained from fitting at that inclination.
    """
    if plot_best:
        # Fit for specified parameters and plot best fit
        result, ci = minimizer(chisqr)
        plot_result(result, ci)
    elif inc_grid:
        # Create grid of inclinations and plot red chi^2
        red_chi2 = []
        phis = []
        for inc in np.arange(5,90,step):
            result, _ = minimizer(chisqr, set_inc=inc)
            red_chi2.append(result.redchi)
            phis.append(inc)
        plot_chi2(phis, red_chi2)

if __name__ == '__main__':
    main()