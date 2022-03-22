import numpy as np
import pandas as pd
import lmfit
import matplotlib.pyplot as plt
import os

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

#********************************** Functions **********************************

def check_if_in_range(lst1, lst2, min, max):
    """
    Check if any value in lst1 or lst2 are outside (min,max).
    Returns list of elements outside range.
    """
    arr1 = np.array(lst1)
    arr1 = arr1[(arr1<min) | (arr1>max)]
    arr2 = np.array(lst2)
    arr2 = arr2[(arr2<min) | (arr2>max)]

    if arr1.size > 0:
        print('List1:', arr1[(arr1<min) | (arr1>max)]) 
    if arr2.size > 0:
        print('List2:', arr2[(arr2<min) | (arr2>max)])

def LC_func(phi, offset, inc, dotM, dotM2, asini, fWR2O):
    """
    Light curve function. Taken from Lamontagne et al. 1996. Equation numbers 
    shown in comments.
    """
    phi2pi = 2*np.pi*phi+np.pi # phi=0 WR in back
    a = asini / np.sin(inc)
    # check_if_in_range(R/a, R2/a, 0, 1)
    #* (9)
    k  = alpha*sigt*dotM / (4*np.pi*mp*v_inf*a)
    k2 = alpha2*sigt*dotM2 / (4*np.pi*mp*v_inf2*a)
    #* (10)
    eps  = np.sin(inc) * np.cos(phi2pi+np.pi)
    eps2 = np.sin(inc) * np.cos(phi2pi)
    # check_if_in_range(eps, eps2, -1, 1) # None
    #* (14)
    A  = 2.5*np.log10(euler)*k / (1+1/fWR2O)
    A2 = 2.5*np.log10(euler)*k2 / (1+fWR2O)
    # beta=1
    #*(15)
    # print(R > a*np.sqrt(1-eps**2), inc*180/np.pi)
    b  = R/a / np.sqrt(1-eps**2)  #! Becomes greater than 1
    for idx, item in enumerate(b): #TODO: Find out why it's unstable at inc > 60
        if item > 1: # Capping it so it doesn't give NaNs
            b[idx] = 0.99
    b2 = R2/a / np.sqrt(1-eps2**2)
    for idx, item in enumerate(b2):
        if item > 1:
            b2[idx] = 0.99

    # check_if_in_range(b2, b2, 0, 1)
    #* Bracket term of (13) follows
    term1  = np.sqrt((1+b)/(1-b))
    term2  = term1 * np.tan(np.arcsin(eps)/2)
    term3  = np.arctan(term1) + np.arctan(term2)
    term4  = 2/np.sqrt((1 - eps**2) * (1 - b**2)) * term3

    term12 = np.sqrt((1+b2) / (1-b2))
    term22 = term12 * np.tan(np.arcsin(eps2)/2.)
    term32 = np.arctan(term12) + np.arctan(term22)
    term42 = 2/np.sqrt((1 - eps2**2) * (1 - b2**2)) * term32 
    #* (13)
    eclipsed_light = offset + A*term4 + A2*term42
    return eclipsed_light

def chisqr(params):
    """Chi^2 function to reduce using LMFit."""
    P      = params['P'].value
    T0     = params['T0'].value
    K1     = params['K1'].value
    K2     = params['K2'].value      
    ecc    = params['ecc'].value

    dotM   = 10**(params['dotM'].value)* Msun / (365*24*3600.) #WR
    dotM2  = 10**(params['dotM2'].value)* Msun /  (365*24*3600.) #O
    inc    = params['inc'].value * np.pi/180
    fWR2O  = params['fWR2O'].value
    offset = params['offset'].value
    asini  = 0.0198 * np.sqrt(1-ecc**2) * (K1+K2) * P * Rsun

    phasesLC = (LC_HJDs-T0)/P - ((LC_HJDs-T0)/P).astype(int) + 1
    lamLC = phasesLC # lambda = nu + omega + 90 (shenar+2021 Tarantula)

    lightmod = LC_func(lamLC, offset, inc, dotM, dotM2, asini, fWR2O)
    return (lightmod-LC_mags)/LC_errs

def minimizer(mini_func, MiniMethod='differential_evolution', set_inc=None):
    """
    Minimize and give final results. If given a specific set_inc,
    does not fit for inclination.
    """
    params = lmfit.Parameters()
    params.add('P', value=Pini, min=Pini - 0.1*Pini, max=Pini + 0.1*Pini, vary=False)
    params.add('T0', value=T0ini, min=T0ini-2*Pini, max=T0ini+0.1*Pini, vary=False) # in HJD
    params.add('K1', value=266.4921, min=110, max=300, vary=False)
    params.add('K2', value=141.8765, min=110, max=150, vary=False)
    params.add('ecc', value=0, min=0, max=1e-4, vary=False)
    params.add('dotM', value=-5.895, min=-8.3, max=-4, vary=True)
    params.add('dotM2', value=-4.695, min=-5.5, max=-3, vary=True)
    params.add('inc', value=54.8, min=30, max=60, vary=True)
    params.add('fWR2O', value=4, min=0.2, max=5, vary=False)
    params.add('offset', value=12.524, min=12.2, max=12.8, vary=True) #* Delta mag in (13)
    if set_inc is None:
        params.add('inc', value=54.87, min=20, max=55, vary=True)
    else:
        params.add('inc', value=set_inc, min=0, max=90, vary=False)

    mini = lmfit.Minimizer(mini_func, params, nan_policy='omit')
    result = mini.minimize(method=MiniMethod)
    ci = None
    if set_inc is None:
        print(lmfit.fit_report(result))
        ci = lmfit.conf_interval(mini, result, sigmas=[1,2])
    return result, ci

def plot_result(result, ci=None):
    """Plot minimization results."""
    P      = result.params['P'].value
    T0     = result.params['T0'].value
    K1     = result.params['K1'].value
    K2     = result.params['K2'].value
    ecc    = result.params['ecc'].value
    dotM   = result.params['dotM'].value
    dotM2  = result.params['dotM2'].value
    inc    = result.params['inc'].value * np.pi/180
    fWR2O  = result.params['fWR2O'].value
    offset = result.params['offset'].value
    asini = 0.0198 * (1-ecc**2)**(1./2.) * (K1+K2) * P  *Rsun
    dotM  = 10**(dotM)* Msun / (365*24*3600.)
    dotM2 = 10**(dotM2)* Msun /  (365*24*3600.)
    
    phi_arr = np.linspace(0, 1, 100)
    LC_phis  = (LC_HJDs-T0)/P - ((LC_HJDs-T0)/P).astype(int) + 1

    LC_sim  = LC_func(phi_arr, offset, inc, dotM, dotM2, asini, fWR2O)
    # LC_sim2 = LC_func(phi_arr, offset, 59.7, dotM, dotM2, asini, fWR2O)
    # LC_sim3 = LC_func(phi_arr, offset, 59.735, dotM, dotM2, asini, fWR2O)

    if ci:
        # For plotting 1 and 2 sigma intervals
        LC_sim_1sigP = LC_func(phi_arr, offset, ci['inc'][1][1]*np.pi/180,
                                dotM, dotM2, asini, fWR2O)
        LC_sim_1sigN = LC_func(phi_arr, offset, ci['inc'][3][1]*np.pi/180,
                                dotM, dotM2, asini, fWR2O)
        LC_sim_2sigP = LC_func(phi_arr, offset, ci['inc'][0][1]*np.pi/180,
                                dotM, dotM2, asini, fWR2O)
        LC_sim_2sigN = LC_func(phi_arr, offset, ci['inc'][4][1]*np.pi/180,
                                dotM, dotM2, asini, fWR2O)
    fig, ax = plt.subplots(nrows=1, figsize=(7,6))
    ax.set_title('Fit on Light-curve')
    ax.errorbar(LC_phis, LC_mags, yerr=LC_errs, fmt='.', color='Green', alpha=0.4)
    ax.plot(phi_arr, LC_sim, color='red', label='Fit')
    # ax.plot(phi_arr, LC_sim2, color='orange', label='Fit')
    # ax.plot(phi_arr, LC_sim3, color='purple', label='Fit')
    if ci:
        # for 1- and 2-sigma region
        ax.plot(phi_arr, LC_sim_1sigP, color='red', linestyle=':', label='1-$\sigma$ conf. level')
        ax.plot(phi_arr, LC_sim_1sigN, color='red', linestyle=':')
        ax.plot(phi_arr, LC_sim_2sigP, color='red', linestyle='-.', label='2-$\sigma$ conf. level')
        ax.plot(phi_arr, LC_sim_2sigN, color='red', linestyle='-.')
        ax.fill_between(phi_arr, LC_sim_1sigP, LC_sim_1sigN, color='red', alpha=0.4)
        ax.fill_between(phi_arr, LC_sim_2sigP, LC_sim_2sigN, color='red', alpha=0.2)
    ax.grid(axis='x', linestyle='--')
    ax.set_ylabel('$M_V$')
    ax.set_xlabel('Phase')
    ax.set_xlim(0,1)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()

def plot_chi2(phis, red_chi2, *args):
    """
    Plots the reduced chi^2 of the fits as a function of inclination.
    """
    step=args[0]
    # np.savetxt('LC_redchi2.txt', red_chi2)
    plt.scatter(phis, red_chi2)
    plt.title('Reduced $\chi^2$ of fit at various inclinations')
    plt.xlabel('Inclination')
    plt.ylabel('Reduced $\chi^2$')
    plt.xlim(0,90)
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
    if inc_grid:
        # Create grid of inclinations and plot red chi^2
        red_chi2 = []
        phis = []
        for inc in np.arange(5,90,step):
            result, _ = minimizer(chisqr, set_inc=inc)
            red_chi2.append(result.redchi)
            phis.append(inc)
            # plot_result(result)
        plot_chi2(phis, red_chi2, step)

if __name__ == '__main__':
    main()