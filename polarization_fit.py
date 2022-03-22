import numpy as np
import matplotlib.pyplot as plt
import lmfit
from LC_fit import plot_chi2

#********************* Define initial guesses for system ***********************
Pini = 1.90753164   # Define initial orbit guess
T0ini = 2459107.55

#************************** Load in Polarimetry data ***************************
pols = np.genfromtxt('BAT99-32_polarimetry_data.txt')
pol_HJDs = pols[:,0] + 2440000
pol_Qs   = pols[:,1]
pol_Us   = pols[:,2]
pol_sig  = pols[:,3]

#********************************** Functions **********************************
def pol_fit(lampol, U_0, Q_0, BigOmega, tau_star, inc):
    """
    Polarization as function of phase, with other parameters.
    As in Moffat+1998.
    """
    lam = lampol * 2*np.pi
    Del_U = -2 * tau_star * np.cos(inc) * np.sin(2*lam)
    Del_Q = -tau_star * ((1 + np.cos(inc)**2) * np.cos(2*lam) - np.sin(inc)**2)
    U = U_0 + Del_Q*np.sin(BigOmega) + Del_U*np.cos(BigOmega)
    Q = Q_0 + Del_Q*np.cos(BigOmega) - Del_U*np.sin(BigOmega)
    return (U, Q)

def chisqr(params):
    """Chi^2 function to reduce using LMFit."""
    Q_0      = params['Q_0'].value
    U_0      = params['U_0'].value
    inc      = params['inc'].value*np.pi/180
    BigOmega = params['BigOmega'].value*np.pi/180
    tau_star = params['tau_star'].value
    T0       = params['T0'].value
    P        = params['P'].value

    phases_pol = (pol_HJDs-T0)/P - ((pol_HJDs-T0)/P).astype(int)+1
    lampol = phases_pol

    U, Q = pol_fit(lampol, U_0, Q_0, BigOmega, tau_star, inc)
    return np.concatenate(((U-pol_Us)/pol_sig, (Q-pol_Qs)/pol_sig))

def minimizer(mini_func, MiniMethod='differential_evolution', set_inc=None):
    """
    Minimize and give final results. If given a specific set_inc,
    does not fit for inclination.
    """
    params = lmfit.Parameters()
    params.add('BigOmega', value=220,  min=0, max=360, vary=True)
    params.add('tau_star', value=0.29,  min=0, max=1, vary=True)
    params.add('Q_0', value=0.64, min=0.5, max=0.75, vary=True)
    params.add('U_0', value=0.11, min=-0.1, max=0.2, vary=True)
    params.add('P', value=Pini, min=Pini-0.7*Pini, max=Pini+0.7*Pini, vary=False)
    params.add('T0', value=T0ini, min=T0ini-0.5*Pini, max=T0ini+0.5*Pini, vary=False) # in HJD
    if set_inc is None:
        params.add('inc', value=54.87,  min=20, max=70, vary=True)
    else:
        params.add('inc', value=set_inc,  min=0, max=90, vary=False)

    mini = lmfit.Minimizer(mini_func, params, nan_policy='omit')
    result = mini.minimize(method=MiniMethod)

    if set_inc is None:
        print(lmfit.fit_report(result))
    return result

def plot_result(result, plot_phase=True, plot_ellipse=True):
    """
    Plot minimization results. Two available plots:
    Phase plot, of phase vs. Stokes params, and ellipse plot, Q vs. U.
    """
    Q_0      = result.params['Q_0'].value
    U_0      = result.params['U_0'].value
    inc      = result.params['inc'].value*np.pi/180
    BigOmega = result.params['BigOmega'].value*np.pi/180
    tau_star = result.params['tau_star'].value
    T0       = result.params['T0'].value
    P        = result.params['P'].value

    phi_arr = np.linspace(0, 1, 100)
    pol_phis = (pol_HJDs-T0)/P - ((pol_HJDs-T0)/P).astype(int)+1
    U_fit, Q_fit = pol_fit(phi_arr, U_0, Q_0, BigOmega, tau_star, inc)
    inc1 = 54.87*np.pi/180
    # inc1 = 42.99*np.pi/180
    U_fit1, Q_fit1 = pol_fit(phi_arr, U_0, Q_0, BigOmega, tau_star, inc1)

    if plot_phase:
        plt.errorbar(pol_phis, pol_Us, yerr=pol_sig, fmt='.', color='darkgreen', alpha=0.4, label='U')
        plt.errorbar(pol_phis, pol_Qs, yerr=pol_sig, fmt='.', color='forestgreen', alpha=0.4, label='Q')
        plt.plot(phi_arr, U_fit, color='red', label='U fit')
        # plt.plot(phi_arr, U_fit1, color='blue', label='U fit1')
        # plt.plot(phi_arr, U_fit2, color='orange', label='U fit2')
        plt.plot(phi_arr, Q_fit, color='coral', label='Q fit')
        # plt.plot(phi_arr, Q_fit1, color='blue', label='Q fit1')
        plt.xlabel('Phase')
        plt.ylabel('Stokes Parameter')
        plt.legend()
        plt.tight_layout()
        plt.show()
    if plot_ellipse:
        plt.errorbar(pol_Qs, pol_Us, yerr=pol_sig, xerr=pol_sig, fmt='o', )
        plt.plot(Q_fit, U_fit, color='r', label=round(inc*180/np.pi, 2))
        # plt.plot(Q_fit1, U_fit1, color='g', label=round(inc1*180/np.pi, 2))
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
        result = minimizer(chisqr)
        plot_result(result)
    elif inc_grid:
        # Create grid of inclinations and plot red chi^2
        red_chi2 = []
        phis = []
        for inc in np.arange(5,90,step):
            result = minimizer(chisqr, set_inc=inc)
            red_chi2.append(result.redchi)
            phis.append(inc)        
        plot_chi2(phis, red_chi2, step)

if __name__ == '__main__':
    main()