import matplotlib.pyplot as plt 
import numpy as np 
import os
import pandas 
import matplotlib.lines as mlines
from scipy.interpolate import interp1d
import scipy.special as sc 
import scipy 

def getFilesInDirectory(directory_path, prefix = None):
    directory = os.fsencode(directory_path)
    files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".dat") or filename.endswith(".data"):
            if prefix is not None:
                if not filename.startswith(prefix):
                    continue 
            files.append(filename)
    return files

masses = [1e-21, 1e-22, 1e-23]
standard_files = ['fc_cdm.dat', 'fc_m21.dat', 'fc_m22.dat', 'fc_m23.dat']
standard_labels = ["CDM", r"$m = 10^{-21}$eV", r"$m = 10^{-22}$eV", r"$m = 10^{-23}$eV"]
colours = ["C0", "C1", "C2", "C3"]

dpi = 600
size=(3.54 * 1.5, 3.54)

plt.rcParams["font.size"] = 8
plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = 'Ubuntu'
#plt.rcParams['font.monospace'] = 'Ubuntu Mono'
#plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 12

a0 = 0.01
z0 = 99
a1 = 1
zmean = 0.9

#Quantum jeans scale as function of scale factor and mass
#Equation (18) from Li 2020
def li_kj(a, m, omega_m = 0.315903, h = 0.7):
  return h*44.7*(6*a*omega_m/0.3)**(0.25) * (100/70*m/(1e-22))**0.5  #Mpc^-1

#For Lambda CDM
scale_factor, comoving_distance = np.loadtxt("splines/a2com.dat", unpack = True)
redshift = 1/scale_factor - 1
z2com = interp1d(redshift, comoving_distance, kind='cubic')

def q0(z0  = 0.9, beta = 1.5):
    q0m = (z0/beta) * sc.gamma(3/beta)
    return 1/q0m 

def q(z, z0  = 0.9, beta = 1.5):
    return q0(z0, beta) * (z/z0)**2 * np.exp(-(z/z0)**beta)

def k2t(z, k):
    com = z2com(z)
    lam = (2*np.pi)/k
    theta = np.arctan(lam/com)
    return theta


def k2ti(k):
    def integrand(z):
        return k2t(z, k)
    zl = 0.01 
    zh = 98
    return scipy.integrate.quad(integrand, zl, zh)[0]/(zh - zl)


def k2l(z, k):
    return np.pi / k2t(z, k)

def k2li(k):
    return np.pi / k2ti(k) 
    
def lj(m):
    return 0.1 * k2l(zmean, li_kj(a0, m, h = 1))

#plt.figure(figsize=size, dpi = dpi)
#plt.ylabel(r"matter trispectrum $T_{\delta}(k, k, k, k)$")
#plt.xlabel(r"momentum $k$ in $h$/Mpc")
#
#path = "../../C/data/trispectrum/equilateral/"
#
#plots = []
#
#p = mlines.Line2D([0], [0], color='steelblue', ls='--',lw=0, label='Tree')
#plots.append(p)
#
#
#for i, file in enumerate(standard_files):
#    k, T = np.loadtxt(path + file, unpack = True, skiprows=1)
#    p, = plt.loglog(k, T, label=standard_labels[i], c=colours[i])
#    if i > 0:
#        plt.axvline(li_kj(a0, masses[i-1]), c=colours[i], ls = "dotted")
#    plots.append(p)
#
#
#
#plt.xlim(1e-3, 1000)
#plt.ylim(1e-24, 1e16)
#leg = plt.legend(handles = plots, loc="lower left")
#leg.get_frame().set_linewidth(0.0)
#plt.savefig("../../figures/c_equilateral_matter_trispectrum.pdf", bbox_inches='tight')
#plt.show()



configs = ["r1=20_r2=10"]
titles = [r"for $k_1$ = $0.2$ $h$/Mpc and $k_2$ = $0.1$ $h$/Mpc", r"for $k_1$ = $20$ $h$/Mpc and $k_2$ = $10$ $h$/Mpc"]


fig, ax = plt.subplots(figsize=size, dpi=dpi)

xticks = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
xlabels = ["$0$", "$\pi/4$", "$\pi/2$", "$3\pi/4$", "$\pi$"]


plots = []

p = mlines.Line2D([0], [0], color='steelblue', ls='--',lw=0, label='Tree')
plots.append(p)
for i, config in enumerate(configs):
    path = "data/bispectrum/angular/tree/"
    files = ['fc_cdm_'+config+'.dat', 'fc_m21_'+config+'.dat', 'fc_m22_'+config+'.dat', 'fc_m23_'+config+'.dat']
    labels = ["CDM", "$m = 10^{-21}$eV", "$m = 10^{-22}$eV", "$m = 10^{-23}$eV"]
    for j, file in enumerate(files):
        theta, bt, btr = np.loadtxt(path + file, skiprows = 1, unpack = True)
        p, = ax.plot(theta, btr, label=labels[j], c = colours[j])
        plots.append(p)

    #path = "data/bispectrum/angular/loop/"
    #files = ['cdm_'+config+'.dat', 'fdm_m21_'+config+'.dat', 'fdm_m22_'+config+'.dat', 'fdm_m23_'+config+'.dat']
    #labels = ["Loop CDM", "Loop $m = 10^{-21}$eV", "Loop $m = 10^{-22}$eV", "Loop $m = 10^{-23}$eV"]#, "Loop $m = 10^{-21}$eV", , "Loop $m = 10^{-23}$eV"]
#
    #for j, file in enumerate(files):
    #    theta, btr, bnlr, sdev = np.loadtxt(path + file, skiprows = 1, unpack = True)
    #    plt.plot(theta, btr + bnlr, label=labels[j], c = colours[j])
##
    #    plt.fill_between(theta, btr + bnlr + sdev, btr + bnlr - sdev, color = colours[j], alpha = 0.5)

    ax.set_xticks(xticks, xlabels)

leg = ax.legend(handles = plots, loc="lower left")
leg.get_frame().set_linewidth(0.0)
ax.set_ylabel("reduced matter bispectrum $Q_{\delta}(k_1, k_2, k_3)$ ")
ax.set_xlabel(r"angle $\theta$")
plt.subplots_adjust(wspace=0)
plt.savefig(f"../../figures/angular_matter_bispectrum_with23.pdf", bbox_inches='tight')
plt.show()
