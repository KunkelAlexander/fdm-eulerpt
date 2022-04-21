# Eulerian perturbation code for FDM and CDM 

## About
This code computes the tree- and loop-level matter power spectra and matter bispectra as well as the tree-level trispectrum in Eulerian perturbation theory for fuzzy dark matter (FDM) and cold dark matter (CDM). For a wonderful review on cosmological perturbation theory, see Bernardeau, F. / Colombi, S. / Gaztanaga, E. / Scoccimarro, R.: Large-Scale Structure of the Universe and Cosmological Perturbation Theory.
It expects initial spectra in the form provided by CAMB (https://github.com/cmbant/CAMB) or axionCAMB (https://github.com/dgrin1/axionCAMB), computes the growth linear growth and decay factors in a dark energy universe for both CDM and FDM and computes the spectrum corrections.

## Usage
The folder "cpp" contains the C++-code that uses the CUBA (http://www.feynarts.de/cuba/) library for carrying out numerical integrations. 
The folder "python" contains python code that uses the vegas (https://vegas.readthedocs.io/en/latest/) library for carrying out numerical integrations.
Both codes are largely redundant, implement mostly the same functions and features and were used for cross-checking results. The C++-code is to be preferred for performance reasons. 

## Dependencies
Python:
- numpy, scipy, matplotlib, vegas, numba
C++:
- cuba, gsl, gslwrap, boost
