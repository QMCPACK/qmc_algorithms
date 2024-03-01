# Wavefunction Functional Forms

## Splines
### Cubic splines
Cubic splines are used in pseudopotentials and the Coulomb potential evaluation

  - [Cubic Splines Basic.ipynb](Wavefunctions/Cubic%20Splines%20Basic.ipynb) - Basics of cubic splines
  - [CubicSplineSolver.ipynb](Wavefunctions/CubicSplineSolver.ipynb) - Derivation and code generation of a cubic spline solver
    - Code generation in QMCPACK: https://github.com/QMCPACK/qmcpack/blob/develop/src/Numerics/codegen/gen_cubic_spline_solver.py
  - [eqn_manip.py](Wavefunctions/eqn_manip.py) - Equation manipuation code to aid derivations
  - [codegen_extras.py](Wavefunctions/codegen_extras.py) - Helper code for code generation

### B-Splines
Bsplines are used to represent periodic single particle orbitals and for Jastrow factors

  - [Explain_Bspline.ipynb](Explain_Bspline.ipynb) - Basics of B-splines


## Single Particle Orbitals
### Gaussian Type Orbitals (GTO)
- [GaussianOrbitals.ipynb](Wavefunctions/GaussianOrbitals.ipynb) - Formulas and normalization
- [gaussian_orbitals.py](Wavefunctions/gaussian_orbitals.py) - Python code for evaluating values and derivatives
- [read_qmcpack.py](Wavefunctions/read_qmcpack.py) - Read basis set information in XML format from QMCPACK input files.
- [MolecularOrbitals.ipynb](Wavefunctions/MolecularOrbitals.ipynb) - Combining the GTOs into molecular orbitals
- [CuspCorrection.ipynb](Wavefunctions/CuspCorrection.ipynb) - Cusp correction to modify GTO-based molecular orbitals to satisfy the cusp condition

### Plane Waves
- [PlaneWaves.ipynb](Wavefunctions/PlaneWaves.ipynb) - Read from the Quantum Espresso HDF format and evaluate
- [read_pw.py](Wavefunctions/read_pw.py) - Standalone Python code to read and evaluate

### Orbital Rotation
- [OrbitalRotation.ipynb](OrbitalRotation.ipynb) - Orbital rotation mixes and optimizes molecular orbitals


## Jastrow factors
- [Pade_Jastrow.ipynb](Wavefunctions/Pade_Jastrow.ipynb) - Simplest form as a Padé approximant
- The section "Bspline for Jastrow" in [Explain_Bspline.ipynb](Wavefunctions/Explain_Bspline.ipynb) - Incorporate boundary conditions for two-body Jastrow.
- [ThreeBodyJastrowPolynomial.ipynb](Wavefunctions/ThreeBodyJastrowPolynomial.ipynb) - Three body polynomial (electron-electron-ion correlation) -
  - [gen_three_body.py](Wavefunctions/gen_three_body.py) - Generate Python or Julia code for three body polynomial of fixed order
