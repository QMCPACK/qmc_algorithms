"""
Perform Ewald sums for Coulomb interactions of ions and electrons in a lattice.
"""
from __future__ import print_function

import numpy as np
import math


# Real space lattice basis
def get_rspace_basis(a):
    """
    Return a 3D real-space basis
    Parameters:
      a -- lattice constant
    """
    a1p = [1.0, 0.0, 0.0]
    a2p = [0.0, 1.0, 0.0]
    a3p = [0.0, 0.0, 1.0]
    a1 = np.array(a1p)*a
    a2 = np.array(a2p)*a
    a3 = np.array(a3p)*a
    return [a1, a2, a3]


# Reciprocal lattice vectors
def get_kspace_basis(basis):
    # Volume factor for reciprocal lattice
    a1, a2, a3 = basis
    vol = a1.dot(np.cross(a2, a3))

    b1 = 2*math.pi*np.cross(a2, a3)/vol
    b2 = 2*math.pi*np.cross(a3, a1)/vol
    b3 = 2*math.pi*np.cross(a1, a2)/vol
    return [b1, b2, b3]


def rspace_sum(rs, qs, basis, kappa, rlim):
    """
    Real space part of the sum
    Parameters:
       rs -- list of particle positions
       qs -- list of particle charges
       basis -- real space basis
       kappa -- splitting parameter
       rlim -- size of lattice (one side of a cube of points)
    """
    rspace_sum = 0.0
    for i in range(len(rs)):
        for j in range(len(rs)):
            q = qs[i]*qs[j]
            r = rs[j] - rs[i]
            for n1 in range(-rlim+1, rlim):
                for n2 in range(-rlim+1, rlim):
                    for n3 in range(-rlim+1, rlim):
                        if i == j and n1 == 0 and n2 == 0 and n3 == 0:
                            continue
                        lat = n1*basis[0] + n2*basis[1] + n3*basis[2]
                        d = r + lat
                        rd = math.sqrt(d.dot(d))
                        rspace_sum += q * math.erfc(kappa*rd)/rd
    return rspace_sum


def kspace_sum(rs, qs, rbasis, kappa, klim):
    """
    Reciprocal space part of the sum
    Parameters:
       rs -- list of particle positions
       qs -- list of particle charges
       rbasis -- reciprocal space basis
       kappa -- splitting parameter
       klim -- size of lattice (one side of a cube of points)
    """
    a1, a2, a3 = basis
    vol = a1.dot(np.cross(a2, a3))
    kspace_sum = 0.0
    for i in range(len(rs)):
        for j in range(len(rs)):
            q = qs[i] * qs[j]
            r = rs[j] - rs[i]
            for n1 in range(-klim+1, klim):
                for n2 in range(-klim+1, klim):
                    for n3 in range(-klim+1, klim):
                        if n1 == 0 and n2 == 0 and n3 == 0:
                            continue
                        rlat = n1*rbasis[0] + n2*rbasis[1] + n3*rbasis[2]
                        kr = rlat.dot(r)
                        k2 = rlat.dot(rlat)
                        fac = math.exp(-k2/(4*kappa**2))/k2
                        kspace_sum += q*fac*math.cos(kr)
    return kspace_sum*4*math.pi/vol


def ewald_sum(rs, qs, basis, kbasis, kappa):
    """
    Perform Ewald sum.
    Parameters:
        rs -- list of particle positions
        qs -- list of charged
        basis -- real space basis
        kbasis -- reciprocal space basis
        kappa -- splitting parameter
    """

    total_q2 = sum([q*q for q in qs])
    total_q = sum(qs)
    a1, a2, a3 = basis
    vol = a1.dot(np.cross(a2, a3))

    old_total = 0.0
    conv_tol = 1e-12  # Convergence test

    print_convergence = False
    if print_convergence:
        print('#lattice size, total, relative difference')
    for lim in range(2, 12):
        rval = 0.5*rspace_sum(rs, qs, basis, kappa, lim)
        kval = 0.5*kspace_sum(rs, qs, kbasis, kappa, lim)

        # Self energy
        se = -total_q2*kappa/math.sqrt(math.pi)

        # Background term.  Non-zero for charged systems.  Useful for
        # summing pieces of systems (just the ions or just the electrons).
        bg = -0.5 * total_q**2 * (math.pi/kappa**2/vol)

        total = rval + kval + se + bg

        rel_diff = 0.5*(total-old_total)/(total+old_total)
        if print_convergence:
            print(lim, rval+kval+se, abs(rel_diff))
        if abs(rel_diff) < conv_tol:
            break
        old_total = total

    if abs(rel_diff) > conv_tol:
        print("not converged!")

    return total


# Particle Positions for BCC H

# Ions
#r0 = np.array([0.0, 0.0, 0.0])
#ri1 = np.array([1.0, 1.0, 1.0])*1.88972614
#a = 3.77945227
# Electrons
#re1 = np.array([0.5, 0.0, 0.0])
#re2 = np.array([0.0, 0.5, 0.0])

#qs = [1.0, 1.0, -1.0, -1.0]
#rs = [r0, ri1, re1, re2]


# Particle positions for simple cubic H
# Ion
r0 = np.array([0.0, 0.0, 0.0])
a = 1.0
# Electron
re1 = np.array([0.5, 0.0, 0.0])
qs = [1.0, -1.0]
rs = [r0, re1]

kappa = 2.0
basis = get_rspace_basis(a)
kbasis = get_kspace_basis(basis)
s = ewald_sum(rs, qs, basis, kbasis, kappa)
print(s)
