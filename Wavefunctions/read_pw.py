from __future__ import print_function

import h5py
import sympy
from sympy.utilities.lambdify import lambdastr
import numpy as np
import math

# Read and evaluate orbitals output from Quantum Espresso using pw2qmcpack.x


class PWOrbitalFile(object):
    def __init__(self):
        self.e, self.de, self.dde = create_eval_funcs()

        self.version = None
        self.number_of_kpoints = 0
        self.number_of_electrons = 0
        self.atom_pos = None
        self.primitive_vectors = None
        self.kspace_basis = None
        self.raw_kpoints = []
        self.kpoints = []
        self.raw_gvectors = None
        self.gvectors = None
        self.coeffs = []
        self.number_of_bands = 0

    def evaluate(self, k_index, band_index, r):
        coeff = self.coeffs[k_index][band_index]
        k = self.kpoints[k_index]
        n = self.gvectors.shape[0]
        return self.e(n, coeff, self.gvectors, k, r)

    def evaluate_all(self, k_index, band_index, r):
        coeff = self.coeffs[k_index][band_index]
        n = self.gvectors.shape[0]
        k = self.kpoints[k_index]
        e = self.e(n, coeff, self.gvectors, k, r)
        #fde =  fake_de(n, coeff, self.gvectors, k, r)
        de = self.de(n, coeff, self.gvectors, k, r)
        dde = self.dde(n, coeff, self.gvectors, k, r)
        return e, de, dde


# Reciprocal lattice vectors
def get_kspace_basis(basis):
    # Volume factor for reciprocal lattice
    a1, a2, a3 = basis
    vol = a1.dot(np.cross(a2, a3))

    pre = 2*math.pi
    b1 = pre*np.cross(a2, a3)/vol
    b2 = pre*np.cross(a3, a1)/vol
    b3 = pre*np.cross(a1, a2)/vol
    return [b1, b2, b3]


def read_pw_file(filename):
    pw = PWOrbitalFile()
    with h5py.File(filename) as f:
        pw.version = f.get('application/version')
        pw.number_of_kpoints = f.get('electrons/number_of_kpoints')[0]
        pw.number_of_electrons = f.get('electrons/number_of_electrons')
        pw.atom_pos = f.get('atoms/positions')
        pw.primitive_vectors = f.get('supercell/primitive_vectors')
        pw.kspace_basis = get_kspace_basis(pw.primitive_vectors)

        pw.raw_kpoints = []
        pw.kpoints = []
        for i in range(pw.number_of_kpoints):
            raw_kpoint = np.array(f.get('electrons/kpoint_%d/reduced_k' % i))
            cell_kpoint = np.array(sum([pw.kspace_basis[j]*raw_kpoint[j] for j in range(3)]))
            pw.raw_kpoints.append(raw_kpoint)
            pw.kpoints.append(cell_kpoint)

        # We are assuming the G vectors are the same for each k-point (?)
        pw.raw_gvectors = np.array(f.get('electrons/kpoint_0/gvectors'))
        pw.gvectors = np.empty(pw.raw_gvectors.shape)
        for i in range(pw.raw_gvectors.shape[0]):
            conv_g = [pw.kspace_basis[j]*pw.raw_gvectors[i, j] for j in range(3)]
            cell_g = np.array(sum(conv_g))
            pw.gvectors[i, :] = cell_g

        pw.coeffs = []
        pw.number_of_bands = f.get('electrons/kpoint_0/spin_0/number_of_states')[0]
        for ik in range(pw.number_of_kpoints):
            coeffs = []
            for ib in range(pw.number_of_bands):
                tmp_c = f.get('electrons/kpoint_%d/spin_0/state_%d/psi_g' % (ik, ib))
                tmp_a = tmp_c[:, 0] + 1j*tmp_c[:, 1]
                coeffs.append(tmp_a)
            pw.coeffs.append(coeffs)

    return pw


# Use symbolic expressions for the orbital sum (Fourier transform)

# For some reason, substitution doesn't work right for Sums.
#  This is a workaround

def sum_subs(s, old, new):
    if isinstance(s, sympy.Sum):
        new_func = s.function.subs(old, new)
        return sympy.Sum(new_func, s.limits)
    else:
        return s.subs(old, new)


# before lambdify, to accept numpy arrays starting at 0
def adjust_sum_limits(e):
    if isinstance(e, sympy.Sum):
        lim = e.limits[0]
        return sympy.Sum(e.function, (lim[0], lim[1]-1, lim[2]-1))
    return e


def create_eval_funcs():
    n = sympy.Symbol('n', integer=True)
    c = sympy.IndexedBase('c')
    g = sympy.IndexedBase('g')
    k = sympy.IndexedBase('k')
    r = sympy.Symbol('r')

    # Later will change r from from Symbol to IndexedBase
    real_r = sympy.IndexedBase('r')

    # Hack to use ':' in array index in lambdify
    all_s = sympy.Symbol(':', integer=True, nonnegative=True)

    i = sympy.Symbol('i', integer=True, nonnegative=True)
    j = sympy.Symbol('j', integer=True, nonnegative=True)

    # Fourier transform
    e = sympy.Sum(c[i]*(sympy.cos((g[i]+k)*r) + sympy.I * sympy.sin((g[i]+k)*r)), (i, 1, n))
    #e = sympy.Sum(c[i]*(sympy.cos(g[i]*r) + 1j * sympy.sin(g[i]*r)), (i, 1, n))

    # gradient
    d_e = e.diff(r)

    # Laplacian
    dd_e = e.diff(r, 2)

    # Mathematical notation often uses a variable 'level of detail' in its
    # representation.  We wish to imitate that here. Otherwise the full details
    # of every symbol need to be carried around and represented from the very
    # start, making some of these expressions visually complicated.
    # Up until now a scalar symbol has been used for r and a 1-D array for g
    # But r is really a length-3 vector, and so is each element of g.
    # The whole of g can be stored as a 2-D array.
    # This step replaces symbols g and r with the more detailed vector versions.

    dot = sympy.Sum((g[i, j] + k[j]) * real_r[j], (j, 0, 2))
    mag = sympy.Sum((g[i, j] + k[j]) * (g[i, j] + k[j]), (j, 0, 2))

    e_vec = sum_subs(e, r*(g[i]+k), dot)
    de_vec = sum_subs(
                      sum_subs(d_e, (g[i]+k)*r, dot), g[i], g[i, all_s])
    dde_vec = sum_subs(
                       sum_subs(dd_e, (g[i]+k)*r, dot), (g[i]+k)**2, mag)

    e_vec = adjust_sum_limits(e_vec)
    de_vec = adjust_sum_limits(de_vec)
    dde_vec = adjust_sum_limits(dde_vec)

    # Use the following to see what Python code is produced by lambdify
    #e_str = lambdastr([n, c, g, k, r] ,e_vec)
    #de_str = lambdastr([n, c, g, k, r], de_vec)
    #dde_str = lambdastr([n, c, g, k, r], dde_vec)
    #print('e lambdastr = ', e_str)
    #print('de lambdastr = ', de_str)
    #print('dde lambdastr = ', dde_str)

    e2 = sympy.lambdify([n, c, g, k, r], e_vec)
    de2 = sympy.lambdify([n, c, g, k, r], de_vec)
    dde2 = sympy.lambdify([n, c, g, k, r], dde_vec)

    # could apply CSE to make evaluating all of them faster

    return e2, de2, dde2


# Expanded version of the output of lambdastr, for debugging
def fake_de(n, c, g, k, r):
    s = 0.0
    for i in range(0, n):
        tmp1 = sum(r[j]*(g[i, j]+k[j]) for j in range(0, 3))
        sin1 = -math.sin(tmp1)
        #cos1 = sympy.I*math.cos(tmp1)
        cos1 = 1j*math.cos(tmp1)
        #print(i,'sin1',sin1,cos1)
        #print('c',c[i])
        #tmp2 = sin1*(g[i,:] + k[:])
        #tmp3 = cos1*(g[i,:] + k[:])
        tmp4 = complex(cos1, sin1)*(g[i, :]+k[:])

        #print(tmp2, tmp3)
        s += (tmp4)*c[i]
    return s
    #sum( (-math.sin((sum(r[j]*g[i, j] for j in range(0, 2+1))))*g[i, :] + sympy.I*math.cos((sum(r[j]*g[i, j] for j in range(0, 2+1))))*g[i, :])*c[i] for i in range(0, n - 1+1))


def test_eval_funcs():
    e, de, dde = create_eval_funcs()

    cc = np.array([complex(1, 0)])
    gg = np.array([[1., 0., 0.]])
    kk = np.array([0., 0., 0.])
    print('shape = ', cc.shape)
    rr = np.array([1., 0., 0.])

    out = e(cc.shape[0], cc, gg, kk, rr)
    print(out)
    d_out = de(cc.shape[0], cc, gg, kk, rr)
    print('grad = ', d_out)

    dd_out = dde(cc.shape[0], cc, gg, kk, rr)
    print('lap = ', dd_out)

if __name__ == '__main__':
    pw = read_pw_file('LiH-arb.pwscf.h5')
    #print (pw.gvectors)
    #print (pw.coeffs[0][0])

    #test_eval_funcs()

    r = np.array([0.0, 0.0, 0.0])
    #v = pw.evaluate(0, 1, r)
    #print(' v = ', v)
    for i in range(36):
        #x = 3.55*i/36 - 2*3.55/12
        x = 3.55*i/36
        r = np.array([x, 0.0, 0.0])
        v, dv, ddv = pw.evaluate_all(0, 0, r)
        print (x, v, dv[0], ddv)
