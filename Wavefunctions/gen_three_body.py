from sympy import *

from sympy.printing.cxxcode import CXX11CodePrinter
from sympy.printing.julia import JuliaCodePrinter
from sympy.printing.pycode import PythonCodePrinter


#
# Three-body Jastrow code generated in Julia or Python
# Output is for fixed sizes of the polynomial expansion
#

def gen_three_body():

    ri = Symbol('r_i')
    rj = Symbol('r_j')
    rij = Symbol('r_ij')

    C = Symbol('C')
    L = Symbol('L')
    gamma = IndexedBase('gamma')
    r = IndexedBase('r')
    l = Symbol('l',integer=True)
    m = Symbol('m',integer=True)
    n = Symbol('n',integer=True)
    N = Symbol('N',integer=True)
    N_ee = Symbol("N_ee",integer=True)
    N_en = Symbol("N_en",integer=True)

    f = (ri - L)**C * (rj -L)**C * Sum(Sum(Sum(gamma[l,m,n]*ri**l *rj**m*rij**n,(l,0,N_en)),(n,0,N_en)),(m,0,N_ee))


    # Concrete values for the expansion of the above sum
    NN_ee = 3
    NN_en = 3

    ff = f.subs(N_en, NN_en).subs(N_ee, NN_ee).doit()
    #print(ff)

    # Constraints on values of gamma

    # Indices are l,m,n
    #  l : e1_N
    #  m : e2_n
    #  n : e1_e2

    # ---------------------------------------------------
    # Symmetric under electron interchange (swap l and m)
    # ---------------------------------------------------

    # Generate substitutions

    sym_subs = {}
    for i1 in range(NN_en+1):
        for i2 in range(i1):
            for i3 in range(NN_ee+1):
                sym_subs[gamma[i2,i1,i3]] = gamma[i1,i2,i3]

    #print(sym_subs)

    # -----------
    # No e-e cusp
    # -----------

    ff_ee = diff(ff, rij).subs(rij,0).subs(rj,ri).subs(sym_subs)
    #print(ff_ee)

    # remove the (ri-L)**C part.
    ff_ee2 = ff_ee.args[1]

    #  Collect powers of ri
    ff_ee3 = collect(expand(ff_ee2), ri)
    #print(ff_ee3)
    # For the expression to be zero for arbitrary ri, each coefficient must be zero separately

    pt_ee = poly(ff_ee3, ri)
    cf_ee = pt_ee.all_coeffs()
    #print(cf_ee2)

    ee_soln = solve(cf_ee)
    print('e-e constraints')
    print(ee_soln)
    print()


    # -----------
    # No e-n cusp
    # -----------

    ff_en = diff(ff,ri).subs(ri, 0).subs(rij, rj)

    ff_en2 = simplify(expand(ff_en))
    #print(ff_en2)

    # remove the (-L)**(C-1) * (rj - L)**C part
    ff_en3 = ff_en2.args[2]
    #print(ff_en3)

    ff_en4 = ff_en3.subs(sym_subs).subs(ee_soln)

    # For the expression to be zero for arbitrary ri, each coefficient must be zero separately

    pt_en = poly(ff_en4, rj)
    cf_en = pt_en.all_coeffs()
    print('e-n constraint equations')
    print(cf_en)

    #en_soln = solve(cf_en)

    en_gamma = {a for a in ff_en4.free_symbols if type(a) is Indexed}
    print('en_gamma = ',en_gamma)
    en_soln = linsolve(cf_en, en_gamma)
    print('e-n solution')
    print(en_soln)
    print('-------')

    en_soln_idx = 0

    # Sometimes {C:0, L:0} is the first solution.  Don't want that one.
    #if len(en_soln) > 1:
    #    if C in en_soln[0].keys():
    #        en_soln_idx = 1
    #  Attempts to add constraints to C to avoid that solution never worked
    #  as expected

    #en_soln2 = en_soln[en_soln_idx]
    en_soln2 = None
    for tmp_en in en_soln:
        en_soln2 = {g:v for g,v in zip(en_gamma, tmp_en)}

    print('e-n constraints')
    print(en_soln2)
    print('-------')

    fout = ff.subs(sym_subs).subs(ee_soln).subs(en_soln2)

    print('Final value')
    print(fout)
    print()
    free_gamma = {a for a in fout.free_symbols if type(a) is Indexed}
    print('Number of free gamma:', len(free_gamma))
    print('Free gamma: ',free_gamma)

    # Replace indexing with variable names
    #gamma_subs = {}
    #for gamma_indexed in free_gamma:
    #    suffix = ''.join([str(j) for j in gamma_indexed.args[1:]])
    #    gamma_name = 'g' + suffix
    #    gamma_subs[gamma_indexed] = gamma_name
    #fout = fout.subs(gamma[1,1,0], Symbol('g110'))

    # Replace 3-dim index with 1-dim contiguous
    gamma_subs = {}
    gbase = IndexedBase('g')
    for idx,gamma_indexed in enumerate(free_gamma):
        gamma_subs[gamma_indexed] = gbase[idx+1]


    fout = fout.subs(gamma_subs)


    if True:
        JC = JuliaCodePrinter(settings={'inline':False})
        s = JC.doprint(fout)
        print('Julia code')
        print(s)

    if False:
        PC = PythonCodePrinter(settings={'inline':False})
        s = PC.doprint(fout)
        print('Python code')
        print(s)


if __name__ == '__main__':
   gen_three_body()
