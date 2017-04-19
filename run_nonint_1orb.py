import pytriqs.utility.mpi as mpi
from pytriqs.archive import HDFArchive
from pytriqs.operators import *
from pytriqs.operators.util.op_struct import set_operator_structure, get_mkind
from pytriqs.operators.util.hamiltonians import h_int_kanamori
from pytriqs.applications.impurity_solvers.cthyb import *
from pytriqs.gf.local import *
from pytriqs.applications.impurity_solvers.cthyb.util import estimate_nfft_buf_size
import numpy as np
from time import time


# Input parameters
beta = 10.0
num_orb = 1
mu = 0.0
epsilon = [0]
V = [np.eye(1)]
spin_names = ("up","dn")
orb_names = range(num_orb)
n_iw = 1024
g_n_l = 25
g2_n_l = 15
g2_n_iw = 10
g2_n_inu = 15
g2_blocks = set([("up","up"),("up","dn")])

mpi.report("Welcome to measure_g2 benchmark.")
gf_struct = set_operator_structure(spin_names,orb_names,True)
mkind = get_mkind(True,None)
H = 0 * n("up", 0) * n("dn", 0)

mpi.report("Constructing the solver...")
S = SolverCore(beta=beta, gf_struct=gf_struct, n_iw=n_iw, n_l=g_n_l)

mpi.report("Preparing the hybridization function...")
delta_w = GfImFreq(indices = orb_names, beta=beta, n_points=n_iw)
delta_w_part = delta_w.copy()
for e, v in zip(epsilon,V):
    delta_w_part << inverse(iOmega_n - e)
    delta_w_part.from_L_G_R(np.transpose(v),delta_w_part,v)
    delta_w += delta_w_part
S.G0_iw << inverse(iOmega_n + mu - delta_w)

mpi.report("Running the simulation for the buffer size estimation...")
p = {}
p["verbosity"] = 1
p["max_time"] = 5*60
p["random_name"] = ""
p["random_seed"] = 123 * mpi.rank + 567
p["length_cycle"] = 20
p["n_warmup_cycles"] = 1000
p["n_cycles"] = 5000
p["use_norm_as_weight"] = False
p["measure_density_matrix"] = False
p["measure_g_tau"] = False
p["measure_g_l"] = False
p["measure_pert_order"] = True
S.solve(h_int=H, **p)
pert_order = S.perturbation_order.copy()

mpi.report("Running two simulations...")
for blockorder, is_first_blockloop in zip(["AABB", "ABBA"], [True, False]):
    p["measure_pert_order"] = is_first_blockloop
    p["measure_g_tau"] = is_first_blockloop
    p["measure_g2_inu"] = True
    p["measure_g2_legendre"] = False
    p["measure_g2_pp"] = True
    p["measure_g2_ph"] = True
    p["measure_g2_block_order"] = blockorder
    p["measure_g2_blocks"] = g2_blocks
    p["measure_g2_n_iw"] = g2_n_iw
    p["measure_g2_n_inu"] = g2_n_inu
    p["measure_g2_n_l"] = g2_n_l
    p["n_warmup_cycles"] = 1000
    p["n_cycles"] = 10000
    p["nfft_buf_sizes"] = estimate_nfft_buf_size(gf_struct, pert_order)
    p["random_seed"] = 123 * mpi.rank + 567
    
    mpi.report("Running the simulation for "+blockorder+"...")
    S.solve(h_int=H, **p)
    
    mpi.report("Saving results...")
    if mpi.is_master_node():
        with HDFArchive("nonint_1orb.h5",'a') as ar:
            ar.create_group(blockorder)
            arbo = ar[blockorder]
            arbo['G2_iw_inu_inup_pp'] = S.G2_iw_inu_inup_pp
            arbo['G2_iw_inu_inup_ph'] = S.G2_iw_inu_inup_ph
            if is_first_blockloop:
                ar['G0_iw'] = S.G0_iw
                ar['G1_tau'] = S.G_tau
                ar['perturbation_order'] = S.perturbation_order
                ar['perturbation_order_total'] = S.perturbation_order_total

