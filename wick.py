import numpy as np, itertools as itt, sys
from pytriqs.gf.local import Block2Gf, MeshImFreq
from pytriqs.gf.local.multivar import GfImFreq_x_ImFreq_x_ImFreqTv4, MeshImFreqImFreqImFreq


class G2_0(Block2Gf):
    def __init__(self, g1_inu, n_iw, n_inu, blocks_to_calculate = [], blockstructure_to_compare_with = "AABB"):
        """
        g1_inu has to be a BlockGf of GfImFreq
        """
        mb = MeshImFreq(g1_inu.beta, "Boson", n_iw)
        mf = MeshImFreq(g1_inu.beta, "Fermion", n_inu)
        blockmesh = MeshImFreqImFreqImFreq(mb, mf, mf)
        blocks = []
        self.gf2_struct = dict()
        for i in g1_inu.indices:
            blocksR = []
            for j in g1_inu.indices:
                blockindex = (i, j)
                m, n = g1_inu[i].data.shape[1], g1_inu[j].data.shape[1]
                blockshape = [m, m, n, n]
                blocksR.append(GfImFreq_x_ImFreq_x_ImFreqTv4(blockmesh, blockshape))
                self.gf2_struct[blockindex] = (m, n)
            blocks.append(blocksR)
        g1_indices = [i for i in g1_inu.indices]
        Block2Gf.__init__(self, g1_indices, g1_indices, blocks)
        self.beta = g1_inu.beta
        # mesh adjustments for negative frequencies and g1/g2 mesh-offsets
        self.n_iw = len(mb)
        self.n_inu = len(mf)
        self.bosonic_mesh = np.array([w for w in self[blockindex].mesh.components[0]])
        self.fermionic_mesh = np.array([w for w in self[blockindex].mesh.components[1]])
        g1_mesh = np.array([w for w in g1_inu.mesh])
        fermionic_mesh_offset = np.argwhere(g1_mesh.imag == self.fermionic_mesh.imag[0])[0, 0]
        bosonic_mesh_offset = np.argwhere(0 == self.bosonic_mesh.imag)[0, 0]
        self.f_mesh_inds = np.arange(fermionic_mesh_offset, self.n_inu + fermionic_mesh_offset)
        self.b_mesh_inds = np.arange(-bosonic_mesh_offset, self.n_iw - bosonic_mesh_offset)
        if len(blocks_to_calculate) == 0:
            blocks_to_calculate = self.indices
        self.blockstruct = blockstructure_to_compare_with
        contractions = {"direct": [1,0,3,2], "exchange": [1,2,3,0]}
        for blockindex in blocks_to_calculate:
            self.set_block(g1_inu, blockindex, contractions)

    def set_block(self, g1, blockindex, contractions):
        for (i_w, w), (i_nu, nu), (i_nup, nup) in self.iterate_mesh_inds():
            if w == 0:
                self[blockindex].data[i_w, i_nu, i_nup, :,:,:,:] += self.beta * self._g_x_g(g1, blockindex, nu, nup, contractions["direct"])
            if nu == nup:
                self[blockindex].data[i_w, i_nu, i_nup, :,:,:,:] -= self.beta * self._g_x_g(g1, blockindex, nup + w, nup, contractions["exchange"])

    def iterate_mesh_inds(self):
        for (i_w, w), (i_nu, nu), (i_nup, nup) in itt.product(enumerate(self.b_mesh_inds),
                                                              enumerate(self.f_mesh_inds),
                                                              enumerate(self.f_mesh_inds)):
            yield (i_w, w), (i_nu, nu), (i_nup, nup)

    def iterate_blockindices(self, blockindex):
        bi = blockindex
        m, n = self.gf2_struct[blockindex]
        assert self.blockstruct in ["AABB", "ABBA"], "blockstructure not recognized"
        for i, j, k, l in itt.product(range(m), range(m), range(n), range(n)):
            if self.blockstruct == "AABB":
                yield (bi[0], i), (bi[0], j), (bi[1], k), (bi[1], l)
            else:
                yield (bi[0], i), (bi[1], j), (bi[1], k), (bi[0], l)

    def _g_x_g(self, g1, blockindex, f1, f2, index_order):
        bi = blockindex
        m, n = self.gf2_struct[blockindex]
        g2_tmp = np.zeros([m, m, n, n], dtype = complex)
        for superindices in self.iterate_blockindices(blockindex):
            i1, j1, k1, l1 = superindices
            [i2, j2, k2, l2] = [superindices[i] for i in index_order]
            if (i2[0] == j2[0]) and (k2[0] == l2[0]):
                g2_tmp[i1[1], j1[1], k1[1], l1[1]] = g1[i2[0]].data[f1, i2[1], j2[1]] * g1[k2[0]].data[f2, k2[1], l2[1]]
        return g2_tmp

    def get_equivalent_indices(self, block, **kwargs):
        m, n = self[block].data.shape[3], self[block].data.shape[5]
        eclasses = []
        for i,j,k,l in itt.product(range(m), range(m), range(n), range(n)):
            is_new_eclass = True
            for i_e, eclass in enumerate(eclasses):
                for i2,j2,k2,l2 in eclass:
                    if np.allclose(self[block].data[:,:,:,i,j,k,l], self[block].data[:,:,:,i2,j2,k2,l2], **kwargs):
                        is_new_eclass = False
                        eclasses[i_e].append((i,j,k,l))
                        break
                if not is_new_eclass:
                    break
            if is_new_eclass:
                eclasses.append([(i,j,k,l)])
        return eclasses

    def get_zero_indices(self, block, **kwargs):
        m, n = self[block].data.shape[3], self[block].data.shape[5]
        zeros = []
        for i,j,k,l in itt.product(range(m), range(m), range(n), range(n)):
            if np.allclose(self[block].data[:,:,:,i,j,k,l], np.zeros(self[block].data.shape[:3]), **kwargs):
                zeros.append((i,j,k,l))
        return zeros
