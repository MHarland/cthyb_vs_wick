import numpy as np, matplotlib, itertools as itt
from pytriqs.gf.local import Block2Gf, GfImFreqTv4


class G2ConstiwPlot:
    def __init__(self, g2, bosonic_frequency_to_plot = 0):
        self.g2 = g2
        self.n = dict()
        for s, b in g2:
            g2mesh = np.array([w.imag for w in g2[s].mesh.components[0]])
            n_iw0 = np.argwhere(g2mesh == 0)[0, 0]
            self.n[s] = bosonic_frequency_to_plot + n_iw0
        self.mesh_nu1 = None
        self.mesh_nu2 = None
        self.data = None
        self.max_absval = None

    def prepare_data(self, block = ('up','up'), index = (0,0,0,0), part = 'real'):
        i,j,k,l = index
        if part == 'real':
            self.data = self.g2[block].data[self.n[block],:,:,i,j,k,l].real
        else:
            assert part == 'imag', 'part unkown'
            self.data = g2[block].data[self.n[block],:,:,i,j,k,l].imag
        self.mesh_nu1 = [nu.imag for nu in self.g2[block].mesh.components[1]]
        self.mesh_nu2 = [nu.imag for nu in self.g2[block].mesh.components[2]]
        self.max_absval = np.max(abs(self.data))

    def draw(self, ax, n_ticks = 6, **kwargs):
        extent = min(self.mesh_nu1), max(self.mesh_nu1), min(self.mesh_nu2), max(self.mesh_nu2)
        opts = {'origin': 'lower', 'cmap': matplotlib.cm.RdBu, 'aspect': 'auto',
                'interpolation': 'none', 'vmin': -self.max_absval, 'vmax':self.max_absval,
                'extent': extent}
        opts.update(kwargs)
        handle = ax.imshow(self.data.transpose(), **opts)
        return handle

class G2iwPlot:
    def __init__(self, g2_iw):
        self.g2 = g2_iw

    def draw(self, ax, realpart = True, indices = [], labelprefix = '', channel = '', blockstruct = '', **kwargs):
        if len(indices) == 0:
            indices = [i for i in self.g2.all_indices]
        if 'colors' in kwargs.keys():
            assert len(kwargs['colors']) == len(indices), "colors and indices must be of same length"
            colors = kwargs.pop('colors')
        else:
            n_c = len(indices)
            colors = [matplotlib.cm.jet(i/float(max(n_c-1,1))) for i in range(n_c)]
        for (s1, s2, i, j, k, l), color in zip(indices, colors):
            x = [w.imag for w in self.g2[(s1, s2)].mesh]
            if realpart:
                y = self.g2[(s1, s2)].data[:, i, j, k, l].real
            else:
                y = self.g2[(s1, s2)].data[:, i, j, k, l].imag
            ax.plot(x, y, label = "$"+labelprefix+s1+s2+str(i)+str(j)+str(k)+str(l)+"$", color = color, **kwargs)
        ax.set_xlabel("$i\\omega_n$")
        ax.set_ylabel("$G_{"+blockstruct+"}^{("+( str(2) if not channel else channel) +")}(i\\omega_n)$")

class TraceInuInup(Block2Gf):
    def __init__(self, g2_iw_inu_inup):
        g2 = g2_iw_inu_inup
        n1, n2 = g2._Block2Gf__indices1, g2._Block2Gf__indices2
        indices = [i for i in g2.indices]
        g2block0 = g2[indices[0]]
        self.beta = g2block0.mesh.components[0].beta
        for n, m in itt.product(n1, n2):
            assert g2[(n, m)].data.shape[3] == g2[(n, m)].data.shape[4] == g2[(n, m)].data.shape[5] == g2[(n, m)].data.shape[6], 'blockstructure not supported (TODO), blocksizes have to equal each other'
        Block2Gf.__init__(self, n1, n2, [[GfImFreqTv4(g2[(n, m)].mesh.components[0], g2[(n, m)].data.shape[3:]) for m in n2] for n in n1])
        self.perform_trace(g2)

    def perform_trace(self, g2):
        norm = float(self.beta**2) # TODO self.beta**2 ???
        for s, b in g2:
            for g2_inu_iw in b.data.transpose(1,2,0,3,4,5,6):
                for g2_iw in g2_inu_iw:
                    self[s].data[:,:,:] += g2_iw / norm
