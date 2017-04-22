import matplotlib, sys, numpy as np, itertools as itt
matplotlib.use("pdf")
from matplotlib import pyplot as plt
from pytriqs.archive import HDFArchive
from pytriqs.gf.local import GfImTime, GfImFreq, BlockGf
from wick import G2ph_0, G2pp_0
from g2plot import G2ConstiwPlot, TraceInuInup, G2iwPlot


n_iw = 10
n_inu = 20

for block, archivename, channel in itt.product([('up','up'), ('up','dn')], sys.argv[1:], ['ph', 'pp']):
    for w in range(2):
        fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2,2)
        plot_kwargs = {'vmin': -1, 'vmax': 1, 'aspect':'equal'}
        for blockstruct, ax in zip(["AABB", "ABBA"], [ax1, ax3]):
            sto = HDFArchive(archivename, 'r')
            g0 = sto["G0_iw"]
            if channel == 'ph':
                g2_0 = G2ph_0(g0, n_iw, n_inu, [], blockstruct)
            elif channel == 'pp':
                g2_0 = G2pp_0(g0, n_iw, n_inu, [], blockstruct)
            g2_0plot = G2ConstiwPlot(g2_0, w)
            g2_0plot.prepare_data(block)
            axhandle = g2_0plot.draw(ax, **plot_kwargs)
            ax.set_title("Exact "+blockstruct)
        for blockstruct, ax in zip(["AABB", "ABBA"], [ax2, ax4]):
            sto = HDFArchive(archivename, 'r')
            g2 = sto[blockstruct]["G2_iw_inu_inup_"+channel]
            g2plot = G2ConstiwPlot(g2, w)
            g2plot.prepare_data(block)
            axhandle = g2plot.draw(ax, **plot_kwargs)
            ax.set_title("CTHYB "+blockstruct)
        outname = archivename[:-3]+"_g2"+channel+"_inuinup_"+block[0]+block[1]+"_w"+str(w)+".pdf"
        plt.savefig(outname)
        print outname+" ready"
        plt.close()

    fig, [ax1, ax2] = plt.subplots(2,1)
    for blockstruct, ax in zip(["AABB", "ABBA"], [ax1, ax2]):
        sto = HDFArchive(archivename, 'r')
        g0 = sto["G0_iw"]
        n_orbs = g0["up"].data.shape[1]
        if channel == 'ph':
            g2_0 = G2ph_0(g0, n_iw, n_inu, [], blockstruct)
        elif channel == 'pp':
            g2_0 = G2pp_0(g0, n_iw, n_inu, [], blockstruct)
        trg2_0 = TraceInuInup(g2_0)
        trg2_0plot = G2iwPlot(trg2_0)
        g2 = sto[blockstruct]["G2_iw_inu_inup_"+channel]
        trg2 = TraceInuInup(g2)
        trg2plot = G2iwPlot(trg2)
        #indices_to_plot = [(block[0],block[1])+indices[0] for indices in g2_0.get_equivalent_indices(block)]
        if n_orbs == 2:
            indices_to_plot = [(block[0],block[1],0,0,0,0), (block[0],block[1],0,0,1,1), (block[0],block[1],0,1,1,0)]
        if n_orbs == 1:
            indices_to_plot = [(block[0],block[1],0,0,0,0)]
        trg2_0plot.draw(ax, True, indices_to_plot, "Exact\_", channel, blockstruct, marker = '+')
        trg2plot.draw(ax, True, indices_to_plot, "CTHYB\_", channel, blockstruct, marker = 'x')
        ax.legend(fontsize = 8)
        ax.set_xlim(left= 0)
    outname = archivename[:-3]+"_g2"+channel+"_iw_"+block[0]+block[1]+".pdf"
    plt.savefig(outname)
    print outname+" ready"
    plt.close()
