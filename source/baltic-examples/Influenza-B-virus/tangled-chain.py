import baltic as bt

from baltic.bt_utils import untangle_trees
from baltic.curonia import plot_tangled_chain

from baltic.bt_utils import clean_axes, desaturate_cmap

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

############
segments=['PB1','PB2','PA','HA','NP','NA','M1','NS1']

trees={} ## dict
for segment in segments:
    print(segment)
    ll=bt.io.load_nexus(f"InfB_{segment}t_ALLs1.mcc.tre", 'time', tipRegex=r'_([0-9\-]+)$') ## load tree
    trees[segment] = ll.collapse_branches(collapseIfFxn=lambda k: k.is_node() and k.traits['posterior'] <= 0.5) ## collapse uncertain nodes

############
treeList = [trees[seg] for seg in segments] ## order trees

treeList = untangle_trees(treeList, iterations=3, bidirectional=False)

#########
fig = plt.figure(figsize=(30, 15), facecolor='w')
gs = GridSpec(1, 1)
ax = plt.subplot(gs[0])

cmap = desaturate_cmap(mpl.cm.Spectral_r, 0.4)
colourDict = {k.name: cmap(k.y/treeList[0].ySpan) for k in treeList[0].get_external()}

plot_tangled_chain(ax, treeList, colourDict=colourDict, treeSpace=10, lw=2, padding=0.3, pointKwargs={'colour': 'w', 'pointSize': 20}, treeKwargs={'colour': 'k', 'zorder': 3}, zorder=1) ## plot trees and tangled lines

plot_tangled_chain(ax, treeList, colourDict={k.name: 'dimgray' for k in treeList[0].get_external()}, treeSpace=10, lw=2.5, padding=0.3, treeKwargs={'colour': 'none'}, zorder=0) ## add outline behind tangled lines

###########
for i, tree in enumerate(treeList):
    xs = tree.get_parameter_list('x') ## get all tree x-coordinates (will have accounted for tanglegram coordinates)

    treeMiddle = (max(xs) + min(xs)) / 2 ## get middle of tree
    ax.text(treeMiddle, treeList[0].ySpan + 10, segments[i], size=20, ha='center', va='bottom') ## add segment label

clean_axes(ax)

plt.savefig('tangled-chain.png', bbox_inches='tight')