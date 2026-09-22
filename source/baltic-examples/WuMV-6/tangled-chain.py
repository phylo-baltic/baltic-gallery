import baltic as bt
from baltic.bt_utils import untangle_trees, clean_axes
from baltic.curonia import plot_tangled_chain

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

###########
segments=['PB1','PB2','PA','NP','gp64','hypothetical','hypothetical2','hypothetical3']

trees={} ## dict
for segment in segments:
    print(segment)
    
    ll=bt.io.load_newick(f"{segment}.rooted.newick", 'divergence') ## treeFile here can alternatively be a path to a local file
    trees[segment] = ll

########
treeList = [trees[seg] for seg in segments]

treeList = untangle_trees(treeList, iterations=3, bidirectional=False)

###########
fig = plt.figure(figsize=(10, 5), facecolor='w')
gs = GridSpec(1, 1, hspace = 0.01, wspace = 0.0)
ax = plt.subplot(gs[0])

plot_tangled_chain(ax, treeList, lw = 8, alpha=0.8, treeSpace=0.01, pointKwargs={'colour': 'w'})

treeList[-1].plot_text(ax, xCoordinateFxn=lambda k: k.x * 1.01, textContentFxn=lambda k: f"{k.name.split('|')[0]}|{k.name.split('|')[-1]}", recomputeCoordinates=False)

###########
for i, tree in enumerate(treeList):
    xs = tree.get_parameter_list('x') ## get all tree x-coordinates (will have accounted for tanglegram coordinates)

    treeMiddle = (max(xs) + min(xs)) / 2 ## get middle of tree
    segName = segments[i]
    if 'hypo' in segName: segName = f"hypo{segName[-1]}" if ('2' in segName or '3' in segName) else 'hypo'
    
    ax.text(treeMiddle, treeList[0].ySpan + 1, segName, size=12, ha='center', va='bottom') ## add segment label

clean_axes(ax)
plt.savefig('tangled-chain.png', bbox_inches='tight')