import baltic as bt
from baltic import bt_utils

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")
#########

ll = bt.io.load_newick('gp64.rooted.newick', 'divergence') ## treeFile here can alternatively be a path to a local file
ll.treeStats() ## report stats about tree

####### this does plotting
fig = plt.figure(figsize=(10, 10), facecolor='w')
gs = GridSpec(1, 1)

ax = plt.subplot(gs[0])

ll.sort_branches()
ll.plot_tree(ax)
ll.plot_points(ax, pointSize=25, colour='k', outlineColour='w', outlineSize=45)
ll.plot_aligned_tip_labels(ax=ax, fontsize=8, textContentFxn=lambda k: k.name.split('|')[0])

#####
bt_utils.plot_scale_bar(ax=ax, xy=(0.0015, 2), tree=ll, textKwargs={'size': 16}, unitText='substitutions per site')

bt_utils.clean_axes(ax=ax)

ax.tick_params(axis='x', labelsize=22)

ax.set_ylim(-1, ll.ySpan + 1)
plt.savefig('simple-tree.png', bbox_inches='tight')