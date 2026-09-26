import baltic as bt
from baltic import bt_utils

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

import numpy as np

ll=bt.io.load_nexus('MERS.mcc.tree', 'time') ## treeFile here can alternatively be a path to a local file
ll.treeStats() ## report stats about tree

##########
fig = plt.figure(figsize=(10, 10), facecolor='w')
gs = GridSpec(1, 1)

ax = plt.subplot(gs[0])

traitName = 'type'

def partitionFxn(node):
    return node.traits[traitName] != node.parent.traits[traitName]

pt = bt_utils.state_collapse_tree(ll, switchFxn=partitionFxn, keepLast=False, adjustEarlyHeights=True)


host_cmap = bt_utils.make_cmap(['#3B76AF', 'grey', '#E68033'])

convertStateProbs = lambda k: {key: value for key, value in zip(k.traits[f"{traitName}.set"], k.traits[f"{traitName}.set.prob"])}

branchColourFxn = lambda k: host_cmap(1.0) if (k.is_leaf() and k.traits[traitName] != k.parent.traits[traitName]) else host_cmap(convertStateProbs(k)['c'] if 'c' in convertStateProbs(k) else 0.0)
pointColourFxn = lambda k: host_cmap(convertStateProbs(k)['c'] if 'c' in convertStateProbs(k) else 0.0)

padNodes = {k: np.log10(k.traits['size']) for k in pt.get_external()}
psFxn = lambda k: 20 + np.log(k.traits['size']) ** 4

pt.plot_tree(ax, colourFxn=branchColourFxn, padNodes=padNodes, width=4, zorder=100)
pt.plot_points(ax, colourFxn=pointColourFxn, padNodes=padNodes, pointSizeFxn=psFxn, outlineColour='w', zorder=101)

calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01', spacing='yearly')
bt_utils.format_time_grid(ax, calendarTimeline, outputFmtFxn=lambda d: bt_utils.convert_date_format(d, '%Y-%m-%d', '%Y'))

calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01', spacing='monthly')
bt_utils.plot_time_grid(ax, calendarTimeline, edgeColour='none', alpha=0.03, zorder=0)

bt_utils.clean_axes(ax, removeTickLabels='y')

[ax.axvline(yr, color='k', ls='--', lw=0.5, zorder=0) for yr in range(2010, 2016)]

ax.set_xlim(bt_utils.calendar_to_decimal_date('2010-01-01'), bt_utils.calendar_to_decimal_date('2015-10-01'))
ax.set_ylim(-1, pt.ySpan + 1)

ax.tick_params(axis='x', labelsize=22)

ax2 = ax.inset_axes([0.1, 0.1, 0.025, 0.5])
mpl.colorbar.ColorbarBase(ax2, cmap=host_cmap, ticks=np.linspace(0.0, 1.0, 5))
ax2.tick_params(size=5, labelsize=14)
ax2.set_ylabel('posterior probability host is camel', size=14)
ax3 = ax2.twinx()
ax3.set_ylim(0, 1)
ax3.set_yticks([0.0, 1.0])
ax3.set_yticklabels(['human', 'camel'])
ax3.tick_params(size=0, labelsize=18)

plt.savefig('state-collapsed-tree-earliest-adjust.png', bbox_inches='tight')