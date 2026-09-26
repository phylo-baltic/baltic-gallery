import baltic as bt
from baltic import bt_utils
from baltic import curonia

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

import numpy as np

ll = bt.io.load_nexus('MERS.mcc.tree', 'time') ## treeFile here can alternatively be a path to a local file
ll.treeStats() ## report stats about tree

####### this does plotting
fig = plt.figure(figsize=(10, 12), facecolor='w')
gs = GridSpec(1, 1)

ax = plt.subplot(gs[0])

traitName = 'type'

host_cmap = bt_utils.make_cmap(['#3B76AF', 'grey', '#E68033'])
convertStateProbs = lambda k: {key: value for key, value in zip(k.traits[f"{traitName}.set"], k.traits[f"{traitName}.set.prob"])}

branchColourFxn = lambda k: host_cmap(1.0) if (k.is_leaf() and k.traits[traitName] != k.parent.traits[traitName]) else host_cmap(convertStateProbs(k)['c'] if 'c' in convertStateProbs(k) else 0.0)
pointColourFxn = lambda k: host_cmap(convertStateProbs(k)['c'] if 'c' in convertStateProbs(k) else 0.0)

ll.sort_branches()
ll.plot_tree(ax, colourFxn=branchColourFxn, width=2)
ll.plot_points(ax, pointSize=15, colourFxn=pointColourFxn, outlineColour='w', outlineSize=25)

curonia.plot_height_95hpds(ax=ax, tree=ll, targetFxn=lambda k: k.is_node() and len(k.leaves) > 20, width=3)

calendarTimeline = bt_utils.generate_calendar_timeline('2008-01-01', '2015-10-01',spacing='yearly')
bt_utils.format_time_grid(ax, calendarTimeline, outputFmtFxn=lambda d: bt_utils.convert_date_format(d, '%Y-%m-%d', '%Y'))

calendarTimeline = bt_utils.generate_calendar_timeline('2008-01-01', '2015-10-01', spacing='monthly')
bt_utils.plot_time_grid(ax, calendarTimeline, edgeColour='none', alpha=0.03, zorder=0)

bt_utils.clean_axes(ax, removeTickLabels='y')

[ax.axvline(yr, color='k', ls='--', lw=0.5, zorder=0) for yr in range(2008, 2016)]

ax.set_xlim(bt_utils.calendar_to_decimal_date('2008-01-01'), bt_utils.calendar_to_decimal_date('2015-10-01'))
ax.set_ylim(-1, ll.ySpan + 1)

ax.tick_params(axis='x', labelsize=22)

plt.savefig('height-95HPD-bars.png', bbox_inches='tight')