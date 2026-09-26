import baltic as bt
from baltic.curonia import plot_gradient_clade_tree
from baltic import bt_utils

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

#################
ll=bt.io.load_nexus('MERS.mcc.tree', 'time') ## treeFile here can alternatively be a path to a local file
ll.treeStats() ## report stats about tree

ll.sort_branches()
############

fig = plt.figure(figsize=(10, 15), facecolor='w')
gs = GridSpec(1, 1)
ax = plt.subplot(gs[0])

chooseClades = lambda k: k.is_node() and k!=ll.root and all([ch.is_node() for ch in k.children]) and all([10 <= len(ch.leaves) <= 40 for ch in k.children]) ## internal nodes that aren't root with nodes as children with each having between 10 and 40 descendant tips
colourFxn = lambda k: 'orange' if k.traits['type'] == 'c' else 'steelblue'
designatedClades = ll.get_internal(chooseClades)

plot_gradient_clade_tree(ax=ax, tree=ll, designatedNodes=designatedClades, colourFxn=colourFxn, tipLen=1.0, treeKwargs={'width': 2}, cladeBranchKwargs={'lw': 2}, outline=True, outlineKwargs={'ls': '--', 'lw': 1})

ll.plot_points(ax, pointSize=15, colourFxn=colourFxn)

##########
calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01', spacing='yearly')
bt_utils.format_time_grid(ax, calendarTimeline, outputFmtFxn=lambda d: bt_utils.convert_date_format(d, '%Y-%m-%d', '%Y'))

calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01', spacing='monthly')
bt_utils.plot_time_grid(ax, calendarTimeline, edgeColour='none', alpha=0.03, zorder=0)

bt_utils.clean_axes(ax)
ax.tick_params(axis='x', labelsize=22)

[ax.axvline(yr, color = 'k', ls = '--', lw = 0.5, zorder = 0) for yr in range(2010, 2016)] ## add lines for years

ax.set_xlim(bt_utils.calendar_to_decimal_date('2010-01-01'), bt_utils.calendar_to_decimal_date('2015-10-01'))
ax.set_ylim(-1, ll.ySpan+1)

plt.savefig('gradient-clade.png', bbox_inches='tight')