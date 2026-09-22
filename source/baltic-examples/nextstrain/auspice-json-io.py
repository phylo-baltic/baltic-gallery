import baltic as bt
from baltic import bt_utils
import json

ll = bt.io.load_nexus('./../MERS-CoV/MERS.mcc.tree', 'time') ## import nexus MCC tree
ll.treeStats()

ll.sort_branches()
auspiceJSON = ll.to_auspice_json() ## convert to auspice JSON

outFile = open('mers-json.json', 'w')
json.dump(auspiceJSON, fp=outFile, indent=4) ## save JSON to file
outFile.close()
##########

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

mpl.use("Agg")
#########

ll, meta = bt.io.load_JSON('mers-json.json', 'time') ## note that importing auspice JSONs results in a tuple - the baltic representation of the tree and a meta dict
ll.treeStats() ## report stats about tree

####### this does plotting
fig = plt.figure(figsize=(10, 15), facecolor='w')
gs = GridSpec(1, 1)

ax = plt.subplot(gs[0])

traitName = 'type'

host_cmap = bt_utils.make_cmap(['#3B76AF', 'grey', '#E68033'])
getCamelProb = lambda k: k.traits[f"{traitName}_confidence"]['c'] if 'c' in k.traits[f"{traitName}_confidence"] else 0.0 ## grab probability of host == camel

branchColourFxn = lambda k: host_cmap(1.0) if (k.is_leaf() and k.traits[traitName] != k.parent.traits[traitName]) else host_cmap(getCamelProb(k))
pointColourFxn = lambda k: host_cmap(getCamelProb(k))

ll.sort_branches()
ll.plot_tree(ax, colourFxn=branchColourFxn)
ll.plot_points(ax, pointSize=15, colourFxn=pointColourFxn, outlineColour='w', outlineSize=25)

########

calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01',spacing='yearly')
bt_utils.format_time_grid(ax, calendarTimeline, outputFmtFxn=lambda d: bt_utils.convert_date_format(d, '%Y-%m-%d', '%Y'))

calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01', spacing='monthly')
bt_utils.plot_time_grid(ax, calendarTimeline, edgeColour='none', alpha=0.03, zorder=0)

bt_utils.clean_axes(ax, removeTickLabels='y')

[ax.axvline(yr, color='k', ls='--', lw=0.5, zorder=0) for yr in range(2010, 2016)]

ax.set_xlim(bt_utils.calendar_to_decimal_date('2010-01-01'), bt_utils.calendar_to_decimal_date('2015-10-01'))
ax.set_ylim(-1, ll.ySpan + 1)

ax.tick_params(axis='x', labelsize=22)

#####
ax2 = ax.inset_axes([0.1, 0.02, 0.025, 0.3])
mpl.colorbar.ColorbarBase(ax2, cmap=host_cmap, ticks=np.linspace(0.0, 1.0, 5))
ax2.tick_params(size=5, labelsize=14)
ax2.set_ylabel('posterior probability host is camel', size=15)
ax3 = ax2.twinx()
ax3.set_ylim(0, 1)
ax3.set_yticks([0.0, 1.0])
ax3.set_yticklabels(['human', 'camel'])
ax3.tick_params(size=0, labelsize=18)

plt.savefig('auspice-json-io.png', bbox_inches='tight')