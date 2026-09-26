import baltic as bt
from baltic import bt_utils
from baltic import curonia

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

#################
ll=bt.io.load_nexus('genome.network.combined.mcc.tree', 'time') ## treeFile here can alternatively be a path to a local file
ll.treeStats() ## report stats about tree
##########

fig = plt.figure(figsize=(10, 8), facecolor='w')
gs = GridSpec(1, 1, hspace=0.0, wspace=0.0)
ax = plt.subplot(gs[0])

ll = ll.reduce_tree([k for k in ll.get_external(onlyLeaves=False) if (k.is_leaf() or (k.is_leaflike() and k.traits['posterior'] >= 0.1))]) ## remove low-probability reassortment edges

ll.sort_branches()

ll = ll._partition_tree(partitionFxn=lambda k: hasattr(k, 'contribution') and k.contribution in ll.Objects) ## label each new subtree originating from a reassortment

for k in ll.Objects:
    states = set([q.traits['partition'] for q in k.get_path_to_root()]) ## traverse from branch to root, check how many unique partitions are seen
    # print(k.index, states, hasattr(k, 'contribution'))
    k.traits['reassortments'] = len(states) - 1
##############

colourCycle = [ "#4c72a5", "#e1c72f", "#d0694a", "#48a365", "#cc79a7", "#77bedb", "#7f6e85", "#ccc197" ]
reassortmentColourFxn = lambda k: colourCycle[k.traits['reassortments'] % len(colourCycle)]

ll.plot_tree(ax, colourFxn=reassortmentColourFxn)

colours = {'SWE': '#237fa4', 'CHN': '#b64949', 'KHM': '#b2822a', 'USA': 'seagreen', 'AUS': '#5e3e75'}

countryColourFxn = lambda k: bt_utils.desaturate(colours[k.name.split('|')[2].split('-')[0]], 0.8)
ll.plot_points(ax, colourFxn=countryColourFxn)
#############

curonia.plot_reticulations(ax=ax, tree=ll, plotSegMatrix=True, segWidth=0.666, segMatrixDist=1.02, 
                           segOrder=['PB1', 'PB2', 'PA', 'NP', 'gp64', 'hypo', 'hypo2', 'hypo3'], 
                           segNames={1: 'PB1', 3: 'PB2', 2: 'PA', 0: 'NP', 4: 'gp64', 7: 'hypo', 5: 'hypo2', 6: 'hypo3'})

nonSwedish = ll.find_MRCA([k for k in ll.get_external() if '|SWE' not in k.name])
bt_utils.plot_tmrca_posterior(ax=ax, 
                              tmrcaFile='genome.network.tmrcas.txt', 
                              tmrcaName='nonSwedish', 
                              burnin=0, 
                              node=nonSwedish, connectNode=True, 
                              kdeWidth=1.8, outlineKwargs={'color': 'k', 'lw': 1})

curonia.plot_height_95hpds(ax, ll, targetFxn=lambda k: k.is_node() and len(k.children) > 1)
##############

ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(3))
ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(1))

calendarTimeline = bt_utils.generate_calendar_timeline('1991-01-01', '2020-02-01', spacing='yearly')
bt_utils.plot_time_grid(ax, [d for i, d in enumerate(calendarTimeline) if i % 2 == 0], alpha=0.03)

ax.set_xlim(1992, 2020)
ax.set_ylim(-1, ll.ySpan + 1)
bt_utils.clean_axes(ax, hideSpines=['left', 'top', 'right'], removeTickLabels='y')

ax.tick_params(labelsize=20)

plt.savefig('reassortment-network.png', bbox_inches='tight')