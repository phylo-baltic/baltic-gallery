import baltic as bt
from baltic import bt_utils
from baltic import curonia

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

import numpy as np

ll = bt.io.load_nexus('Makona_1610_cds_ig.GLM.MCC.tree', 'time')
ll.treeStats() ## report stats about tree

##########

fig = plt.figure(figsize=(10, 10), facecolor='w')
gs = GridSpec(1, 1)

ax = plt.subplot(gs[0])

admin2country = {
'Bo': 'SLE', 'Bombali': 'SLE', 'Bonthe': 'SLE', 'Kailahun': 'SLE', 'Kambia': 'SLE', 'Kenema': 'SLE', 'Koinadugu': 'SLE', 'Kono': 'SLE', 'Moyamba': 'SLE', 'PortLoko': 'SLE', 'Pujehun': 'SLE', 'Tonkolili': 'SLE', 'WesternRural': 'SLE', 'WesternUrban': 'SLE', 
'Bomi': 'LBR', 'Bong': 'LBR', 'Gbarpolu': 'LBR', 'GrandBassa': 'LBR', 'GrandCapeMount': 'LBR', 'GrandGedeh': 'LBR', 'GrandKru': 'LBR', 'Lofa': 'LBR', 'Margibi': 'LBR', 'Maryland': 'LBR', 'Montserrado': 'LBR', 'Nimba': 'LBR', 'RiverCess': 'LBR', 'RiverGee': 'LBR', 'Sinoe': 'LBR', 
'Beyla': 'GIN', 'Boffa': 'GIN', 'Boke': 'GIN', 'Conakry': 'GIN', 'Coyah': 'GIN', 'Dabola': 'GIN', 'Dalaba': 'GIN', 'Dinguiraye': 'GIN', 'Dubreka': 'GIN', 'Faranah': 'GIN', 'Forecariah': 'GIN', 'Fria': 'GIN', 'Gaoual': 'GIN', 'Gueckedou': 'GIN', 'Kankan': 'GIN', 'Kerouane': 'GIN', 'Kindia': 'GIN', 'Kissidougou': 'GIN', 'Koubia': 'GIN', 'Koundara': 'GIN', 'Kouroussa': 'GIN', 'Labe': 'GIN', 'Lelouma': 'GIN', 'Lola': 'GIN', 'Macenta': 'GIN', 'Mali': 'GIN', 'Mamou': 'GIN', 'Mandiana': 'GIN', 'Nzerekore': 'GIN', 'Pita': 'GIN', 'Siguiri': 'GIN', 'Telimele': 'GIN', 'Tougue': 'GIN', 'Yamou': 'GIN', 
'Kedougou': 'SEN', 'Salemata': 'SEN', 'Saraya': 'SEN', 'Velingara': 'SEN', 'Tambacounda': 'SEN', 
'Kenieba': 'MLI', 'Kita': 'MLI', 'Kangaba': 'MLI', 'Kati': 'MLI', 'Yanfolila': 'MLI', 
'Gabu': 'GNB', 'Tombali': 'GNB', 
'SanPedro': 'CIV', 'Folon': 'CIV', 'Kabadougou': 'CIV', 'Cavally': 'CIV', 'Tonkpi': 'CIV', 'Bafing': 'CIV'
} ## this dict translates location names in the tree to the corresponding country

traitName = 'location.states' ## key used to access annotation of interest in tree

colours = {'SLE': '#647397', 'GIN': '#6db69f', 'LBR': '#ea906e'}
colourFxn = lambda k: colours[admin2country[k.traits[traitName]]]

def partitionFxn(node):
    return admin2country[node.traits[traitName]] != admin2country[node.parent.traits[traitName]] ## evaluates to True on branches that migrate to another country

pt = bt_utils.state_collapse_tree(ll, switchFxn=partitionFxn, keepLast=False, adjustEarlyHeights=True)
pt.treeStats()

padNodes = {k: np.log10(k.traits['size']) for k in pt.get_external()} ## this dict connects each tip in state-collapsed tree to a custom amount of y-axis space around that tip
psFxn = lambda k: 10 + np.log(k.traits['size']) ** 4 ## this function scales tip circle sizes in state-collapsed using the 'size' value added to remaining tips' trait dicts 

pt.plot_tree(ax, colourFxn=colourFxn, padNodes=padNodes, width=4, zorder=100)
pt.plot_points(ax, colourFxn=colourFxn, padNodes=padNodes, pointSizeFxn=psFxn, zorder=101)

curonia.plot_height_95hpds(ax=ax, tree=pt, targetFxn=lambda k: k.is_node() and len(k.children) > 1, width=1.2, facecolor='#d5d822', edgecolor='#9d9e42', alpha=0.5) ## this adds node 95% HPD height bars, only adding them to nodes with at least 2 children (because this is a state-collapsed tree that contains all of the original branches some of which are now multitype branches with single descendants) 

bt_utils.clean_axes(ax)

calendarTimeline = bt_utils.generate_calendar_timeline('2014-01-01', '2015-11-01')
bt_utils.format_time_grid(ax, calendarTimeline)
bt_utils.plot_time_grid(ax, calendarTimeline)

ax.tick_params(labelsize=14)

ax.set_xlim(bt_utils.calendar_to_decimal_date('2014-01-01'), bt_utils.calendar_to_decimal_date('2015-12-01'))
ax.set_ylim(0, pt.ySpan)

plt.savefig('height-95HPD-bars.png', bbox_inches='tight')