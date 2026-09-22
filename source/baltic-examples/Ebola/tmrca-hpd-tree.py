import sys

import baltic as bt
from baltic import bt_utils

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

admin2country = {
'Bo': 'SLE', 'Bombali': 'SLE', 'Bonthe': 'SLE', 'Kailahun': 'SLE', 'Kambia': 'SLE', 'Kenema': 'SLE', 'Koinadugu': 'SLE', 'Kono': 'SLE', 'Moyamba': 'SLE', 'PortLoko': 'SLE', 'Pujehun': 'SLE', 'Tonkolili': 'SLE', 'WesternRural': 'SLE', 'WesternUrban': 'SLE', 
'Bomi': 'LBR', 'Bong': 'LBR', 'Gbarpolu': 'LBR', 'GrandBassa': 'LBR', 'GrandCapeMount': 'LBR', 'GrandGedeh': 'LBR', 'GrandKru': 'LBR', 'Lofa': 'LBR', 'Margibi': 'LBR', 'Maryland': 'LBR', 'Montserrado': 'LBR', 'Nimba': 'LBR', 'RiverCess': 'LBR', 'RiverGee': 'LBR', 'Sinoe': 'LBR', 
'Beyla': 'GIN', 'Boffa': 'GIN', 'Boke': 'GIN', 'Conakry': 'GIN', 'Coyah': 'GIN', 'Dabola': 'GIN', 'Dalaba': 'GIN', 'Dinguiraye': 'GIN', 'Dubreka': 'GIN', 'Faranah': 'GIN', 'Forecariah': 'GIN', 'Fria': 'GIN', 'Gaoual': 'GIN', 'Gueckedou': 'GIN', 'Kankan': 'GIN', 'Kerouane': 'GIN', 'Kindia': 'GIN', 'Kissidougou': 'GIN', 'Koubia': 'GIN', 'Koundara': 'GIN', 'Kouroussa': 'GIN', 'Labe': 'GIN', 'Lelouma': 'GIN', 'Lola': 'GIN', 'Macenta': 'GIN', 'Mali': 'GIN', 'Mamou': 'GIN', 'Mandiana': 'GIN', 'Nzerekore': 'GIN', 'Pita': 'GIN', 'Siguiri': 'GIN', 'Telimele': 'GIN', 'Tougue': 'GIN', 'Yamou': 'GIN', 
'Kedougou': 'SEN', 'Salemata': 'SEN', 'Saraya': 'SEN', 'Velingara': 'SEN', 'Tambacounda': 'SEN', 
'Kenieba': 'MLI', 'Kita': 'MLI', 'Kangaba': 'MLI', 'Kati': 'MLI', 'Yanfolila': 'MLI', 
'Gabu': 'GNB', 'Tombali': 'GNB', 
'SanPedro': 'CIV', 'Folon': 'CIV', 'Kabadougou': 'CIV', 'Cavally': 'CIV', 'Tonkpi': 'CIV', 'Bafing': 'CIV'
}

traitName = 'location.states' ## key used to access annotation of interest in tree

colours = {'SLE': '#647397', 'GIN': '#6db69f', 'LBR': '#ea906e'}
colourFxn = lambda k: colours[admin2country[k.traits[traitName]]]


ll=bt.io.load_nexus('Makona_1610_cds_ig.GLM.MCC.tree', 'time') ## treeFile here can alternatively be a path to a local file
ll.treeStats() ## report stats about tree

################# processing
collapseNodes = {}
Nseen = {}

for k in sorted(ll.Objects[1:], key=lambda q: q.absoluteTime): ## skip root because its parent doesn't have a trait
    parTrait, childTrait = k.parent.traits[traitName], k.traits[traitName]
    
    if parTrait != childTrait and parTrait == 'Gueckedou': ## check for all lineages migrating out of Gueckedou

        if childTrait not in Nseen: Nseen[childTrait] = 0
        Nseen[childTrait] += 1 ## remember number of jumps into new location
        
        mark = False
        ## only interested in migrations into certain regions
        if k.absoluteTime < bt_utils.calendar_to_decimal_date('2014-08-01') and childTrait in ['Macenta', 'Kissidougou', 'Lofa', 'Kailahun', 'Conakry', 'Siguiri']:
            mark = True ## this branch will be treated differently when plotting
            sys.stdout.write(f"Branch {k.index} migrating from {parTrait} to {childTrait} with {1 if k.is_leaf() else len(k.leaves)} descendants")
            
            if childTrait == 'Lofa' and k.is_node(): ## need to manually get rid of this early SLE -> LBR migration
                mark = False

        if mark == False: ## branch not marked
            continue

        if Nseen[childTrait] == 1: ## format new name for branch
            if childTrait == 'Conakry':
                newName = f'{childTrait} (GIN)\nGN-1 lineage' ## obscure persistent Conakry lineage
            elif childTrait == 'Kailahun':
                newName = f'{childTrait} (SLE), SL lineages' ## large SLE lineage
            else:
                newName = f'{childTrait} ({admin2country[childTrait]})' ## first time seeing migration into this location
        else:
            newName=f'{childTrait}#{Nseen[childTrait]} ({admin2country[childTrait]})' ## not the first time seeing migration into here
        
        collapseNodes[newName] = k ## remember branch object for collapsing later

for node in collapseNodes: ## iterate over branches designated for collapsing
    if collapseNodes[node].is_node(): ## dealing with node - can collapse into clade
        sys.stdout.write(f"Collapsing node {collapseNodes[node].index} named {node}")
        ll.collapse_subtree_to_clade(cl=collapseNodes[node], givenName=node, widthFunction=lambda k: 1)
    else:
        collapseNodes[node].name = node ## dealing with tip - rename it
################

fig = plt.figure(figsize=(8,12),facecolor='w')
gs = GridSpec(1,1)

ax = plt.subplot(gs[0])

######## plotting
markSupportedNodes = lambda k: k.is_node() and k.traits['posterior'] > 0.50 ## function that targets nodes with >=50% posterior support
GueckedouSubtree = lambda k: (k.is_node() or k.is_leaf()) and k.traits[traitName]=='Gueckedou' ## function that targets branches in Gueckedou
GueckedouTips = lambda k: k.is_leaf() and k.traits[traitName]=='Gueckedou' ## function that targets tips in Gueckedou

ll.plot_tree(ax, targetFxn=GueckedouSubtree, colourFxn=colourFxn, plotClades=False) ## Gueckedou subtree
ll.plot_tree(ax, targetFxn=lambda k: GueckedouSubtree(k)==False, colourFxn=colourFxn, ls=':', plotClades=False) ## parts outside Gueckedou

ll.plot_points(ax, targetFxn=markSupportedNodes, pointSize=25, colour='w', outlineColour=colours['GIN'], outlineSize=45) ## supported node markers
ll.plot_points(ax, targetFxn=GueckedouTips, pointSize=25, colourFxn=colourFxn, outlineSize=50) ## tips in Gueckedou

####### texts
pathEffects = bt_utils.get_path_effects(mainWeight=0.2, outlineWeight=3) ## adds white outline to text
ll.plot_text(ax, targetFxn=lambda k: markSupportedNodes(k) and len(k.leaves) >= 5, 
             xCoordinateFxn=lambda k: k.absoluteTime - 0.005, yCoordinateFxn=lambda k: k.y - 0.25, 
             textContentFxn = lambda k: f"{k.traits['posterior']:.2f}", 
             ha='right', va='top', path_effects=pathEffects) ## add posterior labels for well-supported nodes with at least 5 descendant tips

ll.plot_text(ax, targetFxn=lambda k: k.is_leaflike() and GueckedouTips(k)==False, 
             xCoordinateFxn=lambda k: k.absoluteTime, yCoordinateFxn=lambda k: k.y, 
             textContentFxn = lambda k: k.name, 
             ha='left', va='center', path_effects=pathEffects) ## add posterior labels for well-supported nodes with at least 5 descendant tips

############ node bar
from baltic.bt_utils import plot_node_bar

traitColourDict = {'Macenta': '#75bba5', 'Gueckedou': '#336555'} ## minimal set of traits that need colours if ignoring traits with <0.5% posterior support (otherThres parameter)

plot_node_bar(ax=ax, node=ll.root, traitName=traitName, traitColourDict=traitColourDict, otherThres=0.005, 
              width=0.04, height=23,
              xyFxn=lambda k: (k.absoluteTime - 0.05, k.y - 11.5), connectNode=False, ec='k') ## plot node bar, collapse any trait <0.5% posterior probability into "other" category

###########
rescale = 0.5

GIN_node = ll.get_branches(lambda k: k.is_leaflike() and 'Conakry' in k.name)[0] ## get branch corresponding to the Conakry persistent lineage
bt_utils.plot_tmrca_posterior(ax=ax, tmrcaFile='Makona_1610_cds_ig.GLM.tmrcas', 
                              tmrcaName='Conakry', burnin=10e6, yCoord=-10, fullViolin=False, connectNode=True, node=GIN_node, 
                              kdeWidth=rescale, normalise=False, 
                              violinKwargs={'fc': colours['GIN'], 'alpha': 0.6}, 
                              outlineKwargs={'color': colours['GIN']}, 
                              connectionLineKwargs={'ls': ':', 'color': colours['GIN']}) ## plot TMRCA KDE from file - half of violin, below tree, connected to corresponding node, rescale KDE height without normalising (otherwise all KDEs are same height)

SLE_node = ll.get_branches(lambda k: k.is_leaflike() and 'Kailahun' in k.name)[0]
bt_utils.plot_tmrca_posterior(ax=ax, tmrcaFile='Makona_1610_cds_ig.GLM.tmrcas', 
                              tmrcaName='Kailahun', burnin=10e6, yCoord=-10, fullViolin=False, connectNode=True, node=SLE_node, 
                              kdeWidth=rescale, normalise=False, 
                              violinKwargs={'fc': colours['SLE'], 'alpha': 0.6}, 
                              outlineKwargs={'color': colours['SLE']}, 
                              connectionLineKwargs={'ls': ':', 'color': colours['SLE']})

bt_utils.plot_tmrca_posterior(ax=ax, tmrcaFile='Makona_1610_cds_ig.GLM.tmrcas', 
                              tmrcaName='Root', burnin=10e6, yCoord=-10, fullViolin=False, connectNode=True, node=ll.root, 
                              kdeWidth=rescale, normalise=False, 
                              violinKwargs={'fc': 'k', 'alpha': 0.6}, 
                              outlineKwargs={'color': 'k'}, 
                              connectionLineKwargs={'ls': ':', 'color': 'k'})

########## time grid
calendarTimeline = bt_utils.generate_calendar_timeline('2013-12-01', '2014-09-01')
bt_utils.format_time_grid(ax, calendarTimeline)
bt_utils.plot_time_grid(ax, calendarTimeline, zorder=0)

ax.tick_params(labelsize=14)

bt_utils.clean_axes(ax, hideSpines=['left', 'top', 'right'], removeTickLabels='y')

ax.set_xlim(bt_utils.calendar_to_decimal_date('2013-12-01'), bt_utils.calendar_to_decimal_date('2014-09-01'))
ax.set_ylim(-10, ll.ySpan)

plt.savefig('tmrca-hpd-tree.png', bbox_inches='tight')