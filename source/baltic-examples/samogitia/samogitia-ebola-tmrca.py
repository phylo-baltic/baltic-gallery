import baltic as bt
from baltic import samogitia
from baltic import bt_utils

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")
#############

treesFile = 'Makona_1610_cds_ig.100.trees' ## input
outputFile = 'Makona_1610_cds_ig.100.tmrcas.log.txt' ## output

## this dict defines a pair of TMRCAs to be extracted from the posterior set of trees, defined by 2 or more descendants
tmrcaDict = {'SLE-tmrca': ['EBOV|G3889.1|KR105205|SLE|Kenema|2014-06-19', 'EBOV|G3724|KM233053|SLE|Kailahun|2014-06-05', 'EBOV|G3707|KM233049|SLE|Kailahun|2014-05-31'], 
             'GIN-tmrca': ['EBOV|Gueckedou-C05|KJ660348|GIN|Gueckedou|2014-03-19', 'EBOV|EM_COY_2015_017574||GIN|Boke|2015-06-10']}

samogitia.process_posterior_trees(treesPath=treesFile, processFxn=samogitia.tmrca_worker, outputPath=outputFile, tipNames=tmrcaDict, burnin=0)

##########
fig = plt.figure(figsize=(10, 5), facecolor='w')
gs = GridSpec(1, 1)

ax = plt.subplot(gs[0])

bt_utils.plot_tmrca_posterior(ax=ax, tmrcaFile='Makona_1610_cds_ig.100.tmrcas.log.txt', tmrcaName='SLE-tmrca', fullViolin=False, normalise=False, violinKwargs={'fc': 'royalblue', 'alpha': 0.6}, outlineKwargs={'color': 'royalblue'}, burnin=0)
bt_utils.plot_tmrca_posterior(ax=ax, tmrcaFile='Makona_1610_cds_ig.100.tmrcas.log.txt', tmrcaName='GIN-tmrca', fullViolin=False, normalise=False, violinKwargs={'fc': 'seagreen', 'alpha': 0.6}, outlineKwargs={'color': 'seagreen'}, burnin=0)

######### aesthetics
calendarTimeline = bt_utils.generate_calendar_timeline('2013-10-01', '2014-06-01')
bt_utils.format_time_grid(ax, calendarTimeline)
bt_utils.plot_time_grid(ax, calendarTimeline)
ax.tick_params(labelsize=22)

bt_utils.clean_axes(ax, hideSpines=['left', 'top', 'right'], removeTickLabels='y')
ax.set_ylabel('density', size=20)

ax.set_ylim(bottom = 0)
plt.savefig('samogitia-ebola-tmrca.png', bbox_inches='tight')