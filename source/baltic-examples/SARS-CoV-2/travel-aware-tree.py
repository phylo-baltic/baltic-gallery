import baltic as bt
from baltic import bt_utils

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")
#################

ll=bt.io.load_nexus('latest_country_mcc.tre', 'time', tipRegex=r'\|([0-9\-]+)(\_[a-z_]+)*$')
ll.treeStats() ## report stats about tree
##########

fig = plt.figure(figsize=(10, 5), facecolor='w')
gs = GridSpec(1, 1, hspace=0.0, wspace=0.0)
ax = plt.subplot(gs[0])

##########
cladeTips = ll.get_external(lambda k: k.name in ['hCoV-19/Germany/BY-RKI-I-060971/2021|EPI_ISL_1566470|2021-03-25', 'hCoV-19/Germany/BY-RKI-I-061070/2021|EPI_ISL_1566566|2021-03-26', 'hCoV-19/Germany/un-RKI-I-078686/2021|EPI_ISL_1643902|2021-04-02', 'hCoV-19/Germany/un-RKI-I-078700/2021|EPI_ISL_1643915|2021-04-01', 'hCoV-19/Germany/BY-RKI-I-063055/2021|EPI_ISL_1568272|2021-03-31', 'hCoV-19/Germany/un-RKI-I-066248/2021|EPI_ISL_1571290|2021-03-29', 'hCoV-19/Germany/un-RKI-I-100187/2021|EPI_ISL_1845926|2021-04-05', 'hCoV-19/Germany/un-RKI-I-100182/2021|EPI_ISL_1845921|2021-04-05', 'hCoV-19/Germany/BY-MVP-000005530/2021|EPI_ISL_2094536|2021-04-01', 'hCoV-19/Switzerland/GE-33576177/2021|EPI_ISL_1369646|2021-03-16', 'hCoV-19/Switzerland/GE-33576177/2021|EPI_ISL_1369646|2021-03-16_ancestor_taxon', 'hCoV-19/Belgium/ULG-16502/2021|EPI_ISL_2685881|2021-06-03', 'hCoV-19/Belgium/rega-6642/2021|EPI_ISL_2424358|2021-03-29', 'hCoV-19/Ghana/WACCBIP-TRA702/2020|EPI_ISL_2285857|2021-04-14', 'hCoV-19/Belgium/ULG-12917/2021|EPI_ISL_1241728|2021-03-01', 'hCoV-19/Belgium/ULG-12917/2021|EPI_ISL_1241728|2021-03-01_ancestor_taxon', 'hCoV-19/France/ARA-210013001901/2021|EPI_ISL_1406653|2021-02-26', 'hCoV-19/France/ARA-210013001901/2021|EPI_ISL_1406653|2021-02-26_ancestor_taxon', 'hCoV-19/Italy/VEN-IZSVe-21RS1435-1_BL/2021|EPI_ISL_2638158|2021-05-12', 'hCoV-19/Italy/VEN-IZSVe-21RS1311-2_BL/2021|EPI_ISL_2448513|2021-05-17'])

ca = ll.find_MRCA(cladeTips)
ll = ll.subtree(ca)

traitName = 'country'
colours = {'Italy': 'skyblue', 'France': 'steelblue', 
           'Cameroon': 'maroon', 'Ghana': 'indianred', 
           'Belgium': 'indigo', 'Germany': 'purple', 'Switzerland': 'pink'}
colourFxn = lambda k: colours[k.traits[traitName]]

ll.plot_tree(ax, colourFxn=colourFxn, connectionType='elbow')
ll.plot_points(ax, targetFxn=lambda k: True, colourFxn=colourFxn, outlineColour='w', outlineSize=30, pointSize=15)

pathEffects = bt_utils.get_path_effects(mainColour='none', outlineWeight=2)
ll.plot_text(ax, colourFxn=colourFxn, textContentFxn=lambda k: k.traits[traitName], size=14, path_effects=pathEffects)

########
bt_utils.clean_axes(ax)

calendarTimeline = bt_utils.generate_calendar_timeline('2021-01-01', '2021-07-01')
bt_utils.format_time_grid(ax, calendarTimeline)
bt_utils.plot_time_grid(ax, calendarTimeline)

ax.tick_params(labelsize=16)
plt.savefig('travel-aware-tree.png', bbox_inches='tight')