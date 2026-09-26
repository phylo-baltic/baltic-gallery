import baltic as bt
from baltic import bt_utils
from baltic import curonia

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")
#########

# url = 'https://nextstrain.org/charon/getDataset?prefix=nipah/all'
# ll, meta = bt.io.load_JSON(url, 'divergence') ## this import from URL also works

ll, meta = bt.io.load_JSON('nipah_all.json', 'divergence')

####### this does plotting
fig = plt.figure(figsize=(20, 15), facecolor='w')
gs = GridSpec(1, 2, width_ratios=[1, 3], wspace=0.05)

ax = plt.subplot(gs[0])
#########

import cartopy
import cartopy.crs as ccrs

proj = ccrs.NearsidePerspective(central_longitude=91,central_latitude=20,satellite_height=8000000)
ax2 = plt.subplot(gs[1], projection=proj)

##########
scale='50m'
water='#e8eced'
land='#848E86'

ax2.add_feature(cartopy.feature.LAKES.with_scale(scale), facecolor=water, zorder=0)
ax2.add_feature(cartopy.feature.OCEAN.with_scale(scale), facecolor=water, edgecolor='none', zorder=0)
ax2.add_feature(cartopy.feature.LAND.with_scale(scale), facecolor=land, edgecolor='none', zorder=1)
ax2.add_feature(cartopy.feature.BORDERS.with_scale(scale), edgecolor='w', lw=1, zorder=1)
ax2.gridlines(color='k', linestyle='--', alpha=0.2)
ax2.set_global()

##########
ll.sort_branches(descending=False)

cladeColours = {'I': 'royalblue', 'II': 'seagreen', 'Ia': 'skyblue', 'Ib': 'orange'}
colourFxn = lambda k: cladeColours[k.traits['clade_membership']]

ll.plot_tree(ax, colourFxn=colourFxn, autoSort=False) ## note that padNodes is now provided to all plotting functions - tree, points and text
ll.plot_points(ax, colourFxn=colourFxn, zorder=4)

textContentFxn = lambda k: f"{k.name}|{k.traits.get('country', '')}|{k.traits['date']}"
textEffects = bt_utils.get_path_effects()
ll.plot_aligned_tip_labels(ax=ax, textContentFxn=textContentFxn, colourFxn=colourFxn, path_effects=textEffects, size=8)

######## extract country coordinates, count lineages per location
lonlat = {}

for resolution in meta['geo_resolutions']: ## iterate over resolutions in auspice JSON
    if resolution['key'] == 'country': ## found country
        for country in resolution['demes']: ## iterate over entries
            lonlat[country] = (resolution['demes'][country]['longitude'], resolution['demes'][country]['latitude']) ## extract lat and lon

import numpy as np
tipCoordinates = {}
lineageCounts = {}
for k in ll.get_external():
    country = k.traits.get('country', None)
    tipCoordinates[k.name] = lonlat[country][::-1] if country else (np.nan, np.nan)
    
    if country:
        lineage = k.traits['clade_membership']
        if country not in lineageCounts: lineageCounts[country] = {'I': 0, 'II': 0, 'Ia': 0, 'Ib': 0}
        
        lineageCounts[country][lineage] += 1

#### this bit plots piecharts (plotting is done via Wedges because cartopy has issues with ax.pie
from matplotlib.patches import Wedge

for country in lineageCounts: ## iterate over countries
    lon, lat = lonlat[country]
    linCounts = [lineageCounts[country][clade] for clade in cladeColours]
    
    total = sum(linCounts) ## get total number of lineages in location at this time

    if total < 0.00001:
        continue
        
    probs = np.array([c / total for c in linCounts]) ## grab serotype lineage counts for location
    angles = probs * 360.0

    cs = [cladeColours[clade] for clade in cladeColours] ## serotype colours
    
    scale_factor = 0.4   # adjust for map size
    
    R_total = scale_factor * np.sqrt(total)
    
    theta_start = 0.0
    for ang, color in zip(angles, cs):
        theta_end = theta_start + ang

        w = Wedge(
            center=(lon, lat),
            r=R_total,
            theta1=theta_start,
            theta2=theta_end,
            facecolor=color,
            edgecolor="k",
            linewidth=2,
            transform=ccrs.PlateCarree(),
            zorder=3,
            clip_on=False
        )
        ax2.add_patch(w)

        theta_start = theta_end

### connect tree to map
curonia.connect_tree_to_map(treeAx=ax, mapAx=ax2, tree=ll, tipCoordinates=tipCoordinates, destinationProjection=proj, colourFxn=colourFxn, shoulderPositionFxn=lambda t: ll.treeHeight * 2, zorder=2)

##### aesthetics
bt_utils.plot_scale_bar(ax=ax, xy=(0.0005, 95), L=0.005, style='fancy', fancyWidth=50, tree=ll, textKwargs={'size': 12})
bt_utils.clean_axes(ax=ax)
ax.tick_params(axis='x', labelsize=22)
ax.set_ylim(-1, ll.ySpan + 1)

plt.savefig('nextstrain-nipah-map-piecharts.png', bbox_inches='tight')