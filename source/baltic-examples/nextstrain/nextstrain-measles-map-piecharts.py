import baltic as bt
from baltic import bt_utils
from baltic import curonia

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

mpl.use("Agg")
#########

# url = 'https://nextstrain.org/charon/getDataset?prefix=measles/genome'
# ll, meta = bt.io.load_JSON(url, 'time') ## this import from URL also works

ll, meta = bt.io.load_JSON('measles_genome.json', 'time')

####### this does plotting
fig = plt.figure(figsize=(35, 15), facecolor='w')
gs = GridSpec(1, 2, width_ratios=[1, 6], wspace=0.0)

ax = plt.subplot(gs[0])
#########

import cartopy
import cartopy.crs as ccrs

proj = ccrs.Robinson()
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

############
key = 'clade'

cladeColours = {'A': 'lightgray', 
                'B3': 'deepskyblue', 
                'C1': 'lightgray', 'C2': 'lightgray', 
                'D3': 'lightgray', 'D4': 'seagreen', 'D5': 'lightgray', 'D6': 'lightgray', 'D7': 'lightgray', 'D8': 'royalblue', 'D9': 'lightgray', 'D11': 'lightgray', 
                'G2': 'lightgray', 'G3': 'lightgray', 
                'H1': 'firebrick'}

colourFxn = lambda k: cladeColours[k.traits[key]] if (key in k.traits and k.traits[key] in cladeColours) else 'lightgray'

########## simplify tree by just showing lineage relationships
ll.plot_tree(ax, width=2, zorder=100)
ll.plot_points(ax, colourFxn=colourFxn, outlineColour='w', zorder=101)
######## extract country coordinates, count lineages per location
lonlat = {}

for resolution in meta['geo_resolutions']: ## iterate over resolutions in auspice JSON
    if resolution['key'] == 'country': ## found country
        for country in resolution['demes']: ## iterate over entries
            lonlat[country] = (resolution['demes'][country]['longitude'], resolution['demes'][country]['latitude']) ## extract lat and lon


tipCoordinates = {}
lineageCounts = {}
for k in ll.get_external():
    country = k.traits.get('country', None)
    tipCoordinates[k.name] = lonlat[country][::-1] if country else (np.nan, np.nan)
    
    if country:
        lineage = k.traits[key] if key in k.traits else 'other'
        if country not in lineageCounts: lineageCounts[country] = {}
        
        if lineage not in cladeColours:
            lineage = 'other'
        if lineage not in lineageCounts[country]: lineageCounts[country][lineage] = 0
        
        lineageCounts[country][lineage] += 1 ## count lineage at location

#### this bit plots piecharts (plotting is done via Wedges because cartopy has issues with ax.pie
from matplotlib.patches import Wedge

for country in lineageCounts: ## iterate over countries
    lon, lat = lonlat[country]
    linCounts = [lineageCounts[country][clade] if clade in lineageCounts[country] else 0.0 for clade in cladeColours]
    
    total = sum(linCounts) ## get total number of lineages in location at this time

    if total < 0.00001:
        continue
        
    probs = np.array([c / total for c in linCounts]) ## grab serotype lineage counts for location
    angles = probs * 360.0

    cs = [cladeColours[clade] for clade in cladeColours] ## serotype colours
    
    # scale_factor = 0.4   # adjust for map size
    scale_factor = 50000
    R_total = scale_factor * np.sqrt(total)
    
    theta_start = 0.0
    for ang, color in zip(angles, cs):
        if ang == 0.0:
            continue
        
        theta_end = theta_start + ang

        tLon, tLat = proj.transform_point(lon, lat, src_crs=ccrs.PlateCarree())
        
        w = Wedge(
            center=(tLon, tLat),
            r=R_total,
            theta1=theta_start,
            theta2=theta_end,
            facecolor=color,
            edgecolor="k",
            linewidth=2,
            # transform=ccrs.PlateCarree(), ## using flat piecharts, not projected onto planet's surface
            zorder=3,
            clip_on=False
        )
        ax2.add_patch(w)

        theta_start = theta_end

##### aesthetics
ax2.text(0.01, 0.95, bt_utils.convert_date_format(meta['updated'], '%Y-%m-%d', '%Y %b %d'), ha='left', va='top', size=30, transform=ax2.transAxes) ## time stamp visualisation

bt_utils.plot_scale_bar(ax=ax, xy=(1850, 200), style='fancy', fancyWidth=0.1, tree=ll, textKwargs={'size': 22})
bt_utils.clean_axes(ax=ax)
ax.set_ylim(-10, ll.ySpan + 10)

plt.savefig('nextstrain-measles-map-piecharts.png', bbox_inches='tight')