import baltic as bt
from baltic import bt_utils
from baltic.bt_utils import calendar_to_decimal_date
from baltic import curonia

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

mpl.use("Agg")

fig = plt.figure(figsize=(10, 5), facecolor='w')
gs = GridSpec(1, 1)

ax = plt.subplot(gs[0])

curonia.plot_skygrid(ax=ax, logFile='camel_skygrid.combined.log', burnin=0, mostRecent=calendar_to_decimal_date('2015-08-27'))


calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01',spacing='yearly')
bt_utils.format_time_grid(ax, calendarTimeline, outputFmtFxn=lambda d: bt_utils.convert_date_format(d, '%Y-%m-%d', '%Y'))

calendarTimeline = bt_utils.generate_calendar_timeline('2010-01-01', '2015-10-01', spacing='yearly')
bt_utils.plot_time_grid(ax, calendarTimeline, edgeColour='none', alpha=0.05, zorder=0)

ax.set_ylabel(r'$N_{e}\tau$', size=20)
ax.set_xlabel('time', size=14)

plt.savefig('skygrid-calendar-time.png', bbox_inches='tight')