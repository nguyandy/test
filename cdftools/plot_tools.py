#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
cdftools/plot_tools.py
Tools for plotting with matplotlib
'''
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from cdftools.windrose import WindroseAxes
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    from matplotlib.image import NonUniformImage
    import metpy.calc as mpcalc
    from metpy.plots import SkewT, Hodograph
    from metpy.units import units
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import cmocean
except ImportError:
    import sys
    sys.stderr.write("Please install matplotlib, metpy, \nand pyproj to use this module\n")
    raise


class OceanPlots(object):
    """
    Plotting tools for making Oceanographic plots
    """
    axis_font_default = {
        'fontname': 'DejaVu Sans',
        'size': '14',
        'color': 'black'
    }

    title_font_default = {
        'fontname': 'DejaVu Sans',
        'size': '16',
        'color': 'black',
        'verticalalignment': 'bottom'
    }

    def get_log_tick_labels(self, xmin, xmax):
        '''
        Manually construct logarithmic x-axis ticks and labels

        :param float xmin: minimum value for xaxis
        :param float xmax: maximum value for xaxis
        '''

        def get_ticks(tickindex, tick_num, tick_lab, tol):
            '''
            Recursively calculate the spacing between the logarithmic ticks
            Returns the new tick index, value and label for the axis

            :param int tickindex - index of the tick
            :param list tick_num - List of the tick values
            :param list tick_lab - List of the tick labels
            :param float tol - Acceptable tolerance to allow a subsequent tick
            '''
            eps = np.spacing(1)
            while ((((10 ** (tol + 1)) - tick_num[tickindex]) > eps) and
                   (tick_num[tickindex] < xmax)):
                tickindex += 1
                tick_num.append(tick_num[-1] + 10 ** tol)
                # Determine whether to add a label at this particular tick location
                if (((10 ** (tol + 1)) - tick_num[tickindex]) < eps):
                    tick_lab.append(str(tick_num[tickindex]))
                else:
                    tick_lab.append('')
            return tickindex, tick_num, tick_lab

        # Create some empty lists
        tick_num = []   # Empty list to build ticks
        tick_lab = []   # Empty list to build tick labels
        tol = np.floor(np.log10(xmin))
        factors = 10 ** (np.fix(-tol))
        tick_num.append(np.round(xmin * factors) / factors)
        tickindex = 0
        tick_lab.append(str(xmin))  # First tick label on far left side

        # Loop through tick marks until the last value (righlim)
        while tick_num[tickindex] < xmax:
            tickindex, tick_num, tick_lab = get_ticks(tickindex,
                                                      tick_num,
                                                      tick_lab,
                                                      tol)
            tol += 1
        tick_num[-1] = xmax
        tick_num[0] = xmin
        tick_lab[-1] = str(xmax)
        return tick_lab, tick_num

    def get_time_label(self, ax, dates):
        '''
        Returns a custom date axis format based on time span

        :param plt.ax ax - A matplotlib axis
        :param np.array x - Array of dates
        '''
        def format_func(x, pos=None):
            x = mdates.num2date(x)
            if pos == 0:
                fmt = '%Y-%m-%d %H:%M'
            else:
                fmt = '%H:%M'
            label = x.strftime(fmt)
            # label = label.rstrip("0")
            # label = label.rstrip(".")
            return label
        day_delta = (max(dates) - min(dates)).days

        if day_delta < 1:
            ax.xaxis.set_major_formatter(FuncFormatter(format_func))
        elif day_delta < 5:
            # major = mdates.DayLocator()   # every day
            major = mdates.HourLocator(interval=24)  # every hour
            formt = mdates.DateFormatter('%m-%d %H:%M')
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(formt)
            # ax.xaxis.set_minor_locator(minor)
        elif day_delta < 20:
            major = mdates.DayLocator(interval=2)
            formt = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(formt)
            # major = mdates.AutoDateLocator(interval_multiples=False)
        elif day_delta < 31:
            major = mdates.DayLocator(interval=4)
            formt = mdates.DateFormatter('%Y-%m-%d')
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(formt)
        else:
            # pass
            major = mdates.AutoDateLocator()
            formt = mdates.AutoDateFormatter(major, defaultfmt=u'%Y-%m-%d')
            formt.scaled[1.0] = '%Y-%m-%d'
            formt.scaled[30] = '%Y-%m'
            formt.scaled[1. / 24.] = '%Y-%m-%d %H:%M:%S'
            # formt.scaled[1./(24.*60.)] = FuncFormatter(format_func)
            ax.xaxis.set_major_locator(major)
            ax.xaxis.set_major_formatter(formt)
        return ax

    def plot_time_series(self, fig, ax, x, y, fill=False, title='', ylabel='',
                         title_font={}, axis_font={}, tick_font={}, **kwargs):
        '''
        Returns a timeseries plot
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default

        h = ax.plot(x, y, **kwargs)

        self.get_time_label(ax, x)

        fig.autofmt_xdate()

        if ylabel:
            ax.set_ylabel(ylabel, **axis_font)
        if title:
            ax.set_title(title, **title_font)
        if ylabel.lower() in ('degree', 'degrees', 'degrees_north', 'degrees_east', 'degrees_south', 'degrees_west'):
            ax.set_ylim([0, 360])
        ax.grid(True)
        if fill:
            miny = min(ax.get_ylim())
            ax.fill_between(x, y, miny + 1e-7, facecolor = h[0].get_color(), alpha=0.15)
        # plt.subplots_adjust(top=0.85)
        if tick_font:
            ax.tick_params(**tick_font)


        # plt.tight_layout()

    def plot_stacked_time_series(self, fig, ax, x, y, z, title='', ylabel='',
                                 cbar_title='', title_font={}, axis_font={}, tick_font = {}, invert=True,
                                 **kwargs):
        '''
        Returns a stacked time series plot.

        For example,
        xaxis: time
        yaxis: depth
        zaxis: ocean parameter (currents, temp, density, etc)
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default
        z = np.ma.array(z, mask=np.isnan(z))
        h = plt.pcolormesh(x, y, z, shading='gouraud', **kwargs)
        # h = plt.pcolormesh(x, y, z, **kwargs)
        if ylabel:
            ax.set_ylabel(ylabel, **axis_font)
        if title:
            ax.set_title(title, **title_font)
        plt.axis('tight')
        ax.xaxis_date()
        date_list = mdates.num2date(x)
        self.get_time_label(ax, date_list)
        fig.autofmt_xdate()
        if invert:
            ax.invert_yaxis()
        cbar = plt.colorbar(h)
        if cbar_title:
            cbar.ax.set_ylabel(cbar_title)
        ax.grid(True)
        if tick_font:
            ax.tick_params(**tick_font)

    def plot_stacked_time_series_image(self, fig, ax, x, y, z, tide=None, title='', ylabel='',
                                       cbar_title='', title_font={}, axis_font={}, tick_font = {}, invert=True,
                                       **kwargs):
        '''
        Returns a stacked time series that uses NonUniformImage with regualrly spaced ydata from
        a linear interpolation. Designed to support FRF ADCP data.
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default
        # z = np.ma.array(z, mask=np.isnan(z))

        h = NonUniformImage(ax, interpolation='bilinear', extent=(min(x), max(x), min(y), max(y)),
                            cmap=plt.cm.jet)
        h.set_data(x, y, z)
        ax.images.append(h)
        ax.set_xlim(min(x), max(x))
        ax.set_ylim(min(y), max(y))
        # h = plt.pcolormesh(x, y, z, shading='gouraud', **kwargs)
        # h = plt.pcolormesh(x, y, z, **kwargs)
        if ylabel:
            ax.set_ylabel(ylabel, **axis_font)
        if title:
            ax.set_title(title, **title_font)
        # plt.axis('tight')
        ax.xaxis_date()
        date_list = mdates.num2date(x)
        self.get_time_label(ax, date_list)
        fig.autofmt_xdate()
        if invert:
            ax.invert_yaxis()
        cbar = plt.colorbar(h)
        if cbar_title:
            cbar.ax.set_ylabel(cbar_title)
        ax.grid(True)
        if tick_font:
            ax.tick_params(**tick_font)
        if tide is not None:
            plt.plot(x, tide, 'k', linewidth=1.5)
        plt.show()

    def plot_profile(self, ax, x, y, xlim=[], ylim=[], xlabel='', ylabel='',
                     axis_font={}, tick_font = {}, **kwargs):
        '''
        Returns a single profile plot with depth
        '''

        if not axis_font:
            axis_font = self.axis_font_default

        plt.plot(ax, x, y, **kwargs)
        if xlabel:
            ax.set_xlabel(xlabel, **axis_font)
        if ylabel:
            ax.set_ylabel(ylabel, **axis_font)
        if tick_font:
            ax.tick_params(**tick_font)
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        ax.invert_yaxis()
        ax.xaxis.set_label_position('top')  # this moves the label to the top
        ax.xaxis.set_ticks_position('top')
        ax.grid(True)

    def plot_histogram(self, ax, x, bins, title='', xlabel='', title_font={},
                       axis_font={}, tick_font={}, **kwargs):
        '''
        Returns a histogram plot in number of occurrences
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default
        x = x[~np.isnan(x)]
        plt.hist(ax, x, bins, grid='y', **kwargs)
        if xlabel:
            ax.set_xlabel(xlabel, labelpad=10, **axis_font)
        if tick_font:
            ax.tick_params(**tick_font)
        ax.set_ylabel('No. of Occurrences', **axis_font)
        ax.set_title(title, **title_font)
        # ax.grid(True)

    def plot_2d_histogram(self, ax, x, y, bins=200, title='', xlabel='', ylabel='',
                          title_font={}, axis_font={}, tick_font={}, **kwargs):
        '''
        Returns a 2d histogram plot
        '''
        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default

        # Estimate the 2D histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=bins)

        # H needs to be rotated and flipped
        H = np.rot90(H)
        H = np.flipud(H)

        # Mask zeros
        Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

        # Plot 2D histogram using pcolor
        cmap = plt.cm.jet
        plt.pcolormesh(xedges, yedges, Hmasked, cmap=cmap)
        if xlabel:
            ax.set_xlabel(xlabel, **axis_font)
        if ylabel:
            ax.set_ylabel(ylabel, **axis_font)
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('No. of Occurrences')

    def plot_scatter(self, fig, ax, x, y, title='', xlabel='', ylabel='',
                     title_font={}, axis_font={}, tick_font={}, **kwargs):
        '''
        Returns an x, y scatter plot
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default

        plt.scatter(ax, x, y, **kwargs)
        if xlabel:
            ax.set_xlabel(xlabel, labelpad=10, **axis_font)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=10, **axis_font)
        if tick_font:
            ax.tick_params(**tick_font)
        ax.set_title(title, **title_font)
        ax.grid(True)
        ax.set_aspect(1. / ax.get_data_ratio())  # make axes square

    def plot_3d_scatter(self, fig, ax, x, y, z, title='', xlabel='', ylabel='', zlabel='',
                        title_font={}, axis_font={}, tick_font={}):
        '''
        Returns a 3D scatter plot (think colored scatter plot). x, y and z must all be the same dimension.
        This plot is useful to see the dependence of z variable as a funciton of x and y.
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default

        cmap = plt.cm.jet
        h = plt.scatter(x, y, c=z, cmap=cmap)
        ax.set_aspect(1. / ax.get_data_ratio())  # make axes square
        cbar = plt.colorbar(h, orientation='vertical', aspect=30, shrink=0.9, fraction=0.046, pad=0.04)

        if xlabel:
            ax.set_xlabel(xlabel, labelpad=10, **axis_font)
        if ylabel:
            ax.set_ylabel(ylabel, labelpad=10, **axis_font)
        if zlabel:
            cbar.ax.set_ylabel(zlabel, labelpad=10, **axis_font)
        if tick_font:
            ax.tick_params(**tick_font)
        if title:
            ax.set_title(title, **title_font)
        ax.grid(True)

    def single_rose_axes(self):
        '''
        A quick way to create new windrose axes
        '''
        fig = plt.figure(figsize=(8, 6), facecolor='w', edgecolor='w')
        rect = [0.15, 0.15, 0.7, 0.7]
        ax = WindroseAxes(fig, rect)
        fig.add_axes(ax)
        return fig, ax

    def double_rose_axes(self):
        '''
        Create 1x2 subplot of wind rose axes for comparison
        '''
        fig = plt.figure(figsize=(16, 6), facecolor='w', edgecolor='w')
        ax1 = fig.add_subplot(1, 2, 1, projection='windrose')
        ax2 = fig.add_subplot(1, 2, 2, projection='windrose')
        return fig, ax1, ax2

    def set_legend(self, ax, label=''):
        '''
        Adjust the legend box.
        '''
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        l = ax.legend(title=label, loc='lower left', bbox_to_anchor=(1, 0))
        plt.setp(l.get_texts(), fontsize=10)

    def get_rose_bins(self, magnitude):
        '''
        Return a set of sensible bins for the rose plots
        Beware! This assumes we're plotting winds, wave, currents, so bins are checked
        accordingly. (IE don't worry about negative values)

        :param np.array magnitude - Array of magnitude values to inspect for bins
        '''

        maxval = np.nanmax(magnitude)
        if maxval < 0.6:
            return [0, 0.1, 0.2, 0.3, 0.4]
        elif maxval < 1.5:
            return [0, 0.2, 0.4, 0.6, 0.8]
        elif maxval < 2.5:
            return [0, 0.5, 1.0, 1.5, 2]
        elif maxval < 6:
            return [0, 1, 2, 3, 4]
        elif maxval < 11:
            return [0, 2, 4, 6, 8]
        elif maxval < 24:
            return [0, 4, 8, 12, 16]
        elif maxval < 60:
            return [0, 10, 20, 30, 40]
        elif maxval < 90:
            return [0, 15, 30, 45, 60]
        return [0, 20, 40, 60, 80]


    def plot_rose(self, magnitude, direction, bins=None, nsector=16,
                  title='', title_font={}, legend_title='', normed=True,
                  colormap=cmocean.cm.haline, blowto=False):
        '''
        Return a rose plot (winds, waves, currents)

        This function allows the user to set any colormap from cmocean
        and the direction convention
        '''

        if not title_font:
            title_font = self.title_font_default

        fig, ax = self.single_rose_axes()
        magnitude = magnitude[~np.isnan(magnitude)]
        direction = direction[~np.isnan(direction)]
        cmap = colormap
        if bins is None:
            bins = self.get_rose_bins(magnitude)
        ax.bar(direction, magnitude, bins=bins, normed=normed, cmap=cmap,
               opening=0.9, edgecolor='white', nsector=nsector, blowto=blowto)

        self.set_legend(ax, legend_title)
        ax.set_title(title, **title_font)

        return fig, ax

    def plot_rose_comparison(self, magnitude, direction, magnitude2, direction2, nsector=16,
                             title1='', title2='', title_font={}, legend_title='', normed=True,
                             colormap=cmocean.cm.haline, blowto=False):
        '''
        Returns figure and axes of a side by side rose plot comparison
        '''

        if not title_font:
            title_font = self.title_font_default

        fig, ax1, ax2 = self.double_rose_axes()

        magnitude = magnitude[~np.isnan(magnitude)]
        direction = direction[~np.isnan(direction)]

        magnitude2 = magnitude2[~np.isnan(magnitude2)]
        direction2 = direction2[~np.isnan(direction2)]

        cmap = cmocean.cm.haline

        bins = self.get_rose_bins(np.append(magnitude, magnitude2))
        rmax1 = ax1.bar(direction, magnitude, bins=bins, normed=normed, cmap=cmap,
                       opening=0.9, edgecolor='white', nsector=nsector, blowto=blowto)

        self.set_legend(ax1, legend_title)
        ax1.set_title(title1, **title_font)

        rmax2 = ax2.bar(direction2, magnitude2, bins=bins, normed=normed, cmap=cmap,
                       opening=0.8, edgecolor='white', nsector=nsector, blowto=blowto)

        rmax = max([rmax1, rmax2])
        ax1._update(rmax)
        ax2._update(rmax)
        self.set_legend(ax2, legend_title)
        ax2.set_title(title2, **title_font)

        return fig, ax1, ax2


    def plot_1d_quiver(self, fig, ax, time, u, v, title='', ylabel='',
                       title_font={}, axis_font={}, tick_font={},
                       legend_title="Current magnitude [m/s]", **kwargs):
        '''
        Returns a 1D quiver plot
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default
        # Plot quiver
        magnitude = (u**2 + v**2)**0.5
        maxmag = max(magnitude)
        ax.set_ylim(-maxmag, maxmag)
        dx = time[-1] - time[0]
        ax.set_xlim(time[0] - 0.05 * dx, time[-1] + 0.05 * dx)
        ax.fill_between(time, magnitude, 0, color='k', alpha=0.1)

        # Fake 'box' to be able to insert a legend for 'Magnitude'
        p = ax.add_patch(plt.Rectangle((1, 1), 1, 1, fc='k', alpha=0.1))
        leg1 = ax.legend([p], [legend_title], loc='lower right')
        leg1._drawFrame = False

        # # 1D Quiver plot
        q = ax.quiver(time, 0, u, v, **kwargs)
        plt.quiverkey(q, 0.2, 0.05, 0.2,
                      r'$0.2 \frac{m}{s}$',
                      labelpos='W',
                      fontproperties={'weight': 'bold'})

        ax.xaxis_date()
        date_list = mdates.num2date(time)
        self.get_time_label(ax, date_list)
        fig.autofmt_xdate()

        if ylabel:
            ax.set_ylabel(ylabel, labelpad=20, **axis_font)
        if tick_font:
            ax.tick_params(**tick_font)
        ax.set_title(title, **title_font)

    def plot_2d_wave_spectrum_contour(self, ax, freq, deg, spectrum,
                                      title='Contour 2D Spectrum',
                                      title_font={},
                                      tick_font={},
                                      cbar_title='',
                                      axis_font={},
                                      clim=None,
                                      log=False, **kwargs):
        '''
        Returns an XWaves based 2D wave spectra contour plot
        with up to 12 contour levels for a single spectra
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default
        if not cbar_title:
            cbar_title = ('LOG' + r'$_{10}$' +
                          ' Energy Density (m' + r'$^2$' + '/Hz/Deg)')

        # Need to replace 0's with low energy threshold
        threshold = 1e-8
        spectrum[spectrum < threshold] = threshold

        contour_levels = np.log10(spectrum)
        if len(set(contour_levels.flat)) < 7:
            nlines = len(set(contour_levels.flat))
        else:
            nlines = 12
        # nlines = len(set(contour_levels.flat)) if len(set(contour_levels.flat)) < 7 else 12

        cmap = plt.cm.jet
        if clim:
            levels = np.around(np.linspace(clim[0], clim[1], nlines), 1)
            h = ax.contourf(freq, deg, contour_levels, levels=levels, cmap=cmap, vmin=clim[0], vmax=clim[1])
        else:
            h = ax.contourf(freq, deg, contour_levels, nlines, cmap=cmap)

        cbar = plt.colorbar(h, ax=ax, orientation='horizontal', aspect=30, shrink=0.8)
        cbar.solids.set_edgecolor("face")
        for line in cbar.lines:
            line.set_linewidth(3)
        if cbar_title:
            cbar.ax.set_xlabel(cbar_title)

        ax.set_ylabel('Direction (deg from N)', **axis_font)
        ax.set_ylim([0, 360])
        ax.set_xlabel('Frequency (Hz)', **axis_font)
        ax.set_title(title, **title_font)
        if tick_font:
            ax.tick_params(**tick_font)
        if log:
            ax.set_xscale('log')
            ticklab, ticknum = self.get_log_tick_labels(freq.min(), freq.max())
            ax.set_xticks(ticknum)
            ax.set_xticklabels(ticklab)
        ax.grid(b=False)

    def plot_2d_wave_spectrum_polar(self, ax, freq, deg, spectrum,
                                    title='Polar 2D Spectrum', title_font={},
                                    cbar_title='', clim=None, tick_font={}):
        '''
        Returns a polar wave spectra plot for a single spectra
        '''

        if not title_font:
            title_font = self.title_font_default
        if not cbar_title:
            cbar_title = ('LOG' + r'$_{10}$' +
                          ' Energy Density (m' + r'$^2$' + '/Hz/Deg(from N))')

        # fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        theta_angles = np.arange(0, 360, 45)
        theta_labels = ['N', 'N-E', 'E', 'S-E', 'S', 'S-W', 'W', 'N-W']
        ax.set_thetagrids(angles=theta_angles, labels=theta_labels)

        # Need to replace 0's with low energy threshold
        threshold = 1e-8
        spectrum[spectrum < threshold] = threshold
        spectrum = np.log10(spectrum)
        # radius is log of freq
        if max(freq) < 0:
            ymin = np.log10(0.05)
            zeniths = freq - ymin
        else:
            ymin = 0.0
            zeniths = freq
        # zeniths = np.divide(1.0, zeniths)

        # Add angles to wrap contours around 360 deg
        deg = np.append(deg[0] - (deg[-1] - deg[-2]) / 2.0, deg)
        # deg = np.append(0, deg)
        deg = np.append(deg, deg[-1] + (deg[-1] - deg[-2]) / 2.0)

        if max(deg) < 7.0:  # Radians
            azimuths = deg
        else:
            azimuths = np.radians(deg)

        theta, r = np.meshgrid(azimuths, zeniths)

        # Add 2 columns to spectrum now
        new_col = np.array((spectrum[:, 0] + spectrum[:, -1]) / 2.0)
        z = np.concatenate((np.mat(new_col).T, spectrum), axis=1)
        z = np.concatenate((z, np.mat(new_col).T), axis=1)

        contour_levels = np.log10(z)
        nlevels = len((contour_levels)) if len((contour_levels)) < 7 else 12
        values = np.array(z)
        if clim:
            levels = np.around(np.linspace(clim[0], clim[1], nlevels), 1)
            h = ax.contourf(theta, r, values, levels=levels, vmin=clim[0], vmax=clim[1])
        else:
            h = ax.contourf(theta, r, values, nlevels)

        ax.set_title(title, **title_font)

        # Create the new radial tick marks and labels
        periods = [10, 5, 3.3, 2.5, 2]
        ticks = [np.divide(1.0, period) for period in periods]
        labels = [str(period) + ' s' for period in periods]
        ax.set_rgrids(ticks, labels, angle=22.5, size=8, fontweight='bold')

        cbar = plt.colorbar(h, ax=ax, orientation='horizontal',
                            aspect=20, shrink=0.45)
        if cbar_title:
            cbar.ax.set_xlabel(cbar_title)

        plt.tight_layout()

    def plot_spectrograph(self, fig, ax, time, freq, spectrum,
                          ylabel='Frequency (Hz)',
                          title='Wave Energy Spectrograph',
                          cbar_title='', tick_font={},
                          title_font={}, axis_font={}, **kwargs):
        '''
        Returns a colored wave spectrograph plot

        xaxis: time
        yaxis: frequency
        zaxis: wave energy (m^2/Hz)
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default
        if not cbar_title:
            cbar_title = 'LOG' + r'$_{10}$' + ' (m' + r'$^2$' + '/Hz)'

        h = plt.pcolormesh(time, freq, np.log10(spectrum), shading='gouraud')

        ax.set_ylabel(ylabel, **axis_font)
        ax.set_title(title, **title_font)

        ax.xaxis_date()
        date_list = mdates.num2date(time)
        self.get_time_label(ax, date_list)
        fig.autofmt_xdate()
        ax.set_xlim([time.min(), time.max()])
        ax.set_ylim([freq.min(), freq.max()])

        cbar = plt.colorbar(h, orientation='horizontal',
                            aspect=20, shrink=0.85, pad=0.2)

        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        labels = ax2.get_yticks().tolist()
        new_labels = []
        for label in labels:
            if float(label) > 0:
                new_labels.append(str(np.floor( 1/float(label) * 10 )/10))
            else:
                new_labels.append('')

        ax2.set_yticklabels(new_labels)
        ax2.set_ylabel('Period (s)', **axis_font)
        if cbar_title:
            cbar.ax.set_xlabel(cbar_title)
        if tick_font:
            ax.tick_params(**tick_font)

    def plot_frequency_spectrum(self, ax, freq, spectrum_1d,
                                title='Frequency Spectrum', title_font={},
                                axis_font={}, tick_font={}, log=False, **kwargs):
        '''
        Returns a 1D wave frequency spectrum plot (log or linear axis)
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default

        if log:
            plt.loglog(freq, spectrum_1d, basex=10, basey=10, **kwargs)
            ax.set_xlim([freq.min(), freq.max()])
            ticklab, ticknum = self.get_log_tick_labels(freq.min(), freq.max())
            ax.set_xticks(ticknum)
            ax.set_xticklabels(ticklab)
        else:
            plt.plot(ax, freq, spectrum_1d, **kwargs)
            ax.set_xlim([freq.min(), freq.max()])
            # plt.fill_between(ax, freq, spectrum_1d, **kwargs)

        # # Now plot the Pierson Moskowitz spectra
        # alpha = 0.0081
        # beta = -0.74
        # g = 9.81
        # wo = -g/

        ax.grid(True)
        ax.set_xlabel('Frequency (Hz)', **axis_font)
        ax.set_ylabel('Wave Energy (' + r'$m^2$' + '/Hz)', **axis_font)
        ax.set_title(title, **title_font)
        if tick_font:
            ax.tick_params(**tick_font)
        plt.tight_layout()

    def plot_2d_wave_spectrum_cartesian(self, ax, freq, deg, spectrum,
                                        title='Cartesian 2D Spectrum',
                                        title_font={}, axis_font={}, tick_font={}, **kwargs):
        '''
        Returns an XWaves 2D wave spectrum plot on a 3D cartesian coordinate
        '''


        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default

        X, Y = np.meshgrid(freq, deg)
        if np.shape(X) != np.shape(spectrum):
            spectrum = spectrum.T
            if np.shape(X) != np.shape(spectrum):
                raise Exception('Dimension mismatch!')
        ax.plot_surface(X, Y, spectrum, cmap=plt.cm.jet, rstride=1, cstride=1,
                        linewidth=0, antialiased=False)
        ax.set_ylim([0, 360])
        ax.set_ylabel('Direction (deg from N)', **axis_font)
        ax.set_xlim([freq.min(), freq.max()])
        ax.set_xlabel('Frequency (Hz)', **axis_font)
        ax.set_zlabel('Energy Density (m' + r'$^2$' + '/Hz/deg)', **axis_font)
        ax.set_title(title, **title_font)
        if tick_font:
            ax.tick_params(**tick_font)
        ax.set_axis_bgcolor((1, 1, 1))

    def make_patch_spines_invisible(self, ax):
        ax.set_frame_on(True)
        ax.patch.set_visible(False)
        for sp in ax.spines.itervalues():
            sp.set_visible(False)

    def set_spine_direction(self, ax, direction):
        if direction in ["right", "left"]:
            ax.yaxis.set_ticks_position(direction)
            ax.yaxis.set_label_position(direction)
        elif direction in ["top", "bottom"]:
            ax.xaxis.set_ticks_position(direction)
            ax.xaxis.set_label_position(direction)
        else:
            raise ValueError("Unknown Direction: %s" % (direction,))

        ax.spines[direction].set_visible(True)

    def plot_multiple_xaxes(self, ax, xdata, ydata, colors, ylabel='Depth (m)', title='', title_font={},
                            axis_font={}, tick_font={}, **kwargs):
        '''
        Returns a time series plot with up to 6 x axes

        Acknowledgment: This program is based on code written by Jae-Joon Lee,
        URL= http://matplotlib.svn.sourceforge.net/viewvc/matplotlib/trunk/matplotlib/
        examples/pylab_examples/multiple_yaxis_with_spines.py?revision=7908&view=markup
        '''

        if not title_font:
            title_font = self.title_font_default
        if not axis_font:
            axis_font = self.axis_font_default

        n_vars = len(xdata)
        if n_vars > 6:
            raise Exception('This code currently handles a maximum of 6 independent variables.')

        # Generate the plot.
        # Use twiny() to create extra axes for all dependent variables except the first
        # (we get the first as part of the ax axes).
        x_axis = n_vars * [0]
        x_axis[0] = ax
        for i in range(1, n_vars):
            x_axis[i] = ax.twiny()

        ax.spines["top"].set_visible(False)
        self.make_patch_spines_invisible(x_axis[1])
        self.set_spine_direction(x_axis[1], "top")

        offset = [1.10, -0.1, -0.20, 1.20]
        spine_directions = ["top", "bottom", "bottom", "top", "top", "bottom"]

        count = 0
        for i in range(2, n_vars):
            x_axis[i].spines[spine_directions[count]].set_position(("axes", offset[count]))
            self.make_patch_spines_invisible(x_axis[i])
            self.set_spine_direction(x_axis[i], spine_directions[count])
            count += 1

        # Adjust the axes left/right accordingly
        if n_vars >= 4:
            plt.subplots_adjust(bottom=0.2, top=0.8)
        elif n_vars == 3:
            plt.subplots_adjust(bottom=0.0, top=0.8)

        # Label the y-axis:
        ax.set_ylabel(ylabel, **axis_font)
        for ind, key in enumerate(xdata):
            x_axis[ind].plot(xdata[key], ydata, colors[ind], **kwargs)
            # Label the x-axis and set text color:
            x_axis[ind].set_xlabel(key, **axis_font)
            x_axis[ind].xaxis.label.set_color(colors[ind])
            x_axis[ind].spines[spine_directions[ind]].set_color(colors[ind])

            for obj in x_axis[ind].xaxis.get_ticklines():
                # `obj` is a matplotlib.lines.Line2D instance
                obj.set_color(colors[ind])
                obj.set_markeredgewidth(2)

            for obj in x_axis[ind].xaxis.get_ticklabels():
                obj.set_color(colors[ind])
                obj.set_size(12)
                obj.set_weight(600)

        ax.invert_yaxis()
        ax.grid(True)
        ax.set_title(title, y=1.23, **title_font)
        if tick_font:
            ax.tick_params(**tick_font)

    def plot_multiple_yaxes(self, fig, ax, data, colors, title,
                            axis_font={}, title_font={}, tick_font={}, **kwargs):
        '''
        Plot a timeseries with multiple y-axes

        ydata is a python dictionary of all the data to plot. Key values are used as plot labels

        Acknowledgment: This program is based on code written by Jae-Joon Lee,
        URL= http://matplotlib.svn.sourceforge.net/viewvc/matplotlib/trunk/matplotlib/
        examples/pylab_examples/multiple_yaxis_with_spines.py?revision=7908&view=markup

        http://matplotlib.org/examples/axes_grid/demo_parasite_axes2.html
        '''

        if not axis_font:
            axis_font = self.axis_font_default
        if not title_font:
            title_font = self.title_font_default

        n_vars = len(data)
        if n_vars > 6:
            raise ValueError('This code currently handles a maximum of four independent variables.')

        # Generate the plot.
        # Use twinx() to create extra axes for all dependent variables except the first
        # (we get the first as part of the ax axes).

        y_axis = n_vars * [0]
        y_axis[0] = ax
        for i in xrange(1, n_vars):
            y_axis[i] = ax.twinx()

        ax.spines["top"].set_visible(False)
        self.make_patch_spines_invisible(y_axis[1])

        self.set_spine_direction(y_axis[1], "top")

        # Define the axes position offsets for each 'extra' axis
        offset = [1.10, -0.10, 1.20, -0.20]
        spine_directions = ["left", "right", "left", "right", "left", "right"]
        count = 0
        for i in xrange(2, n_vars):
            y_axis[i].spines[spine_directions[count + 1]].set_position(("axes", offset[count]))
            self.make_patch_spines_invisible(y_axis[i])
            self.set_spine_direction(y_axis[i], spine_directions[count + 1])
            count += 1

        # Adjust the axes left/right accordingly
        if n_vars >= 4:
            plt.subplots_adjust(left=0.2, right=0.8)
        elif n_vars == 3:
            plt.subplots_adjust(left=0.0, right=0.8)

        primary_date_list = None
        for ind, key in enumerate(data):
            date_list, ydata = data[key]
            # Our x-axis will be based off the first available variable
            if primary_date_list is None:
                primary_date_list = date_list
            y_axis[ind].plot(date_list, ydata, colors[ind], **kwargs)
            # Label the x-axis and set text color:
            y_axis[ind].set_ylabel(key, **axis_font)
            y_axis[ind].yaxis.label.set_color(colors[ind])
            y_axis[ind].spines[spine_directions[ind]].set_color(colors[ind])
            y_axis[ind].tick_params(axis='y', colors=colors[ind])

        self.get_time_label(ax, primary_date_list)

        fig.autofmt_xdate()
        ax.set_title(title, **title_font)
        ax.grid(True)
        if tick_font:
            ax.tick_params(**tick_font)
        plt.tight_layout()

    def plot_skew_t_log_p(self, fig, pressure, temperature, dewpoint,
                          windspeed, winddir, title, title_font={},
                          axis_font={}, tick_font={}, rotation=45):
        """
        Returns a skewT logP plot using the MetPy module given a figure,
        pressure, temperature, dewpoint, windspeed, and wind direction.

        :param matplotlib figure fig   - matplotlib figure object on which to plot
        :param numpy array pressure    - array of pressure (units: hPa)
        :param numpy array temperature - array of temperature (units: degC)
        :param numpy array dewpoint    - array of dewpoints (units: degC)
        :param numpy array windspeed   - array of windspeeds (units: kts)
        :param numpy array winddir     - array of wind directions (units: degrees)
        :param str title               - title string
        :param dict title_font         - dict of fontsize for the title;
                                         e.g. `{'fontsize': 12}`
        :param dict axis_font          - dict of fontsize for axes labels; e.g.
                                         `{'size': 10}`
        :param dict tick_font          - dict of fontsize for tick labels;
                                         e.g. `{'labelsize': 8}`
        :param float rotation          - rotation of temperature lines in degrees;
                                         default 45.

        notes
        -----
        It should be noted that when passing a figure into this function, the
        figure should be generated by `plt.figure()` and not `plt.subplots().`
        `plt.subplots()` returns both a figure object and axis object, and if
        both are present the function will also attempt to plot the newly
        created axis, which will result in overlapping y-tick and x-tick labels.

        Regarding rotation: adjusting the rotation of the temperature line may
        yield a better aspect ratio depending on the data extents.
        """

        if not axis_font:
            axis_font = self.axis_font_default
        if not title_font:
            title_font = self.title_font_default
        xlab = 'Temperature ($^\circ$C)'
        ylab = 'Pressure (hPa)'

        p = pressure * units.hPa
        Td = dewpoint * units.degC
        T = temperature * units.degC
        wind_speed = windspeed * units.knots
        wind_dir = winddir * units.degrees
        u, v = mpcalc.wind_components(wind_speed, wind_dir)

        # Plot the data using normal plotting functions, in this case using
        # log scaling in Y, as dictated by the typical meteorological plot
        skew = SkewT(fig, rotation=rotation)
        plt.ylim(ymin=1050, ymax=100)  # conventional y limits
        skew.plot(p, T, 'r', clip_on=True)
        skew.plot(p, Td, 'g', clip_on=True)
        skew.plot_barbs(p, u, v, y_clip_radius=0)  # don't allow barbs past y-axis

        # Calculate LCL height and plot as black dot
        lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
        skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

        # Calculate full parcel profile and add to plot as black line
        prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
        skew.plot(p, prof, 'k', linewidth=1.5)

        # Shade areas of CAPE and CIN
        skew.shade_cin(p, T, prof)
        skew.shade_cape(p, T, prof)

        # plot adiabats and mixing lines
        skew.plot_dry_adiabats(
            t0=np.arange(-40, 120, 5)*units.degC,
            linewidths=0.9
        )
        skew.plot_moist_adiabats(
            t0=np.arange(-12, 40, 5)*units.degC,
            linewidths=0.9
        )
        skew.plot_mixing_lines(linewidths=1.2)
        skew.ax.set_xlim(-33, 32)  # conventional x limits

        # plot hodograph
        # 30% size of each axis, set bbox to leave room for barbs
        ax_hod = inset_axes(
            skew.ax, '30%', '30%',
            bbox_to_anchor=(0,0,.9,.98),
            bbox_transform=skew.ax.transAxes,
            borderpad=0.0
        )
        h = Hodograph(ax_hod, component_range=80.)
        h.add_grid(increment=20) # grid lines on hodogrpah
        h.plot_colormapped(u, v, np.hypot(u, v)) # plots with colored lines

        if tick_font:
            skew.ax.tick_params(**tick_font)
        skew.ax.set_xlabel(xlab, **axis_font)
        skew.ax.set_ylabel(ylab, **axis_font)
        skew.ax.set_title(title, **title_font)
        return fig
