
print ""
print "#-----------------------------------------------------------------------"
print "Continuum-subtraction script,    created by A. Sanchez-Monge (v.2015/09)"


################################################################################
#
# Importing packages and general setup:

import os
import sys

import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import stats
from pylab import *
from scipy.optimize import leastsq
from scipy.optimize import curve_fit

import fits_cutout
import kdestats

sys.dont_write_bytecode = True


################################################################################
#
# Creating the list of options that can be used in this script:

pars = argparse.ArgumentParser(description = "Continuum-subtraction script (v.2015/09)")
pars.add_argument('--cutout', action='store_true', default=False, dest='cutout', help='Create a cutout of the original fits file')
pars.add_argument('--plots', action='store_true', default=False, dest='plots', help='Create plots of the KDE distribution on a pixel basis')
pars.add_argument('--messages', action='store_true', default=False, dest='messages', help='Show intermediate messages')
op = pars.parse_args()


################################################################################
#
# Defining some general variables, directories and functions:

os.system('mkdir -p datacubes/')
os.system('mkdir -p cutout/')
os.system('mkdir -p continuum/')
os.system('mkdir -p continuum/plots')

datacubes_path = "datacubes/"
cutout_path = "cutout/"
continuum_path = "continuum/"
plots_path = "continuum/plots/"

# Definition of the Gaussian function (version 1 of the fit)
fitfunc = lambda p, x: p[0]*exp(-0.5*((x-p[1])/p[2])**2.)
errfunc = lambda p, x, y: (y - fitfunc(p, x))

# Definition of the Gaussian function (version 2 of the fit, currently not used)
def gaussian(x, a, mu, sigma):
    return a * np.exp(-(1./2.) * np.power((x - mu)/sigma, 2.))

# Noise level of your data cubes (in units of the fits files)
rms_noise = 0.02

# Number of bins you want to use to sample your distribution of the spectra
number_bins = 100


################################################################################
#
# List of files/datacubes to be analyzed:

#input_files = ['SgrB2M_HF_spw0_image', 'SgrB2M_HF_spw1_image', 'SgrB2M_HF_spw2_image', 'SgrB2M_HF_spw3_image', 'SgrB2M_HF_spw4_image', 'SgrB2M_HF_spw5_image', 'SgrB2M_HF_spw6_image', 'SgrB2M_HF_spw7_image', 'SgrB2M_HF_spw8_image', 'SgrB2M_HF_spw9_image', 'SgrB2M_HF_spw10_image', 'SgrB2M_HF_spw11_image', 'SgrB2M_HF_spw16_image', 'SgrB2M_HF_spw17_image', 'SgrB2M_HF_spw18_image', 'SgrB2M_HF_spw19_image']
#input_files = ['SgrB2N_LF_spw0_image', 'SgrB2N_LF_spw1_image', 'SgrB2N_LF_spw2_image', 'SgrB2N_LF_spw3_image', 'SgrB2N_LF_spw4_image', 'SgrB2N_LF_spw5_image', 'SgrB2N_LF_spw6_image', 'SgrB2N_LF_spw7_image', 'SgrB2N_LF_spw8_image', 'SgrB2N_LF_spw9_image', 'SgrB2N_LF_spw10_image', 'SgrB2N_LF_spw11_image', 'SgrB2N_LF_spw16_image', 'SgrB2N_LF_spw17_image', 'SgrB2N_LF_spw18_image', 'SgrB2N_LF_spw19_image', 'SgrB2N_HF_spw0_image', 'SgrB2N_HF_spw1_image', 'SgrB2N_HF_spw2_image', 'SgrB2N_HF_spw3_image', 'SgrB2N_HF_spw4_image', 'SgrB2N_HF_spw5_image', 'SgrB2N_HF_spw6_image', 'SgrB2N_HF_spw7_image', 'SgrB2N_HF_spw8_image', 'SgrB2N_HF_spw9_image', 'SgrB2N_HF_spw10_image', 'SgrB2N_HF_spw11_image', 'SgrB2N_HF_spw16_image', 'SgrB2N_HF_spw17_image', 'SgrB2N_HF_spw18_image', 'SgrB2N_HF_spw19_image']
#input_files = ['SgrB2N_LF_spw0_image']
#input_files = ['SgrB2M_HF_spw0_b07_restfreq', 'SgrB2M_HF_spw1_b07_restfreq', 'SgrB2M_HF_spw2_b07_restfreq', 'SgrB2M_HF_spw3_b07_restfreq', 'SgrB2M_HF_spw4_b07_restfreq', 'SgrB2M_HF_spw5_b07_restfreq', 'SgrB2M_HF_spw6_b07_restfreq', 'SgrB2M_HF_spw7_b07_restfreq', 'SgrB2M_HF_spw8_b07_restfreq', 'SgrB2M_HF_spw9_b07_restfreq', 'SgrB2M_HF_spw10_b07_restfreq', 'SgrB2M_HF_spw11_b07_restfreq', 'SgrB2M_HF_spw12_b07_restfreq', 'SgrB2M_HF_spw13_b07_restfreq', 'SgrB2M_HF_spw14_b07_restfreq', 'SgrB2M_HF_spw15_b07_restfreq', 'SgrB2M_HF_spw16_b07_restfreq', 'SgrB2M_HF_spw17_b07_restfreq', 'SgrB2M_HF_spw18_b07_restfreq', 'SgrB2M_HF_spw19_b07_restfreq']
#input_files = ['SgrB2M_LF_spw0_b07_restfreq', 'SgrB2M_LF_spw1_b07_restfreq', 'SgrB2M_LF_spw2_b07_restfreq', 'SgrB2M_LF_spw3_b07_restfreq', 'SgrB2M_LF_spw4_b07_restfreq', 'SgrB2M_LF_spw5_b07_restfreq', 'SgrB2M_LF_spw6_b07_restfreq', 'SgrB2M_LF_spw7_b07_restfreq', 'SgrB2M_LF_spw8_b07_restfreq', 'SgrB2M_LF_spw9_b07_restfreq', 'SgrB2M_LF_spw10_b07_restfreq', 'SgrB2M_LF_spw11_b07_restfreq', 'SgrB2M_LF_spw12_b07_restfreq', 'SgrB2M_LF_spw13_b07_restfreq', 'SgrB2M_LF_spw14_b07_restfreq', 'SgrB2M_LF_spw15_b07_restfreq', 'SgrB2M_LF_spw16_b07_restfreq', 'SgrB2M_LF_spw17_b07_restfreq', 'SgrB2M_LF_spw18_b07_restfreq', 'SgrB2M_LF_spw19_b07_restfreq']
#input_files = ['SgrB2N_HF_spw0_b07_restfreq', 'SgrB2N_HF_spw1_b07_restfreq', 'SgrB2N_HF_spw2_b07_restfreq', 'SgrB2N_HF_spw3_b07_restfreq', 'SgrB2N_HF_spw4_b07_restfreq', 'SgrB2N_HF_spw5_b07_restfreq', 'SgrB2N_HF_spw6_b07_restfreq', 'SgrB2N_HF_spw7_b07_restfreq', 'SgrB2N_HF_spw8_b07_restfreq', 'SgrB2N_HF_spw9_b07_restfreq', 'SgrB2N_HF_spw10_b07_restfreq', 'SgrB2N_HF_spw11_b07_restfreq', 'SgrB2N_HF_spw12_b07_restfreq', 'SgrB2N_HF_spw13_b07_restfreq', 'SgrB2N_HF_spw14_b07_restfreq', 'SgrB2N_HF_spw15_b07_restfreq', 'SgrB2N_HF_spw16_b07_restfreq', 'SgrB2N_HF_spw17_b07_restfreq', 'SgrB2N_HF_spw18_b07_restfreq', 'SgrB2N_HF_spw19_b07_restfreq']
#input_files = ['SgrB2N_LF_spw0_b07_restfreq', 'SgrB2N_LF_spw1_b07_restfreq', 'SgrB2N_LF_spw2_b07_restfreq', 'SgrB2N_LF_spw3_b07_restfreq', 'SgrB2N_LF_spw4_b07_restfreq', 'SgrB2N_LF_spw5_b07_restfreq', 'SgrB2N_LF_spw6_b07_restfreq', 'SgrB2N_LF_spw7_b07_restfreq', 'SgrB2N_LF_spw8_b07_restfreq', 'SgrB2N_LF_spw9_b07_restfreq', 'SgrB2N_LF_spw10_b07_restfreq', 'SgrB2N_LF_spw11_b07_restfreq', 'SgrB2N_LF_spw12_b07_restfreq', 'SgrB2N_LF_spw13_b07_restfreq', 'SgrB2N_LF_spw14_b07_restfreq', 'SgrB2N_LF_spw15_b07_restfreq', 'SgrB2N_LF_spw16_b07_restfreq', 'SgrB2N_LF_spw17_b07_restfreq', 'SgrB2N_LF_spw18_b07_restfreq', 'SgrB2N_LF_spw19_b07_restfreq']
#input_files = ['SgrB2M_HF_spw0_b07_restfreq', 'SgrB2M_HF_spw1_b07_restfreq', 'SgrB2M_HF_spw2_b07_restfreq', 'SgrB2M_HF_spw3_b07_restfreq', 'SgrB2M_HF_spw4_b07_restfreq', 'SgrB2M_HF_spw5_b07_restfreq', 'SgrB2M_HF_spw6_b07_restfreq', 'SgrB2M_HF_spw7_b07_restfreq', 'SgrB2M_HF_spw8_b07_restfreq', 'SgrB2M_HF_spw9_b07_restfreq', 'SgrB2M_HF_spw10_b07_restfreq', 'SgrB2M_HF_spw11_b07_restfreq', 'SgrB2M_HF_spw12_b07_restfreq', 'SgrB2M_HF_spw13_b07_restfreq', 'SgrB2M_HF_spw14_b07_restfreq', 'SgrB2M_HF_spw15_b07_restfreq', 'SgrB2M_HF_spw16_b07_restfreq', 'SgrB2M_HF_spw17_b07_restfreq', 'SgrB2M_HF_spw18_b07_restfreq', 'SgrB2M_HF_spw19_b07_restfreq', 'SgrB2M_LF_spw0_b07_restfreq', 'SgrB2M_LF_spw1_b07_restfreq', 'SgrB2M_LF_spw2_b07_restfreq', 'SgrB2M_LF_spw3_b07_restfreq', 'SgrB2M_LF_spw4_b07_restfreq', 'SgrB2M_LF_spw5_b07_restfreq', 'SgrB2M_LF_spw6_b07_restfreq', 'SgrB2M_LF_spw7_b07_restfreq', 'SgrB2M_LF_spw8_b07_restfreq', 'SgrB2M_LF_spw9_b07_restfreq', 'SgrB2M_LF_spw10_b07_restfreq', 'SgrB2M_LF_spw11_b07_restfreq', 'SgrB2M_LF_spw12_b07_restfreq', 'SgrB2M_LF_spw13_b07_restfreq', 'SgrB2M_LF_spw14_b07_restfreq', 'SgrB2M_LF_spw15_b07_restfreq', 'SgrB2M_LF_spw16_b07_restfreq', 'SgrB2M_LF_spw17_b07_restfreq', 'SgrB2M_LF_spw18_b07_restfreq', 'SgrB2M_LF_spw19_b07_restfreq', 'SgrB2N_HF_spw0_b07_restfreq', 'SgrB2N_HF_spw1_b07_restfreq', 'SgrB2N_HF_spw2_b07_restfreq', 'SgrB2N_HF_spw3_b07_restfreq', 'SgrB2N_HF_spw4_b07_restfreq', 'SgrB2N_HF_spw5_b07_restfreq', 'SgrB2N_HF_spw6_b07_restfreq', 'SgrB2N_HF_spw7_b07_restfreq', 'SgrB2N_HF_spw8_b07_restfreq', 'SgrB2N_HF_spw9_b07_restfreq', 'SgrB2N_HF_spw10_b07_restfreq', 'SgrB2N_HF_spw11_b07_restfreq', 'SgrB2N_HF_spw12_b07_restfreq', 'SgrB2N_HF_spw13_b07_restfreq', 'SgrB2N_HF_spw14_b07_restfreq', 'SgrB2N_HF_spw15_b07_restfreq', 'SgrB2N_HF_spw16_b07_restfreq', 'SgrB2N_HF_spw17_b07_restfreq', 'SgrB2N_HF_spw18_b07_restfreq', 'SgrB2N_HF_spw19_b07_restfreq', 'SgrB2N_LF_spw0_b07_restfreq', 'SgrB2N_LF_spw1_b07_restfreq', 'SgrB2N_LF_spw2_b07_restfreq', 'SgrB2N_LF_spw3_b07_restfreq', 'SgrB2N_LF_spw4_b07_restfreq', 'SgrB2N_LF_spw5_b07_restfreq', 'SgrB2N_LF_spw6_b07_restfreq', 'SgrB2N_LF_spw7_b07_restfreq', 'SgrB2N_LF_spw8_b07_restfreq', 'SgrB2N_LF_spw9_b07_restfreq', 'SgrB2N_LF_spw10_b07_restfreq', 'SgrB2N_LF_spw11_b07_restfreq', 'SgrB2N_LF_spw12_b07_restfreq', 'SgrB2N_LF_spw13_b07_restfreq', 'SgrB2N_LF_spw14_b07_restfreq', 'SgrB2N_LF_spw15_b07_restfreq', 'SgrB2N_LF_spw16_b07_restfreq', 'SgrB2N_LF_spw17_b07_restfreq', 'SgrB2N_LF_spw18_b07_restfreq', 'SgrB2N_LF_spw19_b07_restfreq']
input_files = ['SgrB2N_LF_spw16_b07_restfreq']
extension = '.fits'


################################################################################
#
# Using cutout to create new fits with smaller sizes:

if op.cutout is True:
    print ""
    print "--cutout: Create a cutout of the original fits files"
    
    tmp_files = []

    for file_name in input_files:

        original_fitsfile = datacubes_path + file_name + extension
        central_xpixel = 261
        central_ypixel = 292
        number_pixels = 6
        cutout_fitsfile = cutout_path + file_name + '_cutout' + extension
        fits_cutout.cutout(original_fitsfile, central_xpixel, central_ypixel, number_pixels, cutout_fitsfile)
        tmp_path = 'cutout/'
        tmp_file = file_name + '_cutout'
        tmp_files.append(tmp_file)

if op.cutout is False:
    
    tmp_files = []

    for file_name in input_files:
    
        tmp_path = 'datacubes/'
        tmp_file = file_name
        tmp_files.append(tmp_file)


################################################################################
#
# Reading the header of the fits files, and determining the continuum level:

for tmp_file in tmp_files:

    print ""
    print "+++ READING THE HEADER OF " + tmp_path + tmp_file + extension
    
    #
    # reading the header information
    fitsfile = fits.open(tmp_path+tmp_file+extension)
    
    header = fitsfile[0].header
    data = fitsfile[0].data
    
    nxpix = header.get('NAXIS1')
    nypix = header.get('NAXIS2')
    nchan = header.get('NAXIS3')
    npolz = header.get('NAXIS4')

    #
    # loop through the polarizations, channels and positions
    # and determination of the continuum level
    print ""
    print "+++ DETERMINING THE CONTINUUM LEVEL"
    continuum_flux_mean = []
    continuum_flux_maximum = []
    continuum_flux_Gaussian = []
    continuum_flux_KDEmax = []
    
    for polz in range(1):
        continuum_flux_mean.append([])
        continuum_flux_maximum.append([])
        continuum_flux_Gaussian.append([])
        continuum_flux_KDEmax.append([])
        
        for chan in range(1):
            continuum_flux_mean[0].append([])
            continuum_flux_maximum[0].append([])
            continuum_flux_Gaussian[0].append([])
            continuum_flux_KDEmax[0].append([])
            
            for ypix in range(nypix):
                print "... analyzing column " + str(ypix+1) + " out of " + str(nypix)
                continuum_flux_mean[0][0].append([])
                continuum_flux_maximum[0][0].append([])
                continuum_flux_Gaussian[0][0].append([])
                continuum_flux_KDEmax[0][0].append([])
                
                for xpix in range(nxpix):
                	
                	# reading the spectrum
                    if op.messages is True:
                        print "  . corresponding to the pixel " + str(xpix+1) + "," + str(ypix+1)
                    chans = []
                    freqs = []
                    for channel in range(nchan):
                        chans.append(channel)
                        freqs.append((header.get('CRVAL3') + (channel - header.get('CRPIX3') - 1) * header.get('CDELT3')) / 1.e9) # in GHz

    # determining the continuum level
                    flux = data[0, 0:nchan, ypix, xpix]
                    
    # creating a general histogram of the flux data
    # main variables are:
    #   all_hist     - counts in each bin of the histogram
    #   all_bins     - location of the bins (fluxes)
    #   all_number_* - index of the array
                    all_hist, all_bin_edges = np.histogram(flux, number_bins)
                    all_bin_width = abs(all_bin_edges[1]-all_bin_edges[0])
                    all_bins = all_bin_edges[0:len(all_bin_edges)-1]
                    all_bins = [x + all_bin_width/2. for x in all_bins]
                    all_number_max_array = (np.where(all_hist == all_hist.max())[0])
                    all_number_max = all_number_max_array[0]
                    all_hist_max = all_hist[all_number_max]
                    all_bins_max = (all_bin_edges[all_number_max] + all_bin_width/2.)

    # CONTINUUM FLUX as the maximum of the histogram
                    maximum_flux = all_bins_max
                    if op.messages is True:
                        print "    flux of maximum   = " + str(int(maximum_flux*1.e5)/1.e5)
                    
    # Gaussian fit around the maximum of the distribution
    # determining the range to fit the Gaussian function
                    all_number_left  = (np.where(((all_hist == 0) & (all_bins <= all_bins_max)) | (all_bins == all_bins[0]))[0]).max()
                    all_number_right = (np.where(((all_hist == 0) & (all_bins >= all_bins_max)) | (all_bins == all_bins[number_bins-1]))[0]).min()
                    emission_absorption_ratio = abs(all_number_right-all_number_max)/abs(all_number_left-all_number_max)
                    if (emission_absorption_ratio >= 0.6):
                        lower_all_bins = all_bins_max - 9. * rms_noise
                        upper_all_bins = all_bins_max + 1. * rms_noise
                    if (emission_absorption_ratio <= 0.4):
                        lower_all_bins = all_bins_max - 1. * rms_noise
                        upper_all_bins = all_bins_max + 9. * rms_noise
                    if ((emission_absorption_ratio > 0.4) and (emission_absorption_ratio < 0.6)):
                        lower_all_bins = all_bins_max - 3. * rms_noise
                        upper_all_bins = all_bins_max + 3. * rms_noise
                    sel_bins_array = np.where((all_bins >= lower_all_bins) & (all_bins <= upper_all_bins))[0]
                    sel_bins = all_bins[sel_bins_array[0]:sel_bins_array[len(sel_bins_array)-1]+1]
                    sel_hist = all_hist[sel_bins_array[0]:sel_bins_array[len(sel_bins_array)-1]+1]
                    sel_flux = flux[(flux >= lower_all_bins) & (flux <= upper_all_bins)]

    # CONTINUUM FLUX as the mean of the "Gaussian" distribution
                    mean_flux = np.mean(sel_flux)
                    variance_flux = np.var(sel_flux)
                    sigma_flux = np.sqrt(variance_flux)
                    if op.messages is True:
                        print "    flux of mean      = " + str(int(mean_flux*1.e5)/1.e5)

    # CONTINUUM FLUX as the peak of the "Gaussian" fit
                    init = [all_hist.max(), all_bins_max, rms_noise]
                    out = leastsq( errfunc, init, args=(sel_bins, sel_hist))
                    c = out[0]
                    Gaussian_flux = c[1]
                    # alternative Gaussian fit
                    #xdata = np.linspace(min(flux), max(flux), number_bins*10.)
                    #ydata = gaussian(xdata, all_hist.max(), all_bins_max, rms_noise)
                    #popt, pcov = curve_fit(gaussian, sel_bins, sel_hist)
                    #new_fit = gaussian(xdata, *popt)
                    #Gaussian_flux = popt[1]
                    if op.messages is True:
                        print "    flux of Gaussian  = " + str(int(Gaussian_flux*1.e5)/1.e5)
                	
    # CONTINUUM FLUX as the maximum of a KDE distribution
                    KDE_bandwidth = 2. * rms_noise
                    scipy_kde = stats.gaussian_kde(flux, bw_method=KDE_bandwidth)
                    KDEmax_flux = kdestats.kde_max(scipy_kde)[0]
                    if op.messages is True:
                        print "    flux of KDEmax    = " + str(int(KDEmax_flux*1.e5)/1.e5)

    # CONTINUUM FLUX as the maximum of a lognormal fit
                    #fit_shape, fit_loc, fit_scale = sp.stats.lognorm.fit(sel_flux)
                    #lognorm_flux = np.log(fit_scale)
                    #if op.messages is True:
                    #    print "    flux of LogNormal = " + str(int(lognorm_flux*1.e5)/1.e5)

    # determination of the continuum level from different methods
                    continuum_flux_mean[0][0][ypix].append(mean_flux)
                    continuum_flux_maximum[0][0][ypix].append(maximum_flux)
                    continuum_flux_Gaussian[0][0][ypix].append(Gaussian_flux)
                    continuum_flux_KDEmax[0][0][ypix].append(KDEmax_flux)

	# creation of plots of the histogram distributions and spectra
                    if op.plots is True:
                        fig_file_histogram = plots_path + tmp_file + '_' + str(xpix) + '_' + str(ypix) + '.png'
                        fig1 = plt.figure()

    # plotting the histogram distribution with the continuum levels
                        plt.subplot(2, 1, 1)
                        xx = np.linspace(min(flux), max(flux), number_bins*10.)
                        plt.hist(flux, number_bins, facecolor='w')
                        plt.xlim(min(flux), max(flux))
                        plt.plot(xx, fitfunc(c, xx), 'b-')
                        scipy_kde_plot = stats.gaussian_kde(flux, bw_method=KDE_bandwidth)(xx)
                        plt.plot(xx, 5.*scipy_kde_plot, 'r-')
                        plt.axvline(x=Gaussian_flux, linestyle='-', color='b', linewidth='4.0')
                        plt.axvline(x=maximum_flux, linestyle='--', color='k', linewidth='1.5')
                        plt.axvline(x=mean_flux, linestyle='--', color='y', linewidth='1.5')
                        plt.axvline(x=KDEmax_flux, linestyle='--', color='r', linewidth='1.5')
                        plt.axvline(x=lower_all_bins, linestyle=':', color='k', linewidth='0.5')
                        plt.axvline(x=upper_all_bins, linestyle=':', color='k', linewidth='0.5')
                        plt.title('Histogram of the spectrum SgrB2-N at pixel (' + str(xpix+1) + ',' + str(ypix+1) + ')')
                        x_text = flux.min()+0.7*(flux.max()-flux.min())
                        y_text = all_hist_max
                        plt.axhline(y=0.91*y_text, xmin=0.58, xmax=0.68, linestyle='-', color='b', linewidth='4.0')
                        plt.text(x_text, 0.90*y_text, "Gaussian fit flux")
                        plt.axhline(y=0.81*y_text, xmin=0.58, xmax=0.68, linestyle='--', color='k', linewidth='1.5')
                        plt.text(x_text, 0.80*y_text, "maximum flux")
                        plt.axhline(y=0.71*y_text, xmin=0.58, xmax=0.68, linestyle='--', color='y', linewidth='1.5')
                        plt.text(x_text, 0.70*y_text, "mean flux")
                        plt.axhline(y=0.61*y_text, xmin=0.58, xmax=0.68, linestyle='--', color='r', linewidth='1.5')
                        plt.text(x_text, 0.60*y_text, "KDE max flux")
                        plt.xlabel('Intensity (Jy/beam)')
                        plt.ylabel('Counts')

    # plotting the spectra with the continuum levels
                        plt.subplot(2, 1, 2)
                        plt.plot(freqs, flux, 'k-')
                        plt.axhline(y=Gaussian_flux, linestyle='-', color='b', linewidth='4.0')
                        plt.axhline(y=maximum_flux, linestyle='--', color='k', linewidth='1.5')
                        plt.axhline(y=mean_flux, linestyle='--', color='y', linewidth='1.5')
                        plt.axhline(y=KDEmax_flux, linestyle='--', color='r', linewidth='1.5')
                        #plt.title('Spectrum of SgrB2-N at pixel (' + str(xpix+1) + ',' + str(ypix+1) + ')')
                        plt.xlabel('Frequency (GHz)')
                        plt.ylabel('Intensity (Jy/beam)')
                        
                        fig1.savefig(fig_file_histogram)
                        plt.close(fig1)

    #
    # writing the output continuum fits file
    print " "
    continuum_file_mean = continuum_path + tmp_file + '_continuum_mean' + extension
    continuum_file_maximum = continuum_path + tmp_file + '_continuum_maximum' + extension
    continuum_file_Gaussian = continuum_path + tmp_file + '_continuum_Gaussian' + extension
    continuum_file_KDEmax = continuum_path + tmp_file + '_continuum_KDEmax' + extension
    os.system('rm -rf ' + continuum_file_mean)
    os.system('rm -rf ' + continuum_file_maximum)
    os.system('rm -rf ' + continuum_file_Gaussian)
    os.system('rm -rf ' + continuum_file_KDEmax)
    print "+++ CONTINUUM FILEs CREATED: " + continuum_file_mean
    print "                             " + continuum_file_maximum
    print "                             " + continuum_file_Gaussian
    print "                             " + continuum_file_KDEmax
    print " "
    fits.writeto(continuum_file_mean, np.float32(continuum_flux_mean), header=header, clobber=True)
    fits.writeto(continuum_file_maximum, np.float32(continuum_flux_maximum), header=header, clobber=True)
    fits.writeto(continuum_file_Gaussian, np.float32(continuum_flux_Gaussian), header=header, clobber=True)
    fits.writeto(continuum_file_KDEmax, np.float32(continuum_flux_KDEmax), header=header, clobber=True)
	
