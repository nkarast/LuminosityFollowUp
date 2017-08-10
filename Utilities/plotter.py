
import sys, os
BIN = os.path.expanduser("/afs/cern.ch/user/l/lumimod/lumimod/a2017_luminosity_followup/")
sys.path.append(BIN)

import matplotlib
from colorsys import hsv_to_rgb
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MultipleLocator
import pandas as pd
import numpy as np
import numpy.ma as ma
import pickle
import time
from operator import add
from scipy.constants import c as clight
from scipy.optimize import curve_fit
import gzip
import tarfile
from glob import glob
from logging import *
from datetime import datetime
import Utilities.readYamlDB as db
import config
from collections import OrderedDict
import seaborn as sns
import matplotlib.dates as mdates
from dateutil import tz
import itertools
from scipy.integrate import cumtrapz


class Plotter(object):

	def __init__(self, filln, resc='', savedir="/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/SB_analysis/fill_<FILLNUMBER>/plots/"):
		self.filln = int(filln)
		self.folder = config.stableBeams_folder+config.fill_dir.replace("<FILLNUMBER>", str(filln)) #'/eos/user/l/lumimod/2017/backup/fill_5848/' #

		bunches_dict = { (5830, 5837) : 985, 
						 (5837, 5841) : 1225, 
						 (5842, 5844) : 1561,
						 (5845, 5849) : 1741,
						 (5849, 5850) : 2029,
						 (5856, 5857) : 2173,
						 (5862, 5866) : 2317,
						 (5867, 5880) : 2460,      
						 (5880, 5890) : 2556,
						 (5880, 5890) : 2556,
						 (5919, 5921) : 2,      
						 (5930, 5933) : 51,      
						 (5934, 5935) : 601,      
						 (5942, 5947) : 1357,      
						 (5950, 5951) : 2173,      
						 (5952, 5953) : 2371,      

		}

		self.bunches = None
		for key in bunches_dict.keys():
			if self.filln >= key[0] and self.filln < key[1]:
				self.bunches = bunches_dict[key]






		info("# Plotter : Firing up plotter for fill [{}]".format(self.filln))

		cycle_filename 		 = self.folder+config.Cycle_filename.replace("<FILLNUMBER>", str(filln)).replace("<RESC>", str(resc)) 
		sb_filename    		 = self.folder+config.SB_filename.replace("<FILLNUMBER>", str(filln)).replace("<RESC>", str(resc))
		lumi_filename  		 = self.folder+config.Lumi_filename.replace("<FILLNUMBER>", str(filln)).replace("<RESC>", str(resc))
		massi_filename 		 = self.folder+config.Massi_filename.replace("<FILLNUMBER>", str(filln))
		lifetime_filename    = sb_filename.replace('.pkl.gz', '_lifetime.pkl.gz')
		cycleModel_filename  = self.folder+config.Cycle_model_filename.replace("<FILLNUMBER>", str(filln)).replace("<RESC>", str(resc))
		sbfits_filename      = self.folder+config.SB_fits_filename.replace("<FILLNUMBER>", str(filln)).replace("<RESC>", str(resc))
		# sbmodel_filename     = self.folder+config.SB_model_filename.replace("<FILLNUMBER>", str(filln)).replace("<RESC>", str(resc))

		self.savedir 		 = savedir.replace("<FILLNUMBER>", str(self.filln))

		basicConfig(format='%(asctime)s %(levelname)s : %(message)s', filename=None, level=20)


		try:
			with gzip.open(cycle_filename, 'rb') as fid:
				self.filln_CycleDict           = pickle.load(fid)
		except:
			warn("#Plotter : Issues loading Cycle Data")

		try:
			with gzip.open(sb_filename, 'rb') as fid:
				self.filln_StableBeamsDict           = pickle.load(fid)
		except:
			warn("#Plotter : Issues loading SB Data")

		try:
			with gzip.open(lumi_filename, 'rb') as fid:
				self.filln_LumiCalcDict           = pickle.load(fid)
		except:
			warn("#Plotter : Issues loading Lumi Calc Data")

		try:
			with gzip.open(massi_filename, 'rb') as fid:
				self.filln_LumiMeasDict           = pickle.load(fid)
		except:
			warn("#Plotter : Issues loading Lumi Measured Data")

		try:
			with gzip.open(lifetime_filename, 'rb') as fid:
				self.filln_LifetimeDict           = pickle.load(fid)
		except:
			warn("#Plotter : Issues loading Lifetime Data")

		try:
			with gzip.open(cycleModel_filename, 'rb') as fid:
				self.filln_CycleModelDict           = pickle.load(fid)
		except:
			warn("#Plotter : Issues loading Cycle Model Data")

		try:
			with gzip.open(sbfits_filename, 'rb') as fid:
				self.filln_SBFitsDict           = pickle.load(fid)
		except:
			warn("#Plotter : Issues loading SB Fits Data")

		self.fills_bmodes_file = config.fills_bmodes_file
		self.bmodes, self.filln_list = self.getBmodesDF()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def getFilledSlotsArray(self, dict_intervals_two_beams, beam, cycle, cycleTime,  mask_invalid=True):
		'''
		Returns the array with the filled slots.
		Inputs : beam           : beam string ('beam_1', 'beam_2')
				 cycle          : cycle string ('injection', 'flattop')
				 cycleTime      : cycle step string ('injection_start', 'injection_end', 'flattop_start', 'flattop_end')
				 mask_invalid   : boolean True/False to mask invalid values in the output array
		Returns: filled_slots array
		'''
		return ma.masked_invalid(dict_intervals_two_beams[beam][cycle]['filled_slots'])

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def convertToLocalTime(self, df_timeRange, timezone="Europe/Zurich", scalar=False):
		'''
		Converts a Series (Pandas Column) in Local Time and returns it.
		Input  : df_timeRange : Pandas Series with timestamp data
				 timezone     : timezone (tz) string - defaults to Europe/Zurich
				 scalar 	  : if it is a single value and not list/array
		Returns: converted pandas Series
		'''
		return np.array(pd.to_datetime(np.array(df_timeRange), unit='s', utc=True).tz_convert(timezone).tz_localize(None))

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	
	def plot_mean_and_spread(self, axc, x_vect, ymat, color = 'k', alpha=1, label=None, shade=False):
		'''
		Function to plot the average value and the +/- 2sigma range of a given
		vector on a given axis

		Inputs : axc : axis on which you want me to plot stuff
				 x_vect: array of x-axis values (any iterable object)
				 y_mat : array of y-axis values for which the std and mean will be Calculated
				 color : the color for the lines
				 alpha : the alpha of the lines
				 label : label to add to this axis
				 shade : plot in shade the +/- 1 standard deviation band
		Returns: None -- simply adds stuff on the given pl.axis
		'''
		avg = np.nanmean(ymat, axis=1)
		std = np.nanstd(ymat, axis=1)

		if shade:
			axc.fill_between(x_vect, avg-std, avg+std, alpha=.3, color=color, label=None)
		else:
			axc.plot(x_vect, avg-1*std, '--', color=color, linewidth=1, alpha=alpha, label=None)
			axc.plot(x_vect, avg+1*std, '--', color=color, linewidth=1, alpha=alpha, label=None)
		axc.plot(x_vect, avg, color=color, linewidth=2, alpha=alpha, label=label)

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def getCycleDataTimes(self, filln):
		'''
		Returns : for cycle t_start_fill, t_end_fill, t_fill_len, t_ref
		'''
		t_start_fill = self.bmodes['t_startfill'][filln]
		t_end_fill   = self.bmodes['t_endfill'][filln]
		t_fill_len   = t_end_fill - t_start_fill
		t_ref        = t_start_fill

		return t_start_fill, t_end_fill, t_fill_len, t_ref

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	
	def getSBDataTimes(self, filln):
		'''
		Function to return times for the SB data
		Input : filln : fill number
		Returns: t_start_STABLE, t_end_STABLE, time_range, N_steps
		'''
		t_start_STABLE = self.bmodes['t_start_STABLE'][filln]
		t_end_STABLE   = self.bmodes['t_endfill'][filln]
		time_range     = np.arange(t_start_STABLE, t_end_STABLE-15*60, config.t_step_sec)
		N_steps        = len(time_range)
		return t_start_STABLE, t_end_STABLE, time_range, N_steps

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def getBmodesDF(self):
		'''
		Function to get bmodes from fills_bmodes file and for first/last fill and min time
		in stable beams. Also takes into account rescaling and crossing angle changes.
		Input:  fills_bmodes_file : (string) filename of the bmodes file
				min_time_SB       : (float) minimum time in stable beams requirement
				first_fill        : (float) first fill number
				last_fill         : (float) last fill number
				crossingAngleChange:(bool) consider crossing angle changes
				XingAngle         : (dict) crossing angle dictionary fills: xing angle
				doRescale         : (bool) consider BSRT rescaling
				resc_period       : (list) list of tuples for (orig,resc) periods
		Returns bmodes dataframe and fill number list
		'''

		## Get the bmodes information into a pandas dataframe
		with open(self.fills_bmodes_file, 'rb') as fid:
			dict_fill_bmodes = pickle.load(fid)
		bmodes = pd.DataFrame.from_dict(dict_fill_bmodes, orient='index') ##index is the fill number

		## Clean up bmodes df depending on SB flag and SB duration
		## Same clean up as the one done in 001c loop
		#if config.min_time_SB > 0 :
		#	bmodes = bmodes[:][(bmodes['t_start_STABLE'] > 0) & (bmodes['t_stop_STABLE']-bmodes['t_start_STABLE']>= config.min_time_SB)]

		bmodes = bmodes.ix[config.first_fill:config.last_fill]

		## Get the fill list
		filln_list = bmodes.index.values

		## Make a column with the "period" the fill belongs to
		bmodes['period'] = np.nan
		for key in config.periods:
			## key = 'A', 'B', 'C'
			bmodes['period'].loc[filln_list[np.logical_and(np.less(filln_list,config.periods[key][1]), np.greater_equal(filln_list,config.periods[key][0]))]]=str(key)

		bmodes['CrossingAngle_ATLAS'] = np.nan
		bmodes['CrossingAngle_CMS'] = np.nan
		for key in config.XingAngle:
			bmodes['CrossingAngle_ATLAS'].loc[filln_list[np.logical_and(np.less(filln_list,key[1]), np.greater_equal(filln_list,key[0]))]]=config.XingAngle[key][0]
			bmodes['CrossingAngle_CMS'].loc[filln_list[np.logical_and(np.less(filln_list,key[1]), np.greater_equal(filln_list,key[0]))]]=config.XingAngle[key][1]

		######################################################     RESCALING     ############################################################
		if config.doRescale:
			#import BSRT_calib_rescale as BSRT_calib
			bmodes['rescaledPeriod'] = np.nan
			for res in config.resc_period:
				bmodes['rescaledPeriod'][bmodes['period']==res[0]]=res[1]
		else:
			#import BSRT_calib as BSRT_calib
			bmodes['rescaledPeriod'] = bmodes['period'] # @TODO IS THIS NEEDED?


		filln_list = bmodes.index.values

		return bmodes, filln_list

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *	

	def colorprog(self, i_prog, Nplots, v1 = .9, v2 = 1.):
		if hasattr(Nplots, '__len__'):
			Nplots = len(Nplots)
		return hsv_to_rgb(float(i_prog)/float(Nplots), v1, v2)

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *	

	########################################################################################################################
	#
	#													CYCLE PLOTS
	#
	#########################################################################################################################
	
	def plotCycleEmittances(self, save=False, batch=False, show_only=None):
		'''
		show_only == 'injection', 'startRamp', 'endRamp', 'startSB'
		'''
		dict_intervals_two_beams = self.filln_CycleDict

		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		showAll = True
		if show_only is not None:
			showAll = False

		##### BBB Emittances
		info('#plotCycleEmittances : Fill {} -> Making Cycle Emittances bbb plot...'.format(self.filln))
		pl.close('all')
		fig_bbbemit = pl.figure("Emittances B1", figsize=(14, 7))
		fig_bbbemit.set_facecolor('w')
		ax_b1_h = pl.subplot(2,1,1)
		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1 -  H
		if showAll or show_only == 'injection':
			ax_b1_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith'])), '.', color='blue',   markersize=8, label='Injected')
		if showAll or show_only == 'startRamp':
			ax_b1_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith'])),   '.', color='orange', markersize=8, label='Start Ramp')
		if showAll or show_only == 'endRamp':
			ax_b1_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith'])),  '.', color='green',  markersize=8, label='End Ramp')
		if showAll or show_only == 'startSB':
			ax_b1_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b1_h.set_ylabel("B1 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b1_h.minorticks_on()
		ax_b1_h.set_ylim(1,3)
		# ax_b1_h.set_ylim(0,6)
		ax_b1_h.text(0.5, 0.9, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])))
                                                                                                                                                         ), horizontalalignment='center', verticalalignment='top', transform=ax_b1_h.transAxes,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b1_h.get_position()
		ax_b1_h.set_position([box.x0, box.y0, box.width*0.8, box.height])
		# Put a legend to the right of the current axis
		ax_b1_h.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, numpoints=1)
		ax_b1_h.grid('on', which='both')
                

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1 -  V

		ax_b1_v = pl.subplot(2,1,2, sharex=ax_b1_h, sharey=ax_b1_h)
		if showAll or show_only == 'injection':
			ax_b1_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",     "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv'])), '.', color='blue',   markersize=8, label='Injected')
		if showAll or show_only == 'startRamp':		
			ax_b1_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",     "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv'])),   '.', color='orange', markersize=8, label='Start Ramp')
		if showAll or show_only == 'endRamp':
			ax_b1_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",  "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv'])),  '.', color='green',  markersize=8, label='End Ramp')
		if showAll or show_only == 'startSB':
			ax_b1_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",  "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b1_v.set_ylabel("B1 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b1_v.minorticks_on()
		ax_b1_v.set_ylim(1,3)
		# ax_b1_v.set_ylim(0,6)
		ax_b1_v.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')
		ax_b1_v.text(0.5, 0.9, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv']))),
					  np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv']))
                                                                                                                                                        )), horizontalalignment='center', verticalalignment='top', transform=ax_b1_v.transAxes,
                bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)
		## Shrink current axis by 20%
		box = ax_b1_v.get_position()
		ax_b1_v.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		# Put a legend to the right of the current axis
		ax_b1_v.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, numpoints=1)
		ax_b1_v.grid('on', which='both')

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2  - H
		fig_bbbemit2 = pl.figure("Emittances B2", figsize=(14, 7))
		fig_bbbemit2.set_facecolor('w')
		ax_b2_h = pl.subplot(2,1,1)

		if showAll or show_only == 'injection':
			ax_b2_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith'])), '.', color='blue',   markersize=8, label='Injected')
		if showAll or show_only == 'startRamp':		
			ax_b2_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith'])),   '.', color='orange', markersize=8, label='Start Ramp')
		if showAll or show_only == 'endRamp':
			ax_b2_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith'])),  '.', color='green',  markersize=8, label='End Ramp')
		if showAll or show_only == 'startSB':
			ax_b2_h.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_end",     mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b2_h.set_ylabel("B2 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b2_h.minorticks_on()
		ax_b2_h.set_ylim(1,3)
		# ax_b2_h.set_ylim(0,6)
		ax_b2_h.text(0.5, 0.9, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith']))),
					  np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith']))),
                                                                                                                                                        np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])))), horizontalalignment='center', verticalalignment='top', transform=ax_b2_h.transAxes,
                             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)
		## Shrink current axis by 20%
		box = ax_b2_h.get_position()
		ax_b2_h.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		# Put a legend to the right of the current axis
		ax_b2_h.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, numpoints=1)
		ax_b2_h.grid('on', which='both')

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2  - V
		ax_b2_v = pl.subplot(2,1,2, sharex=ax_b2_h, sharey=ax_b2_h)

		if showAll or show_only == 'injection':
			ax_b2_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv'])), '.', color='blue',   markersize=8, label='Injected')
		if showAll or show_only == 'startRamp':		
			ax_b2_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv'])),   '.', color='orange', markersize=8, label='Start Ramp')
		if showAll or show_only == 'endRamp':
			ax_b2_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv'])),  '.', color='green',  markersize=8, label='End Ramp')
		if showAll or show_only == 'startSB':
			ax_b2_v.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b2_v.set_ylabel("B2 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b2_v.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')
		ax_b2_v.minorticks_on()
		ax_b2_v.set_ylim(1,3)
		# ax_b2_v.set_ylim(0,6)
		ax_b2_v.text(0.5, 0.9, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv']))),
                                                                                                                                                        np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv']))),
                                                                                                                                                        np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])))), horizontalalignment='center', verticalalignment='top', transform=ax_b2_v.transAxes,
                             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)
		## Shrink current axis by 20%
		box = ax_b2_v.get_position()
		ax_b2_v.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		# Put a legend to the right of the current axis
		ax_b2_v.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, numpoints=1)
		ax_b2_v.grid('on', which='both')

		# tref string
		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)

		fig_bbbemit.suptitle(subtitle, fontsize=16, fontweight='bold')
		fig_bbbemit2.suptitle(subtitle, fontsize=16, fontweight='bold')
		pl.subplots_adjust(hspace=0.5, left=0.1, right=0.8)#, right=0.02, left=0.01)


		if save:
			filename = self.savedir+"fill_{}_cycle_emittancesbbb_<BEAM>.pdf".format(self.filln)
			print filename
			fig_bbbemit.savefig(filename.replace("<BEAM>", 'b1'),  dpi=300)
			fig_bbbemit2.savefig(filename.replace("<BEAM>", 'b2'), dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def getCycleEmittanceTable(self, save=False, batch=False, print_table=True):

		dict_intervals_two_beams = self.filln_CycleDict
		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		
		def rd(a, b):
			return ((b-a)/a)*100.0
		def calcEmittanceBlowUps(b1h, b1v, b2h, b2v, filln, save):
			steps = ['Injection - Start Ramp', 'Start Ramp - End Ramp', 'End Ramp - Stable Beams', 'Injection - Stable Beams']

			lists = [b1h, b1v, b2h, b2v]
			beams = ['B1H', 'B1V', 'B2H', 'B2V']

			str_output   = """\\begin{tabular}{l|c|c|c|c}\n"""+"""{} & B1H [%] & B1V [%] & B2H [%] & B2V [%]\\\\\n\hline\\\\\n""".format(filln)
			str_template = """<STEP> & <B1H>  & <B1V>  & <B2H>  & <B2V>"""

			out_dict = OrderedDict()
			for beam in range(len(beams)):

				in_values = lists[beam]
				out_values = []
				out_values.append(rd(in_values[0], in_values[1]))
				out_values.append(rd(in_values[1], in_values[2]))
				out_values.append(rd(in_values[2], in_values[3]))
				out_values.append(rd(in_values[0], in_values[3]))

				out_dict.update({beams[beam] : zip(steps, out_values)})

				if print_table:
					print '---- FILL ', filln , ' ----'
					print 'Beam : ', beams[beam]
					for step, val in zip(steps, out_values):
						print step, ' {:.1f}'.format(val)
			
			if save:

				l1 = """{} & {:.2f}  & {:.2f}  & {:.2f}  & {:.2f}\\\\\n""".format(steps[0], out_dict['B1H'][0][1], out_dict['B1V'][0][1], out_dict['B2H'][0][1], out_dict['B2V'][0][1])
				l2 = """{} & {:.2f}  & {:.2f}  & {:.2f}  & {:.2f}\\\\\n""".format(steps[1], out_dict['B1H'][1][1], out_dict['B1V'][1][1], out_dict['B2H'][1][1], out_dict['B2V'][1][1])
				l3 = """{} & {:.2f}  & {:.2f}  & {:.2f}  & {:.2f}\\\\\n""".format(steps[2], out_dict['B1H'][2][1], out_dict['B1V'][2][1], out_dict['B2H'][2][1], out_dict['B2V'][2][1])
				l4 = """{} & {:.2f}  & {:.2f}  & {:.2f}  & {:.2f}\\\\\n""".format(steps[3], out_dict['B1H'][3][1], out_dict['B1V'][3][1], out_dict['B2H'][3][1], out_dict['B2V'][3][1])

				l5 = """\end{tabular}\n"""
				str_output = "".join((str_output, l1))
				str_output = "".join((str_output, l2))
				str_output = "".join((str_output, l3))
				str_output = "".join((str_output, l4))
				str_output = "".join((str_output, l5))
				print str_output

			return out_dict

		b1h_INJ = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith'])))
		b1h_FB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith'])))
		b1h_FT  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith'])))
		b1h_SB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])))

		b1v_INJ = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv'])))
		b1v_FB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv'])))
		b1v_FT  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv'])))
		b1v_SB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])))

		
		b2h_INJ = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith'])))
		b2h_FB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith'])))
		b2h_FT  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith'])))
		b2h_SB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])))
		

		b2v_INJ = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv'])))
		b2v_FB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv'])))
		b2v_FT  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv'])))
		b2v_SB  = np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])))

		dict_emit = OrderedDict()
		dict_emit.update({self.filln : [[b1h_INJ, b1h_FB, b1h_FT, b1h_SB],[b1v_INJ, b1v_FB, b1v_FT, b1v_SB],[b2h_INJ, b2h_FB, b2h_FT, b2h_SB],[b2v_INJ, b2v_FB, b2v_FT, b2v_SB]]})
		print dict_emit

		out_dict = {}
		for key in dict_emit.keys():
			out_dict = calcEmittanceBlowUps(dict_emit[key][0], dict_emit[key][1], dict_emit[key][2], dict_emit[key][3], key, save)

		return out_dict

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	
	def plotCycleEmittanceViolins(self, save=False, batch=False, return_df =False):

		dict_intervals_two_beams = self.filln_CycleDict
		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		Delta_t_b1_inj = (np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['time_meas']))
		Delta_t_b2_inj = (np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['time_meas']))

		Delta_t_b1_he = (np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['time_meas']))
		Delta_t_b2_he = (np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['time_meas']))


		mask_b1_inj = Delta_t_b1_inj > 5.0*60.0
		mask_b2_inj = Delta_t_b2_inj > 5.0*60.0

		mask_b1_he = Delta_t_b1_he > 5.0*60.0
		mask_b2_he = Delta_t_b2_he > 5.0*60.0

		b1h_INJ = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith'])[mask_b1_inj]
		b1h_FB  = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith'])[mask_b1_inj]
		b1h_FT  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith'])[mask_b1_he]
		b1h_SB  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])[mask_b1_he]

		b1v_INJ = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv'])[mask_b1_inj]
		b1v_FB  = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv'])[mask_b1_inj]
		b1v_FT  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv'])[mask_b1_he]
		b1v_SB  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])[mask_b1_he]

		
		b2h_INJ = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith'])[mask_b2_inj]
		b2h_FB  = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith'])[mask_b2_inj]
		b2h_FT  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith'])[mask_b2_he]
		b2h_SB  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])[mask_b2_he]
		

		b2v_INJ = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv'])[mask_b2_inj]
		b2v_FB  = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv'])[mask_b2_inj]
		b2v_FT  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv'])[mask_b2_he]
		b2v_SB  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])[mask_b2_he]

		b1h_emit = [b1h_INJ, b1h_FB, b1h_FT, b1h_SB]
		b1v_emit = [b1h_INJ, b1v_FB, b1v_FT, b1v_SB]
		
		b2h_emit = [b2h_INJ, b2h_FB, b2h_FT, b2h_SB]
		b2v_emit = [b2v_INJ, b2v_FB, b2v_FT, b2v_SB]

		emit  = []
		cycle = []
		beam  = []
		plane = []
		slot  = []

		# start with b1h
		emit.append(b1h_INJ)
		cycle.append(['Injection']*len(b1h_INJ))
		beam.append(['B1']*len(b1h_INJ))
		plane.append(['Horizontal']*len(b1h_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1h_FB)
		cycle.append(['Start Ramp']*len(b1h_FB))
		beam.append(['B1']*len(b1h_FB))
		plane.append(['Horizontal']*len(b1h_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1h_FT)
		cycle.append(['End Ramp']*len(b1h_FT))
		beam.append(['B1']*len(b1h_FT))
		plane.append(['Horizontal']*len(b1h_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b1h_SB)
		cycle.append(['Stable Beams']*len(b1h_SB))
		beam.append(['B1']*len(b1h_SB))
		plane.append(['Horizontal']*len(b1h_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		# b1v
		emit.append(b1v_INJ)
		cycle.append(['Injection']*len(b1v_INJ))
		beam.append(['B1']*len(b1v_INJ))
		plane.append(['Vertical']*len(b1v_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1v_FB)
		cycle.append(['Start Ramp']*len(b1v_FB))
		beam.append(['B1']*len(b1v_FB))
		plane.append(['Vertical']*len(b1v_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1v_FT)
		cycle.append(['End Ramp']*len(b1v_FT))
		beam.append(['B1']*len(b1v_FT))
		plane.append(['Vertical']*len(b1v_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b1v_SB)
		cycle.append(['Stable Beams']*len(b1v_SB))
		beam.append(['B1']*len(b1v_SB))
		plane.append(['Vertical']*len(b1v_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		# b2h
		emit.append(b2h_INJ)
		cycle.append(['Injection']*len(b2h_INJ))
		beam.append(['B2']*len(b2h_INJ))
		plane.append(['Horizontal']*len(b2h_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())


		emit.append(b2h_FB)
		cycle.append(['Start Ramp']*len(b2h_FB))
		beam.append(['B2']*len(b2h_FB))
		plane.append(['Horizontal']*len(b2h_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())

		emit.append(b2h_FT)
		cycle.append(['End Ramp']*len(b2h_FT))
		beam.append(['B2']*len(b2h_FT))
		plane.append(['Horizontal']*len(b2h_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b2h_SB)
		cycle.append(['Stable Beams']*len(b2h_SB))
		beam.append(['B2']*len(b2h_SB))
		plane.append(['Horizontal']*len(b2h_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		# b2v
		emit.append(b2v_INJ)
		cycle.append(['Injection']*len(b2v_INJ))
		beam.append(['B2']*len(b2v_INJ))
		plane.append(['Vertical']*len(b2v_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())

		emit.append(b2v_FB)
		cycle.append(['Start Ramp']*len(b2v_FB))
		beam.append(['B2']*len(b2v_FB))
		plane.append(['Vertical']*len(b2v_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())

		emit.append(b2v_FT)
		cycle.append(['End Ramp']*len(b2v_FT))
		beam.append(['B2']*len(b2v_FT))
		plane.append(['Vertical']*len(b2v_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b2v_SB)
		cycle.append(['Stable Beams']*len(b2v_SB))
		beam.append(['B2']*len(b2v_SB))
		plane.append(['Vertical']*len(b2v_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		emit = list(itertools.chain.from_iterable(emit))
		cycle = list(itertools.chain.from_iterable(cycle))
		beam = list(itertools.chain.from_iterable(beam))
		plane = list(itertools.chain.from_iterable(plane))
		slot = list(itertools.chain.from_iterable(slot))

		df_emit = pd.DataFrame()
		df_emit['Emittance'] = pd.Series(emit, dtype='float')
		df_emit['Cycle']     = pd.Series(cycle, dtype='category')
		df_emit['Plane']     = pd.Series(plane, dtype='category')
		df_emit['Beam']      = pd.Series(beam, dtype='category')
		df_emit['Slot']      = pd.Series(slot, dtype='int')
		df_emit['fill']		 = [int(self.filln)]*len(df_emit)
		df_emit['bunches']	 = [int(self.bunches)]*len(df_emit)

		df_emit = df_emit.dropna(axis=0, how='any')


		# df_relative_differences = pd.DataFrame()
		# B1H_INJ_FB = (b1h_FB - b1h_INJ)
		# B1H_FB_FT  = (b1h_FT - b1h_FB)
		# B1H_FT_SB  = (b1h_SB - b1h_FT)
		# B1H_INJ_SB = (b1h_SB - b1h_INJ)

		# B1V_INJ_FB = (b1v_FB - b1v_INJ)
		# B1V_FB_FT  = (b1v_FT - b1v_FB)
		# B1V_FT_SB  = (b1v_SB - b1v_FT)
		# B1V_INJ_SB = (b1v_SB - b1v_INJ)


		# B2H_INJ_FB = (b2h_FB - b2h_INJ)
		# B2H_FB_FT  = (b2h_FT - b2h_FB)
		# B2H_FT_SB  = (b2h_SB - b2h_FT)
		# B2H_INJ_SB = (b2h_SB - b2h_INJ)

		# B2V_INJ_FB = (b2v_FB - b2v_INJ)
		# B2V_FB_FT  = (b2v_FT - b2v_FB)
		# B2V_FT_SB  = (b2v_SB - b2v_FT)
		# B2V_INJ_SB = (b2v_SB - b2v_INJ)


		# B1H_INJ_FB = B1H_INJ_FB[~np.isnan(B1H_INJ_FB)].tolist()
		# B1H_FB_FT  =  B1H_FB_FT[~np.isnan(B1H_FB_FT)].tolist()
		# B1H_FT_SB  =  B1H_FT_SB[~np.isnan(B1H_FT_SB)].tolist()
		# B1H_INJ_SB = B1H_INJ_SB[~np.isnan(B1H_INJ_SB)].tolist()
		# B1V_INJ_FB = B1V_INJ_FB[~np.isnan(B1V_INJ_FB)].tolist()
		# B1V_FB_FT  =  B1V_FB_FT[~np.isnan(B1V_FB_FT)].tolist()
		# B1V_FT_SB  =  B1V_FT_SB[~np.isnan(B1V_FT_SB)].tolist()
		# B1V_INJ_SB = B1V_INJ_SB[~np.isnan(B1V_INJ_SB)].tolist()
		# B2H_INJ_FB = B2H_INJ_FB[~np.isnan(B2H_INJ_FB)].tolist()
		# B2H_FB_FT  =  B2H_FB_FT[~np.isnan(B2H_FB_FT)].tolist()
		# B2H_FT_SB  =  B2H_FT_SB[~np.isnan(B2H_FT_SB)].tolist()
		# B2H_INJ_SB = B2H_INJ_SB[~np.isnan(B2H_INJ_SB)].tolist()
		# B2V_INJ_FB = B2V_INJ_FB[~np.isnan(B2V_INJ_FB)].tolist()
		# B2V_FB_FT  =  B2V_FB_FT[~np.isnan(B2V_FB_FT)].tolist()
		# B2V_FT_SB  =  B2V_FT_SB[~np.isnan(B2V_FT_SB)].tolist()
		# B2V_INJ_SB = B2V_INJ_SB[~np.isnan(B2V_INJ_SB)].tolist()



		# blowup = []
		# bcycle  = []
		# bbeam   = []
		# bplane  = []
		# print blowup


		# # start with b1h
		# blowup.append(B1H_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B1H_INJ_FB))
		# bbeam.append(['B1']*len(B1H_INJ_FB))
		# bplane.append(['Horizontal']*len(B1H_INJ_FB))

		# blowup.append(B1H_FB_FT)
		# bcycle.append(['FB2FT']*len(B1H_FB_FT))
		# bbeam.append(['B1']*len(B1H_FB_FT))
		# bplane.append(['Horizontal']*len(B1H_FB_FT))
		
		# blowup.append(B1H_FT_SB)
		# bcycle.append(['FT2SB']*len(B1H_FT_SB))
		# bbeam.append(['B1']*len(B1H_FT_SB))
		# bplane.append(['Horizontal']*len(B1H_FT_SB))

		# blowup.append(B1H_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B1H_INJ_SB))
		# bbeam.append(['B1']*len(B1H_INJ_SB))
		# bplane.append(['Horizontal']*len(B1H_INJ_SB))

		# # b1v
		# blowup.append(B1V_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B1V_INJ_FB))
		# bbeam.append(['B1']*len(B1V_INJ_FB))
		# bplane.append(['Vertical']*len(B1V_INJ_FB))

		# blowup.append(B1V_FB_FT)
		# bcycle.append(['FB2FT']*len(B1V_FB_FT))
		# bbeam.append(['B1']*len(B1V_FB_FT))
		# bplane.append(['Vertical']*len(B1V_FB_FT))
		
		# blowup.append(B1V_FT_SB)
		# bcycle.append(['FT2SB']*len(B1V_FT_SB))
		# bbeam.append(['B1']*len(B1V_FT_SB))
		# bplane.append(['Vertical']*len(B1V_FT_SB))

		# blowup.append(B1V_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B1V_INJ_SB))
		# bbeam.append(['B1']*len(B1V_INJ_SB))
		# bplane.append(['Vertical']*len(B1V_INJ_SB))


		# # start with b2h
		# blowup.append(B2H_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B2H_INJ_FB))
		# bbeam.append(['B2']*len(B2H_INJ_FB))
		# bplane.append(['Horizontal']*len(B2H_INJ_FB))

		# blowup.append(B2H_FB_FT)
		# bcycle.append(['FB2FT']*len(B2H_FB_FT))
		# beam.append(['B2']*len(B2H_FB_FT))
		# bplane.append(['Horizontal']*len(B2H_FB_FT))
		
		# blowup.append(B2H_FT_SB)
		# bcycle.append(['FT2SB']*len(B2H_FT_SB))
		# bbeam.append(['B2']*len(B2H_FT_SB))
		# bplane.append(['Horizontal']*len(B2H_FT_SB))

		# blowup.append(B2H_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B2H_INJ_SB))
		# bbeam.append(['B2']*len(B2H_INJ_SB))
		# bplane.append(['Horizontal']*len(B2H_INJ_SB))

		# # b2v
		# blowup.append(B2V_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B2V_INJ_FB))
		# bbeam.append(['B2']*len(B2V_INJ_FB))
		# bplane.append(['Vertical']*len(B2V_INJ_FB))

		# blowup.append(B2V_FB_FT)
		# bcycle.append(['FB2FT']*len(B2V_FB_FT))
		# bbeam.append(['B2']*len(B2V_FB_FT))
		# bplane.append(['Vertical']*len(B2V_FB_FT))
		
		# blowup.append(B2V_FT_SB)
		# bcycle.append(['FT2SB']*len(B2V_FT_SB))
		# bbeam.append(['B2']*len(B2V_FT_SB))
		# bplane.append(['Vertical']*len(B2V_FT_SB))

		# blowup.append(B2V_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B2V_INJ_SB))
		# bbeam.append(['B2']*len(B2V_INJ_SB))
		# bplane.append(['Vertical']*len(B2V_INJ_SB))


		# blowup = list(itertools.chain.from_iterable(blowup))
		# bcycle = list(itertools.chain.from_iterable(bcycle))
		# bbeam = list(itertools.chain.from_iterable(bbeam))
		# bplane = list(itertools.chain.from_iterable(bplane))

		# bfill = [self.filln] * len(bplane)
		# bbunches = [self.bunches] * len(bplane)


		# df_blowup = pd.DataFrame()
		# df_blowup['Emittance'] = pd.Series(blowup, dtype='float')
		# df_blowup['Cycle']     = pd.Series(bcycle, dtype='category')
		# df_blowup['Plane']     = pd.Series(bplane, dtype='category')
		# df_blowup['Beam']      = pd.Series(bbeam, dtype='category')
		# df_blowup['Fill']	   = pd.Series(bfill, dtype='int')
		# df_blowup['Bunches']   = pd.Series(bbunches, dtype='int')



		mean_values_b1 = []
		for ncycle in ['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams']:
			for nplane in ['Horizontal', 'Vertical']:
				mean_values_b1.append(df_emit['Emittance'][(df_emit['Cycle']==ncycle) & (df_emit['Plane']==nplane) & (df_emit['Beam']=='B1')].mean())
				# print np.array(df_emit['Emittance'][(df_emit['Cycle']==ncycle) & (df_emit['Plane']==nplane) & (df_emit['Beam']=='B1')])[0]
		

		mean_values_b2 = []
		for ncycle in ['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams']:
			for nplane in ['Horizontal', 'Vertical']:
				mean_values_b2.append(df_emit['Emittance'][(df_emit['Cycle']==ncycle) & (df_emit['Plane']==nplane) & (df_emit['Beam']=='B2')].mean())

		mean_labels_b1 = [str(np.round(s, 2))+'$\mu$m' for s in mean_values_b1]
		mean_labels_b2 = [str(np.round(s, 2))+'$\mu$m' for s in mean_values_b2] 

		print mean_labels_b1, mean_labels_b2

		pos = range(len(mean_labels_b1))
		
		sns.set_style("whitegrid")
		pl.close('all')

		fig_emit_volin = pl.figure('Violins', figsize=(15,8))
		ax_b1 = pl.subplot(211)
		ax_b1 = sns.violinplot(x="Cycle", y="Emittance", hue="Plane", data=df_emit[df_emit['Beam']=='B1'], palette=['#71aaf2', '#ff8a3d'], split=True, order=['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams'])
		ax_b1.get_axes().xaxis.set_ticklabels([])
		ax_b1.set_ylim(1,4)
		ax_b1.set_ylabel('B1 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
		ax_b1.set_xlabel('')


		ax_b2 = pl.subplot(212)#, sharex=ax_b1h, sharey=ax_b1h)
		ax_b2 = sns.violinplot(x="Cycle", y="Emittance", hue="Plane", data=df_emit[df_emit['Beam']=='B2'],  palette=['#71aaf2', '#ff8a3d'], split=True, order=['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams'])
		ax_b2.get_axes().xaxis.set_ticklabels([])
		ax_b2.set_ylim(1,4)
		ax_b2.set_ylabel('B2 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
		ax_b2.get_axes().xaxis.set_ticklabels(['INJECTION', 'Start RAMP', 'End RAMP', 'Start STABLE BEAMS'])
		ax_b2.set_xlabel('')
		for tick in ax_b2.xaxis.get_major_ticks():
			tick.label.set_fontsize(16)
			tick.label.set_fontweight('bold')

		for tick in ax_b1.yaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		for tick in ax_b2.yaxis.get_major_ticks():
			tick.label.set_fontsize(14)

		ax_b1.legend(loc=2, fontsize=14)
		ax_b2.legend(loc=2, fontsize=14)

		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)
		fig_emit_volin.suptitle(subtitle, fontsize=16, fontweight='bold')

		for tick,label in zip(pos,ax_b1.get_xticklabels()):
			ax_b1.text(pos[tick]-0.15, mean_values_b1[2*tick],   mean_labels_b1[2*tick],    horizontalalignment='center', size='small', color='k', weight='semibold')
			ax_b1.text(pos[tick]+0.15, mean_values_b1[2*tick+1], mean_labels_b1[2*tick+1],  horizontalalignment='center', size='small', color='k', weight='semibold')
		for tick,label in zip(pos,ax_b2.get_xticklabels()):
			ax_b2.text(pos[tick]-0.15, mean_values_b2[2*tick],   mean_labels_b2[2*tick], 	 horizontalalignment='center', size='small', color='k', weight='semibold')
			ax_b2.text(pos[tick]+0.15, mean_values_b2[2*tick+1], mean_labels_b2[2*tick+1],  horizontalalignment='center', size='small', color='k', weight='semibold')

		if save:
			filename = self.savedir+"fill_{}_cycle_emittancesViolin_HV.pdf".format(self.filln)
			fig_emit_volin.savefig(filename,  dpi=300)
			
		if not batch:
			pl.show()

		if return_df:
			return df_emit

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleIntensities(self, save=False, batch=False):
		dict_intervals_two_beams = self.filln_CycleDict

		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		#################
		pl.close('all')
		info('#plotCycleIntensities : Fill {} -> Making Cycle Intensities bbb plot...'.format(filln))
		fig_bbbintens = pl.figure("Intensities", figsize=(14, 7))
		fig_bbbintens.set_facecolor('w')

		ax_b1 = pl.subplot(2,1,1)

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['intensity'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['intensity']  )),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['intensity'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",      mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['intensity']  )),    '.', color='red',    markersize=8, label='Start SB')
		ax_b1.set_ylabel("B1 Intensity [ppb]", fontsize=14, fontweight='bold')
		ax_b1.minorticks_on()
		ax_b1.text(0.5, 0.9, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['intensity']  ))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['intensity']  ))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['intensity']  ))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['intensity']  )))), horizontalalignment='center', verticalalignment='top', transform=ax_b1.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b1.get_position()
		ax_b1.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b1.grid('on', which='both')
		ax_b1.set_ylim(0.5e11, 1.7e11)

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2
		ax_b2 = pl.subplot(2,1,2)
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['intensity'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['intensity']  )),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_start",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['intensity'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_end",      mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['intensity']  )),    '.', color='red',    markersize=8, label='Start SB')
		ax_b2.set_ylabel("B2 Intensity [ppb]", fontsize=14, fontweight='bold')
		ax_b2.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')
		ax_b2.minorticks_on()
		ax_b2.text(0.5, 0.9, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['intensity']  ))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['intensity']  ))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['intensity']  ))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['intensity']  )))), horizontalalignment='center', verticalalignment='center', transform=ax_b2.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b2.get_position()
		ax_b2.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b2.grid('on', which='both')
		ax_b2.set_ylim(0.5e11, 1.7e11)
		# tref string
		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)

		fig_bbbintens.suptitle(subtitle, fontsize=16, fontweight='bold')

		if save:
			filename = self.savedir+"fill_{}_cycle_intensitiesbbb.pdf".format(self.filln)
			print filename
			fig_bbbintens.savefig(filename,  dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleBrightness(self, save=False, batch=False):
		dict_intervals_two_beams = self.filln_CycleDict

		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		##########
		pl.close('all')
		info('#plotCycleBrightness : Fill {} -> Making Cycle Brightness bbb plot...'.format(filln))
		fig_bbbbright = pl.figure("Brightness", figsize=(14, 7))
		fig_bbbbright.set_facecolor('w')
		ax_b1 = pl.subplot(2,1,1)
		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['brightness'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['brightness'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['brightness'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b1.set_ylabel("B1 Brightness [p/$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b1.minorticks_on()
		ax_b1.text(0.5, 0.9, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['brightness'])))), horizontalalignment='center', verticalalignment='top', transform=ax_b1.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b1.get_position()
		ax_b1.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b1.grid('on', which='both')


		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2
		ax_b2 = pl.subplot(2,1,2)
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['brightness'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['brightness'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['brightness'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['brightness'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b2.set_ylabel("B2 Brightness [p/$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b2.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')
		ax_b2.minorticks_on()
		ax_b2.text(0.5, 0.9, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['brightness'])))), horizontalalignment='center', verticalalignment='center', transform=ax_b2.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b2.get_position()
		ax_b2.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b2.grid('on', which='both')

		# tref string
		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)

		fig_bbbbright.suptitle(subtitle, fontsize=16, fontweight='bold')
		if self.savePlots:
			filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))+"fill_{}_cycle_brightnessbbb".format(filln)+self.plotFormat
			pl.savefig(filename, dpi=self.plotDpi)

		if save:
			filename = self.savedir+"fill_{}_cycle_brightnessbbb.pdf".format(self.filln)
			print filename
			fig_bbbbright.savefig(filename,  dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleBunchLength(self, save=False, batch=False):
		dict_intervals_two_beams = self.filln_CycleDict

		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)
		#################### ------- #
		pl.close('all')
		info('#plotCycleBunchLength : Fill {} -> Making Cycle Bunch Length bbb plot...'.format(self.filln))
		fig_bbbblength = pl.figure("blength", figsize=(14, 7))
		fig_bbbblength.set_facecolor('w')
		ax_b1 = pl.subplot(2,1,1)

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['blength']))/1.0e-9, '.', color='blue',   markersize=8, label='Injected')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['blength']))/1.0e-9,   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['blength']))/1.0e-9,  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['blength']))/1.0e-9,    '.', color='red',    markersize=8, label='Start SB')
		ax_b1.set_ylabel("B1 Bunch Length [ns]", fontsize=12, fontweight='bold')
		ax_b1.minorticks_on()
		ax_b1.text(0.5, 0.1, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['blength']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['blength']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['blength']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['blength'])))), horizontalalignment='center', verticalalignment='top', transform=ax_b1.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b1.get_position()
		ax_b1.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b1.grid('on', which='both')
		ax_b1.set_ylim(0.5, 1.5)


		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2
		ax_b2 = pl.subplot(2,1,2)
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['blength']))/1.0e-9, '.', color='blue',   markersize=8, label='Injected')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['blength']))/1.0e-9,   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['blength']))/1.0e-9,  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2.plot(self.getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['blength']))/1.0e-9,    '.', color='red',    markersize=8, label='Start SB')
		ax_b2.set_ylabel("B2 Bunch Length [ns]", fontsize=12, fontweight='bold')
		ax_b2.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')
		ax_b2.minorticks_on()
		ax_b2.text(0.5, 0.1, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['blength']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['blength']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['blength']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['blength']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['blength'])))), horizontalalignment='center', verticalalignment='center', transform=ax_b2.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b2.get_position()
		ax_b2.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b2.grid('on', which='both')
		ax_b2.set_ylim(0.5, 1.5)

		# tref string
		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)

		fig_bbbblength.suptitle(subtitle, fontsize=16, fontweight='bold')

		if save:
			filename = self.savedir+"fill_{}_cycle_blengthbbb.pdf".format(self.filln)
			print filename
			fig_bbbblength.savefig(filename,  dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleTime(self, save=False, batch=False):
		dict_intervals_two_beams = self.filln_CycleDict

		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)
		#####-----
		pl.close('all')
		info('#plotCycleTime : Fill {} -> Making Cycle Time bbb plot...'.format(filln))
		fig_bbbtime = pl.figure("time", figsize=(14, 7))
		fig_bbbtime.set_facecolor('w')
		# injection
		time_b1_inj_atStart = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['time_meas'])
		time_b2_inj_atStart = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['time_meas'])
		time_b1_inj_atEnd   = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['time_meas'])
		time_b2_inj_atEnd   = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['time_meas'])


		# flattop
		time_b1_ft_atStart = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['time_meas'])
		time_b2_ft_atStart = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['time_meas'])
		time_b1_ft_atEnd   = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['time_meas'])
		time_b2_ft_atEnd   = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['time_meas'])


		time_b1_inj = (time_b1_inj_atEnd - time_b1_inj_atStart)/60.
		time_b1_ft  = (time_b1_ft_atEnd  - time_b1_ft_atStart)/60.
		time_b2_inj = (time_b2_inj_atEnd - time_b2_inj_atStart)/60.
		time_b2_ft  = (time_b2_ft_atEnd  - time_b2_ft_atStart)/60.

		mean_time_b1_inj = np.nanmean(time_b1_inj)
		mean_time_b1_ft  = np.nanmean(time_b1_ft)
		mean_time_b2_inj = np.nanmean(time_b2_inj)
		mean_time_b2_ft  = np.nanmean(time_b2_ft)

		std_time_b1_inj = (np.nanstd(time_b1_inj_atStart) + np.nanstd(time_b1_inj_atEnd))/60.
		std_time_b1_ft  = (np.nanstd(time_b1_ft_atStart) + np.nanstd(time_b1_ft_atEnd))/60.

		std_time_b2_inj = (np.nanstd(time_b2_inj_atStart) + np.nanstd(time_b2_inj_atEnd))/60.
		std_time_b2_ft  = (np.nanstd(time_b2_ft_atStart) + np.nanstd(time_b2_ft_atEnd))/60.

		ax_b1 = pl.subplot(2,1,1)

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",  "at_start",  mask_invalid=True), time_b1_inj, '.', color='blue',   markersize=8, label='Injection')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",      mask_invalid=True), time_b1_ft,    '.', color='red',    markersize=8, label='Flat Top')
		ax_b1.set_ylabel("B1 Time [min]", fontsize=12, fontweight='bold')
		ax_b1.minorticks_on()
		ax_b1.text(0.5, 0.9, "Injection: {:.2f}$\pm${:.2f} | Flat Top : {:.2f}$\pm${:.2f}".format(mean_time_b1_inj, std_time_b1_inj,mean_time_b1_ft,std_time_b1_ft ),

				#(np.mean(time_b1_inj)/60., (np.std(time_b1_inj_atEnd)+np.std(time_b1_inj_atStart))/60.,
				#        np.mean(time_b1_ft_atEnd - time_b1_ft_atStart)/60.), (np.std(time_b1_ft_atEnd)+np.std(time_b1_ft_atStart))/60.),
						horizontalalignment='center', verticalalignment='top', transform=ax_b1.transAxes,
						bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)

		## Shrink current axis by 20%
		box = ax_b1.get_position()
		ax_b1.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b1.grid('on', which='both')


		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2
		ax_b2 = pl.subplot(2,1,2)
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",  "at_start",  mask_invalid=True), (time_b2_inj_atEnd-time_b2_inj_atStart)/60., '.', color='blue',   markersize=8, label='Injection')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_end",      mask_invalid=True), (time_b2_ft_atEnd-time_b2_ft_atStart)/60.,    '.', color='red',    markersize=8, label='Flat Top')
		ax_b2.set_ylabel("B2 Time [min]", fontsize=12, fontweight='bold')
		ax_b2.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')
		ax_b2.minorticks_on()
		ax_b2.text(0.5, 0.9, "Injection: {:.2f}$\pm${:.2f} | Flat Top : {:.2f}$\pm${:.2f}".format(mean_time_b2_inj, std_time_b2_inj,mean_time_b2_ft,std_time_b2_ft ),
						horizontalalignment='center', verticalalignment='top', transform=ax_b2.transAxes,
						bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)

		## Shrink current axis by 20%
		box = ax_b2.get_position()
		ax_b2.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_b2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14, numpoints=1)
		ax_b2.grid('on', which='both')

		# tref string
		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)

		fig_bbbtime.suptitle(subtitle, fontsize=16, fontweight='bold')

		if save:
			filename = self.savedir+"fill_{}_cycle_timebbb.pdf".format(self.filln)
			print filename
			fig_bbbtime.savefig(filename,  dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleHistos(self, save=False, batch=False):
		dict_intervals_two_beams = self.filln_CycleDict

		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)
		## ---
		pl.close('all')
		info('#plotCycleHistos : Fill {} -> Making Cycle Histograms bbb plot...'.format(filln))
		fig_hist = pl.figure("Histograms", figsize=(18, 9))
		fig_hist.clf()
		fig_hist.set_facecolor('w')

		ax_emit_b1_h = pl.subplot(4,2,1)
		ax_emit_b1_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith'])),    range=(0, 5),   bins=50, color='blue',   histtype='step', lw=2,  label='Injected')
		ax_emit_b1_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith'])),      range=(0, 5),   bins=50, color='orange', histtype='step', lw=2,  label='Start Ramp')
		ax_emit_b1_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith'])),     range=(0, 5),   bins=50, color='green',  histtype='step', lw=2,  label='End Ramp')
		ax_emit_b1_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])),       range=(0, 5),   bins=50, color='red',    histtype='step', lw=2,  label='Start SB')
		ax_emit_b1_h.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_emit_b1_h.set_xlabel('B1 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]', fontweight='bold', fontsize=8)
		## Shrink current axis by 20%
		box = ax_emit_b1_h.get_position()
		ax_emit_b1_h.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_emit_b1_h.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_emit_b1_h.grid('on', which='both')
		ax_emit_b1_h.text(0.5, 1.1, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])))), horizontalalignment='center', verticalalignment='center', transform=ax_emit_b1_h.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)


		ax_emit_b1_v = pl.subplot(4,2,3, sharex=ax_emit_b1_h , sharey=ax_emit_b1_h)
		ax_emit_b1_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv'])),    range=(0, 5),   bins=50, color='blue',    histtype='step', lw=2, label='Injected')
		ax_emit_b1_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv'])),      range=(0, 5),   bins=50, color='orange',  histtype='step', lw=2, label='Start Ramp')
		ax_emit_b1_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv'])),     range=(0, 5),   bins=50, color='green',   histtype='step', lw=2, label='End Ramp')
		ax_emit_b1_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])),       range=(0, 5),   bins=50, color='red',     histtype='step', lw=2, label='Start SB')
		ax_emit_b1_v.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_emit_b1_v.set_xlabel('B1 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]', fontweight='bold', fontsize=8)
		ax_emit_b1_v.text(0.5, 1.1, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])))), horizontalalignment='center', verticalalignment='center', transform=ax_emit_b1_v.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)
		## Shrink current axis by 20%
		box = ax_emit_b1_v.get_position()
		ax_emit_b1_v.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_emit_b1_v.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_emit_b1_v.grid('on', which='both')


		ax_emit_b2_h = pl.subplot(4,2,5,  sharex=ax_emit_b1_h , sharey=ax_emit_b1_h)
		ax_emit_b2_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith'])),    range=(0, 5),   bins=50, color='blue',   histtype='step', lw=2, label='Injected')
		ax_emit_b2_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith'])),      range=(0, 5),   bins=50, color='orange', histtype='step', lw=2, label='Start Ramp')
		ax_emit_b2_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith'])),     range=(0, 5),   bins=50, color='green',  histtype='step', lw=2, label='End Ramp')
		ax_emit_b2_h.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])),       range=(0, 5),   bins=50, color='red',    histtype='step', lw=2, label='Start SB')
		ax_emit_b2_h.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_emit_b2_h.set_xlabel('B2 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]', fontweight='bold', fontsize=8)
		ax_emit_b2_h.text(0.5, 1.1, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])))), horizontalalignment='center', verticalalignment='center', transform=ax_emit_b2_h.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)
		# ax_emit_b2_h.minorticks_on()
		## Shrink current axis by 20%
		box = ax_emit_b2_h.get_position()
		ax_emit_b2_h.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_emit_b2_h.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_emit_b2_h.grid('on', which='both')

		ax_emit_b2_v = pl.subplot(4,2,7, sharex=ax_emit_b1_h,       sharey=ax_emit_b1_h)
		ax_emit_b2_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv'])),    range=(0, 5),   bins=50, color='blue',    histtype='step', lw=2, label='Injected')
		ax_emit_b2_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv'])),      range=(0, 5),   bins=50, color='orange',  histtype='step', lw=2, label='Start Ramp')
		ax_emit_b2_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv'])),     range=(0, 5),   bins=50, color='green',   histtype='step', lw=2, label='End Ramp')
		ax_emit_b2_v.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])),       range=(0, 5),   bins=50, color='red',     histtype='step', lw=2, label='Start SB')
		ax_emit_b2_v.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_emit_b2_v.set_xlabel('B2 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]', fontweight='bold', fontsize=8)
		ax_emit_b2_v.text(0.5, 1.1, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])))), horizontalalignment='center', verticalalignment='center', transform=ax_emit_b2_v.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)
		# ax_emit_b2_v.minorticks_on()
		## Shrink current axis by 20%
		box = ax_emit_b2_v.get_position()
		ax_emit_b2_v.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis

		# ax_emit_b1_h.set_ylim(0,5)
		# ax_emit_b1_v.set_ylim(0,5)
		# ax_emit_b2_h.set_ylim(0,5)
		# ax_emit_b2_v.set_ylim(0,5)

		ax_emit_b2_v.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_emit_b2_v.grid('on', which='both')

		##### ==== Intensities
		ax_intens_b1 = pl.subplot(4,2,2)
		ax_intens_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['intensity'])),    range=(0.0e11, 1.5e11),   bins=50, color='blue',   histtype='step', lw=2, label='Injected')
		ax_intens_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['intensity'])),      range=(0.0e11, 1.5e11),   bins=50, color='orange', histtype='step', lw=2, label='Start Ramp')
		ax_intens_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['intensity'])),     range=(0.0e11, 1.5e11),   bins=50, color='green',  histtype='step', lw=2, label='End Ramp')
		ax_intens_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['intensity'])),       range=(0.0e11, 1.5e11),   bins=50, color='red',    histtype='step', lw=2, label='Start SB')
		ax_intens_b1.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_intens_b1.set_xlabel('B1 Intensity [ppb]', fontweight='bold', fontsize=8)
		ax_intens_b1.text(0.6, 1.1, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['intensity']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['intensity']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['intensity'])))), horizontalalignment='center', verticalalignment='center', transform=ax_intens_b1.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)
		# ax_intens_b1.minorticks_on()
		## Shrink current axis by 20%
		box = ax_intens_b1.get_position()
		ax_intens_b1.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_intens_b1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_intens_b1.grid('on', which='both')


		ax_intens_b2 = pl.subplot(4,2,4)
		ax_intens_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['intensity'])),    range=(0.0e11, 1.5e11),   bins=50, color='blue',   histtype='step', lw=2, label='Injected')
		ax_intens_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['intensity'])),      range=(0.0e11, 1.5e11),   bins=50, color='orange', histtype='step', lw=2, label='Start Ramp')
		ax_intens_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['intensity'])),     range=(0.0e11, 1.5e11),   bins=50, color='green',  histtype='step', lw=2, label='End Ramp')
		ax_intens_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['intensity'])),       range=(0.0e11, 1.5e11),   bins=50, color='red',    histtype='step', lw=2, label='Start SB')
		ax_intens_b2.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_intens_b2.set_xlabel('B2 Intensity [ppb]', fontweight='bold', fontsize=8)
		ax_intens_b2.text(0.6, 1.1, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['intensity']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['intensity']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['intensity']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['intensity']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['intensity'])))), horizontalalignment='center', verticalalignment='center', transform=ax_intens_b2.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)
		# ax_intens_b2.minorticks_on()
		## Shrink current axis by 20%
		box = ax_intens_b2.get_position()
		ax_intens_b2.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_intens_b2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_intens_b2.grid('on', which='both')

		##### ==== Brightness
		ax_bright_b1 = pl.subplot(4,2,6)
		ax_bright_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['brightness'])),    range=(0.0e11, 1.5e11),   bins=50, color='blue',   histtype='step', lw=2, label='Injected')
		ax_bright_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['brightness'])),      range=(0.0e11, 1.5e11),   bins=50, color='orange', histtype='step', lw=2, label='Start Ramp')
		ax_bright_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness'])),     range=(0.0e11, 1.5e11),   bins=50, color='green',  histtype='step', lw=2, label='End Ramp')
		ax_bright_b1.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['brightness'])),       range=(0.0e11, 1.5e11),   bins=50, color='red',    histtype='step', lw=2, label='Start SB')
		ax_bright_b1.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_bright_b1.set_xlabel('B1 Brightness [p/$\mathbf{\mu}$m]', fontweight='bold', fontsize=8)
		ax_bright_b1.text(0.6, 1.1, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['brightness']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['brightness']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['brightness'])))), horizontalalignment='center', verticalalignment='center', transform=ax_bright_b1.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)
		# ax_bright_b1.minorticks_on()
		## Shrink current axis by 20%
		box = ax_bright_b1.get_position()
		ax_bright_b1.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_bright_b1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_bright_b1.grid('on', which='both')


		ax_bright_b2 = pl.subplot(4,2,8)
		ax_bright_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['brightness'])),    range=(0.0e11, 1.5e11),   bins=50, color='blue',   histtype='step',lw=2, label='Injected')
		ax_bright_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['brightness'])),      range=(0.0e11, 1.5e11),   bins=50, color='orange', histtype='step',lw=2, label='Start Ramp')
		ax_bright_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['brightness'])),     range=(0.0e11, 1.5e11),   bins=50, color='green',  histtype='step',lw=2, label='End Ramp')
		ax_bright_b2.hist(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['brightness'])),       range=(0.0e11, 1.5e11),   bins=50, color='red',    histtype='step',lw=2, label='Start SB')
		ax_bright_b2.set_ylabel('Entries', fontweight='bold', fontsize=8)
		ax_bright_b2.set_xlabel('B2 Brightness [p/$\mathbf{\mu}$m]', fontweight='bold', fontsize=8)
		ax_bright_b2.text(0.6, 1.1, "Injected: {:.2e}$\pm${:.2e} | Start Ramp: {:.2e}$\pm${:.2e} | End Ramp: {:.2e}$\pm${:.2e} | Start SB: {:.2e}$\pm${:.2e}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['brightness']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['brightness']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['brightness'])))), horizontalalignment='center', verticalalignment='center', transform=ax_bright_b2.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=7)
		# ax_bright_b2.minorticks_on()
		## Shrink current axis by 20%
		box = ax_bright_b2.get_position()
		ax_bright_b2.set_position([box.x0, box.y0, box.width*0.9, box.height])
		# Put a legend to the right of the current axis
		ax_bright_b2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
		ax_bright_b2.grid('on', which='both')

		pl.subplots_adjust(left=0.1, wspace=0.5, hspace=0.7)


		# tref string
		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)
		fig_hist.suptitle(subtitle, fontsize=16, fontweight='bold')

		if save:
			filename = self.savedir+"fill_{}_cycle_histos.pdf".format(self.filln)
			print filename
			fig_hist.savefig(filename,  dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleModelEmittance(self, save=False, batch=False):
		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		t_fill_len = t_end_fill - t_start_fill
		t_ref = t_start_fill
		t_min = 10.0*60.0
		
		pl.close('all')
		for interval, tag in zip(['Injection', 'he_before_SB'], ["INJ", "FT"]):

			for beam_n in [1,2]:
				
				info("plotCycleModelEmittance : Making Emittance plots for {} {}...".format(tag, beam_n))
				# Emittance plot
				fig_emit = pl.figure('emittances_{}'.format(TAG), figsize=(15,7))
				Delta_t = (np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_end']['time_meas'])-np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['time_meas']))
				filled_slots = self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['filled_slots']
				mask_valid_bunches = Delta_t>t_min

				Delta_emit_meas_H   = (np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_end']['emith'])-np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['emith']))
				growth_rate_meas_H  = Delta_emit_meas_H/Delta_t
				Delta_emit_model_H  = (np.array(self.filln_CycleModelDict['beam_{}'.format(beam_n)][interval]['at_end']['emith'])-np.array(self.filln_CycleModelDict['beam_{}'.format(beam_n)][interval]['at_start']['emith']))
				growth_rate_model_H = Delta_emit_model_H/Delta_t

				Delta_emit_meas_V   = (np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_end']['emitv'])-np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['emitv']))
				growth_rate_meas_V  = Delta_emit_meas_V/Delta_t
				Delta_emit_model_V  = (np.array(self.filln_CycleModelDict['beam_{}'.format(beam_n)][interval]['at_end']['emitv'])-np.array(self.filln_CycleModelDict['beam_{}'.format(beam_n)][interval]['at_start']['emitv']))
				growth_rate_model_V = Delta_emit_model_V/Delta_t

				ax_em_h = fig_emit.add_subplot(211)
				ax_em_h.plot(filled_slots[mask_valid_bunches], 3600.*growth_rate_meas_H[mask_valid_bunches], 'bo', markersize=6, label='Measured')
				ax_em_h.plot(filled_slots[mask_valid_bunches], 3600.*growth_rate_model_H[mask_valid_bunches],'gx', markersize=6 , label='Model')
				ax_em_h.set_ylim(-1, 3)
				ax_em_h.set_ylabel("B{} {}".format(beam_n, tag)+" $\mathbf{\epsilon_{H}}$ Growth [$\mathbf{\mu}$m/h]", fontsize=14, fontweight='bold')
				ax_em_h.grid('on')
				ax_em_h.legend(loc='best', numpoints=1)

				ax_em_v = fig_emit.add_subplot(212)
				ax_em_v.plot(filled_slots[mask_valid_bunches], 3600.*growth_rate_meas_V[mask_valid_bunches], 'bo', markersize=6,  label='Measured')
				ax_em_v.plot(filled_slots[mask_valid_bunches], 3600.*growth_rate_model_V[mask_valid_bunches],'gx', markersize=6 , label='Model')
				ax_em_v.set_ylim(-1, 3)
				ax_em_v.set_xlabel('Bunch Slot [25ns]', fontsize=14, fontweight='bold')
				ax_em_v.set_ylabel('B{} {}'.format(beam_n, tag)+' $\mathbf{\epsilon_{V}}$ Growth [$\mathbf{\mu}$m/h]', fontsize=14, fontweight='bold')
				ax_em_v.grid('on')
				# tref string
				tref_string = datetime.fromtimestamp(t_ref)
				subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)
				fig_emit.suptitle(subtitle, fontsize=16, fontweight='bold')
				ax_em_v.legend(loc='best', numpoints=1)

				if save:
					filename = self.savedir+"fill_{}_cycleModel_emittanceGrowth_B{}_{}.pdf".format(self.filln, beam_n, tag)
					print filename
					fig_emit.savefig(filename,  dpi=300)

				if not batch:
					pl.show()
	
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleModelEmittanceViolins(self, save=False, batch=False, return_df =False):

		dict_intervals_two_beams = self.filln_CycleDict
		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		Delta_t_b1_inj = (np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['time_meas']))
		Delta_t_b2_inj = (np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['time_meas']))

		Delta_t_b1_he = (np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['time_meas']))
		Delta_t_b2_he = (np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['time_meas'])-np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['time_meas']))


		mask_b1_inj = Delta_t_b1_inj > 5.0*60.0
		mask_b2_inj = Delta_t_b2_inj > 5.0*60.0

		mask_b1_he = Delta_t_b1_he > 5.0*60.0
		mask_b2_he = Delta_t_b2_he > 5.0*60.0

		b1h_INJ = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith'])[mask_b1_inj]
		b1h_FB  = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith'])[mask_b1_inj]
		b1h_FT  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith'])[mask_b1_he]
		b1h_SB  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])[mask_b1_he]

		b1v_INJ = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv'])[mask_b1_inj]
		b1v_FB  = np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv'])[mask_b1_inj]
		b1v_FT  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv'])[mask_b1_he]
		b1v_SB  = np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])[mask_b1_he]

		
		b2h_INJ = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith'])[mask_b2_inj]
		b2h_FB  = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith'])[mask_b2_inj]
		b2h_FT  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith'])[mask_b2_he]
		b2h_SB  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])[mask_b2_he]
		

		b2v_INJ = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv'])[mask_b2_inj]
		b2v_FB  = np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv'])[mask_b2_inj]
		b2v_FT  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv'])[mask_b2_he]
		b2v_SB  = np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])[mask_b2_he]

		b1h_emit = [b1h_INJ, b1h_FB, b1h_FT, b1h_SB]
		b1v_emit = [b1h_INJ, b1v_FB, b1v_FT, b1v_SB]
		
		b2h_emit = [b2h_INJ, b2h_FB, b2h_FT, b2h_SB]
		b2v_emit = [b2v_INJ, b2v_FB, b2v_FT, b2v_SB]

		emit  = []
		cycle = []
		beam  = []
		plane = []
		slot  = []

		# start with b1h
		emit.append(b1h_INJ)
		cycle.append(['Injection']*len(b1h_INJ))
		beam.append(['B1']*len(b1h_INJ))
		plane.append(['Horizontal']*len(b1h_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1h_FB)
		cycle.append(['Start Ramp']*len(b1h_FB))
		beam.append(['B1']*len(b1h_FB))
		plane.append(['Horizontal']*len(b1h_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1h_FT)
		cycle.append(['End Ramp']*len(b1h_FT))
		beam.append(['B1']*len(b1h_FT))
		plane.append(['Horizontal']*len(b1h_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b1h_SB)
		cycle.append(['Stable Beams']*len(b1h_SB))
		beam.append(['B1']*len(b1h_SB))
		plane.append(['Horizontal']*len(b1h_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		# b1v
		emit.append(b1v_INJ)
		cycle.append(['Injection']*len(b1v_INJ))
		beam.append(['B1']*len(b1v_INJ))
		plane.append(['Vertical']*len(b1v_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1v_FB)
		cycle.append(['Start Ramp']*len(b1v_FB))
		beam.append(['B1']*len(b1v_FB))
		plane.append(['Vertical']*len(b1v_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['filled_slots'])).tolist())

		emit.append(b1v_FT)
		cycle.append(['End Ramp']*len(b1v_FT))
		beam.append(['B1']*len(b1v_FT))
		plane.append(['Vertical']*len(b1v_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b1v_SB)
		cycle.append(['Stable Beams']*len(b1v_SB))
		beam.append(['B1']*len(b1v_SB))
		plane.append(['Vertical']*len(b1v_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['filled_slots'])).tolist())

		# b2h
		emit.append(b2h_INJ)
		cycle.append(['Injection']*len(b2h_INJ))
		beam.append(['B2']*len(b2h_INJ))
		plane.append(['Horizontal']*len(b2h_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())


		emit.append(b2h_FB)
		cycle.append(['Start Ramp']*len(b2h_FB))
		beam.append(['B2']*len(b2h_FB))
		plane.append(['Horizontal']*len(b2h_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())

		emit.append(b2h_FT)
		cycle.append(['End Ramp']*len(b2h_FT))
		beam.append(['B2']*len(b2h_FT))
		plane.append(['Horizontal']*len(b2h_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b2h_SB)
		cycle.append(['Stable Beams']*len(b2h_SB))
		beam.append(['B2']*len(b2h_SB))
		plane.append(['Horizontal']*len(b2h_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		# b2v
		emit.append(b2v_INJ)
		cycle.append(['Injection']*len(b2v_INJ))
		beam.append(['B2']*len(b2v_INJ))
		plane.append(['Vertical']*len(b2v_INJ))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())

		emit.append(b2v_FB)
		cycle.append(['Start Ramp']*len(b2v_FB))
		beam.append(['B2']*len(b2v_FB))
		plane.append(['Vertical']*len(b2v_FB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['filled_slots'])).tolist())

		emit.append(b2v_FT)
		cycle.append(['End Ramp']*len(b2v_FT))
		beam.append(['B2']*len(b2v_FT))
		plane.append(['Vertical']*len(b2v_FT))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		emit.append(b2v_SB)
		cycle.append(['Stable Beams']*len(b2v_SB))
		beam.append(['B2']*len(b2v_SB))
		plane.append(['Vertical']*len(b2v_SB))
		slot.append(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['filled_slots'])).tolist())

		emit = list(itertools.chain.from_iterable(emit))
		cycle = list(itertools.chain.from_iterable(cycle))
		beam = list(itertools.chain.from_iterable(beam))
		plane = list(itertools.chain.from_iterable(plane))
		slot = list(itertools.chain.from_iterable(slot))

		df_emit = pd.DataFrame()
		df_emit['Emittance'] = pd.Series(emit, dtype='float')
		df_emit['Cycle']     = pd.Series(cycle, dtype='category')
		df_emit['Plane']     = pd.Series(plane, dtype='category')
		df_emit['Beam']      = pd.Series(beam, dtype='category')
		df_emit['Slot']      = pd.Series(slot, dtype='int')
		df_emit['fill']		 = [int(self.filln)]*len(df_emit)
		df_emit['bunches']	 = [int(self.bunches)]*len(df_emit)

		df_emit = df_emit.dropna(axis=0, how='any')


		# df_relative_differences = pd.DataFrame()
		# B1H_INJ_FB = (b1h_FB - b1h_INJ)
		# B1H_FB_FT  = (b1h_FT - b1h_FB)
		# B1H_FT_SB  = (b1h_SB - b1h_FT)
		# B1H_INJ_SB = (b1h_SB - b1h_INJ)

		# B1V_INJ_FB = (b1v_FB - b1v_INJ)
		# B1V_FB_FT  = (b1v_FT - b1v_FB)
		# B1V_FT_SB  = (b1v_SB - b1v_FT)
		# B1V_INJ_SB = (b1v_SB - b1v_INJ)


		# B2H_INJ_FB = (b2h_FB - b2h_INJ)
		# B2H_FB_FT  = (b2h_FT - b2h_FB)
		# B2H_FT_SB  = (b2h_SB - b2h_FT)
		# B2H_INJ_SB = (b2h_SB - b2h_INJ)

		# B2V_INJ_FB = (b2v_FB - b2v_INJ)
		# B2V_FB_FT  = (b2v_FT - b2v_FB)
		# B2V_FT_SB  = (b2v_SB - b2v_FT)
		# B2V_INJ_SB = (b2v_SB - b2v_INJ)


		# B1H_INJ_FB = B1H_INJ_FB[~np.isnan(B1H_INJ_FB)].tolist()
		# B1H_FB_FT  =  B1H_FB_FT[~np.isnan(B1H_FB_FT)].tolist()
		# B1H_FT_SB  =  B1H_FT_SB[~np.isnan(B1H_FT_SB)].tolist()
		# B1H_INJ_SB = B1H_INJ_SB[~np.isnan(B1H_INJ_SB)].tolist()
		# B1V_INJ_FB = B1V_INJ_FB[~np.isnan(B1V_INJ_FB)].tolist()
		# B1V_FB_FT  =  B1V_FB_FT[~np.isnan(B1V_FB_FT)].tolist()
		# B1V_FT_SB  =  B1V_FT_SB[~np.isnan(B1V_FT_SB)].tolist()
		# B1V_INJ_SB = B1V_INJ_SB[~np.isnan(B1V_INJ_SB)].tolist()
		# B2H_INJ_FB = B2H_INJ_FB[~np.isnan(B2H_INJ_FB)].tolist()
		# B2H_FB_FT  =  B2H_FB_FT[~np.isnan(B2H_FB_FT)].tolist()
		# B2H_FT_SB  =  B2H_FT_SB[~np.isnan(B2H_FT_SB)].tolist()
		# B2H_INJ_SB = B2H_INJ_SB[~np.isnan(B2H_INJ_SB)].tolist()
		# B2V_INJ_FB = B2V_INJ_FB[~np.isnan(B2V_INJ_FB)].tolist()
		# B2V_FB_FT  =  B2V_FB_FT[~np.isnan(B2V_FB_FT)].tolist()
		# B2V_FT_SB  =  B2V_FT_SB[~np.isnan(B2V_FT_SB)].tolist()
		# B2V_INJ_SB = B2V_INJ_SB[~np.isnan(B2V_INJ_SB)].tolist()



		# blowup = []
		# bcycle  = []
		# bbeam   = []
		# bplane  = []
		# print blowup


		# # start with b1h
		# blowup.append(B1H_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B1H_INJ_FB))
		# bbeam.append(['B1']*len(B1H_INJ_FB))
		# bplane.append(['Horizontal']*len(B1H_INJ_FB))

		# blowup.append(B1H_FB_FT)
		# bcycle.append(['FB2FT']*len(B1H_FB_FT))
		# bbeam.append(['B1']*len(B1H_FB_FT))
		# bplane.append(['Horizontal']*len(B1H_FB_FT))
		
		# blowup.append(B1H_FT_SB)
		# bcycle.append(['FT2SB']*len(B1H_FT_SB))
		# bbeam.append(['B1']*len(B1H_FT_SB))
		# bplane.append(['Horizontal']*len(B1H_FT_SB))

		# blowup.append(B1H_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B1H_INJ_SB))
		# bbeam.append(['B1']*len(B1H_INJ_SB))
		# bplane.append(['Horizontal']*len(B1H_INJ_SB))

		# # b1v
		# blowup.append(B1V_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B1V_INJ_FB))
		# bbeam.append(['B1']*len(B1V_INJ_FB))
		# bplane.append(['Vertical']*len(B1V_INJ_FB))

		# blowup.append(B1V_FB_FT)
		# bcycle.append(['FB2FT']*len(B1V_FB_FT))
		# bbeam.append(['B1']*len(B1V_FB_FT))
		# bplane.append(['Vertical']*len(B1V_FB_FT))
		
		# blowup.append(B1V_FT_SB)
		# bcycle.append(['FT2SB']*len(B1V_FT_SB))
		# bbeam.append(['B1']*len(B1V_FT_SB))
		# bplane.append(['Vertical']*len(B1V_FT_SB))

		# blowup.append(B1V_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B1V_INJ_SB))
		# bbeam.append(['B1']*len(B1V_INJ_SB))
		# bplane.append(['Vertical']*len(B1V_INJ_SB))


		# # start with b2h
		# blowup.append(B2H_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B2H_INJ_FB))
		# bbeam.append(['B2']*len(B2H_INJ_FB))
		# bplane.append(['Horizontal']*len(B2H_INJ_FB))

		# blowup.append(B2H_FB_FT)
		# bcycle.append(['FB2FT']*len(B2H_FB_FT))
		# beam.append(['B2']*len(B2H_FB_FT))
		# bplane.append(['Horizontal']*len(B2H_FB_FT))
		
		# blowup.append(B2H_FT_SB)
		# bcycle.append(['FT2SB']*len(B2H_FT_SB))
		# bbeam.append(['B2']*len(B2H_FT_SB))
		# bplane.append(['Horizontal']*len(B2H_FT_SB))

		# blowup.append(B2H_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B2H_INJ_SB))
		# bbeam.append(['B2']*len(B2H_INJ_SB))
		# bplane.append(['Horizontal']*len(B2H_INJ_SB))

		# # b2v
		# blowup.append(B2V_INJ_FB)
		# bcycle.append(['INJ2FB']*len(B2V_INJ_FB))
		# bbeam.append(['B2']*len(B2V_INJ_FB))
		# bplane.append(['Vertical']*len(B2V_INJ_FB))

		# blowup.append(B2V_FB_FT)
		# bcycle.append(['FB2FT']*len(B2V_FB_FT))
		# bbeam.append(['B2']*len(B2V_FB_FT))
		# bplane.append(['Vertical']*len(B2V_FB_FT))
		
		# blowup.append(B2V_FT_SB)
		# bcycle.append(['FT2SB']*len(B2V_FT_SB))
		# bbeam.append(['B2']*len(B2V_FT_SB))
		# bplane.append(['Vertical']*len(B2V_FT_SB))

		# blowup.append(B2V_INJ_SB)
		# bcycle.append(['INJ2SB']*len(B2V_INJ_SB))
		# bbeam.append(['B2']*len(B2V_INJ_SB))
		# bplane.append(['Vertical']*len(B2V_INJ_SB))


		# blowup = list(itertools.chain.from_iterable(blowup))
		# bcycle = list(itertools.chain.from_iterable(bcycle))
		# bbeam = list(itertools.chain.from_iterable(bbeam))
		# bplane = list(itertools.chain.from_iterable(bplane))

		# bfill = [self.filln] * len(bplane)
		# bbunches = [self.bunches] * len(bplane)


		# df_blowup = pd.DataFrame()
		# df_blowup['Emittance'] = pd.Series(blowup, dtype='float')
		# df_blowup['Cycle']     = pd.Series(bcycle, dtype='category')
		# df_blowup['Plane']     = pd.Series(bplane, dtype='category')
		# df_blowup['Beam']      = pd.Series(bbeam, dtype='category')
		# df_blowup['Fill']	   = pd.Series(bfill, dtype='int')
		# df_blowup['Bunches']   = pd.Series(bbunches, dtype='int')



		mean_values_b1 = []
		for ncycle in ['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams']:
			for nplane in ['Horizontal', 'Vertical']:
				mean_values_b1.append(df_emit['Emittance'][(df_emit['Cycle']==ncycle) & (df_emit['Plane']==nplane) & (df_emit['Beam']=='B1')].mean())
				# print np.array(df_emit['Emittance'][(df_emit['Cycle']==ncycle) & (df_emit['Plane']==nplane) & (df_emit['Beam']=='B1')])[0]
		

		mean_values_b2 = []
		for ncycle in ['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams']:
			for nplane in ['Horizontal', 'Vertical']:
				mean_values_b2.append(df_emit['Emittance'][(df_emit['Cycle']==ncycle) & (df_emit['Plane']==nplane) & (df_emit['Beam']=='B2')].mean())

		mean_labels_b1 = [str(np.round(s, 2))+'$\mu$m' for s in mean_values_b1]
		mean_labels_b2 = [str(np.round(s, 2))+'$\mu$m' for s in mean_values_b2] 

		print mean_labels_b1, mean_labels_b2

		pos = range(len(mean_labels_b1))
		
		sns.set_style("whitegrid")
		pl.close('all')

		fig_emit_volin = pl.figure('Violins', figsize=(15,8))
		ax_b1 = pl.subplot(211)
		ax_b1 = sns.violinplot(x="Cycle", y="Emittance", hue="Plane", data=df_emit[df_emit['Beam']=='B1'], palette=['#71aaf2', '#ff8a3d'], split=True, order=['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams'])
		ax_b1.get_axes().xaxis.set_ticklabels([])
		ax_b1.set_ylim(1,4)
		ax_b1.set_ylabel('B1 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
		ax_b1.set_xlabel('')


		ax_b2 = pl.subplot(212)#, sharex=ax_b1h, sharey=ax_b1h)
		ax_b2 = sns.violinplot(x="Cycle", y="Emittance", hue="Plane", data=df_emit[df_emit['Beam']=='B2'],  palette=['#71aaf2', '#ff8a3d'], split=True, order=['Injection', 'Start Ramp', 'End Ramp', 'Stable Beams'])
		ax_b2.get_axes().xaxis.set_ticklabels([])
		ax_b2.set_ylim(1,4)
		ax_b2.set_ylabel('B2 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
		ax_b2.get_axes().xaxis.set_ticklabels(['INJECTION', 'Start RAMP', 'End RAMP', 'Start STABLE BEAMS'])
		ax_b2.set_xlabel('')
		for tick in ax_b2.xaxis.get_major_ticks():
			tick.label.set_fontsize(16)
			tick.label.set_fontweight('bold')

		for tick in ax_b1.yaxis.get_major_ticks():
			tick.label.set_fontsize(14)
		for tick in ax_b2.yaxis.get_major_ticks():
			tick.label.set_fontsize(14)

		ax_b1.legend(loc=2, fontsize=14)
		ax_b2.legend(loc=2, fontsize=14)

		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(self.filln, tref_string)
		fig_emit_volin.suptitle(subtitle, fontsize=16, fontweight='bold')

		for tick,label in zip(pos,ax_b1.get_xticklabels()):
			ax_b1.text(pos[tick]-0.15, mean_values_b1[2*tick],   mean_labels_b1[2*tick],    horizontalalignment='center', size='small', color='k', weight='semibold')
			ax_b1.text(pos[tick]+0.15, mean_values_b1[2*tick+1], mean_labels_b1[2*tick+1],  horizontalalignment='center', size='small', color='k', weight='semibold')
		for tick,label in zip(pos,ax_b2.get_xticklabels()):
			ax_b2.text(pos[tick]-0.15, mean_values_b2[2*tick],   mean_labels_b2[2*tick], 	 horizontalalignment='center', size='small', color='k', weight='semibold')
			ax_b2.text(pos[tick]+0.15, mean_values_b2[2*tick+1], mean_labels_b2[2*tick+1],  horizontalalignment='center', size='small', color='k', weight='semibold')

		if save:
			filename = self.savedir+"fill_{}_cycle_emittancesViolin_HV.pdf".format(self.filln)
			fig_emit_volin.savefig(filename,  dpi=300)
			
		if not batch:
			pl.show()

		if return_df:
			return df_emit

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleModelBunchLength(self, save=False, batch=False):
		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		t_fill_len = t_end_fill - t_start_fill
		t_ref = t_start_fill
		t_min = 10.0*60.0
		

		pl.close('all')
		for interval, tag in zip(['Injection', 'he_before_SB'], ["INJ", "FT"]):
			info("plotCycleModelBunchLength : Making Bunch Length plots for {}...".format(tag))

			fig_blen = pl.figure('bucnh length', figsize=(15,7))

			Delta_t_b1 = (np.array(self.filln_CycleDict['beam_1'][interval]['at_end']['time_meas'])-np.array(self.filln_CycleDict['beam_1'][interval]['at_start']['time_meas']))
			filled_slots_b1 = self.filln_CycleDict['beam_1'][interval]['filled_slots']
			mask_valid_bunches_b1 = Delta_t_b1>t_min

			Delta_blength_meas_b1   = (np.array(self.filln_CycleDict['beam_1'][interval]['at_end']['blength'])-np.array(self.filln_CycleDict['beam_1'][interval]['at_start']['blength']))
			growth_bl_rate_meas_b1  = Delta_blength_meas_b1/Delta_t_b1

			Delta_blength_model_b1  = (np.array(self.filln_CycleModelDict['beam_1'][interval]['at_end']['blength'])-np.array(self.filln_CycleModelDict['beam_1'][interval]['at_start']['blength']))
			growth_bl_rate_model_b1 = Delta_blength_model_b1/Delta_t_b1

			Delta_t_b2 = (np.array(self.filln_CycleDict['beam_2'][interval]['at_end']['time_meas'])-np.array(self.filln_CycleDict['beam_2'][interval]['at_start']['time_meas']))
			filled_slots_b2 = self.filln_CycleDict['beam_2'][interval]['filled_slots']
			mask_valid_bunches_b2 = Delta_t_b2>t_min

			Delta_blength_meas_b2   = (np.array(self.filln_CycleDict['beam_2'][interval]['at_end']['blength'])-np.array(self.filln_CycleDict['beam_2'][interval]['at_start']['blength']))
			growth_bl_rate_meas_b2  = Delta_blength_meas_b2/Delta_t_b2

			Delta_blength_model_b2  = (np.array(self.filln_CycleModelDict['beam_2'][interval]['at_end']['blength'])-np.array(self.filln_CycleModelDict['beam_2'][interval]['at_start']['blength']))
			growth_bl_rate_model_b2 = Delta_blength_model_b2/Delta_t_b2

			ax_blen_b1 = fig_blen.add_subplot(211)
			ax_blen_b1.plot(filled_slots_b1[mask_valid_bunches_b1], 1.0e12*3600.*growth_bl_rate_meas_b1[mask_valid_bunches_b1],  'bo', markersize=6,  label='Measured')
			ax_blen_b1.plot(filled_slots_b1[mask_valid_bunches_b1], 1.0e12*3600.*growth_bl_rate_model_b1[mask_valid_bunches_b1], 'gx', markersize=6 , label="Model")
			ax_blen_b1.set_ylabel('B1 {}'.format(tag)+' Bunch Length Slope [ps/h]'.format(tag=tag), fontsize=10, fontweight='bold')
			ax_blen_b1.grid('on')
			ax_blen_b1.legend(loc='best', numpoints=1)


			ax_blen_b2 = fig_blen.add_subplot(212)
			ax_blen_b2.plot(filled_slots_b2[mask_valid_bunches_b2], 1.0e12*3600.*growth_bl_rate_meas_b2[mask_valid_bunches_b2], 'bo', markersize=6, label='Measured')
			ax_blen_b2.plot(filled_slots_b2[mask_valid_bunches_b2], 1.0e12*3600.*growth_bl_rate_model_b2[mask_valid_bunches_b2],'gx', markersize=6 , label="Model")
			ax_blen_b2.set_xlabel('Bunch Slot [25ns]', fontsize=14, fontweight='bold')
			ax_blen_b2.set_ylabel('B2 '.format(tag)+' Bunch Length Slope [ps/h]', fontsize=10, fontweight='bold')
			ax_blen_b2.grid('on')
			ax_blen_b2.legend(loc='best', numpoints=1)
			# tref string
			tref_string = datetime.fromtimestamp(t_ref)
			subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)
			fig_blen.suptitle(subtitle, fontsize=16, fontweight='bold')


			if save:
				filename = self.savedir+"fill_{}_cycleModel_bunch_length_{}.pdf".format(self.filln, tag)
				print filename
				fig_blen.savefig(filename,  dpi=300)

			if not batch:
				pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	########################################################################################################################
	#
	#													STABLE BEAMS PLOTS
	#
	#########################################################################################################################

		
	def plotStableBeamsCrossingAngle(self, save=False, batch=False):
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)

		info("# plotStableBeamsCrossingAngle : Fill {} -> Making Emittances B1 plot...".format(self.filln))
		fig_xing = pl.figure(0, figsize=(15, 7))
		fig_xing.canvas.set_window_title('CrossingAngle')
		fig_xing.set_facecolor('w')
		ax = pl.subplot(111)
		ax.plot(self.convertToLocalTime((time_range).astype(int)), 1.0e6*self.filln_StableBeamsDict['xing_angle'][1]/2.0, 'b', linestyle='-', drawstyle='steps', label='IP1', lw=3)
		ax.plot(self.convertToLocalTime((time_range).astype(int)), 1.0e6*self.filln_StableBeamsDict['xing_angle'][5]/2.0, 'r', linestyle=':', drawstyle='steps', label='IP5', lw=2)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		ax.set_xlabel('Time', fontsize=14, fontweight='bold')
		ax.set_ylabel('Half Crossing Angle [$\mathbf{\mu}$rad]', fontsize=14, fontweight='bold')
		ax.grid('on')
		ax.legend(loc='best', fontsize=12)
		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_xing.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_xing.savefig(self.savedir+"fill_{}_crossingAngle.pdf".format(self.filln) , dpi=300)
			
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsEmittances(self, save=False, batch=False, return_df = False):
		## first get the times for the fill
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)

		## Then load the necessary stuff from the SB dictionary

		eh_interp_coll               = self.filln_StableBeamsDict['eh_interp_coll']
		ev_interp_coll               = self.filln_StableBeamsDict['ev_interp_coll']
		eh_interp_raw_coll           = self.filln_StableBeamsDict['eh_interp_raw_coll']
		ev_interp_raw_coll           = self.filln_StableBeamsDict['ev_interp_raw_coll']
		slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
		eh_interp_noncoll            = self.filln_StableBeamsDict['eh_interp_noncoll']
		ev_interp_noncoll            = self.filln_StableBeamsDict['ev_interp_noncoll']
		eh_interp_raw_noncoll        = self.filln_StableBeamsDict['eh_interp_raw_noncoll']
		ev_interp_raw_noncoll        = self.filln_StableBeamsDict['ev_interp_raw_noncoll']
		slots_filled_noncoll         = self.filln_StableBeamsDict['slots_filled_noncoll']
		time_range                   = self.filln_StableBeamsDict['time_range']

		## start plotting...
		pl.close('all')

		## Figure : Emittances B1
		info("# plotStableBeamsEmittances : Fill {} -> Making Emittances B1 plot...".format(self.filln))
		fig_em1 = pl.figure(1, figsize=(15,7))
		fig_em1.canvas.set_window_title('Emittances B1')
		fig_em1.set_facecolor('w')
		ax_ne1h = pl.subplot(2,3,(2,3))
		ax_ne1h_t = pl.subplot(2,3,1, sharey=ax_ne1h)
		ax_share = ax_ne1h
		ax_share_t = ax_ne1h_t
		axy_share = ax_ne1h
		ax_ne1v = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
		ax_ne1v_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)
		fig_em1.subplots_adjust(wspace=0.5, hspace=0.5)


		## Figure : Emittances B2
		info("# plotStableBeamsEmittances : Fill {} -> Making Emittances B2 plot...".format(self.filln))
		fig_em2 = pl.figure(2, figsize=(15,7))
		fig_em2.canvas.set_window_title('Emittances B2')
		fig_em2.set_facecolor('w')
		ax_ne2h = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
		ax_ne2v = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
		ax_ne2h_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
		ax_ne2v_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)
		fig_em2.subplots_adjust(wspace=0.5, hspace=0.5)

		## Figure : Emittances B1 Raw
		info("# plotStableBeamsEmittances : Fill {} -> Making Emittances B1 Raw plot...".format(self.filln))
		fig_em1_raw = pl.figure(101, figsize=(15,7))
		fig_em1_raw.canvas.set_window_title('Emittances B1 raw')
		fig_em1_raw.set_facecolor('w')
		ax_ne1h_raw = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
		ax_ne1h_t_raw = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
		ax_ne1v_raw = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
		ax_ne1v_t_raw = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)
		fig_em1_raw.subplots_adjust(wspace=0.5, hspace=0.5)


		## Figure : Emittances B2 Raw
		info("# plotStableBeamsEmittances : Fill {} -> Making Emittances B2 Raw plot...".format(self.filln))
		fig_em2_raw = pl.figure(102, figsize=(15,7))
		fig_em2_raw.canvas.set_window_title('Emittances B2 raw')
		fig_em2_raw.set_facecolor('w')
		ax_ne2h_raw = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
		ax_ne2v_raw = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
		ax_ne2h_t_raw = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
		ax_ne2v_t_raw = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)
		fig_em2_raw.subplots_adjust(wspace=0.5, hspace=0.5)

		# self.plot_mean_and_spread(ax_ne1h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_noncoll[1], color='grey', alpha=.5)
		# self.plot_mean_and_spread(ax_ne1v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_noncoll[1], color='grey', alpha=.5)
		# self.plot_mean_and_spread(ax_ne2h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_noncoll[2], color='grey', alpha=.5)
		# self.plot_mean_and_spread(ax_ne2v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_noncoll[2], color='grey', alpha=.5)

		# self.plot_mean_and_spread(ax_ne1h_t, (time_range-t_start_STABLE)/3600., eh_interp_noncoll[1], color='grey', alpha=.5)
		# self.plot_mean_and_spread(ax_ne1v_t, (time_range-t_start_STABLE)/3600., ev_interp_noncoll[1], color='grey', alpha=.5)
		# self.plot_mean_and_spread(ax_ne2h_t, (time_range-t_start_STABLE)/3600., eh_interp_noncoll[2], color='grey', alpha=.5)
		# self.plot_mean_and_spread(ax_ne2v_t, (time_range-t_start_STABLE)/3600., ev_interp_noncoll[2], color='grey', alpha=.5)

		# self.plot_mean_and_spread(ax_ne1h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_coll[1])
		# self.plot_mean_and_spread(ax_ne1v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_coll[1])
		# self.plot_mean_and_spread(ax_ne2h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_coll[2])
		# self.plot_mean_and_spread(ax_ne2v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_coll[2])

		# self.plot_mean_and_spread(ax_ne1h_t, (time_range-t_start_STABLE)/3600., eh_interp_coll[1])
		# self.plot_mean_and_spread(ax_ne1v_t, (time_range-t_start_STABLE)/3600., ev_interp_coll[1])
		# self.plot_mean_and_spread(ax_ne2h_t, (time_range-t_start_STABLE)/3600., eh_interp_coll[2])
		# self.plot_mean_and_spread(ax_ne2v_t, (time_range-t_start_STABLE)/3600., ev_interp_coll[2])

		self.plot_mean_and_spread(ax_ne1h_t_raw, self.convertToLocalTime((time_range).astype(int)), eh_interp_raw_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne1v_t_raw, self.convertToLocalTime((time_range).astype(int)), ev_interp_raw_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2h_t_raw, self.convertToLocalTime((time_range).astype(int)), eh_interp_raw_noncoll[2], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2v_t_raw, self.convertToLocalTime((time_range).astype(int)), ev_interp_raw_noncoll[2], color='grey', alpha=.5)
		# ax_ne2v_t_raw.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		# ax_ne2v_t_raw.xticks(rotation=30)
		

		self.plot_mean_and_spread(ax_ne1h_t, self.convertToLocalTime((time_range).astype(int)), eh_interp_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne1v_t, self.convertToLocalTime((time_range).astype(int)), ev_interp_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2h_t, self.convertToLocalTime((time_range).astype(int)), eh_interp_noncoll[2], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2v_t, self.convertToLocalTime((time_range).astype(int)), ev_interp_noncoll[2], color='grey', alpha=.5)
		# ax_ne2v_t.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))


		self.plot_mean_and_spread(ax_ne1h_t_raw, self.convertToLocalTime((time_range).astype(int)), eh_interp_raw_coll[1])
		self.plot_mean_and_spread(ax_ne1v_t_raw, self.convertToLocalTime((time_range).astype(int)), ev_interp_raw_coll[1])
		self.plot_mean_and_spread(ax_ne2h_t_raw, self.convertToLocalTime((time_range).astype(int)), eh_interp_raw_coll[2])
		self.plot_mean_and_spread(ax_ne2v_t_raw, self.convertToLocalTime((time_range).astype(int)), ev_interp_raw_coll[2])
		# ax_ne2v_t_raw.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
		

		self.plot_mean_and_spread(ax_ne1h_t, self.convertToLocalTime((time_range).astype(int)), eh_interp_coll[1])
		self.plot_mean_and_spread(ax_ne1v_t, self.convertToLocalTime((time_range).astype(int)), ev_interp_coll[1])
		self.plot_mean_and_spread(ax_ne2h_t, self.convertToLocalTime((time_range).astype(int)), eh_interp_coll[2])
		self.plot_mean_and_spread(ax_ne2v_t, self.convertToLocalTime((time_range).astype(int)), ev_interp_coll[2])
		# ax_ne2v_t.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

		


		for ax in [ax_ne1h_t_raw, ax_ne1v_t_raw, ax_ne2h_t_raw, ax_ne2v_t_raw, ax_ne1h_t, ax_ne1v_t, ax_ne2h_t, ax_ne2v_t, ax_ne1h_t_raw, ax_ne1v_t_raw, ax_ne2h_t_raw, ax_ne2v_t_raw, ax_ne1h_t, ax_ne1v_t, ax_ne2h_t, ax_ne2v_t]:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			pl.setp( ax.xaxis.get_majorticklabels(), rotation=30 )



		for i_time in range(N_steps):
			colorcurr = self.colorprog(i_prog=i_time, Nplots=N_steps)
			ax_ne1h.plot(slots_filled_coll[1], eh_interp_coll[1][i_time, :], '.', color=colorcurr)
			ax_ne1v.plot(slots_filled_coll[1], ev_interp_coll[1][i_time, :], '.', color=colorcurr)
			ax_ne2h.plot(slots_filled_coll[2], eh_interp_coll[2][i_time, :], '.', color=colorcurr)
			ax_ne2v.plot(slots_filled_coll[2], ev_interp_coll[2][i_time, :], '.', color=colorcurr)

			ax_ne1h_raw.plot(slots_filled_coll[1], eh_interp_raw_coll[1][i_time, :], '.', color=colorcurr)
			ax_ne1v_raw.plot(slots_filled_coll[1], ev_interp_raw_coll[1][i_time, :], '.', color=colorcurr)
			ax_ne2h_raw.plot(slots_filled_coll[2], eh_interp_raw_coll[2][i_time, :], '.', color=colorcurr)
			ax_ne2v_raw.plot(slots_filled_coll[2], ev_interp_raw_coll[2][i_time, :], '.', color=colorcurr)

			ax_ne1h_raw.plot(slots_filled_noncoll[1], eh_interp_raw_noncoll[1][i_time, :], 'x', color=colorcurr)
			ax_ne1v_raw.plot(slots_filled_noncoll[1], ev_interp_raw_noncoll[1][i_time, :], 'x', color=colorcurr)
			ax_ne2h_raw.plot(slots_filled_noncoll[2], eh_interp_raw_noncoll[2][i_time, :], 'x', color=colorcurr)
			ax_ne2v_raw.plot(slots_filled_noncoll[2], ev_interp_raw_noncoll[2][i_time, :], 'x', color=colorcurr)

			ax_ne1h.plot(slots_filled_noncoll[1], eh_interp_noncoll[1][i_time, :], 'x', color=colorcurr)
			ax_ne1v.plot(slots_filled_noncoll[1], ev_interp_noncoll[1][i_time, :], 'x', color=colorcurr)
			ax_ne2h.plot(slots_filled_noncoll[2], eh_interp_noncoll[2][i_time, :], 'x', color=colorcurr)
			ax_ne2v.plot(slots_filled_noncoll[2], ev_interp_noncoll[2][i_time, :], 'x', color=colorcurr)

		ax_ne1h_t.set_ylim(0, 5)
		ax_ne1v_t.set_ylim(0, 5)
		ax_ne2h_t.set_ylim(0, 5)
		ax_ne2v_t.set_ylim(0, 5)


		for sp in [ax_ne1h, ax_ne1v, ax_ne2h, ax_ne2v, ax_ne1h_raw, ax_ne1v_raw, ax_ne2h_raw, ax_ne2v_raw]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

		for sp in [ax_ne1h_t, ax_ne1v_t, ax_ne2h_t, ax_ne2v_t, ax_ne1h_t_raw, ax_ne1v_t_raw, ax_ne2h_t_raw, ax_ne2v_t_raw]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel('Time', fontsize=14, fontweight='bold')

		ax_ne1h_t.set_ylabel('Emittance B1H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne1v_t.set_ylabel('Emittance B1V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2h_t.set_ylabel('Emittance B2H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2v_t.set_ylabel('Emittance B2V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')

		ax_ne1h_t_raw.set_ylabel('Emittance B1H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne1v_t_raw.set_ylabel('Emittance B1V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2h_t_raw.set_ylabel('Emittance B2H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2v_t_raw.set_ylabel('Emittance B2V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')

		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		for ff in [fig_em1, fig_em2, fig_em1_raw, fig_em2_raw]:
			ff.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_em1.savefig(self.savedir+"fill_{}_b1Emittances.pdf".format(self.filln) , dpi=300)
			fig_em2.savefig(self.savedir+"fill_{}_b2Emittances.pdf".format(self.filln) , dpi=300)
			fig_em1_raw.savefig(self.savedir+"fill_{}_b1EmittancesRaw.pdf".format(self.filln) , dpi=300)
			fig_em2_raw.savefig(self.savedir+"fill_{}_b2EmittancesRaw.pdf".format(self.filln) , dpi=300)

		if not batch:
			pl.show()

		if return_df:
			emit_b1h_coll = eh_interp_coll[1][-4]
			emit_b1v_coll = ev_interp_coll[1][-4]

			emit_b2h_coll = eh_interp_coll[2][-4]
			emit_b2v_coll = ev_interp_coll[2][-4]

			emit_b1h_noncoll = eh_interp_noncoll[1][-4]
			emit_b1v_noncoll = ev_interp_noncoll[2][-4]

			emit_b2h_noncoll = eh_interp_noncoll[1][-4]
			emit_b2v_noncoll = ev_interp_noncoll[2][-4]

			emit_list  = []
			cycle_list = []
			plane_list = []
			beam_list  = []

			emit_list.append(emit_b1h_coll)
			cycle_list.append(['End Stable Beams']*len(emit_b1h_coll))
			plane_list.append(['Horizontal']*len(emit_b1h_coll))
			beam_list.append(['B1']*len(emit_b1h_coll))

			emit_list.append(emit_b1v_coll)
			cycle_list.append(['End Stable Beams']*len(emit_b1v_coll))
			plane_list.append(['Vertical']*len(emit_b1v_coll))
			beam_list.append(['B1']*len(emit_b1v_coll))

			emit_list.append(emit_b2h_coll)
			cycle_list.append(['End Stable Beams']*len(emit_b2h_coll))
			plane_list.append(['Horizontal']*len(emit_b2h_coll))
			beam_list.append(['B2']*len(emit_b2h_coll))

			emit_list.append(emit_b2v_coll)
			cycle_list.append(['End Stable Beams']*len(emit_b2v_coll))
			plane_list.append(['Vertical']*len(emit_b2v_coll))
			beam_list.append(['B2']*len(emit_b2v_coll))

			emit_list = list(itertools.chain(*emit_list))
			cycle_list = list(itertools.chain(*cycle_list))
			plane_list = list(itertools.chain(*plane_list))
			beam_list = list(itertools.chain(*beam_list))


			df_emit = pd.DataFrame()
			df_emit['Emittance'] = pd.Series(emit_list, dtype='float')
			df_emit['Cycle']     = pd.Series(cycle_list, dtype='category')
			df_emit['Plane']     = pd.Series(plane_list, dtype='category')
			df_emit['Beam']      = pd.Series(beam_list, dtype='category')

			return df_emit

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsIntensity(self, save=False, batch=False):
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)
		b_inten_interp_coll          = self.filln_StableBeamsDict['b_inten_interp_coll']
		slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']

		b_inten_interp_noncoll       = self.filln_StableBeamsDict['b_inten_interp_noncoll']
		slots_filled_noncoll         = self.filln_StableBeamsDict['slots_filled_noncoll']
		time_range                   = self.filln_StableBeamsDict['time_range']

		## start plotting...
		pl.close('all')
		## Figure : Bunch intensity
		info("# plotStableBeamsIntensity : Fill {} -> Making Bunch Intensity plot...".format(self.filln))
		fig_int = pl.figure(3, figsize=(15,7))
		fig_int.canvas.set_window_title('Bunch intensity')
		fig_int.set_facecolor('w')
		bx_nb1 = pl.subplot(2,3,(2,3))#, sharex=ax_share)
		bx_nb2 = pl.subplot(2,3,(5,6), sharex=bx_nb1)

		bx_nb1_t = pl.subplot(2,3,1, sharey=bx_nb1)
		bx_nb2_t = pl.subplot(2,3,4, sharex=bx_nb1_t, sharey=bx_nb2)
		fig_int.subplots_adjust(wspace=0.5, hspace=0.5)

		self.plot_mean_and_spread(bx_nb1_t, self.convertToLocalTime(time_range.astype(int)), b_inten_interp_coll[1])
		self.plot_mean_and_spread(bx_nb2_t, self.convertToLocalTime(time_range.astype(int)), b_inten_interp_coll[2])

		self.plot_mean_and_spread(bx_nb1_t, self.convertToLocalTime(time_range.astype(int)), b_inten_interp_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(bx_nb2_t, self.convertToLocalTime(time_range.astype(int)), b_inten_interp_noncoll[2], color='grey', alpha=.5)
		for ax in [bx_nb1_t, bx_nb2_t]:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			pl.setp( ax.xaxis.get_majorticklabels(), rotation=30 )

		for i_time in range(N_steps):
			colorcurr = self.colorprog(i_prog=i_time, Nplots=N_steps)

			bx_nb1.plot(slots_filled_coll[1], b_inten_interp_coll[1][i_time, :], '.', color=colorcurr)
			bx_nb2.plot(slots_filled_coll[2], b_inten_interp_coll[2][i_time, :], '.', color=colorcurr)

			bx_nb1.plot(slots_filled_noncoll[1], b_inten_interp_noncoll[1][i_time, :], 'x', color=colorcurr)
			bx_nb2.plot(slots_filled_noncoll[2], b_inten_interp_noncoll[2][i_time, :], 'x', color=colorcurr)

		for sp in [bx_nb1, bx_nb2]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

		for sp in [bx_nb1_t, bx_nb2_t]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel('Time', fontsize=14, fontweight='bold')

		bx_nb1_t.set_ylabel('Intensity B1 [p/b]'  , fontsize=14, fontweight='bold')
		bx_nb2_t.set_ylabel('Intensity B2 [p/b]'  , fontsize=14, fontweight='bold')

		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_int.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_int.savefig(self.savedir+"fill_{}_bunchIntensity.pdf".format(self.filln) , dpi=300)
		if not batch:
			pl.show()
			
	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsBunchLength(self, save=False, batch=False):
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)

		bl_interp_m_coll             = self.filln_StableBeamsDict['bl_interp_m_coll']
		slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']

		bl_interp_m_noncoll          = self.filln_StableBeamsDict['bl_interp_m_noncoll']
		slots_filled_noncoll         = self.filln_StableBeamsDict['slots_filled_noncoll']
		time_range                   = self.filln_StableBeamsDict['time_range']
		## start plotting...
		pl.close('all')
		## Figure : Bunch Length
		info("# plotStableBeamsBunchLength : Fill {} -> Making Bunch Length plot...".format(self.filln))
		fig_bl = pl.figure(4, figsize=(15,7))
		fig_bl.canvas.set_window_title('Bunch length')
		fig_bl.set_facecolor('w')
		bx_bl1 = pl.subplot(2,3,(2,3))#, sharex=ax_share)
		bx_bl2 = pl.subplot(2,3,(5,6), sharex=bx_bl1)
		bx_bl1_t = pl.subplot(2,3,1, sharey=bx_bl1)
		bx_bl2_t = pl.subplot(2,3,4, sharex=bx_bl1_t, sharey=bx_bl1)
		fig_bl.subplots_adjust(wspace=0.5, hspace=0.5)

		self.plot_mean_and_spread(bx_bl1_t, self.convertToLocalTime(time_range.astype(int)), bl_interp_m_coll[1]*4/clight*1e9)
		self.plot_mean_and_spread(bx_bl2_t, self.convertToLocalTime(time_range.astype(int)), bl_interp_m_coll[2]*4/clight*1e9)
		self.plot_mean_and_spread(bx_bl1_t, self.convertToLocalTime(time_range.astype(int)), bl_interp_m_noncoll[1]*4/clight*1e9, color='grey', alpha=.5)
		self.plot_mean_and_spread(bx_bl2_t, self.convertToLocalTime(time_range.astype(int)), bl_interp_m_noncoll[2]*4/clight*1e9, color='grey', alpha=.5)
		for ax in [bx_bl1_t, bx_bl2_t]:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			pl.setp( ax.xaxis.get_majorticklabels(), rotation=30 )

		for i_time in range(N_steps):
			colorcurr = self.colorprog(i_prog=i_time, Nplots=N_steps)

			bx_bl1.plot(slots_filled_noncoll[1], bl_interp_m_noncoll[1][i_time, :]*4/clight*1e9, 'x', color=colorcurr)
			bx_bl2.plot(slots_filled_noncoll[2], bl_interp_m_noncoll[2][i_time, :]*4/clight*1e9, 'x', color=colorcurr)			
			
			bx_bl1.plot(slots_filled_coll[1], bl_interp_m_coll[1][i_time, :]*4/clight*1e9, '.', color=colorcurr)
			bx_bl2.plot(slots_filled_coll[2], bl_interp_m_coll[2][i_time, :]*4/clight*1e9, '.', color=colorcurr)
		
		for sp in [bx_bl1, bx_bl2]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

		for sp in [bx_bl1_t, bx_bl2_t]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel('Time', fontsize=14, fontweight='bold')
		bx_bl1_t.set_ylabel('Bunch length B1 [ns]', fontsize=14, fontweight='bold')
		bx_bl2_t.set_ylabel('Bunch length B1 [ns]', fontsize=14, fontweight='bold')

		bx_bl1_t.set_ylim(0.8, 1.2)
		bx_bl2_t.set_ylim(0.8, 1.2)

		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_bl.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_bl.savefig(self.savedir+"fill_{}_bunchLength.pdf".format(self.filln) , dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsCalculatedLuminosity(self, save=False, batch=False):
		## first get the times for the fill
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)

		## load info from calculated lumi dictionary
		lumi_bbb_ATLAS_invm2         = self.filln_LumiCalcDict['ATLAS']['bunch_lumi']
		lumi_bbb_CMS_invm2           = self.filln_LumiCalcDict['CMS']['bunch_lumi']
		slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
		time_range                   = self.filln_StableBeamsDict['time_range']

		info("# plotStableBeamsCalculatedLuminosity : Fill {} -> Making Expected BBB Luminosities plot...".format(self.filln))
		fig_lumi_calc = pl.figure(5, figsize=(15,7))
		fig_lumi_calc.canvas.set_window_title('Expected bbb lumis')
		fig_lumi_calc.set_facecolor('w')
		ax_ATLAS_calc = pl.subplot(2,3,(2,3))#, sharex=ax_share)
		ax_CMS_calc = pl.subplot(2,3,(5,6), sharex=ax_ATLAS_calc, sharey=ax_ATLAS_calc)
		ax_ATLAS_calc_t = pl.subplot(2,3,1, sharey=ax_ATLAS_calc)
		ax_CMS_calc_t = pl.subplot(2,3,4, sharex=ax_ATLAS_calc_t, sharey=ax_ATLAS_calc)
		fig_lumi_calc.subplots_adjust(wspace=0.5, hspace=0.5)

		for i_time in range(0, N_steps):
			colorcurr = self.colorprog(i_prog=i_time, Nplots=N_steps)
			ax_ATLAS_calc.plot(slots_filled_coll[1], lumi_bbb_ATLAS_invm2[i_time, :], '.', color=colorcurr)
			ax_CMS_calc.plot(slots_filled_coll[1],   lumi_bbb_CMS_invm2[i_time, :],   '.', color=colorcurr)

		self.plot_mean_and_spread(ax_ATLAS_calc_t, self.convertToLocalTime(time_range.astype(int)), lumi_bbb_ATLAS_invm2)
		self.plot_mean_and_spread(ax_CMS_calc_t,   self.convertToLocalTime(time_range.astype(int)), lumi_bbb_CMS_invm2)
		for ax in [ax_ATLAS_calc_t, ax_CMS_calc_t]:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			pl.setp( ax.xaxis.get_majorticklabels(), rotation=30 )

		for sp in [ax_ATLAS_calc, ax_CMS_calc]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

		for sp in [ax_ATLAS_calc_t, ax_CMS_calc_t]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel('Time', fontsize=14, fontweight='bold')

		ax_ATLAS_calc_t.set_ylabel('Calc. Luminosity ATLAS [m$\mathbf{^{-2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')
		ax_CMS_calc_t.set_ylabel('Calc. Luminosity CMS [m$\mathbf{^{-2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')


		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_lumi_calc.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_lumi_calc.savefig(self.savedir+"fill_{}_calcLumi.pdf".format(self.filln) , dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsMeasuredLuminosity(self, save=False, batch=False):
		## first get the times for the fill
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)
		print type(t_start_STABLE), t_start_STABLE.shape 

		slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
		time_range                   = self.filln_StableBeamsDict['time_range']

		info("# plotStableBeamsMeasuredLuminosity : Fill {} -> Making Measured BBB Luminosities plot...".format(self.filln))
		fig_lumi_meas   = pl.figure(7, figsize=(15,7))
		fig_lumi_meas.canvas.set_window_title('Measured bbb lumi raw')
		fig_lumi_meas.set_facecolor('w')

		ax_ATLAS_meas   = pl.subplot(2,3,(2,3))#, sharex=ax_share, sharey=ax_ATLAS_calc)
		ax_CMS_meas     = pl.subplot(2,3,(5,6), sharex=ax_ATLAS_meas, sharey=ax_ATLAS_meas)
		ax_ATLAS_meas_t = pl.subplot(2,3,1, sharey=ax_ATLAS_meas)
		ax_CMS_meas_t   = pl.subplot(2,3,4, sharex=ax_ATLAS_meas_t, sharey=ax_ATLAS_meas)
		fig_lumi_meas.subplots_adjust(wspace=0.5, hspace=0.5)

		self.plot_mean_and_spread(ax_ATLAS_meas_t, self.convertToLocalTime(time_range.astype(int)), self.filln_LumiMeasDict['ATLAS']['bunch_lumi'])
		self.plot_mean_and_spread(ax_CMS_meas_t,   self.convertToLocalTime(time_range.astype(int)), self.filln_LumiMeasDict['CMS']['bunch_lumi'])
		for ax in [ax_ATLAS_meas_t, ax_CMS_meas_t]:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			pl.setp( ax.xaxis.get_majorticklabels(), rotation=30 )

		for i_time in range(0, N_steps):
			colorcurr = self.colorprog(i_prog=i_time, Nplots=N_steps)
			ax_ATLAS_meas.plot(slots_filled_coll[1], self.filln_LumiMeasDict['ATLAS']['bunch_lumi'][i_time, :], '.', color=colorcurr)
			ax_CMS_meas.plot(slots_filled_coll[1],   self.filln_LumiMeasDict['CMS']['bunch_lumi'][i_time, :],   '.', color=colorcurr)

			for sp in [ax_ATLAS_meas, ax_CMS_meas]:
				sp.grid('on')
				sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

			for sp in [ax_ATLAS_meas_t, ax_CMS_meas_t]:
				sp.grid('on')
				sp.set_xlabel('Time', fontsize=14, fontweight='bold')
				sp.set_xlim(pd.to_datetime(t_start_STABLE, unit='s', utc=True).tz_convert('Europe/Zurich').tz_localize(None), pd.to_datetime(t_end_STABLE, unit='s', utc=True).tz_convert('Europe/Zurich').tz_localize(None))
				

			ax_ATLAS_meas_t.set_ylabel('Meas. Luminosity ATLAS [m$\mathbf{^{-2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')
			ax_CMS_meas_t.set_ylabel('Meas. Luminosity CMS [m$\mathbf{^{-2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')
			ax_CMS_meas.set_xlim(0, 3564)


		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_lumi_meas.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_lumi_meas.savefig(self.savedir+"fill_{}_measLumi.pdf".format(self.filln) , dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsTotalLuminosity(self, save=False, batch=False):
		## first get the times for the fill
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)
		# print t_start_STABLE, t_end_STABLE, time_range, N_steps

		## load info from calculated lumi dictionary
		lumi_bbb_ATLAS_invm2         = self.filln_LumiCalcDict['ATLAS']['bunch_lumi']
		lumi_bbb_CMS_invm2           = self.filln_LumiCalcDict['CMS']['bunch_lumi']

		info("# plotStableBeamsTotalLuminosity : Fill {} -> Making Total Luminosity plot...".format(self.filln))
		fig_total = pl.figure(8, figsize=(15,7))
		fig_total.canvas.set_window_title('Total Luminosity')
		fig_total.set_facecolor('w')
		# ax   = pl.subplot(111)
		pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1),       color='b', linewidth=2., label="$\mathcal{L}^{meas}_{ATLAS}$")
		pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(lumi_bbb_ATLAS_invm2,              axis=1), '--', color='b', linewidth=2., label="$\mathcal{L}^{calc}_{ATLAS}$")
		pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(self.filln_LumiMeasDict['CMS']['bunch_lumi'],   axis=1),       color='r', linewidth=2., label="$\mathcal{L}^{meas}_{CMS}$")
		pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(lumi_bbb_CMS_invm2,                axis=1), '--', color='r', linewidth=2., label="$\mathcal{L}^{calc}_{CMS}$")
		ax = pl.gca()
		ax.legend(loc='best', prop={"size":12})
		ax.set_xlim(-.5, None)
		ax.set_ylim(0, None)
		ax.set_ylabel('Luminosity [cm$\mathbf{^{-2}}$ s$\mathbf{^{-1}}$]', fontweight='bold', fontsize=14) ## NK: cm??? ATLAS??
		ax.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
		ax.grid('on')
		
		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_total.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_total.savefig(self.savedir+"fill_{}_totalLumi.pdf".format(self.filln) , dpi=300)
		pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsIntensityLifetime(self, save=False, batch=False):
		## first get the times for the fill
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)
		b_inten_interp_coll          = self.filln_StableBeamsDict['b_inten_interp_coll']

		dict_lifetime = self.filln_LifetimeDict
		if len(dict_lifetime.keys()) == 0:
			dict_lifetime = {1:{}, 2: {}}
			dNdt_bbb            = {}
			dt = self.filln_StableBeamsDict['time_range'][1]-self.filln_StableBeamsDict['time_range'][0]
			tau_Np_bbb          = {}
			for beam_n in [1,2 ]:
				dNp_bbb                   = -(b_inten_interp_coll[beam_n][:-1,:]) + (b_inten_interp_coll[beam_n][1:,:])
				dNdt_bbb[beam_n]          = (np.abs(dNp_bbb)/dt)
				tau_Np_bbb[beam_n]        = -1/((dNp_bbb/dt)/b_inten_interp_coll[beam_n][:-1,:])
				dict_lifetime[beam_n]['tau_Np_bbb'] = tau_Np_bbb[beam_n]

		# Figure Intensity Lifetime
		info("# plotStableBeamsIntensityLifetime : Fill {} -> Making BBB Lifetime...".format(self.filln))
		fig_bbb_tau = pl.figure('bbb_tau', figsize=(15,7))
		fig_bbb_tau.set_facecolor('w')
		ax_b1_bbbtau = pl.subplot(211)
		ax_b2_bbbtau = pl.subplot(212)

		self.plot_mean_and_spread(ax_b1_bbbtau, self.convertToLocalTime(time_range[0:-1].astype(int)), dict_lifetime[1]['tau_Np_bbb']/3600., label='Beam 1 - $\\tau_{N_{p}^{0}}$'+'={:.2f}h'.format(np.mean(dict_lifetime[1]['tau_Np_bbb']/3600., axis=1)[0]), color='b', shade=True)
		ax_b1_bbbtau.grid('on')
		ax_b1_bbbtau.set_xlabel('Time', fontsize=14, fontweight='bold')
		ax_b1_bbbtau.set_ylabel("$\mathbf{\\tau_{N_{p}}}$ [h]", fontsize=14, fontweight='bold')
		ax_b1_bbbtau.legend(loc='best')

		self.plot_mean_and_spread(ax_b2_bbbtau, self.convertToLocalTime(time_range[0:-1].astype(int)), dict_lifetime[2]['tau_Np_bbb']/3600., label='Beam 2 - $\\tau_{N_{p}^{0}}$'+'={:.2f}h'.format(np.mean(dict_lifetime[2]['tau_Np_bbb']/3600., axis=1)[0]), color='r', shade=True)
		ax_b2_bbbtau.grid('on')
		ax_b2_bbbtau.set_xlabel('Time', fontsize=14, fontweight='bold')
		ax_b2_bbbtau.set_ylabel("$\mathbf{\\tau_{N_{p}}}$ [h]", fontsize=14, fontweight='bold')
		ax_b2_bbbtau.legend(loc='best')
		pl.subplots_adjust(hspace=0.5, wspace=0.5)#, hspace=0.7)
		
		for ax in [ax_b1_bbbtau, ax_b2_bbbtau]:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			# pl.setp( ax.xaxis.get_majorticklabels(), rotation=30 )

		ax_b1_bbbtau.set_ylim(10,60)
		ax_b2_bbbtau.set_ylim(10,60)


		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_bbb_tau.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_bbb_tau.savefig(self.savedir+"fill_{}_bbbIntensityTau.pdf".format(self.filln) , dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsNormalizedLosses(self, save=False, batch=False):
		## first get the times for the fill
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)

		info("# plotStableBeamsNormalizedLosses : Fill {} -> Making BBB Normalized Losses plot...".format(self.filln))
		fig_bbblosses = pl.figure('bbb_losses', figsize=(15,7))
		fig_bbblosses.set_facecolor('w')
		ax_b1 = pl.subplot(211)
		ax_b2 = pl.subplot(212)

		self.plot_mean_and_spread(ax_b1, self.convertToLocalTime(time_range[0:-1].astype(int)), self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, label='Beam 1 - $\sigma_{eff}^{0}$'+'={:.1f}mb'.format(np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1)[0]), color='b', shade=True)
		ax_b1.axhline(80, xmin=0, xmax=1, color='black', label='$\sigma_{\mathrm{inel}}$ = 80mb')
		ax_b1.grid('on')
		ax_b1.set_xlabel('Time', fontsize=14, fontweight='bold')
		ax_b1.set_ylabel("$\mathbf{\left(\\frac{dN}{dt}\\right)\slash\mathcal{L}}$ [mb]", fontsize=14, fontweight='bold')
		ax_b1.legend(loc='best')

		self.plot_mean_and_spread(ax_b2, self.convertToLocalTime(time_range[0:-1].astype(int)), self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, label='Beam 2 - $\sigma_{eff}^{0}$'+'={:.1f}mb'.format(np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1)[0]), color='r', shade=True)
		ax_b2.axhline(80, xmin=0, xmax=1, color='black', label='$\sigma_{\mathrm{inel}}$ = 80mb')
		ax_b2.grid('on')
		ax_b2.set_xlabel('Time', fontsize=14, fontweight='bold')
		ax_b2.set_ylabel("$\mathbf{\left(\\frac{dN}{dt}\\right)\slash\mathcal{L}}$ [mb]", fontsize=14, fontweight='bold')
		ax_b2.legend(loc='best')
		pl.subplots_adjust(hspace=0.5)#, hspace=0.7)
		for ax in [ax_b1, ax_b2]:
			ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
			# pl.setp( ax.xaxis.get_majorticklabels(), rotation=30 )

		ax_b1.set_ylim(60, 200)
		ax_b2.set_ylim(60, 200)

		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_bbblosses.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_bbblosses.savefig(self.savedir+"fill_{}_bbbLosses.pdf".format(self.filln) , dpi=300)
		if not batch:
			pl.show()

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeamsLuminosityLifetime(self, save=False, batch=False, return_result=False, turn_around_time_h=3.5):

		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)
		slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
		time_range                   = self.filln_StableBeamsDict['time_range']

		info("# plotStableBeamsLuminosityLifetime : Fill {} -> Making Luminosity Lifetime Total plot...".format(self.filln))
		fig_tauLumi_tot = pl.figure('tauLumiTotal', figsize=(15,7))
		tau_tot_AT  = pl.subplot(211)
		tau_tot_AT.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['ATLAS']['tau_lumi_calc_coll_full']/3600., c='g', marker='o', markersize=4, ls='None', label='ATLAS - Total Calculated')
		tau_tot_AT.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['ATLAS']['tau_lumi_meas_coll_full']/3600., c='k', marker='o', markersize=4, ls='None', label='ATLAS - Total Measured')
		tau_tot_AT.grid('on')
		tau_tot_AT.set_ylabel('$\mathbf{\\tau_{\mathcal{L}}}$ [h]', fontsize=14, fontweight='bold')
		tau_tot_AT.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
		# tau_tot_AT.text(0.5, 1.0, "ATLAS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
		# tau_tot_AT.text(0.1, 0.9, "ATLAS", horizontalalignment='center', verticalalignment='top', transform=tau_tot_AT.transAxes,
		#              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)

		tau_tot_AT.legend(loc='best', numpoints=1)

		tau_tot_CMS = pl.subplot(212)
		# print '------> Slots : ', slots_filled_coll[1][0], ' , shape : ', np.array(slots_filled_coll[1]).shape
		# print '------> Tau Total : ', (self.filln_SBFitsDict['CMS']['tau_lumi_calc_coll_full']/3600.)[0], ' , shape : ', np.array(self.filln_SBFitsDict['CMS']['tau_lumi_calc_coll_full']/3600.).shape
		tau_tot_CMS.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['CMS']['tau_lumi_calc_coll_full']/3600., c='g', marker='o', markersize=4, ls='None',  label='CMS - Total Calculated')#ls='dotted',
		tau_tot_CMS.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['CMS']['tau_lumi_meas_coll_full']/3600., c='k', marker='o', markersize=4, ls='None',  label='CMS - Total Measured')  #ls='dotted',
		tau_tot_CMS.grid('on')
		tau_tot_CMS.set_ylabel('$\mathbf{\\tau_{\mathcal{L}}}$ [h]', fontsize=14, fontweight='bold')
		tau_tot_CMS.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
		# tau_tot_CMS.text(0.5, 1.0, "CMS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
		# tau_tot_CMS.text(0.1, 0.9, "CMS", horizontalalignment='center', verticalalignment='top', transform=tau_tot_CMS.transAxes,
		#              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)
		tau_tot_CMS.legend(loc='best', numpoints=1)
		pl.subplots_adjust(hspace=0.5)

		mean_tau_atlas_measured   = np.mean(-1.0*self.filln_SBFitsDict['ATLAS']['tau_lumi_meas_coll_full']/3600.)
		mean_tau_cms_measured     = np.mean(-1.0*self.filln_SBFitsDict['CMS']['tau_lumi_meas_coll_full']/3600.)
		mean_tau_atlas_calculated =	np.mean(-1.0*self.filln_SBFitsDict['ATLAS']['tau_lumi_calc_coll_full']/3600.)
		mean_tau_cms_calculated   = np.mean(-1.0*self.filln_SBFitsDict['CMS']['tau_lumi_calc_coll_full']/3600.)
		turn_around_time_h        = turn_around_time_h

		average_measured_lifetime_between_ATLAS_CMS   = (mean_tau_atlas_measured+mean_tau_cms_measured)/2.0
		average_calculated_lifetime_between_ATLAS_CMS = (mean_tau_atlas_calculated+mean_tau_cms_calculated)/2.0
		
		optimal_fill_length_measured   = average_measured_lifetime_between_ATLAS_CMS*np.log(1.0+np.sqrt(2*turn_around_time_h/average_measured_lifetime_between_ATLAS_CMS) + turn_around_time_h/average_measured_lifetime_between_ATLAS_CMS)
		optimal_fill_length_calculated = average_calculated_lifetime_between_ATLAS_CMS*np.log(1.0+np.sqrt(2*turn_around_time_h/average_calculated_lifetime_between_ATLAS_CMS) + turn_around_time_h/average_calculated_lifetime_between_ATLAS_CMS)

		optimal_fill_length_ATLAS_measured = mean_tau_atlas_measured*np.log(1.0+np.sqrt(2*turn_around_time_h/mean_tau_atlas_measured) + turn_around_time_h/mean_tau_atlas_measured)
		optimal_fill_length_CMS_measured   = mean_tau_cms_measured*np.log(1.0+np.sqrt(2*turn_around_time_h/mean_tau_cms_measured) + turn_around_time_h/mean_tau_cms_measured)

		info("# plotStableBeamsLuminosityLifetime : Fit Total : Mean Measured ATLAS      = {:.2f} h ".format(mean_tau_atlas_measured))
		info("# plotStableBeamsLuminosityLifetime : Fit Total : Mean Measured CMS        = {:.2f} h ".format(mean_tau_cms_measured))
		info("# plotStableBeamsLuminosityLifetime : Fit Total : Mean Calculated ATLAS    = {:.2f} h ".format(mean_tau_atlas_calculated))
		info("# plotStableBeamsLuminosityLifetime : Fit Total : Mean calculated CMS      = {:.2f} h ".format(mean_tau_cms_calculated))
		info("# plotStableBeamsLuminosityLifetime : ------------------------------------------------")
		info("# plotStableBeamsLuminosityLifetime : Turn-around time                     = {:.2f} h ".format(turn_around_time_h))
		info("# plotStableBeamsLuminosityLifetime : ------------------------------------------------")
		info("# plotStableBeamsLuminosityLifetime : Optimal Fill Length - Measured Avg   = {:.2f} h ".format(optimal_fill_length_measured))
		info("# plotStableBeamsLuminosityLifetime : Optimal Fill Length - Calculated Avg = {:.2f} h ".format(optimal_fill_length_calculated))
		info("# plotStableBeamsLuminosityLifetime : ------------------------------------------------")
		info("# plotStableBeamsLuminosityLifetime : Optimal Fill Length - ATLAS Measured = {:.2f} h ".format(optimal_fill_length_ATLAS_measured))
		info("# plotStableBeamsLuminosityLifetime : Optimal Fill Length - CMS Measured   = {:.2f} h ".format(optimal_fill_length_CMS_measured))



		# Only for the fitted time
		info("# plotStableBeamsLuminosityLifetime : Fill {} -> Making Luminosity Lifetime (Fit) plot...".format(self.filln))
		fig_tauLumi_fit = pl.figure('tauLumiFit', figsize=(15,7))
		tau_fit_AT  = pl.subplot(211)
		tau_fit_AT.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['ATLAS']['tau_lumi_calc_coll']/3600., c='g', marker='o', markersize=4, ls='None', label='ATLAS - {}h Calculated'.format(int(config.t_fit_length/3600.)))
		tau_fit_AT.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['ATLAS']['tau_lumi_meas_coll']/3600., c='k', marker='o', markersize=4, ls='None', label='ATLAS - {}h Measured'.format(int(config.t_fit_length/3600.)))
		tau_fit_AT.grid('on')
		tau_fit_AT.set_ylabel('$\mathbf{\\tau_{\mathcal{L}}}$ [h]', fontsize=14, fontweight='bold')
		tau_fit_AT.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
		# tau_fit_AT.text(0.5, 1.0, "ATLAS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
		# tau_fit_AT.text(0.1, 0.9, "ATLAS", horizontalalignment='center', verticalalignment='top', transform=tau_fit_AT.transAxes,
		#              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)
		tau_fit_AT.legend(loc='best', numpoints=1)

		tau_fit_CMS = pl.subplot(212)
		tau_fit_CMS.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['CMS']['tau_lumi_calc_coll']/3600., c='g', marker='o', markersize=4, ls='None', label='CMS - {}h Calculated'.format(int(config.t_fit_length/3600.)))
		tau_fit_CMS.plot(slots_filled_coll[1], -1.0*self.filln_SBFitsDict['CMS']['tau_lumi_meas_coll']/3600., c='k', marker='o', markersize=4, ls='None', label='CMS - {}h Measured'.format(int(config.t_fit_length/3600.)))
		tau_fit_CMS.grid('on')
		tau_fit_CMS.set_ylabel('$\mathbf{\\tau_{\mathcal{L}}}$ [h]', fontsize=14, fontweight='bold')
		tau_fit_CMS.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
		# tau_fit_CMS.text(0.5, 1.0, "CMS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
		# tau_fit_CMS.text(0.1, 0.9, "CMS", horizontalalignment='center', verticalalignment='top', transform=tau_fit_CMS.transAxes,
		#              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)
		tau_fit_CMS.legend(loc='best', numpoints=1)
		pl.subplots_adjust(hspace=0.5, wspace=0.5)

		for sp in [tau_tot_AT, tau_tot_CMS, tau_fit_AT, tau_fit_CMS]:
			sp.set_ylim(5, 30)

		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
		fig_tauLumi_tot.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)
		fig_tauLumi_fit.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

		if save:
			fig_tauLumi_tot.savefig(self.savedir+"fill_{}_TotalLumiLifetime.pdf".format(self.filln) , dpi=300)
			fig_tauLumi_fit.savefig(self.savedir+"fill_{}_{}hFitLumiLifetime.pdf".format(self.filln, str(int(config.t_fit_length/3600.)).replace(".","dot") ) , dpi=300)
		if batch is False:
			# pl.show()
			pass

		if return_result: 
			return mean_tau_atlas_measured, mean_tau_cms_measured, mean_tau_atlas_calculated, mean_tau_cms_calculated, turn_around_time_h, optimal_fill_length_measured, optimal_fill_length_calculated, optimal_fill_length_ATLAS_measured, optimal_fill_length_CMS_measured

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	# def plotStableBeamsEmittanceGrowth(self, save=False, batch=False):
	# 	t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)
	# 	slots_filled_coll               = self.filln_StableBeamsDict['slots_filled_coll']
	# 	slots_filled_noncoll            = self.filln_StableBeamsDict['slots_filled_noncoll']
	# 	time_range                      = self.filln_StableBeamsDict['time_range']

	# 	info("# plotStableBeamsEmittanceGrowth : Fill {} -> Making B1 Emittance Growth plot...".format(self.filln))
	# 	tau_tot_B1_coll = pl.figure('tauEmit_B1', figsize=(15,7))
	# 	tau_tot_B1_coll_H  = pl.subplot(211)
	# 	tau_tot_B1_coll_V  = pl.subplot(212)

	# 	for i_time in range(N_steps):
	# 		colorcurr = self.colorprog(i_prog=i_time, Nplots=N_steps)
	# 		tau_tot_B1_coll_H.plot(slots_filled_coll[1], self.filln_SBFitsDict['beam_1']['tau_emith_coll_full'][i_time, :]/3600, '.', color=colorcurr)
	# 		tau_tot_B1_noncoll_H.plot(slots_filled_noncoll[1], self.filln_SBFitsDict['beam_1']['tau_emith_noncoll_full'][i_time, :]/3600, '.', color=colorcurr)



	# 	# tau_tot_B1_coll_H.plot(slots_filled_coll[1],    np.mean(self.filln_SBFitsDict['beam_1']['tau_emith_coll_full']/3600., axis=0), c='g', marker='o', markersize=4, ls='None', label="Colliding")
	# 	# tau_tot_B1_coll_H.plot(slots_filled_noncoll[1], np.mean(self.filln_SBFitsDict['beam_1']['tau_emith_noncoll_full']/3600., axis=0), c='k', marker='o', markersize=4, ls='None', label='Non-Colliding')
	# 	# tau_tot_B1_coll_H.grid('on')
	# 	# # tau_tot_B1_coll_H.set_ylabel('B1 $\mathbf{\\tau_{\mathcal{\epsilon_{H}}}}$ [h]', fontsize=14, fontweight='bold')
	# 	# tau_tot_B1_coll_H.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
		
	# 	# tau_tot_B1_coll_V.plot(slots_filled_coll[1], self.filln_SBFitsDict['beam_1']['tau_emitv_coll_full']/3600., c='g', marker='o', markersize=4, ls='None', label="Colliding")
	# 	# tau_tot_B1_coll_V.plot(slots_filled_noncoll[1], self.filln_SBFitsDict['beam_1']['tau_emitv_noncoll_full']/3600., c='k', marker='o', markersize=4, ls='None', label="Non-Colliding")
	# 	# tau_tot_B1_coll_V.grid('on')
	# 	# # tau_tot_B1_coll_V.set_ylabel('B1 $\mathbf{\\tau_{\mathcal{\epsilon_{V}}}}$ [h]', fontsize=14, fontweight='bold')
	# 	# tau_tot_B1_coll_V.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
	# 	# pl.subplots_adjust(hspace=0.5, wspace=0.5)

	# 	# tau_tot_B1_coll_H.legend(loc='best', numpoints=1)
	# 	# tau_tot_B1_coll_V.legend(loc='best', numpoints=1)

	# 	# info("# plotStableBeamsEmittanceGrowth : Fill {} -> Making B2 Emittance Growth plot...".format(self.filln))
	# 	# tau_tot_B2_coll = pl.figure('tauEmit_B2', figsize=(15,7))
	# 	# tau_tot_B2_coll_H  = pl.subplot(211)
	# 	# tau_tot_B2_coll_V  = pl.subplot(212)
	# 	# tau_tot_B2_coll_H.plot(slots_filled_coll[1], self.filln_SBFitsDict['beam_2']['tau_emith_coll_full']/3600., c='g', marker='o', markersize=4, ls='None', label="Colliding")
	# 	# tau_tot_B2_coll_H.plot(slots_filled_noncoll[1], self.filln_SBFitsDict['beam_2']['tau_emith_noncoll_full']/3600., c='k', marker='o', markersize=4, ls='None', label='Non-Colliding')
	# 	# tau_tot_B2_coll_H.grid('on')
	# 	# # tau_tot_B2_coll_H.set_ylabel('B2 $\mathbf{\\tau_{\mathcal{\epsilon_{H}}}}$ [h]', fontsize=14, fontweight='bold')
	# 	# tau_tot_B2_coll_H.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
		
	# 	# tau_tot_B2_coll_V.plot(slots_filled_coll[1], self.filln_SBFitsDict['beam_2']['tau_emitv_coll_full']/3600., c='g', marker='o', markersize=4, ls='None', label="Colliding")
	# 	# tau_tot_B2_coll_V.plot(slots_filled_noncoll[1], self.filln_SBFitsDict['beam_2']['tau_emitv_noncoll_full']/3600., c='k', marker='o', markersize=4, ls='None', label="Non-Colliding")
	# 	# tau_tot_B2_coll_V.grid('on')
	# 	# # tau_tot_B2_coll_V.set_ylabel('B2 $\mathbf{\\tau_{\mathcal{\epsilon_{V}}}}$ [h]', fontsize=14, fontweight='bold')
	# 	# tau_tot_B2_coll_V.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
	# 	# pl.subplots_adjust(hspace=0.5, wspace=0.5)

	# 	# tau_tot_B2_coll_H.legend(loc='best', numpoints=1)
	# 	# tau_tot_B2_coll_V.legend(loc='best', numpoints=1)

	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B1H Colliding     = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_1']['tau_emith_coll_full']/3600.)))
	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B1V Colliding     = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_1']['tau_emitv_coll_full']/3600.)))
	# 	info("# plotStableBeamsEmittanceGrowth : ------------------------------------------------")
	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B2H Colliding     = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_2']['tau_emith_coll_full']/3600.)))
	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B2V Colliding     = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_2']['tau_emitv_coll_full']/3600.)))
	# 	info("# plotStableBeamsEmittanceGrowth : ------------------------------------------------")		
	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B1H Non-Colliding = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_1']['tau_emith_noncoll_full']/3600.)))
	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B1V Non-Colliding = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_1']['tau_emitv_noncoll_full']/3600.)))
	# 	info("# plotStableBeamsEmittanceGrowth : ------------------------------------------------")
	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B2H Non-Colliding = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_2']['tau_emith_noncoll_full']/3600.)))
	# 	info("# plotStableBeamsEmittanceGrowth : Fit Total : Mean B2V Non-Colliding = {:.2f} h ".format(np.mean(self.filln_SBFitsDict['beam_2']['tau_emitv_noncoll_full']/3600.)))



	# 	# for sp in [tau_tot_B1_coll_H, tau_tot_B1_coll_V, tau_tot_B2_coll_H, tau_tot_B2_coll_V]:
	# 	# 	sp.set_ylim(5, 30)

	# 	tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
	# 	tau_tot_B1_coll.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)
	# 	tau_tot_B2_coll.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(self.filln, tref_string), fontsize=18)

	# 	if save:
	# 		tau_tot_B1_coll.savefig(self.savedir+"fill_{}_TotalEmittanceGrowthB1.pdf".format(self.filln) , dpi=300)
	# 		tau_tot_B2_coll.savefig(self.savedir+"fill_{}_TotalEmittanceGrowthB2.pdf".format(self.filln))
	# 	if not batch:
	# 		pl.show()

	# 	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *



	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	########################################################################################################################
	#
	#													STABLE BEAMS MODEL PLOTS
	#
	#########################################################################################################################

	# todo

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	########################################################################################################################
	#
	#													DRIVERS
	#
	#########################################################################################################################

	def plotCycle(self, save=False, batch=False):
		self.plotCycleEmittances(save, batch)
		self.getCycleEmittanceTable(save, batch)
		self.plotCycleEmittanceViolins(save, batch)
		self.plotCycleIntensities(save, batch)
		self.plotCycleBrightness(save, batch)
		self.plotCycleBunchLength(save, batch)
		self.plotCycleTime(save, batch)
		self.plotCycleHistos(save, batch)

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotCycleModel(self, save=False, batch=False):
		self.plotCycleModelEmittance(save, batch)
		self.plotCycleModelBunchLength(save, batch)

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotStableBeams(self, save=False, batch=False):
		self.plotStableBeamsEmittances(save, batch)
		self.plotStableBeamsIntensity(save, batch)
		self.plotStableBeamsBunchLength(save, batch)
		self.plotStableBeamsCalculatedLuminosity(save, batch)
		self.plotStableBeamsMeasuredLuminosity(save, batch)
		self.plotStableBeamsTotalLuminosity(save, batch)
		self.plotStableBeamsNormalizedLosses(save, batch)
		self.plotStableBeamsIntensityLifetime(save, batch)
		self.plotStableBeamsLuminosityLifetime(save, batch)

	# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

	def plotAllAndSave(self):
		self.plotCycle(True, True)
		self.plotCycleModel(True, True)
		self.plotStableBeams(True, True)

	########################################################################################################################
	#
	#													STATISTICS DATAFRAME
	#
	#########################################################################################################################

	def statisticsDataFrame(self, save=False):
		df_name      = config.stableBeams_folder+"/statsPlotterDF.pkl"
		df_emit_name = config.stableBeams_folder+"/statsEmittancesDF.pkl"

		if os.path.exists(df_name):
			info('# statisticsDataFrame: DF [{}] exists! Loading it...'.format(df_name))
			#with pd.HDFStore(df_name) as store:
			df = pd.read_pickle(df_name)

		else:
			df = pd.DataFrame()

		if os.path.exists(df_emit_name):
			info('# statisticsDataFrame: DF [{}] exists! Loading it...'.format(df_emit_name))
			#with pd.HDFStore(df_name) as store:
			df_emit = pd.read_pickle(df_emit_name)

		else:
			df_emit = pd.DataFrame()
		

		tmp_dict = OrderedDict()

		########################################################	
		#	
		# TIME AND FILL INFO
		#
		########################################################	
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(self.filln)
		t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(self.filln)

		
		#---- filling...
		tmp_dict.update({'fill' 			: int(self.filln)})
		tmp_dict.update({'bunches' 			: int(self.bunches)})
		tmp_dict.update({'t_start_fill' 	: int(t_start_fill)})
		tmp_dict.update({'t_end_fill' 		: int(t_end_fill)})
		tmp_dict.update({'t_fill_len' 		: int(t_fill_len)})
		tmp_dict.update({'t_start_STABLE'   : int(t_start_STABLE)})
		tmp_dict.update({'t_end_STABLE'     : int(t_end_STABLE)})
		tmp_dict.update({'t_STABLE_len_h'      : (t_end_STABLE-t_start_STABLE)/3600.})



		########################################################	
		#	
		# CYCLE INFO
		#
		########################################################


		# Cycle Emittances
		cycle_emit_dict = self.getCycleEmittanceTable(print_table=False)


		# tmp_dict.update({'mean_delta_cycle_emit_B1H_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emith'])     -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1H_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith']) -np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1H_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1H_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emith'])))})
		
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv'])      -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv']) -np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv'])))})
		
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emith'])      -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith']) -np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emith'])))})
		
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv'])      -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv']) -np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv'])))})

		# tmp_dict.update({'mean_delta_cycle_emit_B1H_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emith'])     -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1H_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith']) -np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1H_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1H_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv'])      -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv']) -np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B1V_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emith'])      -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith']) -np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2H_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith'])   -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emith'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_INJ2FB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv'])      -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_FB2FT' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv']) -np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_FT2SB' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv'])))})
		# tmp_dict.update({'mean_delta_cycle_emit_B2V_INJ2SB': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv'])   -np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv'])))})


		cycle_emit_B1H_INJ =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emith']))
		cycle_emit_B1H_FB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emith']))
		cycle_emit_B1H_FT  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith']))
		cycle_emit_B1H_SB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith']))

		cycle_emit_B1V_INJ =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv']))
		cycle_emit_B1V_FB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv']))
		cycle_emit_B1V_FT  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv']))
		cycle_emit_B1V_SB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv']))
		
		cycle_emit_B2H_INJ =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emith']))
		cycle_emit_B2H_FB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emith']))
		cycle_emit_B2H_FT  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith']))
		cycle_emit_B2H_SB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith']))
		
		cycle_emit_B2V_INJ =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv']))
		cycle_emit_B2V_FB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv']))
		cycle_emit_B2V_FT  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv']))
		cycle_emit_B2V_SB  =  ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv']))




		mean_cycle_emit_B1H_INJ	 = np.mean(cycle_emit_B1H_INJ)
		mean_cycle_emit_B1H_FB 	 = np.mean(cycle_emit_B1H_FB)
		mean_cycle_emit_B1H_FT 	 = np.mean(cycle_emit_B1H_FT)
		mean_cycle_emit_B1H_SB 	 = np.mean(cycle_emit_B1H_SB)
		mean_cycle_emit_B1V_INJ	 = np.mean(cycle_emit_B1V_INJ)
		mean_cycle_emit_B1V_FB 	 = np.mean(cycle_emit_B1V_FB)
		mean_cycle_emit_B1V_FT 	 = np.mean(cycle_emit_B1V_FT)
		mean_cycle_emit_B1V_SB 	 = np.mean(cycle_emit_B1V_SB)
		mean_cycle_emit_B2H_INJ	 = np.mean(cycle_emit_B2H_INJ)
		mean_cycle_emit_B2H_FB 	 = np.mean(cycle_emit_B2H_FB)
		mean_cycle_emit_B2H_FT 	 = np.mean(cycle_emit_B2H_FT)
		mean_cycle_emit_B2H_SB 	 = np.mean(cycle_emit_B2H_SB)
		mean_cycle_emit_B2V_INJ	 = np.mean(cycle_emit_B2V_INJ)
		mean_cycle_emit_B2V_FB 	 = np.mean(cycle_emit_B2V_FB)
		mean_cycle_emit_B2V_FT 	 = np.mean(cycle_emit_B2V_FT)
		mean_cycle_emit_B2V_SB 	 = np.mean(cycle_emit_B2V_SB)




		demit_B1H_INJ2FB 	=	mean_cycle_emit_B1H_FB - mean_cycle_emit_B1H_INJ
		demit_B1H_FB2FT 	=	mean_cycle_emit_B1H_FT - mean_cycle_emit_B1H_FB
		demit_B1H_FT2SB 	=	mean_cycle_emit_B1H_SB - mean_cycle_emit_B1H_FT
		demit_B1H_INJ2SB 	=	mean_cycle_emit_B1H_SB - mean_cycle_emit_B1H_INJ
		demit_B1V_INJ2FB 	=	mean_cycle_emit_B1V_FB - mean_cycle_emit_B1V_INJ
		demit_B1V_FB2FT 	=	mean_cycle_emit_B1V_FT - mean_cycle_emit_B1V_FB
		demit_B1V_FT2SB 	=	mean_cycle_emit_B1V_SB - mean_cycle_emit_B1V_FT
		demit_B1V_INJ2SB 	=	mean_cycle_emit_B1V_SB - mean_cycle_emit_B1V_INJ
		demit_B2H_INJ2FB 	=	mean_cycle_emit_B2H_FB - mean_cycle_emit_B2H_INJ
		demit_B2H_FB2FT 	=	mean_cycle_emit_B2H_FT - mean_cycle_emit_B2H_FB
		demit_B2H_FT2SB 	=	mean_cycle_emit_B2H_SB - mean_cycle_emit_B2H_FT
		demit_B2H_INJ2SB 	=	mean_cycle_emit_B2H_SB - mean_cycle_emit_B2H_INJ
		demit_B2V_INJ2FB 	=	mean_cycle_emit_B2V_FB - mean_cycle_emit_B2V_INJ
		demit_B2V_FB2FT 	=	mean_cycle_emit_B2V_FT - mean_cycle_emit_B2V_FB
		demit_B2V_FT2SB 	=	mean_cycle_emit_B2V_SB - mean_cycle_emit_B2V_FT
		demit_B2V_INJ2SB 	=	mean_cycle_emit_B2V_SB - mean_cycle_emit_B2V_INJ


		tmp_dict.update({'demit_B1H_INJ2FB' :	demit_B1H_INJ2FB })
 		tmp_dict.update({'demit_B1H_FB2FT'	:	demit_B1H_FB2FT 	 })
		tmp_dict.update({'demit_B1H_FT2SB'	:	demit_B1H_FT2SB 	 })
		tmp_dict.update({'demit_B1H_INJ2SB'	:	demit_B1H_INJ2SB	 })
		tmp_dict.update({'demit_B1V_INJ2FB'	:	demit_B1V_INJ2FB	 })
		tmp_dict.update({'demit_B1V_FB2FT'	:	demit_B1V_FB2FT 	 })
		tmp_dict.update({'demit_B1V_FT2SB'	:	demit_B1V_FT2SB 	 })
		tmp_dict.update({'demit_B1V_INJ2SB'	:	demit_B1V_INJ2SB	 })
		tmp_dict.update({'demit_B2H_INJ2FB'	:	demit_B2H_INJ2FB	 })
		tmp_dict.update({'demit_B2H_FB2FT'	:	demit_B2H_FB2FT 	 })
		tmp_dict.update({'demit_B2H_FT2SB'	:	demit_B2H_FT2SB 	 })
		tmp_dict.update({'demit_B2H_INJ2SB'	:	demit_B2H_INJ2SB	 })
		tmp_dict.update({'demit_B2V_INJ2FB'	:	demit_B2V_INJ2FB	 })
		tmp_dict.update({'demit_B2V_FB2FT'	:	demit_B2V_FB2FT 	 })
		tmp_dict.update({'demit_B2V_FT2SB'	:	demit_B2V_FT2SB 	 })
		tmp_dict.update({'demit_B2V_INJ2SB'	:	demit_B2V_INJ2SB	 })




		tmp_dict.update({'mean_cycle_emit_B1H_INJ' : mean_cycle_emit_B1H_INJ })
		tmp_dict.update({'mean_cycle_emit_B1H_FB' : mean_cycle_emit_B1H_FB  })
		tmp_dict.update({'mean_cycle_emit_B1V_INJ' : mean_cycle_emit_B1V_INJ })
		tmp_dict.update({'mean_cycle_emit_B1V_FB' : mean_cycle_emit_B1V_FB  })
		tmp_dict.update({'mean_cycle_emit_B1H_FT' : mean_cycle_emit_B1H_FT  })
		tmp_dict.update({'mean_cycle_emit_B1H_SB' : mean_cycle_emit_B1H_SB  })
		tmp_dict.update({'mean_cycle_emit_B1V_FT' : mean_cycle_emit_B1V_FT  })
		tmp_dict.update({'mean_cycle_emit_B1V_SB' : mean_cycle_emit_B1V_SB  })
		tmp_dict.update({'mean_cycle_emit_B2H_INJ' : mean_cycle_emit_B2H_INJ })
		tmp_dict.update({'mean_cycle_emit_B2H_FB' : mean_cycle_emit_B2H_FB  })
		tmp_dict.update({'mean_cycle_emit_B2V_INJ' : mean_cycle_emit_B2V_INJ })
		tmp_dict.update({'mean_cycle_emit_B2V_FB' : mean_cycle_emit_B2V_FB  })
		tmp_dict.update({'mean_cycle_emit_B2H_FT' : mean_cycle_emit_B2H_FT  })
		tmp_dict.update({'mean_cycle_emit_B2H_SB' : mean_cycle_emit_B2H_SB  })
		tmp_dict.update({'mean_cycle_emit_B2V_FT' : mean_cycle_emit_B2V_FT  })
		tmp_dict.update({'mean_cycle_emit_B2V_SB' : mean_cycle_emit_B2V_SB  })

		tmp_dict.update({'std_cycle_emit_B1H_INJ': np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B1H_FB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B1V_INJ': np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv'])))})
		tmp_dict.update({'std_cycle_emit_B1V_FB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv'])))})
		tmp_dict.update({'std_cycle_emit_B1H_FT' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B1H_SB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B1V_FT' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv'])))})
		tmp_dict.update({'std_cycle_emit_B1V_SB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv'])))})
		tmp_dict.update({'std_cycle_emit_B2H_INJ': np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B2H_FB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B2V_INJ': np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv'])))})
		tmp_dict.update({'std_cycle_emit_B2V_FB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv'])))})
		tmp_dict.update({'std_cycle_emit_B2H_FT' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B2H_SB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith'])))})
		tmp_dict.update({'std_cycle_emit_B2V_FT' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv'])))})
		tmp_dict.update({'std_cycle_emit_B2V_SB' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv'])))})


		tmp_dict.update({'emit_relDiff_B1H_Injection_Start_Ramp'	: cycle_emit_dict['B1H'][0][1]})	
		tmp_dict.update({'emit_relDiff_B1H_Start_Ramp_End_Ramp'		: cycle_emit_dict['B1H'][1][1]})	
		tmp_dict.update({'emit_relDiff_B1H_End_Ramp_Start_SB'		: cycle_emit_dict['B1H'][2][1]})	
		tmp_dict.update({'emit_relDiff_B1H_Injection_Start_SB'		: cycle_emit_dict['B1H'][3][1]})	
		tmp_dict.update({'emit_relDiff_B1V_Injection_Start_Ramp'	: cycle_emit_dict['B1V'][0][1]})
		tmp_dict.update({'emit_relDiff_B1V_Start_Ramp_End_Ramp'		: cycle_emit_dict['B1V'][1][1]})
		tmp_dict.update({'emit_relDiff_B1V_End_Ramp_Start_SB'		: cycle_emit_dict['B1V'][2][1]})
		tmp_dict.update({'emit_relDiff_B1V_Injection_Start_SB'		: cycle_emit_dict['B1V'][3][1]})
		tmp_dict.update({'emit_relDiff_B2H_Injection_Start_Ramp'	: cycle_emit_dict['B2H'][0][1]})
		tmp_dict.update({'emit_relDiff_B2H_Start_Ramp_End_Ramp'		: cycle_emit_dict['B2H'][1][1]})
		tmp_dict.update({'emit_relDiff_B2H_End_Ramp_Start_SB'		: cycle_emit_dict['B2H'][2][1]})
		tmp_dict.update({'emit_relDiff_B2H_Injection_Start_SB'		: cycle_emit_dict['B2H'][3][1]})
		tmp_dict.update({'emit_relDiff_B2V_Injection_Start_Ramp'	: cycle_emit_dict['B2V'][0][1]})
		tmp_dict.update({'emit_relDiff_B2V_Start_Ramp_End_Ramp'		: cycle_emit_dict['B2V'][1][1]})
		tmp_dict.update({'emit_relDiff_B2V_End_Ramp_Start_SB'		: cycle_emit_dict['B2V'][2][1]})
		tmp_dict.update({'emit_relDiff_B2V_Injection_Start_SB'		: cycle_emit_dict['B2V'][3][1]})

		tmp_dict.update({'filled_slots_Injection_b1': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['filled_slots'])))})
		tmp_dict.update({'filled_slots_Injection_b2': np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['filled_slots'])))})
		tmp_dict.update({'filled_slots_FlatTop_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['filled_slots'])))})
		tmp_dict.update({'filled_slots_FlatTop_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['filled_slots'])))})

		tmp_dict.update({'mean_cycle_intensity_INJ_b1' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['intensity'])))})
		tmp_dict.update({'mean_cycle_intensity_INJ_b2' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['intensity'])))})
		tmp_dict.update({'mean_cycle_intensity_FB_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['intensity'])))})
		tmp_dict.update({'mean_cycle_intensity_FB_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['intensity'])))})
		tmp_dict.update({'mean_cycle_intensity_FT_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['intensity'])))})
		tmp_dict.update({'mean_cycle_intensity_FT_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['intensity'])))})
		tmp_dict.update({'mean_cycle_intensity_SB_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['intensity'])))})
		tmp_dict.update({'mean_cycle_intensity_SB_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['intensity'])))})
		
		tmp_dict.update({'std_cycle_intensity_INJ_b1' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['intensity'])))})
		tmp_dict.update({'std_cycle_intensity_INJ_b2' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['intensity'])))})
		tmp_dict.update({'std_cycle_intensity_FB_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['intensity'])))})
		tmp_dict.update({'std_cycle_intensity_FB_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['intensity'])))})
		tmp_dict.update({'std_cycle_intensity_FT_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['intensity'])))})
		tmp_dict.update({'std_cycle_intensity_FT_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['intensity'])))})
		tmp_dict.update({'std_cycle_intensity_SB_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['intensity'])))})
		tmp_dict.update({'std_cycle_intensity_SB_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['intensity'])))})


		tmp_dict.update({'mean_cycle_brightness_INJ_b1' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['brightness'])))})
		tmp_dict.update({'mean_cycle_brightness_INJ_b2' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['brightness'])))})
		tmp_dict.update({'mean_cycle_brightness_FB_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['brightness'])))})
		tmp_dict.update({'mean_cycle_brightness_FB_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['brightness'])))})
		tmp_dict.update({'mean_cycle_brightness_FT_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['brightness'])))})
		tmp_dict.update({'mean_cycle_brightness_FT_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['brightness'])))})
		tmp_dict.update({'mean_cycle_brightness_SB_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['brightness'])))})
		tmp_dict.update({'mean_cycle_brightness_SB_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['brightness'])))})
		
		tmp_dict.update({'std_cycle_brightness_INJ_b1' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['brightness'])))})
		tmp_dict.update({'std_cycle_brightness_INJ_b2' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['brightness'])))})
		tmp_dict.update({'std_cycle_brightness_FB_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['brightness'])))})
		tmp_dict.update({'std_cycle_brightness_FB_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['brightness'])))})
		tmp_dict.update({'std_cycle_brightness_FT_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['brightness'])))})
		tmp_dict.update({'std_cycle_brightness_FT_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['brightness'])))})
		tmp_dict.update({'std_cycle_brightness_SB_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['brightness'])))})
		tmp_dict.update({'std_cycle_brightness_SB_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['brightness'])))})


		tmp_dict.update({'mean_cycle_blength_INJ_b1' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['blength'])))})
		tmp_dict.update({'mean_cycle_blength_INJ_b2' : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['blength'])))})
		tmp_dict.update({'mean_cycle_blength_FB_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['blength'])))})
		tmp_dict.update({'mean_cycle_blength_FB_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['blength'])))})
		tmp_dict.update({'mean_cycle_blength_FT_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['blength'])))})
		tmp_dict.update({'mean_cycle_blength_FT_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['blength'])))})
		tmp_dict.update({'mean_cycle_blength_SB_b1'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['blength'])))})
		tmp_dict.update({'mean_cycle_blength_SB_b2'  : np.mean(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['blength'])))})
		
		tmp_dict.update({'std_cycle_blength_INJ_b1' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_start']['blength'])))})
		tmp_dict.update({'std_cycle_blength_INJ_b2' : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_start']['blength'])))})
		tmp_dict.update({'std_cycle_blength_FB_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['Injection']['at_end']['blength'])))})
		tmp_dict.update({'std_cycle_blength_FB_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['Injection']['at_end']['blength'])))})
		tmp_dict.update({'std_cycle_blength_FT_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['blength'])))})
		tmp_dict.update({'std_cycle_blength_FT_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['blength'])))})
		tmp_dict.update({'std_cycle_blength_SB_b1'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['blength'])))})
		tmp_dict.update({'std_cycle_blength_SB_b2'  : np.std(ma.masked_invalid(np.array(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['blength'])))})

		emit  = []
		cycle = []
		beam  = []
		plane = []
		slot  = []
		#startwithb1h
		emit.append(cycle_emit_B1H_FB-cycle_emit_B1H_INJ)
		cycle.append(['INJ2FB']*len(cycle_emit_B1H_FB-cycle_emit_B1H_INJ))
		beam.append(['B1']*len(cycle_emit_B1H_FB-cycle_emit_B1H_INJ))
		plane.append(['Horizontal']*len(cycle_emit_B1H_FB-cycle_emit_B1H_INJ))


		emit.append(cycle_emit_B1H_FT-cycle_emit_B1H_FB)
		cycle.append(['FB2FT']*len(cycle_emit_B1H_FT-cycle_emit_B1H_FB))
		beam.append(['B1']*len(cycle_emit_B1H_FT-cycle_emit_B1H_FB))
		plane.append(['Horizontal']*len(cycle_emit_B1H_FT-cycle_emit_B1H_FB))


		emit.append(cycle_emit_B1H_SB-cycle_emit_B1H_FT)
		cycle.append(['FT2SB']*len(cycle_emit_B1H_SB-cycle_emit_B1H_FT))
		beam.append(['B1']*len(cycle_emit_B1H_SB-cycle_emit_B1H_FT))
		plane.append(['Horizontal']*len(cycle_emit_B1H_SB-cycle_emit_B1H_FT))


		emit.append(cycle_emit_B1H_SB-cycle_emit_B1H_INJ)
		cycle.append(['INJ2SB']*len(cycle_emit_B1H_SB-cycle_emit_B1H_INJ))
		beam.append(['B1']*len(cycle_emit_B1H_SB-cycle_emit_B1H_INJ))
		plane.append(['Horizontal']*len(cycle_emit_B1H_SB-cycle_emit_B1H_INJ))


		#b1v
		emit.append(cycle_emit_B1V_FB-cycle_emit_B1V_INJ)
		cycle.append(['INJ2FB']*len(cycle_emit_B1V_FB-cycle_emit_B1V_INJ))
		beam.append(['B1']*len(cycle_emit_B1V_FB-cycle_emit_B1V_INJ))
		plane.append(['Vertical']*len(cycle_emit_B1V_FB-cycle_emit_B1V_INJ))


		emit.append(cycle_emit_B1V_FT-cycle_emit_B1V_FB)
		cycle.append(['FB2FT']*len(cycle_emit_B1V_FT-cycle_emit_B1V_FB))
		beam.append(['B1']*len(cycle_emit_B1V_FT-cycle_emit_B1V_FB))
		plane.append(['Vertical']*len(cycle_emit_B1V_FT-cycle_emit_B1V_FB))


		emit.append(cycle_emit_B1V_SB-cycle_emit_B1V_FT)
		cycle.append(['FT2SB']*len(cycle_emit_B1V_SB-cycle_emit_B1V_FT))
		beam.append(['B1']*len(cycle_emit_B1V_SB-cycle_emit_B1V_FT))
		plane.append(['Vertical']*len(cycle_emit_B1V_SB-cycle_emit_B1V_FT))


		emit.append(cycle_emit_B1V_SB-cycle_emit_B1V_INJ)
		cycle.append(['INJ2SB']*len(cycle_emit_B1V_SB-cycle_emit_B1V_INJ))
		beam.append(['B1']*len(cycle_emit_B1V_SB-cycle_emit_B1V_INJ))
		plane.append(['Vertical']*len(cycle_emit_B1V_SB-cycle_emit_B1V_INJ))


		#b2h
		emit.append(cycle_emit_B2H_FB-cycle_emit_B2H_INJ)
		cycle.append(['INJ2FB']*len(cycle_emit_B2H_FB-cycle_emit_B2H_INJ))
		beam.append(['B2']*len(cycle_emit_B2H_FB-cycle_emit_B2H_INJ))
		plane.append(['Horizontal']*len(cycle_emit_B2H_FB-cycle_emit_B2H_INJ))

		emit.append(cycle_emit_B2H_FT-cycle_emit_B2H_FB)
		cycle.append(['FB2FT']*len(cycle_emit_B2H_FT-cycle_emit_B2H_FB))
		beam.append(['B2']*len(cycle_emit_B2H_FT-cycle_emit_B2H_FB))
		plane.append(['Horizontal']*len(cycle_emit_B2H_FT-cycle_emit_B2H_FB))


		emit.append(cycle_emit_B2H_SB-cycle_emit_B2H_FT)
		cycle.append(['FT2SB']*len(cycle_emit_B2H_SB-cycle_emit_B2H_FT))
		beam.append(['B2']*len(cycle_emit_B2H_SB-cycle_emit_B2H_FT))
		plane.append(['Horizontal']*len(cycle_emit_B2H_SB-cycle_emit_B2H_FT))


		emit.append(cycle_emit_B2H_SB-cycle_emit_B2H_INJ)
		cycle.append(['INJ2SB']*len(cycle_emit_B2H_SB-cycle_emit_B2H_INJ))
		beam.append(['B2']*len(cycle_emit_B2H_SB-cycle_emit_B2H_INJ))
		plane.append(['Horizontal']*len(cycle_emit_B2H_SB-cycle_emit_B2H_INJ))


		#b2v
		emit.append(cycle_emit_B2V_FB-cycle_emit_B2V_INJ)
		cycle.append(['INJ2FB']*len(cycle_emit_B2V_FB-cycle_emit_B2V_INJ))
		beam.append(['B2']*len(cycle_emit_B2V_FB-cycle_emit_B2V_INJ))
		plane.append(['Vertical']*len(cycle_emit_B2V_FB-cycle_emit_B2V_INJ))


		emit.append(cycle_emit_B2V_FT-cycle_emit_B2V_FB)
		cycle.append(['FB2FT']*len(cycle_emit_B2V_FT-cycle_emit_B2V_FB))
		beam.append(['B2']*len(cycle_emit_B2V_FT-cycle_emit_B2V_FB))
		plane.append(['Vertical']*len(cycle_emit_B2V_FT-cycle_emit_B2V_FB))


		emit.append(cycle_emit_B2V_SB-cycle_emit_B2V_FT)
		cycle.append(['FT2SB']*len(cycle_emit_B2V_SB-cycle_emit_B2V_FT))
		beam.append(['B2']*len(cycle_emit_B2V_SB-cycle_emit_B2V_FT))
		plane.append(['Vertical']*len(cycle_emit_B2V_SB-cycle_emit_B2V_FT))


		emit.append(cycle_emit_B2V_SB-cycle_emit_B2V_INJ)
		cycle.append(['INJ2SB']*len(cycle_emit_B2V_SB-cycle_emit_B2V_INJ))
		beam.append(['B2']*len(cycle_emit_B2V_SB-cycle_emit_B2V_INJ))
		plane.append(['Vertical']*len(cycle_emit_B2V_SB-cycle_emit_B2V_INJ))


		emit=list(itertools.chain.from_iterable(emit))
		cycle=list(itertools.chain.from_iterable(cycle))
		beam=list(itertools.chain.from_iterable(beam))
		plane=list(itertools.chain.from_iterable(plane))


		df_emit_tmp=pd.DataFrame()
		df_emit_tmp['Emittance']= pd.Series(emit,dtype='float')
		df_emit_tmp['Cycle']	= pd.Series(cycle,dtype='category')
		df_emit_tmp['Plane']	= pd.Series(plane,dtype='category')
		df_emit_tmp['Beam']		= pd.Series(beam,dtype='category')
		df_emit_tmp['fill']		= [int(self.filln)]*len(df_emit_tmp)
		df_emit_tmp['bunches']	= [int(self.bunches)]*len(df_emit_tmp)
		df_emit_tmp.set_index('fill', drop=False)


		if int(self.filln) in df_emit.index.values:
			warn('# statisticsDataFrame : Updating Statistics DF')
			# df = df.drop(df.index[int(self.filln)])
			df = df.drop([int(self.filln)]) #(df.loc[int(self.filln)])

		df_emit = df_emit.append(df_emit_tmp)

		df_emit.to_pickle(df_emit_name)




		########################################################	
		#	
		# STABLE BEAMS INFO
		#
		########################################################
		
		# ----------------------------------     LOSSES AND LIFETIME    ----------------------------------
		# losses start/end
		init_bbb_losses_b1        				= np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1)[0]
		init_bbb_losses_b2        				= np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1)[0]
		init_bbb_losses_b1_std     				= np.std(np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1))
		init_bbb_losses_b2_std     				= np.std(np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1))

		if len(np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1))>5 and self.filln != 5870:
			t4steps_before_end_bbb_losses_b1 		= np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1)[-4]
			t4steps_before_end_bbb_losses_b2 		= np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1)[-4]

		else:
			t4steps_before_end_bbb_losses_b1 		= np.nan
			t4steps_before_end_bbb_losses_b2 		= np.nan	

		half_length_bbb_losses_b1 				= np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1)[int(len(np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1))/2.0)]
		half_length_bbb_losses_b2 				= np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1)[int(len(np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1))/2.0)]


		# lifetime start/end
		init_bbb_lifetime_b1      			  = np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)[0]
		init_bbb_lifetime_b2      			  = np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1)[0]
		init_bbb_lifetime_b1_std   			  = np.std(np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1))
		init_bbb_lifetime_b2_std   			  = np.std(np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1))
		if len(np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1))>2:
			t4steps_before_end_bbb_lifetime_b1    	= np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)[-4]
			t4steps_before_end_bbb_lifetime_b2    	= np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1)[-4]
		else:
			t4steps_before_end_bbb_lifetime_b1 		= np.nan
			t4steps_before_end_bbb_lifetime_b2 		= np.nan	
	
		half_len_bbb_lifetime_b1    		  	= np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)[int(len(np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1))/2.0)]
		half_len_bbb_lifetime_b2    		  	= np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1)[int(len(np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1))/2.0)]
	

		# lifetime and losses within fill
		if len(np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)) > 25 : #int(self.filln) not in [5862, 5870] :
			t1h_lifetime_b1      			= np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)[12]
			t1h_lifetime_b2      			= np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1)[12]

			t2h_lifetime_b1      			= np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)[24]
			t2h_lifetime_b2      			= np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1)[24]

			t1h_losses_b1      			  	= np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1)[12]
			t1h_losses_b2      			  	= np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1)[12]

			t2h_losses_b1      			  	= np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1)[24]
			t2h_losses_b2      			  	= np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1)[24]
	
		elif len(np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)) > 13 : #int(self.filln) not in [5862, 5870] :
			t1h_lifetime_b1      			= np.mean(self.filln_LifetimeDict[1]['tau_Np_bbb']/3600., axis=1)[12]
			t1h_lifetime_b2      			= np.mean(self.filln_LifetimeDict[2]['tau_Np_bbb']/3600., axis=1)[12]

			t2h_lifetime_b1      			= np.nan
			t2h_lifetime_b2      			= np.nan

			t1h_losses_b1      			  	= np.mean(self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, axis=1)[12]
			t1h_losses_b2      			  	= np.mean(self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, axis=1)[12]

			t2h_losses_b1      			  	= np.nan
			t2h_losses_b2      			  	= np.nan
			
		else:
			t1h_lifetime_b1      			= np.nan
			t1h_lifetime_b2      			= np.nan

			t2h_lifetime_b1      			= np.nan
			t2h_lifetime_b2      			= np.nan

			t1h_losses_b1      			  	= np.nan
			t1h_losses_b2      			  	= np.nan

			t2h_losses_b1      			  	= np.nan
			t2h_losses_b2      			  	= np.nan

		tmp_dict.update({'0h_losses_b1' : init_bbb_losses_b1})
		tmp_dict.update({'0h_losses_b2' : init_bbb_losses_b2})
		tmp_dict.update({'std_losses_b1' : init_bbb_losses_b1_std})
		tmp_dict.update({'std_losses_b2' : init_bbb_losses_b2_std})


		tmp_dict.update({'t4steps_before_end_losses_b1' : t4steps_before_end_bbb_losses_b1})
		tmp_dict.update({'t4steps_before_end_losses_b2' : t4steps_before_end_bbb_losses_b2})
		tmp_dict.update({'half_length_losses_b1' : half_length_bbb_losses_b1})
		tmp_dict.update({'half_length_losses_b2' : half_length_bbb_losses_b2})
		tmp_dict.update({'1h_losses_b1' : t1h_losses_b1})
		tmp_dict.update({'1h_losses_b2' : t1h_losses_b2})
		tmp_dict.update({'2h_losses_b1' : t2h_losses_b1})
		tmp_dict.update({'2h_losses_b2' : t2h_losses_b2})
		

		tmp_dict.update({'0h_intensity_lifetime_b1' : init_bbb_lifetime_b1})
		tmp_dict.update({'0h_intensity_lifetime_b2' : init_bbb_lifetime_b2})
		tmp_dict.update({'std_intensity_lifetime_b1' : init_bbb_lifetime_b1_std})
		tmp_dict.update({'std_intensity_lifetime_b2' : init_bbb_lifetime_b2_std})
		tmp_dict.update({'t4steps_before_end_intensity_lifetime_b1' : t4steps_before_end_bbb_lifetime_b1})
		tmp_dict.update({'t4steps_before_end_intensity_lifetime_b2' : t4steps_before_end_bbb_lifetime_b2})
		tmp_dict.update({'half_length_intensity_lifetime_b1' : half_len_bbb_lifetime_b1})
		tmp_dict.update({'half_length_intensity_lifetime_b2' : half_len_bbb_lifetime_b2})
		tmp_dict.update({'1h_intensity_lifetime_b1' : t1h_lifetime_b1})
		tmp_dict.update({'1h_intensity_lifetime_b2' : t1h_lifetime_b2})
		tmp_dict.update({'2h_intensity_lifetime_b1' : t2h_lifetime_b1})
		tmp_dict.update({'2h_intensity_lifetime_b2' : t2h_lifetime_b2})



		
		# ----------------------------------     LUMINOSITY     ----------------------------------
			

		# integrated luminosity at the end of the fill in inverse femptobarn
		tmp_dict.update({'ATLAS_int_lumi_invfb_tend' : (cumtrapz(y=np.sum(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1), x=(time_range-t_start_STABLE))*1.0e-43)[-1]})
		tmp_dict.update({'CMS_int_lumi_invfb_tend'   : (cumtrapz(y=np.sum(self.filln_LumiMeasDict['CMS']['bunch_lumi'], axis=1), x=(time_range-t_start_STABLE))*1.0e-43)[-1]})

		# peak luminosity at the start of SB
		tmp_dict.update({'ATLAS_peak_lumi_t0h' : np.sum(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1)[0]})
		tmp_dict.update({'CMS_peak_lumi_t0h'   : np.sum(self.filln_LumiMeasDict['CMS']['bunch_lumi'], axis=1)[0]})
		
		if len(np.mean(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1)) > 6:
			tmp_dict.update({'lumi_ATLAS_measured_meanBunch_4tsteps'   : np.mean(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1)[4]})
			tmp_dict.update({'lumi_CMS_measured_meanBunch_4tsteps'     : np.mean(self.filln_LumiMeasDict['CMS']['bunch_lumi'], axis=1)[4]})
			tmp_dict.update({'lumi_ATLAS_calculated_meanBunch_4tsteps' : np.mean(self.filln_LumiCalcDict['ATLAS']['bunch_lumi'], axis=1)[4]})
			tmp_dict.update({'lumi_CMS_calculated_meanBunch_4tsteps'   : np.mean(self.filln_LumiCalcDict['CMS']['bunch_lumi'], axis=1)[4]})
			# tmp_dict.update({'std_lumi_ATLAS_measured_meanBunch_4tsteps'   : np.std(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1)})
			# tmp_dict.update({'std_lumi_CMS_measured_meanBunch_4tsteps'     : np.std(self.filln_LumiMeasDict['CMS']['bunch_lumi'], axis=1)})
			# tmp_dict.update({'std_lumi_ATLAS_calculated_meanBunch_4tsteps' : np.std(self.filln_LumiCalcDict['ATLAS']['bunch_lumi'], axis=1)})
			# tmp_dict.update({'std_lumi_CMS_calculated_meanBunch_4tsteps'   : np.std(self.filln_LumiCalcDict['CMS']['bunch_lumi'], axis=1)})


		else:
			tmp_dict.update({'lumi_ATLAS_measured_meanBunch_4tsteps'   : np.nan })
			tmp_dict.update({'lumi_CMS_measured_meanBunch_4tsteps'     : np.nan })
			tmp_dict.update({'lumi_ATLAS_calculated_meanBunch_4tsteps' : np.nan })
			tmp_dict.update({'lumi_CMS_calculated_meanBunch_4tsteps'   : np.nan })


		# luminosity lifetime and optimal fill length
		mean_tau_atlas_measured, mean_tau_cms_measured, mean_tau_atlas_calculated, mean_tau_cms_calculated, turn_around_time_h, optimal_fill_length_measured, optimal_fill_length_calculated, optimal_fill_length_ATLAS_measured_3dot5h, optimal_fill_length_CMS_measured_3dot5h = self.plotStableBeamsLuminosityLifetime(save=False, batch=True, return_result=True, turn_around_time_h=3.5)

		_, _, _, _, _, _, _, optimal_fill_length_ATLAS_measured_7dot0h, optimal_fill_length_CMS_measured_7dot0h = self.plotStableBeamsLuminosityLifetime(save=False, batch=True, return_result=True, turn_around_time_h=7.0)

		tmp_dict.update({'fullfit_mean_luminosity_lifetime_ATLAS_measured' 		: mean_tau_atlas_measured})
		tmp_dict.update({'fullfit_mean_luminosity_lifetime_CMS_measured' 		: mean_tau_cms_measured})
		tmp_dict.update({'fullfit_mean_luminosity_lifetime_ATLAS_calculated' 	: mean_tau_atlas_calculated})
		tmp_dict.update({'fullfit_mean_luminosity_lifetime_CMS_calculated' 		: mean_tau_cms_calculated})
		tmp_dict.update({'fullfit_optimal_fill_length_measured' 				: optimal_fill_length_measured})
		tmp_dict.update({'fullfit_optimal_fill_length_calculated' 			    : optimal_fill_length_calculated})
		tmp_dict.update({'fullfit_optimal_fill_length_ATLAS_measured_3dot5turn' : optimal_fill_length_ATLAS_measured_3dot5h})
		tmp_dict.update({'fullfit_optimal_fill_length_CMS_measured_3dot5turn'   : optimal_fill_length_CMS_measured_3dot5h })
		tmp_dict.update({'fullfit_optimal_fill_length_ATLAS_measured_7dot0turn' : optimal_fill_length_ATLAS_measured_7dot0h})
		tmp_dict.update({'fullfit_optimal_fill_length_CMS_measured_7dot0turn'   : optimal_fill_length_CMS_measured_7dot0h })


		# ----------------------------------     EMITTANCE     ----------------------------------

		# average emittances at start of SB
		tmp_dict.update({'emit_B1H_StartSB_noncoll': np.nanmean(self.filln_StableBeamsDict['eh_interp_noncoll'][1], axis=1)[0]})
		tmp_dict.update({'emit_B1V_StartSB_noncoll': np.nanmean(self.filln_StableBeamsDict['ev_interp_noncoll'][1], axis=1)[0]})
		tmp_dict.update({'emit_B2H_StartSB_noncoll': np.nanmean(self.filln_StableBeamsDict['eh_interp_noncoll'][2], axis=1)[0]})
		tmp_dict.update({'emit_B2V_StartSB_noncoll': np.nanmean(self.filln_StableBeamsDict['ev_interp_noncoll'][2], axis=1)[0]})

		tmp_dict.update({'emit_B1H_StartSB_coll'	: np.nanmean(self.filln_StableBeamsDict['eh_interp_coll'][1], axis=1)[0]})
		tmp_dict.update({'emit_B1V_StartSB_coll'	: np.nanmean(self.filln_StableBeamsDict['ev_interp_coll'][1], axis=1)[0]})
		tmp_dict.update({'emit_B2H_StartSB_coll'	: np.nanmean(self.filln_StableBeamsDict['eh_interp_coll'][2], axis=1)[0]})
		tmp_dict.update({'emit_B2V_StartSB_coll'	: np.nanmean(self.filln_StableBeamsDict['ev_interp_coll'][2], axis=1)[0]})


		tmp_dict.update({'std_emit_B1H_StartSB_noncoll': np.nanstd(self.filln_StableBeamsDict['eh_interp_noncoll'][1], axis=1)[0]})
		tmp_dict.update({'std_emit_B1V_StartSB_noncoll': np.nanstd(self.filln_StableBeamsDict['ev_interp_noncoll'][1], axis=1)[0]})
		tmp_dict.update({'std_emit_B2H_StartSB_noncoll': np.nanstd(self.filln_StableBeamsDict['eh_interp_noncoll'][2], axis=1)[0]})
		tmp_dict.update({'std_emit_B2V_StartSB_noncoll': np.nanstd(self.filln_StableBeamsDict['ev_interp_noncoll'][2], axis=1)[0]})

		tmp_dict.update({'std_emit_B1H_StartSB_coll': np.nanstd(self.filln_StableBeamsDict['eh_interp_coll'][1], axis=1)[0]})
		tmp_dict.update({'std_emit_B1V_StartSB_coll': np.nanstd(self.filln_StableBeamsDict['ev_interp_coll'][1], axis=1)[0]})
		tmp_dict.update({'std_emit_B2H_StartSB_coll': np.nanstd(self.filln_StableBeamsDict['eh_interp_coll'][2], axis=1)[0]})
		tmp_dict.update({'std_emit_B2V_StartSB_coll': np.nanstd(self.filln_StableBeamsDict['ev_interp_coll'][2], axis=1)[0]})




		# calculate emittance growth rates at sb as END-START/TIME SB
		Delta_t = (time_range[-1]- time_range[0])/3600.
		Delta_EH1_coll = (np.mean(self.filln_StableBeamsDict['eh_interp_coll'][1], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['eh_interp_coll'][1], axis=1)[0])/Delta_t
		Delta_EV1_coll = (np.mean(self.filln_StableBeamsDict['ev_interp_coll'][1], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['ev_interp_coll'][1], axis=1)[0])/Delta_t

		Delta_EH2_coll = (np.mean(self.filln_StableBeamsDict['eh_interp_coll'][2], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['eh_interp_coll'][2], axis=1)[0])/Delta_t
		Delta_EV2_coll = (np.mean(self.filln_StableBeamsDict['ev_interp_coll'][2], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['ev_interp_coll'][2], axis=1)[0])/Delta_t
		#--
		Delta_EH1_noncoll = (np.mean(self.filln_StableBeamsDict['eh_interp_noncoll'][1], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['eh_interp_noncoll'][1], axis=1)[0])/Delta_t
		Delta_EV1_noncoll = (np.mean(self.filln_StableBeamsDict['ev_interp_noncoll'][1], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['ev_interp_noncoll'][1], axis=1)[0])/Delta_t

		Delta_EH2_noncoll = (np.mean(self.filln_StableBeamsDict['eh_interp_noncoll'][2], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['eh_interp_noncoll'][2], axis=1)[0])/Delta_t
		Delta_EV2_noncoll = (np.mean(self.filln_StableBeamsDict['ev_interp_noncoll'][2], axis=1)[-1] - np.mean(self.filln_StableBeamsDict['ev_interp_noncoll'][2], axis=1)[0])/Delta_t

		tmp_dict.update({'emit_B1H_SB_Growth_coll' : Delta_EH1_coll})
		tmp_dict.update({'emit_B1V_SB_Growth_coll' : Delta_EV1_coll})
		tmp_dict.update({'emit_B2H_SB_Growth_coll' : Delta_EH2_coll})
		tmp_dict.update({'emit_B2V_SB_Growth_coll' : Delta_EV2_coll})

		tmp_dict.update({'emit_B1H_SB_Growth_noncoll' : Delta_EH1_noncoll})
		tmp_dict.update({'emit_B1V_SB_Growth_noncoll' : Delta_EV1_noncoll})
		tmp_dict.update({'emit_B2H_SB_Growth_noncoll' : Delta_EH2_noncoll})
		tmp_dict.update({'emit_B2V_SB_Growth_noncoll' : Delta_EV2_noncoll})

		# exponent fit in emit evolution -> calculate tau -> get inverse
		tmp_dict.update({'fullfit_inv_tau_emit_B1H_SB_coll'	: 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_1']['tau_emith_coll_full']))/3600. })
		tmp_dict.update({'fullfit_inv_tau_emit_B1V_SB_coll'	: 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_1']['tau_emitv_coll_full']))/3600. })
		tmp_dict.update({'fullfit_inv_tau_emit_B2H_SB_coll'	: 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_2']['tau_emith_coll_full']))/3600. })
		tmp_dict.update({'fullfit_inv_tau_emit_B2V_SB_coll'	: 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_2']['tau_emitv_coll_full']))/3600. })

		tmp_dict.update({'fullfit_inv_tau_emit_B1H_SB_noncoll' : 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_1']['tau_emith_noncoll_full']))/3600. })
		tmp_dict.update({'fullfit_inv_tau_emit_B1V_SB_noncoll' : 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_1']['tau_emitv_noncoll_full']))/3600. })
		tmp_dict.update({'fullfit_inv_tau_emit_B2H_SB_noncoll' : 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_2']['tau_emith_noncoll_full']))/3600. })
		tmp_dict.update({'fullfit_inv_tau_emit_B2V_SB_noncoll' : 1.0/np.mean(ma.masked_invalid(self.filln_SBFitsDict['beam_2']['tau_emitv_noncoll_full']))/3600. })


		# ----------------------------------     SLOTS      ----------------------------------

		# tmp_dict.update({'colliding_slots_SB_b1'    : len(self.filln_StableBeamsDict['slots_filled_coll'][1])})
		# tmp_dict.update({'colliding_slots_SB_b2'    : len(self.filln_StableBeamsDict['slots_filled_coll'][2])})
		# tmp_dict.update({'noncolliding_slots_SB_b1' : len(self.filln_StableBeamsDict['slots_filled_noncoll'][1])})
		# tmp_dict.update({'noncolliding_slots_SB_b2' : len(self.filln_StableBeamsDict['slots_filled_noncoll'][2])})


		#------- testing



		for key in tmp_dict.keys():
			print key, type(tmp_dict[key])	
			if 	type(tmp_dict[key])	== np.ndarray:
				print tmp_dict[key].shape

		
		temp_df = pd.DataFrame(tmp_dict, index=[self.filln])

		# temp_df['test'] = np.nan; temp_df['test'] = temp_df['test'].astype(object); 


		# temp_df.loc[self.filln, 'test'] = self.filln_StableBeamsDict['eh_interp_noncoll'][1]

		
		if int(self.filln) in df.index.values:
			warn('# statisticsDataFrame : Updating Statistics DF')
			# df = df.drop(df.index[int(self.filln)])
			df = df.drop([int(self.filln)]) #(df.loc[int(self.filln)])

		df = df.append(temp_df)

		df.to_pickle(df_name)
		# with pd.HDFStore(df_name) as store:
		# 	store['df'] = df
			# store the df of cycle emit
			# store the df of lumi!






if __name__ == '__main__':
	fill = None
	if len(sys.argv)> 1:
		fill = sys.argv[1]
	# pl.close('all')
	plotter = Plotter(int(fill))

