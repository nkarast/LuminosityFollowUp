################################################################################
##
##   LumiFollowUp: A module that implements the creation of
##             Cycle/Stable Beams/Calculated Lumi and Massi Files
##             for the needs of BE/ABP team for LHC Luminosity Follow-up
##
##   DISCLAIMER : This work is *heavily* based on the work of F. Antoniou and
##                G. Iadarola
##
##   Who to blame : Nikos Karastathis ( nkarast <at> cern <dot> ch )
##   Version : 0.3 (28/04/2017)
##
################################################################################


import sys, os
## For LHC Measurement Tools
BIN = os.path.expanduser("/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/")
sys.path.append(BIN)
## For BSRT rescale
BIN = os.path.expanduser("/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/LumiModel_FollowUp/")
sys.path.append(BIN)

## Where the code is:
BIN = os.path.expanduser("/afs/cern.ch/user/l/lumimod/lumimod/a2017_luminosity_followup/")
sys.path.append(BIN)

# lumi model
BIN = os.path.expanduser("/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/PyLHCLumiModel/")
sys.path.append(BIN)

import matplotlib
matplotlib.use('Agg')  #### needed for batch jobs
print("# LumiFollowUp - Setting Matplotlib Backend to Agg.")
import matplotlib.pyplot as pl
from matplotlib.ticker import MultipleLocator

import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_FBCT as FBCT
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.LHC_BSRT as BSRT
import LHCMeasurementTools.LHC_BQM as BQM
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.LHC_Lumi as LUMI
import LHCMeasurementTools.LHC_Lumi_bbb as LUMI_bbb
# import BSRT_calib_rescale as BSRT_calib
import LHCMeasurementTools.BSRT_calib_rescale as BSRT_calib
from IBSmodel_GY import IBSmodel
from LumiModel_convoluted_values_addelastic_flexible_bl_IBScorr import LumiModel

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
import argparse
import socket
import Utilities.readYamlDB as db
import config


class LumiFollowUp(object):
	def __init__(self, debug=True, batch=True,  FORMAT='%(asctime)s %(levelname)s : %(message)s',
				loglevel = 20, logfile = None, fills_bmodes_file = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/fills_and_bmodes.pkl', ## @TODO bmodes df to be saved?
				min_time_SB = 30*60, first_fill = 5005, last_fill = 5456+1, t_fit_length = 2.*3600.,
				t_step_sec = 10*60, intensity_threshold = 3.0e10, enable_smoothing_BSRT = True,
				avg_time_smoothing = 3.0*3600.0, periods = {'A': (5005,  5256), 'B': (5256,     5405), 'C': (5405,     5456+1)},
				doRescale = True, resc_period = [('A', 'C'), ('B','C'), ('C', 'C')], add_resc_string = '',
				BASIC_DATA_FILE = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/fill_basic_data_csvs/basic_data_fill_<FILLNUMBER>.csv',
				BBB_DATA_FILE = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_<FILLNUMBER>.csv',
				BBB_LUMI_DATA_FILE = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/fill_bunchbybunch_data_csvs/bunchbybunch_lumi_data_fill_<FILLNUMBER>.csv',
				makedirs = True, overwriteFiles = False, SB_dir = 'SB_analysis/',
				fill_dir = "fill_<FILLNUMBER>/", plot_dir = "plots/",
				SB_filename = "fill_<FILLNUMBER><RESC>.pkl.gz",
				SB_fits_filename ="fill_<FILLNUMBER>_fits<RESC>.pkl.gz",
				SB_model_filename ="fill_<FILLNUMBER>_sbmodel<RESC>.pkl.gz", SB_models = ['EmpiricalBlowupLosses', 'EmpiricalBlowupBOff', 'IBSBOff', 'IBSLosses'],
				SB_burnoff_filename="fill_<FILLNUMBER>_sigma_burnoff<RESC>.pkl.gz",
				Cycle_filename = "fill_<FILLNUMBER>_cycle<RESC>.pkl.gz",
				Cycle_model_filename = "fill_<FILLNUMBER>_cycle_model<RESC>.pkl.gz",
				Lumi_filename = "fill_<FILLNUMBER>_lumi_calc<RESC>.pkl.gz",
				Massi_filename = 'fill_<FILLNUMBER>_lumi_meas.pkl.gz',
				saveDict = True, savePandas = False, cases=[1], correction_factor_1h=[1.], correction_factor_1v=[1.],
				correction_factor_2h=[1.], correction_factor_2v=[1.],
				force=False, frev = 11245.5, gammaFT = 6927.64, gammaFB=479.6,  tauSRxy_FT=64.7*3600., tauSRxy_FB=np.inf, tauSRl_FT=32.35*3600,
				sigmaBOff_m2=80.0*1.0e-31, sigma_el_m2=29.7*1.0e-31,
				VRF_FT=10.0e06, VRF_FB=6.0e06, betastar_m = 0.40,
				crossingAngleChange = True, XingAngle = {(5005,  5330):2*185e-6, (5330, 5456+1): 2*140e-6},
				savePlots = True, fig_tuple = (17, 10), plotFormat = ".pdf",
				plotDpi = 300, myfontsize = 16, n_skip = 1,  ## xrange step for time
				makePlotTarball = False, doOnly = False, fill=None,
				doCyclePlots=True, doCycleModelPlots=True, doSBPlots=True, doSBModelPlots=True, doSummaryPlots = False, doPlots = False, submit=False,
				fill_yaml_database = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/LumiModel_FollowUp/autoScriptTesting/fill_db.yaml', fill_year=2016,
				massi_afs_path='/afs/cern.ch/user/l/lpc/w0/<YEAR>/measurements/', massi_exp_folders=['ATLAS/', 'CMS/lumi/']):
		'''
		Constructor for the class LumiFollowUp. It contains a long list of initialization
		variables all set as default to something:
		The input options are:
			--- init options ---
			debug   = True      ## run in debug mode
			batch   = False     ## run in batch mode (do not show plots)
			FORMAT  =\'%(asctime)s %(levelname)s : %(message)s\'  ## format for the logger strings
			loglevel = 20   ## logger level (10=debug, 20=info, 30=warn, 40=error, 50=critical)
			logfile = None  ## logfile to store output

			--- input data files ---
			fills_bmodes_file = '../fills_and_bmodes.pkl'   ## beam modes
			BASIC_DATA_FILE = '../fill_basic_data_csvs/basic_data_fill_<FILLNUMBER>.csv'  ## beam data
			BBB_DATA_FILE = '../fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_<FILLNUMBER>.csv' ## bunch by bunch data

			--- fill & Stable beams info
			fill = None                  ## if it is not None, replace filln_list with fill only
			min_time_SB = 30*60          ## minimum time at SB each fill should have in seconds
			first_fill = 5005            ## the number of the first fill
			last_fill = 5456             ## the number of the last fill
			t_step_sec = 10*60           ## time step in seconds
			intensity_threshold = 3.0e10 ## intensity threshold in ppb
			enable_smoothing_BSRT = True ## should we enable BSRT smoothing
			avg_time_smoothing = 3.0*3600.0 ## avg time smoothing in seconds (here 3h)

			--- BSRT Rescaling info ---
			doRescale = True    ## should I do rescale
			resc_period = [('A', 'C'), ('B','C'), ('C', 'C')]  ## To which period should I rescale
			add_resc_string = ''    ## additional string to be appended in files

			--- Folder and filenames ---
			makedirs = True             ## if the dir does not exist should I make it?
			overwriteFiles = False      ## should I overwrite files if they exist?
			SB_dir = 'SB_analysis/'     ## SB directory
			fill_dir = "fill_<FILLNUMBER>/"                     ## name of the fill dir under SB dir
			plot_dir = "plots/"                                 ## name of the plot dir under fill dir
			SB_filename = "fill_<FILLNUMBER><RESC>.pkl.gz"      ## name of the SB data file
			Cycle_filename = "fill_<FILLNUMBER>_cycle<RESC>.pkl.gz"     ## name of the cycle data file
			Lumi_filename = "fill_<FILLNUMBER>_lumi_calc<RESC>.pkl.gz"  ## name of the calc lumi file
			Massi_filename = 'fill_<FILLNUMBER>_lumi_meas.pkl.gz'       ## name of the massi (measured lumi) file

			--- Fill List Loop ---
			doSubmit = None     ## submits the loop in HTCondor
			submitQueue = None  ## defines which HTCondor queue to use
			loopStep   = None   ## if used the after the filln list is defined, the loop
								##       is broken into jobs of <loopStep> fills

			--- Machine Changes and parameters ---
			frev = 11245.5      ## revolution frequency in Hz
			gamma = 6927.64     ## relativistic gamma for 6.5TeV
			betastar_m = 0.40   ## beta star in metres
			crossingAngleChange = True ## was the crossing angle changed during the year
			XingAngle = {(-np.inf,  5330):2*185e-6, (5330, np.inf): 2*140e-6}   ## dict with periods of fill numbers as keys
																				## and values of full xing angle as items


			--- Output & Plot info
			saveDict = True     ## Save the dictionary per fill in pickles?
			savePandas = True     ## Save the pandas per fill in pickles?
			force = False       ## If True forces to run all steps of analysis and overwrite files
								## Other options to force specific steps : ['cycle', 'sb', 'lumi', 'massi']
								## lumimod forces at least 'massi' if the fill db does not exist
			doOnly = None       ## ['cycle', 'sb', 'lumi', 'massi'] forces only one of these to run
			savePlots = True    ## Should I save the plots (if I make them)?
			fig_tuple = (17, 10)  ## figsize tuple
			plotFormat = ".pdf"     ## figure save format
			plotDpi = 300           ## figure save dpi
			myfontsize = 16         ## my fontsize
			n_skip = 1              ## xrange step for time
			makePlotTarball = False ## should I make a tarball of all plots per fill?
		'''

		## --- initialization options
		self.debug      = debug
		self.batch      = batch
		self.FORMAT     = FORMAT
		self.loglevel   = loglevel
		if self.debug:
			self.loglevel = 10
		self.logfile    = logfile

		## --- input data files
		self.fills_bmodes_file       = fills_bmodes_file
		self.BASIC_DATA_FILE         = BASIC_DATA_FILE
		self.BBB_DATA_FILE           = BBB_DATA_FILE
		self.BBB_LUMI_DATA_FILE	 	 = BBB_LUMI_DATA_FILE

		##  --- fill & Stable beams info
		self.fill                    = fill
		self.min_time_SB             = min_time_SB
		self.first_fill              = first_fill
		self.last_fill               = last_fill
		self.t_step_sec              = t_step_sec
		self.intensity_threshold     = intensity_threshold
		self.enable_smoothing_BSRT   = enable_smoothing_BSRT
		## if self.enable_smoothing_BSRT:
		##     from statsmodels.nonparametric.smoothers_lowess import lowess
		self.avg_time_smoothing      = avg_time_smoothing
		self.periods                 = periods
		self.cases                   = cases
		self.correction_factor_1h    = correction_factor_1h
		self.correction_factor_1v    = correction_factor_1v
		self.correction_factor_2h    = correction_factor_2h
		self.correction_factor_2v    = correction_factor_2v
		self.t_fit_length            = t_fit_length


		## --- BSRT Rescaling info
		self.doRescale               = doRescale
		self.resc_period             = resc_period
		if doRescale:
			self.resc_string         = '_rescaled_{}<TO>'.format(add_resc_string)
		else:
			self.resc_string         = ''

		## --- Folder and filenames
		self.makedirs                = makedirs
		self.overwriteFiles          = overwriteFiles
		self.SB_dir                  = SB_dir
		self.fill_dir                = SB_dir+fill_dir
		self.plot_dir                = self.fill_dir+plot_dir
		self.SB_filename             = SB_filename
		self.SB_fits_filename        = SB_fits_filename
		self.SB_model_filename       = SB_model_filename
		self.SB_models               = SB_models
		self.SB_burnoff_filename     = SB_burnoff_filename
		self.Cycle_filename          = Cycle_filename
		self.Cycle_model_filename    = Cycle_model_filename
		self.Lumi_filename           = Lumi_filename
		self.Massi_filename          = Massi_filename

		## --- Machine Changes and parameters
		self.frev                    = frev
		self.gammaFT                 = gammaFT
		self.gammaFB                 = gammaFB
		self.tauSRxy_FT              = tauSRxy_FT
		self.tauSRl_FT               = tauSRl_FT
		self.tauSRxy_FB              = tauSRxy_FB
		self.sigmaBOff_m2            = sigmaBOff_m2
		self.sigma_el_m2             = sigma_el_m2
		self.VRF_FT                  = VRF_FT
		self.VRF_FB                  = VRF_FB
		self.betastar_m              = betastar_m
		self.crossingAngleChange     = crossingAngleChange
		self.XingAngle               = XingAngle

		## --- Output & Plot info
		self.saveDict                = saveDict
		self.savePandas              = savePandas
		self.doOnly                  = doOnly
		self.force                   = force
		if self.doOnly != False:
			self.force = self.doOnly
		if self.doOnly == True:
			raise ValueError("# LumiFollowUp : Unrecognised option for doOnly argument [doOnly={}]".format(doOnly))
		if self.force == True:
			## force everything
			self.forceCycle          = True
			self.forceCycle_model    = True
			self.forceSB             = True
			self.forceMeasLumi       = True
			self.forceCalcLumi       = True
			self.forceSBModel        = True
			self.forceSBFits         = True
			self.forceLifetime       = True
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format('all'))
			self.overwriteFiles      = True
		elif self.force == 'cycle':
			self.forceCycle          = True
			self.forceCycle_model    = True
			self.forceSB             = False
			self.forceMeasLumi       = False
			self.forceCalcLumi       = False
			self.forceSBModel        = False
			self.forceSBFits         = False
			self.forceLifetime       = False
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
			self.overwriteFiles      = True
		elif self.force == 'sb':
			self.forceCycle          = False
			self.forceCycle_model    = False
			self.forceSB             = True
			self.forceMeasLumi       = False
			self.forceCalcLumi       = False
			self.forceSBModel        = False
			self.forceSBFits         = False
			self.forceLifetime       = False
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
			self.overwriteFiles      = True
		elif self.force == 'massi':
			self.forceCycle          = False
			self.forceCycle_model    = False
			self.forceSB             = False
			self.forceMeasLumi       = True
			self.forceCalcLumi       = False
			self.forceSBModel        = False
			self.forceSBFits         = False
			self.forceLifetime       = False
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
			self.overwriteFiles      = True
		elif self.force == 'lumi':
			self.forceCycle          = False
			self.forceCycle_model    = False
			self.forceCycle_model    = False
			self.forceSB             = False
			self.forceMeasLumi       = False
			self.forceCalcLumi       = True
			self.forceSBModel        = False
			self.forceSBFits         = False
			self.forceLifetime       = False
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
			self.overwriteFiles      = True
		elif self.force == 'model':
			self.forceCycle          = False
			self.forceCycle_model    = False
			self.forceCycle_model    = False
			self.forceSB             = False
			self.forceMeasLumi       = False
			self.forceCalcLumi       = False
			self.forceSBModel        = True
			self.forceSBFits         = False
			self.forceLifetime       = False
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
			self.overwriteFiles      = True
		elif self.force == 'lifetime':
			self.forceCycle          = False
			self.forceCycle_model    = False
			self.forceCycle_model    = False
			self.forceSB             = False
			self.forceMeasLumi       = False
			self.forceCalcLumi       = False
			self.forceSBModel        = False
			self.forceSBFits         = False
			self.forceLifetime       = True
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
			self.overwriteFiles      = True
		elif self.force == 'fits':
			self.forceCycle          = False
			self.forceCycle_model    = False
			self.forceCycle_model    = False
			self.forceSB             = False
			self.forceMeasLumi       = False
			self.forceCalcLumi       = False
			self.forceSBModel        = False
			self.forceSBFits         = True
			self.forceLifetime       = False
			warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
			self.overwriteFiles      = True
		else:
			self.forceCycle          = False
			self.forceCycle_model    = False
			self.forceSB             = False
			self.forceMeasLumi       = False
			self.forceCalcLumi       = False
			self.forceSBModel        = False
			self.forceSBFits         = False
			self.forceLifetime       = False

		self.savePlots               = savePlots
		self.doCyclePlots            = doCyclePlots
		self.doCycleModelPlots       = doCycleModelPlots
		self.doSBPlots               = doSBPlots
		self.doSBModelPlots          = doSBModelPlots
		self.doSummaryPlots          = doSummaryPlots
		self.doPlots                 = doPlots
		# if self.doPlots == True: ## flag to set all plotting to true
		#   self.doCyclePlots        = True
		#   self.doCycleModelPlots   = True
		#   self.doSBPlots           = True
		#   self.doSummaryPlots      = True
		#   self.doSBModelPlots      = True
		# elif self.doPlots == False:
		#   self.doCyclePlots        = False
		#   self.doCycleModelPlots   = False
		#   self.doSBPlots           = False
		#   self.doSummaryPlots      = False
		#   self.doSBModelPlots      = False


		self.fig_tuple               = fig_tuple
		self.plotFormat              = plotFormat
		self.plotDpi                 = plotDpi
		self.myfontsize              = myfontsize
		self.n_skip                  = n_skip
		self.makePlotTarball         = makePlotTarball

		## --- initialize logger
		self.init_logger(self.FORMAT, self.logfile, self.loglevel)

		## --- get general info from BMODES file into the bmodes dataframe
		self.bmodes, self.filln_list = self.getBmodesDF()
		if self.fill is not None:
			self.filln_list = self.fill ##np.flatten(np.array(self.fill)).tolist()

		## This is for matplotlib backend:
		self.submit = submit
		hostname = socket.gethostname()
		fatal("LUMI FOLLOW UP : YOUR HOSTAME IS : {}".format(hostname))
		if not self.submit:
			if 'mac' in hostname:
				newBackend = 'MacOSX'
			elif 'lxplus' in hostname:
				newBackend = 'Qt5Agg'
			else:
				newBackend = 'Agg'

			warn("# LumiFollowUp : Reloading matplotlib and switching backend to {}.".format(newBackend))
			pl.switch_backend(newBackend)

		## Database parameters:
		self.fill_yaml_database         = fill_yaml_database
		self.fill_year                  = fill_year
		self.massi_afs_path             = massi_afs_path
		self.massi_exp_folders          = massi_exp_folders
		self.skipMassi                  = None


		## --- These dictionaries here are placeholders. Per fill they will store
		## --- info to be accessible for all functions.
		self.filln_CycleDict           = {}
		self.filln_CycleModelDict      = {}
		self.filln_CycleModelInj2SBDict= {}
		self.filln_StableBeamsDict     = {}
		self.filln_SBFitsDict          = {}
		self.filln_SBModelDict         = {}
		self.filln_LumiCalcDict        = {}
		self.filln_LumiMeasDict        = {}
		self.filln_LifetimeDict        = {}

		self.summaryLumi               = {}

	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def convertToLocalTime(self, df_timeRange, timezone="Europe/Zurich"):
		'''
		Converts a Series (Pandas Column) in Local Time and returns it.
		Input  : df_timeRange : Pandas Series with timestamp data
				 timezone     : timezone (tz) string - defaults to Europe/Zurich
		Returns: converted pandas Series
		'''
		return pd.to_datetime(np.array(df_timeRange), unit='s', utc=True).tz_convert(timezone).tz_localize(None)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def getFillnList(self):
		'''
		Returns the filln_list of the object
		Input   : None
		Returns : filln_list : list with the fill numbers
		'''
		return self.filln_list
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def setFillnList(self, mlist):
		'''
		Sets the filln_list of the object.
		Input  : mlist : list with the fill numbers
		Returns: None
		'''
		self.filln_list = mlist
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def init_logger(self, FORMAT, logfile, loglevel):
		'''
		Function to initialize the logger from logging module
		Inputs : FORMAT   : Format of the logging output
				 logfile  : Logfile to populate with the log info or None (for STDOUT)
				 loglevel : Log level : 10 = debug, 20 = info, 30 = warn, 40 = error, 50 = fatal
		Returns: None
		'''
		basicConfig(format=FORMAT, filename=logfile, level=loglevel)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
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
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
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
		if self.min_time_SB > 0 :
			bmodes = bmodes[:][(bmodes['t_start_STABLE'] > 0) & (bmodes['t_stop_STABLE']-bmodes['t_start_STABLE']>= self.min_time_SB)]

		bmodes = bmodes.ix[self.first_fill:self.last_fill]

		## Get the fill list
		filln_list = bmodes.index.values

		## Make a column with the "period" the fill belongs to
		bmodes['period'] = np.nan
		for key in self.periods:
			## key = 'A', 'B', 'C'
			bmodes['period'].loc[filln_list[np.logical_and(np.less(filln_list,self.periods[key][1]), np.greater_equal(filln_list,self.periods[key][0]))]]=str(key)

		bmodes['CrossingAngle_ATLAS'] = np.nan
		bmodes['CrossingAngle_CMS'] = np.nan
		for key in self.XingAngle:
			bmodes['CrossingAngle_ATLAS'].loc[filln_list[np.logical_and(np.less(filln_list,key[1]), np.greater_equal(filln_list,key[0]))]]=self.XingAngle[key][0]
			bmodes['CrossingAngle_CMS'].loc[filln_list[np.logical_and(np.less(filln_list,key[1]), np.greater_equal(filln_list,key[0]))]]=self.XingAngle[key][1]

		######################################################     RESCALING     ############################################################
		if self.doRescale:
			import BSRT_calib_rescale as BSRT_calib
			bmodes['rescaledPeriod'] = np.nan
			for res in self.resc_period:
				bmodes['rescaledPeriod'][bmodes['period']==res[0]]=res[1]
		else:
			import BSRT_calib as BSRT_calib
			bmodes['rescaledPeriod'] = bmodes['period'] # @TODO IS THIS NEEDED?


		filln_list = bmodes.index.values

		return bmodes, filln_list
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def getTimberDic(self, filln):
		'''
		Function to get the dictionary from Timber data
		Input :     filln            = fill number
					BASIC_DATA_FILE  = basic data file for timber
					BBB_DATA_FILE    = bunch by bunch data file
					debug            = debug bool to be used as verbose option
		Returns:    timber_dic       = the dictionary from timber data
		'''
		info('Getting Timber Data for fill {}'.format(filln))
		timber_dic = {}
		timber_dic.update(tm.parse_timber_file(self.BASIC_DATA_FILE.replace('<FILLNUMBER>',str(filln)), verbose=self.debug))
		timber_dic.update(tm.parse_timber_file(self.BBB_DATA_FILE.replace('<FILLNUMBER>',str(filln)), verbose=self.debug))
		# timber_dic.update(tm.parse_timber_file(self.BBB_LUMI_DATA_FILE.replace('<FILLNUMBER>',str(filln)), verbose=self.debug))
		return timber_dic
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def getBSRTCalibDic(self, filln):
		'''
		Fucntion to get the dictionary from the BSRT calibration.
		Input:      filln   = fill number
		Returns:    bsrt_calib_dict = dictionary from the BSRT calibration
		'''
		if self.bmodes['period'][filln]!=self.bmodes['rescaledPeriod'][filln]:
			bsrt_calib_dict = BSRT_calib.emittance_dictionary(filln=filln, rescale=True, period = self.bmodes['rescaledPeriod'][filln])
		else:
			bsrt_calib_dict = BSRT_calib.emittance_dictionary(filln=filln, rescale=False, period = self.bmodes['period'][filln])
		return bsrt_calib_dict
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def getCycleDataTimes(self, filln):
		'''
		Returns : for cycle t_start_fill, t_end_fill, t_fill_len, t_ref
		'''
		t_start_fill = self.bmodes['t_startfill'][filln]
		t_end_fill   = self.bmodes['t_endfill'][filln]
		t_fill_len   = t_end_fill - t_start_fill
		t_ref        = t_start_fill

		return t_start_fill, t_end_fill, t_fill_len, t_ref
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def getSBDataTimes(self, filln):
		'''
		Function to return times for the SB data
		Input : filln : fill number
		Returns: t_start_STABLE, t_end_STABLE, time_range, N_steps
		'''
		t_start_STABLE = self.bmodes['t_start_STABLE'][filln]
		t_end_STABLE   = self.bmodes['t_endfill'][filln]
		time_range     = np.arange(t_start_STABLE, t_end_STABLE-15*60, self.t_step_sec)
		N_steps        = len(time_range)
		return t_start_STABLE, t_end_STABLE, time_range, N_steps
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def getFracSmoothing(self, filln, avg_time_smoothing, t_end_STABLE, t_start_STABLE):
		'''
		Function to get frac smoothing
		Inputs : filln : fill number
				 avg_time_smoothing : average time smoothing (class member)
				 t_end_STABLE : time that SB ended
				 t_start_STABLE: time that SB started
		Returns: frac_smoothing
		'''
		## Enable a Locally Weighted Scatterplot Smooting (LOWESS)
		frac_smoothing = avg_time_smoothing/(t_end_STABLE-t_start_STABLE)
		if frac_smoothing>1.:
			frac_smoothing=1.
		return frac_smoothing
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def getEmptySBDataDict(self):
		'''
		Returns empty dictionaries for the two beams for SB data
		'''
		eh_interp_raw  = {1:[], 2:[]}
		ev_interp_raw  = {1:[], 2:[]}
		eh_interp      = {1:[], 2:[]}
		ev_interp      = {1:[], 2:[]}
		b_inten_interp = {1:[], 2:[]}
		bl_interp_m    = {1:[], 2:[]}
		slots_filled   = {1:[], 2:[]}

		## Create empty dictionaries for BCT and fast-BCT
		bct_dict  = {1:[], 2:[]}
		fbct_dict = {1:[], 2:[]}
		return eh_interp_raw, ev_interp_raw, eh_interp, ev_interp, b_inten_interp, bl_interp_m, slots_filled, bct_dict, fbct_dict
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def makeCyclePlots(self, dict_intervals_two_beams, filln, t_ref):
		'''
		Function to plot info for cycle
		Input : dict_intervals_two_beams : dictionary holding cycle data
				filln : fill number
				t_ref : reference time for cycle
		Returns: None
		'''

		def getFilledSlotsArray(dict_intervals_two_beams, beam, cycle, cycleTime,  mask_invalid=True):
			'''
			Returns the array with the filled slots.
			Inputs : beam           : beam string ('beam_1', 'beam_2')
					 cycle          : cycle string ('injection', 'flattop')
					 cycleTime      : cycle step string ('injection_start', 'injection_end', 'flattop_start', 'flattop_end')
					 mask_invalid   : boolean True/False to mask invalid values in the output array
			Returns: filled_slots array
			'''
			return ma.masked_invalid(dict_intervals_two_beams[beam][cycle]['filled_slots'])


		##### BBB Emittances
		info('#makeCyclePlots : Fill {} -> Making Cycle Emittances bbb plot...'.format(filln))
		pl.close('all')
		fig_bbbemit = pl.figure("Emittances", figsize=(14, 7))
		fig_bbbemit.set_facecolor('w')
		ax_b1_h = pl.subplot(4,1,1)
		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1 -  H
		ax_b1_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b1_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b1_h.set_ylabel("B1 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b1_h.minorticks_on()
		ax_b1_h.text(0.5, 0.9, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emith']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emith'])))), horizontalalignment='center', verticalalignment='top', transform=ax_b1_h.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)

		## Shrink current axis by 20%
		box = ax_b1_h.get_position()
		ax_b1_h.set_position([box.x0, box.y0, box.width*0.8, box.height])
		# Put a legend to the right of the current axis
		ax_b1_h.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, numpoints=1)
		ax_b1_h.grid('on', which='both')

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1 -  V

		ax_b1_v = pl.subplot(4,1,2)

		ax_b1_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",     "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b1_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",     "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",  "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",  "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b1_v.set_ylabel("B1 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b1_v.minorticks_on()
		ax_b1_v.text(0.5, 0.9, "Injected: {:.2f}$\pm${:.2f} | Start Ramp: {:.2f}$\pm${:.2f} | End Ramp: {:.2f}$\pm${:.2f} | Start SB: {:.2f}$\pm${:.2f}".format(np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv']))),  np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv']))), np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['emitv']))),
					   np.mean(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv']))),   np.std(ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['emitv'])))), horizontalalignment='center', verticalalignment='top', transform=ax_b1_v.transAxes,
					   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=9)
		## Shrink current axis by 20%
		box = ax_b1_v.get_position()
		ax_b1_v.set_position([box.x0, box.y0, box.width * 0.8, box.height])
		# Put a legend to the right of the current axis
		ax_b1_v.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, numpoints=1)
		ax_b1_v.grid('on', which='both')

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2  - H
		ax_b2_h = pl.subplot(4,1,3)

		ax_b2_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emith'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b2_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emith'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emith'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2_h.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_end",     mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emith'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b2_h.set_ylabel("B2 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b2_h.minorticks_on()
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
		ax_b2_v = pl.subplot(4,1,4)

		ax_b2_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['emitv'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b2_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",     "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['emitv'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['emitv'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2_v.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",  "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['emitv'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b2_v.set_ylabel("B2 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]", fontsize=14, fontweight='bold')
		ax_b2_v.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')
		ax_b2_v.minorticks_on()
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
		subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)

		fig_bbbemit.suptitle(subtitle, fontsize=16, fontweight='bold')
		pl.subplots_adjust(hspace=0.5, left=0.1, right=0.8)#, right=0.02, left=0.01)

		if config.savePlots:
			filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))+"fill_{}_cycle_emittancesbbb".format(filln)+self.plotFormat
			print filename
			pl.savefig(filename, dpi=self.plotDpi)


		#################
		pl.close('all')
		info('#makeCyclePlots : Fill {} -> Making Cycle Intensities bbb plot...'.format(filln))
		fig_bbbintens = pl.figure("Intensities", figsize=(14, 7))
		fig_bbbintens.set_facecolor('w')

		ax_b1 = pl.subplot(2,1,1)

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['intensity'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['intensity']  )),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['intensity'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",      mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['intensity']  )),    '.', color='red',    markersize=8, label='Start SB')
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
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['intensity'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['intensity']  )),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_start",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['intensity'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_end",      mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['intensity']  )),    '.', color='red',    markersize=8, label='Start SB')
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
		subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)

		fig_bbbintens.suptitle(subtitle, fontsize=16, fontweight='bold')

		if self.savePlots:
			filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))+"fill_{}_cycle_intensitiesbbb".format(filln)+self.plotFormat
			pl.savefig(filename, dpi=self.plotDpi)

		##########
		pl.close('all')
		info('#makeCyclePlots : Fill {} -> Making Cycle Brightness bbb plot...'.format(filln))
		fig_bbbbright = pl.figure("Brightness", figsize=(14, 7))
		fig_bbbbright.set_facecolor('w')
		ax_b1 = pl.subplot(2,1,1)
		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['brightness'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['brightness'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['brightness'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['brightness'])),    '.', color='red',    markersize=8, label='Start SB')
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
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['brightness'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['brightness'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['brightness'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['brightness'])),    '.', color='red',    markersize=8, label='Start SB')
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
		subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)

		fig_bbbbright.suptitle(subtitle, fontsize=16, fontweight='bold')
		if self.savePlots:
			filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))+"fill_{}_cycle_brightnessbbb".format(filln)+self.plotFormat
			pl.savefig(filename, dpi=self.plotDpi)



		#################### ------- #
		pl.close('all')
		info('#makeCyclePlots : Fill {} -> Making Cycle Bunch Length bbb plot...'.format(filln))
		fig_bbbblength = pl.figure("blength", figsize=(14, 7))
		fig_bbbblength.set_facecolor('w')
		ax_b1 = pl.subplot(2,1,1)

		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 1
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_start']['blength'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['Injection']['at_end']['blength'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_start']['blength'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b1.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_1", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_1']['he_before_SB']['at_end']['blength'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b1.set_ylabel("B1 Bunch Length [p/b]", fontsize=12, fontweight='bold')
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
		ax_b1.set_ylim(0.5e-09, 1.5e-09)


		# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - -
		# beam 2
		ax_b2 = pl.subplot(2,1,2)
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_start']['blength'])), '.', color='blue',   markersize=8, label='Injected')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "Injection",       "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['Injection']['at_end']['blength'])),   '.', color='orange', markersize=8, label='Start Ramp')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_start",  mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_start']['blength'])),  '.', color='green',  markersize=8, label='End Ramp')
		ax_b2.plot(getFilledSlotsArray(dict_intervals_two_beams, "beam_2", "he_before_SB",    "at_end",    mask_invalid=True), ma.masked_invalid(np.array(dict_intervals_two_beams['beam_2']['he_before_SB']['at_end']['blength'])),    '.', color='red',    markersize=8, label='Start SB')
		ax_b2.set_ylabel("B2 Bunch Length [p/b]", fontsize=12, fontweight='bold')
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
		ax_b2.set_ylim(0.5e-09, 1.5e-09)

		# tref string
		tref_string = datetime.fromtimestamp(t_ref)
		subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)

		fig_bbbblength.suptitle(subtitle, fontsize=16, fontweight='bold')

		if self.savePlots:
			filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))+"fill_{}_cycle_blengthbbb".format(filln)+self.plotFormat
			pl.savefig(filename, dpi=config.plotDpi)




		#####-----
		pl.close('all')
		info('#makeCyclePlots : Fill {} -> Making Cycle Time bbb plot...'.format(filln))
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
		subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)

		fig_bbbtime.suptitle(subtitle, fontsize=16, fontweight='bold')

		if self.savePlots:
			filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))+"fill_{}_cycle_timebbb".format(filln)+config.plotFormat
			pl.savefig(filename, dpi=self.plotDpi)

		## ---
		pl.close('all')
		info('#makeCyclePlots : Fill {} -> Making Cycle Histograms bbb plot...'.format(filln))
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


		ax_emit_b1_v = pl.subplot(4,2,3)
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


		ax_emit_b2_h = pl.subplot(4,2,5)
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
		subtitle    = 'Fill {} : Started on {}'.format(filln, tref_string)
		fig_hist.suptitle(subtitle, fontsize=16, fontweight='bold')

		if self.savePlots:
			filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))+"fill_{}_cycle_histos".format(filln)+self.plotFormat
			pl.savefig(filename, dpi=self.plotDpi)

	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def fitfunc(self, x, a, b):
		'''
		Exponential fit function
		Inputs  : x = data
				: a = scaling factor
				: b = exponent factor
		Returns : the result of exponential function
		'''
		return a * np.exp(b * x)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def curve_fit_robust(self, f, x, y):
		'''
		Tries to perform a fit with function f (default exponential) and returns fitted parameters and covariance matrix
		Inputs : f = function to fit with
				 x = x data
				 y = y data
		Returns : popt : fit parameters
				  pcov : convariance matrix of fitted parameters
		'''
		try:
			popt, pcov = curve_fit(f, x, y)
		except ValueError as err:
			 print 'Got Error in curvefit'
			 print err
			 popt  = [np.nan, np.nan]
			 pcov = np.nan
		return popt, pcov
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def checkDirectories(self, filln):
		##############################  CHECK IF THE OUTPUT DIRECTORIES EXIST  ##############################
		## Check if the SB_dir, fill_dir and plot dir exist:
		debug('#checkDirectories : Checking if SB, fill and plots directories exist for fill {}'.format(filln))
		if not os.path.exists(self.SB_dir):
			if self.makedirs:
				c(self.SB_dir)
			else:
				raise IOError('#checkDirectories : SB Analysis directory does not exist & makedirs option is switched off. Exiting...')
		##- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		if not os.path.exists(self.fill_dir.replace('<FILLNUMBER>',str(filln))):
			if self.makedirs:
				os.makedirs(self.fill_dir.replace('<FILLNUMBER>',str(filln)))
			else:
				raise IOError('#checkDirectories : Fill directory for Fill {} does not exist & makedirs option is switched off. Exiting...'.format(filln))
		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		if self.savePlots:
			if not os.path.exists(self.plot_dir.replace('<FILLNUMBER>',str(filln))):
				if self.makedirs:
					os.makedirs(self.plot_dir.replace('<FILLNUMBER>',str(filln)))
				else:
					raise IOError('#checkDirectories : Plots directory for Fill {} does not exist & makedirs option is switched off. Exiting...'.format(filln))
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def checkFiles(self, filln):
		'''
		Checks if for the specific fill number (and resc string) the Cycle files, Massi Files, SB files and Lumi Calc
		files exist.
		If something does not exist or exists but the self.overwriteFiles option is set to True, then it returns
		a boolean set to True for the specific files to be redone.
		Input : filln : fill number
		Returns : getMassi, doSB, doLumiCalc, doCycle booleans
		'''
		getMassi        = False
		doSB            = False
		doSBFits        = False
		doSBModel       = False
		doLifetime      = False
		doLumiCalc      = False
		doCycle         = False
		doCycle_model   = False
		skip            = False
		skipMassi       = False
		getTimber 		= False

		### Check if the fill is in bmodes file...
		if filln not in self.bmodes.index.values:
			fatal("#checkFiles : Fill {} is not in BMODES skipping it...".format(filln))
			os.rmdir(self.plot_dir.replace('<FILLNUMBER>',str(filln)))
			os.rmdir(self.fill_dir.replace('<FILLNUMBER>',str(filln)))
			return getMassi, skipMassi, getTimber, doSB, doSBFits, doSBModel, doLifetime, doLumiCalc, doCycle, doCycle_model, True

		##############################  CHECK IF THE OUTPUT FILES EXIST  ##############################
		## Check if the massi file of the fill exists and flag it
		massi_filename = self.fill_dir+self.Massi_filename
		massi_filename = massi_filename.replace('<FILLNUMBER>',str(filln))

		ATLAS_FILE = self.massi_afs_path.replace('<YEAR>', str(self.fill_year))+self.massi_exp_folders[0]+str(filln)+'.tgz'
		CMS_FILE   = self.massi_afs_path.replace('<YEAR>', str(self.fill_year))+self.massi_exp_folders[1]+str(filln)+'.tgz'

		if not os.path.exists(ATLAS_FILE):
			skipMassi = True
			getTimber = True
		if not os.path.exists(CMS_FILE):
			skipMassi = True
			getTimber = True

		if not skipMassi:
			if os.path.exists(massi_filename):
				if self.overwriteFiles:
					warn("#checkFiles : Dictionary Meas. Lumi pickle [{}] for fill {} already exists! Overwritting it...".format(massi_filename, filln))
					getMassi = True
				else:
					warn("#checkFiles : Dictionary for Meas. Lumi pickle [{}] for fill {} already exists! Skipping it...".format(massi_filename, filln))
			else:
				getMassi = True
			debug('#checkFiles : Checking if dictionary pickle for Massi file [{}] for fill {} exists [{}]'.format(massi_filename, filln, (not getMassi) ))

			if self.savePandas:
				massi_filename = massi_filename.replace('.pkl.gz', '_df.pkl.gz')
				if os.path.exists(massi_filename):
					if self.overwriteFiles:
						warn("#checkFiles : Pandas Meas. Lumi pickle [{}] for fill {} already exists! Overwritting it...".format(massi_filename, filln))
						getMassi = True
					else:
						warn("#checkFiles : Pandas for Meas. Lumi pickle [{}] for fill {} already exists! Skipping it...".format(massi_filename, filln))
				else:
					getMassi = True
				debug('#checkFiles : Checking if Pandas pickle for Massi file [{}] for fill {} exists [{}]'.format(massi_filename, filln, (not getMassi) ))

			##-- also check the database for changed modification date:
			tempGetMassi = db.readYamlDB(filln, year=self.fill_year, yamldb=self.fill_yaml_database, afs_path=self.massi_afs_path, exp_folders=self.massi_exp_folders)
			if getMassi == True:
				## if the file is missing then it will be downloaded, but also force the DB to be updated
				getMassi = True
			elif getMassi == False and tempGetMassi == True:
				## if the file exists but the modification date is changed, force to download it
				getMassi = True
				#rename the old massi file
				info('#checkFiles : Updated Massi Files found - Copying old file to {}'.format(massi_filename.replace('.pkl.gz', '_{}.pkl.gz'.format(datetime.now().strftime("%Y%m%d")))))
				print massi_filename, massi_filename.replace('.pkl.gz', '_{}.pkl.gz'.format(datetime.now().strftime("%Y%m%d")))
				os.rename(massi_filename, massi_filename.replace('.pkl.gz', '_{}.pkl.gz'.format(datetime.now().strftime("%Y%m%d"))))
				if self.savePandas:
					os.rename(massi_filename.replace('.pkl.gz', '_df.pkl.gz'), massi_filename.replace('.pkl.gz', '_{}.pkl.gz'.format(datetime.now().strftime("%Y%m%d"))))
		
		if getTimber:
			massi_filename = self.fill_dir+self.Massi_filename.replace('meas', 'timber_meas')
			if os.path.exists(massi_filename):
				if self.overwriteFiles:
					warn("#checkFiles : Dictionary TIMBER Meas. Lumi pickle [{}] for fill {} already exists! Overwritting it...".format(massi_filename, filln))
					getTimber = True
				else:
					warn("#checkFiles : Dictionary for TIMBER Meas. Lumi pickle [{}] for fill {} already exists! Skipping it...".format(massi_filename, filln))
			else:
				getTimber = True
			debug('#checkFiles : Checking if dictionary pickle for TIMBER Measured Luminosity file [{}] for fill {} exists [{}]'.format(massi_filename, filln, (not getTimber) ))

			if self.savePandas:
				warn("#checkFiles : Timber measured Luminosity in pandas not implemented")
				# massi_filename = massi_filename.replace('.pkl.gz', '_df.pkl.gz')
				# if os.path.exists(massi_filename):
				# 	if self.overwriteFiles:
				# 		warn("#checkFiles : Pandas Meas. Lumi pickle [{}] for fill {} already exists! Overwritting it...".format(massi_filename, filln))
				# 		getTimber = True
				# 	else:
				# 		warn("#checkFiles : Pandas for Meas. Lumi pickle [{}] for fill {} already exists! Skipping it...".format(massi_filename, filln))
				# else:
				# 	getTimber = True
				# debug('#checkFiles : Checking if Pandas pickle for Massi file [{}] for fill {} exists [{}]'.format(massi_filename, filln, (not getMassi) ))



		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		## Check if the SB file of the fill exists and flag it
		SB_filename = self.fill_dir+self.SB_filename
		SB_filename = SB_filename.replace('<FILLNUMBER>',str(filln))
		if self.doRescale:
			if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
				SB_filename = SB_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
			else:
				SB_filename = SB_filename.replace('<RESC>', '').replace("<TO>", '')
		else:
			SB_filename = SB_filename.replace('<RESC>', '')

		if os.path.exists(SB_filename):
			if self.overwriteFiles:
				warn("#checkFiles : Dictionary SB Data pickle [{}] for fill {} already exists! Overwritting it...".format(SB_filename, filln))
				doSB = True
			else:
				warn("#checkFiles : Dictionary SB Data pickle [{}] for fill {} already exists! Skipping it...".format(SB_filename, filln))
		else:
			doSB = True
		debug('#checkFiles : Checking if dictionary pickle of SB file [{}] for fill {} exists [{}]'.format(SB_filename, filln, (not doSB)))

		if self.savePandas:
			SB_filename = SB_filename.replace('.pkl.gz', '_df.pkl.gz')
			if os.path.exists(SB_filename):
				if self.overwriteFiles:
					warn("#checkFiles : Pandas SB Data pickle [{}] for fill {} already exists! Overwritting it...".format(SB_filename, filln))
					doSB = True
				else:
					warn("#checkFiles : Pandas SB Data pickle [{}] for fill {} already exists! Skipping it...".format(SB_filename, filln))
			else:
				doSB = True
			debug('#checkFiles : Checking if pandas pickle of SB file [{}] for fill {} exists [{}]'.format(SB_filename, filln, (not doSB)))

		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		## Check if the lifetime file of the fill exists and flag it
		lt_filename = self.fill_dir+self.SB_filename.replace('.pkl.gz', '_lifetime.pkl.gz')
		lt_filename = lt_filename.replace('<FILLNUMBER>',str(filln))
		if self.doRescale:
			if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
				lt_filename = lt_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
			else:
				lt_filename = lt_filename.replace('<RESC>', '').replace("<TO>", '')
		else:
			lt_filename = lt_filename.replace('<RESC>', '')

		if os.path.exists(lt_filename):
			if self.overwriteFiles:
				warn("#checkFiles : Dictionary SB Data pickle [{}] for fill {} already exists! Overwritting it...".format(lt_filename, filln))
				doLifetime = True
			else:
				warn("#checkFiles : Dictionary SB Data pickle [{}] for fill {} already exists! Skipping it...".format(lt_filename, filln))
		else:
			doLifetime = True
		debug('#checkFiles : Checking if dictionary pickle of SB file [{}] for fill {} exists [{}]'.format(lt_filename, filln, (not doLifetime)))

		if self.savePandas:
			warn('#checkFiles : I do NOT check for SB Lifetime DF -- not yet implemented')

			# lt_filename = lt_filename.replace('.pkl.gz', '_df.pkl.gz')
			# if os.path.exists(lt_filename):
			#   if self.overwriteFiles:
			#       warn("#checkFiles : Pandas SB Data pickle [{}] for fill {} already exists! Overwritting it...".format(lt_filename, filln))
			#       doLifetime = True
			#   else:
			#       warn("#checkFiles : Pandas SB Data pickle [{}] for fill {} already exists! Skipping it...".format(lt_filename, filln))
			# else:
			#   doLifetime = True
			# debug('#checkFiles : Checking if pandas pickle of SB file [{}] for fill {} exists [{}]'.format(lt_filename, filln, (not doLifetime)))

		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		## Check if the lumi calc file of the fill exists and flag it
		lumi_calc_filename = self.fill_dir+self.Lumi_filename
		lumi_calc_filename = lumi_calc_filename.replace('<FILLNUMBER>',str(filln))
		if self.doRescale:
			if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
				lumi_calc_filename = lumi_calc_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
			else:
				lumi_calc_filename = lumi_calc_filename.replace('<RESC>', '').replace("<TO>", '')
		else:
			lumi_calc_filename = lumi_calc_filename.replace('<RESC>', '')

		if os.path.exists(lumi_calc_filename):
			if self.overwriteFiles:
				warn("#checkFiles : Dictionary Lumi Calc pickle [{}] for fill {} already exists! Overwritting it...".format(lumi_calc_filename, filln))
				doLumiCalc = True
			else:
				warn("#checkFiles : Dictionary Lumi Calc pickle [{}] for fill {} already exists! Skipping it...".format(lumi_calc_filename, filln))
		else:
			doLumiCalc = True
		debug('#checkFiles : Checking if dictionary pickle of Lumi Calc file [{}] for fill {} exists [{}]'.format(lumi_calc_filename, filln, (not doLumiCalc) ))

		if self.savePandas:
			lumi_calc_filename=lumi_calc_filename.replace('.pkl.gz', '_df.pkl.gz')
			if os.path.exists(lumi_calc_filename):
				if self.overwriteFiles:
					warn("#checkFiles : Pandas Lumi Calc pickle [{}] for fill {} already exists! Overwritting it...".format(lumi_calc_filename, filln))
					doLumiCalc = True
				else:
					warn("#checkFiles : Pandas Lumi Calc pickle [{}] for fill {} already exists! Skipping it...".format(lumi_calc_filename, filln))
			else:
				doLumiCalc = True
			debug('#checkFiles : Checking if Pandas pickle of Lumi Calc file [{}] for fill {} exists [{}]'.format(lumi_calc_filename, filln, (not doLumiCalc) ))


		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		## Check if the cycle file of the fill exists and flag it
		cycle_filename = self.fill_dir+self.Cycle_filename
		cycle_filename = cycle_filename.replace('<FILLNUMBER>',str(filln))
		if self.doRescale:
			if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
				cycle_filename = cycle_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
			else:
				cycle_filename = cycle_filename.replace('<RESC>', '').replace("<TO>", '')
		else:
			cycle_filename = cycle_filename.replace('<RESC>', '')

		if os.path.exists(cycle_filename):
			if self.overwriteFiles:
				warn("#checkFiles : Dictionary Cycle pickle [{}] for fill {} already exists! Overwritting it...".format(cycle_filename, filln))
				doCycle = True
			else:
				warn("#checkFiles : Dictionary Cycle pickle [{}] for fill {} already exists! Skipping it...".format(cycle_filename, filln))
		else:
			doCycle = True
		debug('#checkFiles : Checking if dictionary pickle of Cycle file [{}] for fill {} exists [{}]'.format(cycle_filename, filln, (not doCycle)  ))

		if self.savePandas:
			cycle_filename = cycle_filename.replace('.pkl.gz', '_df.pkl.gz')
			if os.path.exists(cycle_filename):
				if self.overwriteFiles:
					warn("#checkFiles : Pandas Cycle pickle [{}] for fill {} already exists! Overwritting it...".format(cycle_filename, filln))
					doCycle = True
				else:
					warn("#checkFiles : Pands Cycle pickle [{}] for fill {} already exists! Skipping it...".format(cycle_filename, filln))
			else:
				doCycle = True
			debug('#checkFiles : Checking if Pandas pickle of Cycle file [{}] for fill {} exists [{}]'.format(cycle_filename, filln, (not doCycle)  ))



		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		## Check if the cycle model file of the fill exists and flag it
		cycle_model_filename = self.fill_dir+self.Cycle_model_filename
		cycle_model_filename = cycle_model_filename.replace('<FILLNUMBER>',str(filln))
		if self.doRescale:
			if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
				cycle_model_filename = cycle_model_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
			else:
				cycle_model_filename = cycle_model_filename.replace('<RESC>', '').replace("<TO>", '')
		else:
			cycle_model_filename = cycle_model_filename.replace('<RESC>', '')

		cycle_model_filename_inj2sb = cycle_model_filename.replace('.pkl.gz', '_Inj2SB.pkl.gz')
		if os.path.exists(cycle_model_filename):
			if self.overwriteFiles:
				warn("#checkFiles : Dictionary Cycle model pickle [{}] for fill {} already exists! Overwritting it...".format(cycle_model_filename, filln))
				doCycle_model = True
			else:
				warn("#checkFiles : Dictionary Cycle model pickle [{}] for fill {} already exists! Skipping it...".format(cycle_model_filename, filln))
		else:
			doCycle_model = True
		debug('#checkFiles : Checking if dictionary pickle of Cycle model file [{}] for fill {} exists [{}]'.format(cycle_model_filename, filln, (not doCycle_model)  ))

		if os.path.exists(cycle_model_filename_inj2sb):
			if self.overwriteFiles:
				warn("#checkFiles : Dictionary Cycle model Inj2SB pickle [{}] for fill {} already exists! Overwritting it...".format(cycle_model_filename_inj2sb, filln))
				doCycle_model = True
			else:
				warn("#checkFiles : Dictionary Cycle model Inj2SB pickle [{}] for fill {} already exists! Skipping it...".format(cycle_model_filename_inj2sb, filln))
		else:
			doCycle_model = True
		debug('#checkFiles : Checking if dictionary pickle of Cycle model  Inj2SB file [{}] for fill {} exists [{}]'.format(cycle_model_filename_inj2sb, filln, (not doCycle_model)  ))

		if self.savePandas:
			cycle_model_filename = cycle_model_filename.replace('.pkl.gz', '_df.pkl.gz')
			cycle_model_filename_inj2sb = cycle_model_filename_inj2sb.replace('.pkl.gz', '_df.pkl.gz')
			if os.path.exists(cycle_model_filename):
				if self.overwriteFiles:
					warn("#checkFiles : Pandas Cycle Model pickle [{}] for fill {} already exists! Overwritting it...".format(cycle_model_filename, filln))
					doCycle_model = True
				else:
					warn("#checkFiles : Pands Cycle Model pickle [{}] for fill {} already exists! Skipping it...".format(cycle_model_filename, filln))
			else:
				doCycle_model = True
			debug('#checkFiles : Checking if Pandas pickle of Cycle Model file [{}] for fill {} exists [{}]'.format(cycle_model_filename, filln, (not doCycle_model)  ))

			if os.path.exists(cycle_model_filename_inj2sb):
				if self.overwriteFiles:
					warn("#checkFiles : Pandas Cycle Model pickle [{}] for fill {} already exists! Overwritting it...".format(cycle_model_filename_inj2sb, filln))
					doCycle_model = True
				else:
					warn("#checkFiles : Pands Cycle Model pickle [{}] for fill {} already exists! Skipping it...".format(cycle_model_filename_inj2sb, filln))
			else:
				doCycle_model = True
			debug('#checkFiles : Checking if Pandas pickle of Cycle Model file [{}] for fill {} exists [{}]'.format(cycle_model_filename_inj2sb, filln, (not doCycle_model)  ))

		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		## Check if the fits file of the fill exists and flag it
		sbfits_filename = self.fill_dir+self.SB_fits_filename
		sbfits_filename = sbfits_filename.replace('<FILLNUMBER>',str(filln))
		if self.doRescale:
			if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
				sbfits_filename = sbfits_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
			else:
				sbfits_filename = sbfits_filename.replace('<RESC>', '').replace("<TO>", '')
		else:
			sbfits_filename = sbfits_filename.replace('<RESC>', '')

		if os.path.exists(sbfits_filename):
			if self.overwriteFiles:
				warn("#checkFiles : Dictionary SB Fits pickle [{}] for fill {} already exists! Overwritting it...".format(sbfits_filename, filln))
				doSBFits = True
			else:
				warn("#checkFiles : Dictionary SB Fits pickle [{}] for fill {} already exists! Skipping it...".format(sbfits_filename, filln))
		else:
			doSBFits = True
		debug('#checkFiles : Checking if dictionary pickle of SB Fits file [{}] for fill {} exists [{}]'.format(sbfits_filename, filln, (not doSBFits) ))

		warn('#checkFiles : I do NOT check for SBFits DF')
		# if self.savePandas:
		#   sbfits_filename=sbfits_filename.replace('.pkl.gz', '_df.pkl.gz')
		#   if os.path.exists(sbfits_filename):
		#       if self.overwriteFiles:
		#           warn("#checkFiles : Pandas SB Fits pickle [{}] for fill {} already exists! Overwritting it...".format(sbfits_filename, filln))
		#           doSBFits = True
		#       else:
		#           warn("#checkFiles : Pandas SB Fits pickle [{}] for fill {} already exists! Skipping it...".format(sbfits_filename, filln))
		#   else:
		#       doSBFits = True
		#   debug('#checkFiles : Checking if Pandas pickle of SB Fits file [{}] for fill {} exists [{}]'.format(sbfits_filename, filln, (not doSBFits) ))

		## - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
		## Check if the SB Model file of the fill exists and flag it
		for case in self.cases:

			sbmodel_filename = self.fill_dir+self.SB_model_filename.replace('.pkl.gz', '_case{}.pkl.gz'.format(case))
			sbmodel_filename = sbmodel_filename.replace('<FILLNUMBER>',str(filln))
			info("# checkFiles : Checking model for case {} [{}]".format(case, sbmodel_filename))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					sbmodel_filename = sbmodel_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					sbmodel_filename = sbmodel_filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				sbmodel_filename = sbmodel_filename.replace('<RESC>', '')

			if os.path.exists(sbmodel_filename):
				if self.overwriteFiles:
					warn("#checkFiles : Dictionary SB Model pickle [{}] for fill {} already exists! Overwritting it...".format(sbmodel_filename, filln))
					doSBModel = True
				else:
					warn("#checkFiles : Dictionary SB Model pickle [{}] for fill {} already exists! Skipping it...".format(sbmodel_filename, filln))
			else:
				doSBModel = True
			debug('#checkFiles : Checking if dictionary pickle of SB Model file [{}] for fill {} exists [{}]'.format(sbmodel_filename, filln, (not doSBModel) ))

			if self.savePandas:
				warn('#checkFiles : I do NOT check for SB model PANDAS -- Not implemented yet')
			# sbmodel_filename=sbmodel_filename.replace('.pkl.gz', '_df.pkl.gz')
			# if os.path.exists(sbmodel_filename):
			#   if self.overwriteFiles:
			#       warn("#checkFiles : Pandas SB Model pickle [{}] for fill {} already exists! Overwritting it...".format(sbmodel_filename, filln))
			#       doSBModel = True
			#   else:
			#       warn("#checkFiles : Pandas SB Model pickle [{}] for fill {} already exists! Skipping it...".format(sbmodel_filename, filln))
			# else:
			#   doSBModel = True
			# debug('#checkFiles : Checking if Pandas pickle of SB Model file [{}] for fill {} exists [{}]'.format(sbmodel_filename, filln, (not doSBModel) ))




		return getMassi, skipMassi, getTimber, doSB, doSBFits, doSBModel, doLifetime, doLumiCalc, doCycle, doCycle_model, skip
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runLifetime(self, filln):
		'''
		Create Lifetime files for fill
		Inputs : filln : fill number
		Returns: None
		'''
		self.filln_LifetimeDict.clear()

		# Check that I have the SB info
		if len(self.filln_StableBeamsDict)>0:
			debug("# runLifetime : SB Analysis has ran for this fill, loading the SB dictionary")
		else:
			##populate it from file
			filename = self.fill_dir+self.SB_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			debug("# runLifetime : SB Analysis has NOT ran for this fill, loading the SB dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_StableBeamsDict = pickle.load(fid)

		# Check that I have the meas lumi
		if len(self.filln_LumiMeasDict)>0:
			debug("# runLifetime : Measured Lumi has ran for this fill, loading the Measured Lumi dictionary")
		else:
			##populate it from file
			filename = self.fill_dir+self.Massi_filename
			filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>', '')
			debug("# runLifetime : Measured Lumi Analysis has NOT ran for this fill, loading the Measured Lumi dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_LumiMeasDict = pickle.load(fid)


		b_inten_interp_coll = self.filln_StableBeamsDict['b_inten_interp_coll']

		b_interp_coll_tot   = {}
		tau_BOff_tot        = {}
		tau_BOff_bbb        = {}
		tau_BOff_bct        = {}
		tau_Np_tot          = {}
		tau_Np_bct          = {}
		tau_Np_bbb          = {}
		life_time_Boff_tot  = {}
		life_time_Boff_bct  = {}
		life_time_tot       = {}
		life_time_bct       = {}
		life_time_bbb       = {}
		life_time_Boff_bbb  = {}
		dNdt_bbb            = {}

		losses_bbb          = {}
		losses_bct          = {}
		losses_tot          = {}
		slots               = {}
		time_range          = {}

		dt = self.filln_StableBeamsDict['time_range'][1]-self.filln_StableBeamsDict['time_range'][0]

		dict_lifetime = {1:{}, 2:{}}

		for beam_n in [1, 2]:
			dict_lifetime[beam_n] = {}

			dict_lifetime[beam_n] = {}

			###########################   CALCULATE FOR TOTAL INTENSITY
			# Calculate total intensity
			b_interp_coll_tot[beam_n] = np.sum(b_inten_interp_coll[beam_n], axis=1)

			# Burnoff corrected lifetime (total) is : intensity / (sigma* L)
			tau_BOff_tot[beam_n]      = b_interp_coll_tot[beam_n]/self.sigmaBOff_m2/np.sum((self.filln_LumiMeasDict['ATLAS']['bunch_lumi']+self.filln_LumiMeasDict['CMS']['bunch_lumi']), axis=1)

			# calculate dN (total)
			dNp                       = -(b_interp_coll_tot[beam_n][:-1]) + (b_interp_coll_tot[beam_n][1:])

			# calculate the tau (total) for the dN (total)
			tau_Np_tot[beam_n]        = -1/((dNp/dt)/b_interp_coll_tot[beam_n][:-1])

			###########################   CALCULATE FOR BBB INTENSITY
			# calculate the Burnoff corrected  lifetime tau bbb
			tau_BOff_bbb[beam_n]      = b_inten_interp_coll[beam_n]/self.sigmaBOff_m2/(self.filln_LumiMeasDict['ATLAS']['bunch_lumi']+self.filln_LumiMeasDict['CMS']['bunch_lumi'])
			# calculate dN (bbb)
			dNp_bbb                   = -(b_inten_interp_coll[beam_n][:-1,:]) + (b_inten_interp_coll[beam_n][1:,:])
			# calculate tau (bbb) for dN bbb
			tau_Np_bbb[beam_n]        = -1/((dNp_bbb/dt)/b_inten_interp_coll[beam_n][:-1,:])



			# Calculate losses bbb
			losses_bbb[beam_n]  = (np.abs(dNp_bbb)/dt)/(self.filln_LumiMeasDict['ATLAS']['bunch_lumi']+self.filln_LumiMeasDict['CMS']['bunch_lumi'])[:-1]
			dNdt_bbb[beam_n]    = (np.abs(dNp_bbb)/dt)

			dict_lifetime[beam_n]['life_time_Boff_tot'] = 1/((1/tau_Np_tot[beam_n]-1/tau_BOff_tot[beam_n][0:-1]))
			dict_lifetime[beam_n]['life_time_Boff_bbb'] = 1/((1/tau_Np_bbb[beam_n]-1/tau_BOff_bbb[beam_n][0:-1, :]))
			dict_lifetime[beam_n]['life_time_tot'] = 1/((1/tau_Np_tot[beam_n]))
			dict_lifetime[beam_n]['life_time_bbb'] = 1/((1/tau_Np_bbb[beam_n]))
			dict_lifetime[beam_n]['losses_dndtL_bbb'] = losses_bbb[beam_n]

			dict_lifetime[beam_n]['dndt_bbb']   = dNdt_bbb[beam_n]
			dict_lifetime[beam_n]['tau_Np_tot'] = tau_Np_tot[beam_n]
			dict_lifetime[beam_n]['tau_Np_bbb'] = tau_Np_bbb[beam_n]


			dict_lifetime[1]['slots']   = self.filln_StableBeamsDict['slots_filled_coll'][1]
			dict_lifetime[2]['slots']   = self.filln_StableBeamsDict['slots_filled_coll'][2]
			dict_lifetime['time_range'] = self.filln_StableBeamsDict['time_range']



		self.filln_LifetimeDict = dict_lifetime

		if self.saveDict:
			filename = self.fill_dir+self.SB_filename.replace('.pkl.gz', '_lifetime.pkl.gz')
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			info('# runLifetime : Saving Lifetime dictionary for fill {} into {}'.format(filln, filename ))
			if os.path.exists(filename):
				if self.overwriteFiles:
					warn("# runLifetime : Lifetime Dictionary for fill {} already exists! Overwritting it...".format(filln))
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(dict_lifetime, fid)
				else:
					warn("# runLifetime : Lifetime dictionaryfor fill {} already exists! Skipping it...".format(filln))
			else:
				with gzip.open(filename, 'wb') as fid:
					pickle.dump(dict_lifetime, fid)

		if self.savePandas:
			pass
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runCycleSB(self, filln, doCycle, doSB):
		'''
		Runs the cycle and/or the SB data analysis for the specific fill.
		Inputs : filln   : fill nubmer
				 doCycle : bool to do cycle (also checks if cyclePlots are required)
				 doSB    : bool to do SB analysis
		Returns: None (stores into the self. dictionaries)
		'''

		## first step is to clear the self.filln_dictionaries and get fill times
		if doCycle:
			debug('#runCycleSB : Clearing Cycle dictionary for fill {}'.format(filln))
			self.filln_CycleDict.clear()
			t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(filln)
		if doSB:
			debug('#runCycleSB : Clearing SB dictionary for fill {}'.format(filln))
			self.filln_StableBeamsDict.clear()

		## get timber data for fill
		debug('#runCycleSB : Getting Timber dictionary for fill {}'.format(filln))
		timber_dic = self.getTimberDic(filln)

		## get BSRT calibration dict
		debug('#runCycleSB : Getting BSRT calibration dictionary for fill {}'.format(filln))
		bsrt_calib_dict = self.getBSRTCalibDic(filln)

		## Create empty SB dicts
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(filln)
		eh_interp_raw, ev_interp_raw, eh_interp, ev_interp, b_inten_interp, bl_interp_m, slots_filled, bct_dict, fbct_dict = self.getEmptySBDataDict()
		frac_smoothing = self.getFracSmoothing(filln, self.avg_time_smoothing, t_end_STABLE, t_start_STABLE)

		if doCycle:
			debug('#runCycleSB : Creating Cycle dictionary.')
			dict_intervals_two_beams = {}
			for beam_n in [1, 2]:
				dict_intervals_two_beams['beam_{}'.format(beam_n)] = {}

		debug('#runCycleSB : Looping for the two beams for fill {}'.format(filln))
		for beam_n in [1,2]:
			debug('#runCycleSB : Running for beam {}...'.format(beam_n))
			## Energy, BCT, BSRT, FBCT, BQM are from LHCMeasurementTools
			energy  = Energy.energy(timber_dic, beam=beam_n)
			bct     = BCT.BCT(timber_dic, beam=beam_n)
			bsrt    = BSRT.BSRT(timber_dic, beam=beam_n, calib_dict=bsrt_calib_dict, average_repeated_meas=True)
			fbct    = FBCT.FBCT(timber_dic, beam=beam_n)
			blength = BQM.blength(timber_dic, beam=beam_n)

			## Fill the BCT and FBCT dictionaries from info taken by measurements
			bct_dict[beam_n] = bct
			fbct_dict[beam_n] = fbct

			## Calculate emittances for the specific energy using the BSRT
			bsrt.calculate_emittances(energy)
			## Get bbb evolution from bsrt
			debug('#runCycleSB : Getting BBB emittance evolution.')
			dict_bsrt_bunches, t_bbb, emit_h_bbb, emit_v_bbb, bunch_n_filled = bsrt.get_bbb_emit_evolution()

			if doCycle:

				dict_intervals = dict_intervals_two_beams['beam_{}'.format(beam_n)]
				dict_intervals['Injection']                 = {}
				dict_intervals['Injection']['t_start']      = self.bmodes['t_start_INJPHYS'][filln]
				dict_intervals['Injection']['t_end']        = self.bmodes['t_stop_PRERAMP'][filln]

				dict_intervals['he_before_SB']              = {}
				dict_intervals['he_before_SB']['t_start']   = self.bmodes['t_start_FLATTOP'][filln]
				dict_intervals['he_before_SB']['t_end']     = self.bmodes['t_start_STABLE'][filln]

				for interval in dict_intervals.keys():
					for instance in ['at_start', 'at_end']:
						dict_intervals[interval][instance] = {}
						dict_intervals[interval][instance]['emith'] = []
						dict_intervals[interval][instance]['emitv'] = []
						dict_intervals[interval][instance]['time_meas'] = []
						dict_intervals[interval][instance]['intensity'] = []
						dict_intervals[interval][instance]['blength'] = []

			debug('#runCycleSB : Looping for bunch slots...')
			for slot_bun in bunch_n_filled:
				if doCycle:
					if slot_bun == 0:
						debug('#runCycleSB : Getting Cycle info for bunch slots...')
					for interval in dict_intervals.keys():
						dict_intervals[interval]['filled_slots'] = bunch_n_filled
						mask_obs = np.logical_and(dict_bsrt_bunches[slot_bun]['t_stamp']>dict_intervals[interval]['t_start'], dict_bsrt_bunches[slot_bun]['t_stamp']<dict_intervals[interval]['t_end'])

						if np.sum(mask_obs)>0:
							dict_intervals[interval]['at_start']['emith'].append(dict_bsrt_bunches[slot_bun]['norm_emit_h'][mask_obs][0])
							dict_intervals[interval]['at_start']['emitv'].append(dict_bsrt_bunches[slot_bun]['norm_emit_v'][mask_obs][0])
							dict_intervals[interval]['at_start']['time_meas'].append(dict_bsrt_bunches[slot_bun]['t_stamp'][mask_obs][0])
							dict_intervals[interval]['at_start']['intensity'].append(np.interp(dict_intervals[interval]['at_start']['time_meas'][-1]+100., fbct.t_stamps, fbct.bint[:, slot_bun]))
							dict_intervals[interval]['at_start']['blength'].append(np.interp(dict_intervals[interval]['at_start']['time_meas'][-1]+100., blength.t_stamps, blength.blen[:, slot_bun]))

							dict_intervals[interval]['at_end']['emith'].append(dict_bsrt_bunches[slot_bun]['norm_emit_h'][mask_obs][-1])
							dict_intervals[interval]['at_end']['emitv'].append(dict_bsrt_bunches[slot_bun]['norm_emit_v'][mask_obs][-1])
							dict_intervals[interval]['at_end']['time_meas'].append(dict_bsrt_bunches[slot_bun]['t_stamp'][mask_obs][-1])
							dict_intervals[interval]['at_end']['intensity'].append(np.interp(dict_intervals[interval]['at_end']['time_meas'][-1], fbct.t_stamps, fbct.bint[:, slot_bun]))
							dict_intervals[interval]['at_end']['blength'].append(np.interp(dict_intervals[interval]['at_end']['time_meas'][-1], blength.t_stamps, blength.blen[:, slot_bun]))

						else:
							dict_intervals[interval]['at_start']['emith'].append(np.nan)
							dict_intervals[interval]['at_start']['emitv'].append(np.nan)
							dict_intervals[interval]['at_start']['time_meas'].append(np.nan)
							dict_intervals[interval]['at_start']['intensity'].append(np.nan)
							dict_intervals[interval]['at_start']['blength'].append(np.nan)

							dict_intervals[interval]['at_end']['emith'].append(np.nan)
							dict_intervals[interval]['at_end']['emitv'].append(np.nan)
							dict_intervals[interval]['at_end']['time_meas'].append(np.nan)
							dict_intervals[interval]['at_end']['intensity'].append(np.nan)
							dict_intervals[interval]['at_end']['blength'].append(np.nan)

				##.. make an 1D linear interpolation of bunch intensity from fbct
				fbct_bun_stablebeams = np.interp(time_range, fbct.t_stamps, fbct.bint[:, slot_bun])

				## Check that you are not above the set intensity threshold
				if np.max(fbct_bun_stablebeams) < self.intensity_threshold:
					continue

				if self.enable_smoothing_BSRT:
					from statsmodels.nonparametric.smoothers_lowess import lowess
					if self.debug:
						if slot_bun%100==0:
							info("Fill {}: Bunch Slot : {}".format(filln, slot_bun))
					else:
						if slot_bun%1000==0:
							info("Fill {}: Bunch Slot : {}".format(filln, slot_bun))

				## Create a mask to only take into account bunches after the SQUEEZE
				mask_from_squeeze = dict_bsrt_bunches[slot_bun]['t_stamp']>float(self.bmodes['t_start_SQUEEZE'][filln])

				## ... and perform lowess at the horizontal and vertical bunch slots
				filtered_h = lowess(dict_bsrt_bunches[slot_bun]['norm_emit_h'][mask_from_squeeze], dict_bsrt_bunches[slot_bun]['t_stamp'][mask_from_squeeze], is_sorted=True, frac=frac_smoothing, it=10, delta=0.01*(t_end_STABLE-t_start_STABLE))
				filtered_v = lowess(dict_bsrt_bunches[slot_bun]['norm_emit_v'][mask_from_squeeze], dict_bsrt_bunches[slot_bun]['t_stamp'][mask_from_squeeze], is_sorted=True, frac=frac_smoothing, it=10, delta=0.01*(t_end_STABLE-t_start_STABLE))

				## fill in the interpolated eh/ev
				# print time_range, filtered_h[:,0], filtered_h[:,1]
				eh_interp[beam_n].append(np.interp(time_range, filtered_h[:,0], filtered_h[:,1]))
				ev_interp[beam_n].append(np.interp(time_range, filtered_v[:,0], filtered_v[:,1]))

				## Fill in the raw eh/ev
				eh_interp_raw[beam_n].append(np.interp(time_range,dict_bsrt_bunches[slot_bun]['t_stamp'],dict_bsrt_bunches[slot_bun]['norm_emit_h']))
				ev_interp_raw[beam_n].append(np.interp(time_range,dict_bsrt_bunches[slot_bun]['t_stamp'],dict_bsrt_bunches[slot_bun]['norm_emit_v']))

				## Fill in the bunch intensity, bunch lengths and slots filled
				b_inten_interp[beam_n].append(fbct_bun_stablebeams)
				bl_interp_m[beam_n].append((np.interp(time_range, blength.t_stamps, blength.blen[:, slot_bun]))/4.*clight)
				slots_filled[beam_n].append(slot_bun)

			## outside the slot bun loop
			if doCycle:
				for interval in ['Injection', 'he_before_SB']:
					for moment in ['at_start', 'at_end']:
						dict_intervals[interval][moment]['brightness'] = np.array(dict_intervals[interval][moment]['intensity'])/np.sqrt(np.array(dict_intervals[interval][moment]['emith'])*np.array(dict_intervals[interval][moment]['emitv']))


			## cast to arrays -- Transpose to look like columns
			eh_interp[beam_n] = np.array(eh_interp[beam_n]).T
			ev_interp[beam_n] = np.array(ev_interp[beam_n]).T
			eh_interp_raw[beam_n] = np.array(eh_interp_raw[beam_n]).T
			ev_interp_raw[beam_n] = np.array(ev_interp_raw[beam_n]).T
			b_inten_interp[beam_n] = np.array(b_inten_interp[beam_n]).T
			bl_interp_m[beam_n] = np.array(bl_interp_m[beam_n]).T
			slots_filled[beam_n] = np.array(slots_filled[beam_n]).T

		## outside the beam loop
		## identify colliding - The colliding are the intersections of the filled slots in b1 and b2
		slots_col = list(set(slots_filled[1]) & set(slots_filled[2]))
		mask_colliding = {1:[], 2:[]}
		for beam_n in [1, 2]:
			mask_colliding[beam_n] = np.array(map(lambda slot: slot in slots_col, slots_filled[beam_n]))

		## extract colliding and non_colliding
		eh_interp_coll          = {1:[], 2:[]}
		ev_interp_coll          = {1:[], 2:[]}
		eh_interp_raw_coll      = {1:[], 2:[]}
		ev_interp_raw_coll      = {1:[], 2:[]}
		b_inten_interp_coll     = {1:[], 2:[]}
		bl_interp_m_coll        = {1:[], 2:[]}
		slots_filled_coll       = {1:[], 2:[]}
		slots_filled_coll       = {1:[], 2:[]}

		eh_interp_noncoll       = {1:[], 2:[]}
		ev_interp_noncoll       = {1:[], 2:[]}
		eh_interp_raw_noncoll   = {1:[], 2:[]}
		ev_interp_raw_noncoll   = {1:[], 2:[]}
		b_inten_interp_noncoll  = {1:[], 2:[]}
		bl_interp_m_noncoll     = {1:[], 2:[]}
		slots_filled_noncoll    = {1:[], 2:[]}
		slots_filled_noncoll    = {1:[], 2:[]}
		lifetime                = {1:[], 2:[]}

		tau_emit_h_coll     = {1:[], 2:[]}
		tau_emit_v_coll     = {1:[], 2:[]}
		tau_emit_h_noncoll  = {1:[], 2:[]}
		tau_emit_v_noncoll  = {1:[], 2:[]}
		init_emit_h_coll    = {1:[], 2:[]}
		init_emit_v_coll    = {1:[], 2:[]}
		init_emit_h_noncoll = {1:[], 2:[]}
		init_emit_v_noncoll = {1:[], 2:[]}
		tau_bl_coll         = {1:[], 2:[]}
		tau_bl_noncoll      = {1:[], 2:[]}
		init_bl_coll        = {1:[], 2:[]}
		init_bl_noncoll     = {1:[], 2:[]}
		tau_inten_coll      = {1:[], 2:[]}
		tau_inten_noncoll   = {1:[], 2:[]}
		init_inten_coll     = {1:[], 2:[]}
		init_inten_noncoll  = {1:[], 2:[]}

		tau_emit_h_coll_full     = {1:[], 2:[]}
		tau_emit_v_coll_full     = {1:[], 2:[]}
		tau_emit_h_noncoll_full  = {1:[], 2:[]}
		tau_emit_v_noncoll_full  = {1:[], 2:[]}
		init_emit_h_coll_full    = {1:[], 2:[]}
		init_emit_v_coll_full    = {1:[], 2:[]}
		init_emit_h_noncoll_full = {1:[], 2:[]}
		init_emit_v_noncoll_full = {1:[], 2:[]}
		tau_bl_coll_full         = {1:[], 2:[]}
		tau_bl_noncoll_full      = {1:[], 2:[]}
		init_bl_coll_full        = {1:[], 2:[]}
		init_bl_noncoll_full     = {1:[], 2:[]}
		tau_inten_coll_full      = {1:[], 2:[]}
		tau_inten_noncoll_full   = {1:[], 2:[]}
		init_inten_coll_full     = {1:[], 2:[]}
		init_inten_noncoll_full  = {1:[], 2:[]}



		dt = time_range[1]-time_range[0]
		dNdt_bbb            = {}


		debug('#runCycleSB : Filling the SB dictionary keys...')
		for beam_n in [1, 2]:
			## The colliding are the ones that meet the mask_collinding req
			eh_interp_coll[beam_n]          = eh_interp[beam_n][:, mask_colliding[beam_n]]
			ev_interp_coll[beam_n]          = ev_interp[beam_n][:, mask_colliding[beam_n]]
			eh_interp_raw_coll[beam_n]      = eh_interp_raw[beam_n][:, mask_colliding[beam_n]]
			ev_interp_raw_coll[beam_n]      = ev_interp_raw[beam_n][:, mask_colliding[beam_n]]
			b_inten_interp_coll[beam_n]     = b_inten_interp[beam_n][:, mask_colliding[beam_n]]
			bl_interp_m_coll[beam_n]        = bl_interp_m[beam_n][:, mask_colliding[beam_n]]
			slots_filled_coll[beam_n]       = slots_filled[beam_n][mask_colliding[beam_n]]

			## The non-colliding are the ones that do not meet the mask_collinding req
			eh_interp_noncoll[beam_n]       = eh_interp[beam_n][:, ~mask_colliding[beam_n]]
			ev_interp_noncoll[beam_n]       = ev_interp[beam_n][:, ~mask_colliding[beam_n]]
			eh_interp_raw_noncoll[beam_n]   = eh_interp_raw[beam_n][:, ~mask_colliding[beam_n]]
			ev_interp_raw_noncoll[beam_n]   = ev_interp_raw[beam_n][:, ~mask_colliding[beam_n]]
			b_inten_interp_noncoll[beam_n]  = b_inten_interp[beam_n][:, ~mask_colliding[beam_n]]
			bl_interp_m_noncoll[beam_n]     = bl_interp_m[beam_n][:, ~mask_colliding[beam_n]]
			slots_filled_noncoll[beam_n]    = slots_filled[beam_n][~mask_colliding[beam_n]]


			dNp_bbb                          = -(b_inten_interp_coll[beam_n][:-1,:]) + (b_inten_interp_coll[beam_n][1:,:])
			dNdt_bbb                         = (np.abs(dNp_bbb)/dt)
			lifetime[beam_n]                 = -1/((dNp_bbb/dt)/b_inten_interp_coll[beam_n][:-1,:])


			tau_emit_h_coll[beam_n]         = []
			tau_emit_v_coll[beam_n]         = []
			tau_emit_h_noncoll[beam_n]      = []
			tau_emit_v_noncoll[beam_n]      = []

			tau_emit_h_coll_full[beam_n]         = []
			tau_emit_v_coll_full[beam_n]         = []
			tau_emit_h_noncoll_full[beam_n]      = []
			tau_emit_v_noncoll_full[beam_n]      = []

			init_emit_h_coll[beam_n]        = []
			init_emit_v_coll[beam_n]        = []
			init_emit_h_noncoll[beam_n]     = []
			init_emit_v_noncoll[beam_n]     = []

			init_emit_h_coll_full[beam_n]        = []
			init_emit_v_coll_full[beam_n]        = []
			init_emit_h_noncoll_full[beam_n]     = []
			init_emit_v_noncoll_full[beam_n]     = []

			tau_bl_coll[beam_n]             = []
			tau_bl_noncoll[beam_n]          = []

			tau_bl_coll_full[beam_n]             = []
			tau_bl_noncoll_full[beam_n]          = []

			init_bl_coll[beam_n]            = []
			init_bl_noncoll[beam_n]         = []

			init_bl_coll_full[beam_n]            = []
			init_bl_noncoll_full[beam_n]         = []

			tau_inten_coll[beam_n]          = []
			tau_inten_noncoll[beam_n]       = []

			tau_inten_coll_full[beam_n]          = []
			tau_inten_noncoll_full[beam_n]       = []

			init_inten_coll[beam_n]         = []
			init_inten_noncoll[beam_n]      = []

			init_inten_coll_full[beam_n]         = []
			init_inten_noncoll_full[beam_n]      = []

			trange = time_range
			mask_fit = (trange-trange[0])<self.t_fit_length
			t_tot_emit_fit_length = trange[-1]-trange[0]

			# colliding bunches :
			info('#runCycleSB : Fitting for colliding bunches...')
			slot_list = slots_filled_coll[beam_n]
			for i_slot, slot_numb in enumerate(slot_list):
				#emit h
				ytofit = eh_interp_coll[beam_n][:,i_slot]
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_emit_h_coll[beam_n].append(3600.0*1.0/popt[1])
				tau_emit_h_coll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_emit_h_coll[beam_n].append(popt[0])
				init_emit_h_coll_full[beam_n].append(popt_f[0])

				#emit v
				ytofit = ev_interp_coll[beam_n][:,i_slot]
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_emit_v_coll[beam_n].append(3600.0*1.0/popt[1])
				tau_emit_v_coll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_emit_v_coll[beam_n].append(popt[0])
				init_emit_v_coll_full[beam_n].append(popt_f[0])


				ytofit = bl_interp_m_coll[beam_n][:,i_slot]
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_bl_coll[beam_n].append(3600.0*1.0/popt[1])
				tau_bl_coll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_bl_coll[beam_n].append(popt[0])
				init_bl_coll_full[beam_n].append(popt_f[0])

				ytofit = b_inten_interp_coll[beam_n][:,i_slot]*1.0e-11
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_inten_coll[beam_n].append(3600.0*1.0/popt[1])
				tau_inten_coll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_inten_coll[beam_n].append(popt[0]*1.0e11)
				init_inten_coll_full[beam_n].append(popt_f[0]*1.0e11)

			 # non colliding bunches :
			info('#runCycleSB : Fitting for non-colliding bunches...')
			slot_list = slots_filled_noncoll[beam_n]
			for i_slot, slot_numb in enumerate(slot_list):
				#emit h
				ytofit = eh_interp_noncoll[beam_n][:,i_slot]
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_emit_h_noncoll[beam_n].append(3600.0*1.0/popt[1])
				tau_emit_h_noncoll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_emit_h_noncoll[beam_n].append(popt[0])
				init_emit_h_noncoll_full[beam_n].append(popt_f[0])

				#emit v
				ytofit = ev_interp_noncoll[beam_n][:,i_slot]
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_emit_v_noncoll[beam_n].append(3600.0*1.0/popt[1])
				tau_emit_v_noncoll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_emit_v_noncoll[beam_n].append(popt[0])
				init_emit_v_noncoll_full[beam_n].append(popt_f[0])


				ytofit = bl_interp_m_noncoll[beam_n][:,i_slot]
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_bl_noncoll[beam_n].append(3600.0*1.0/popt[1])
				tau_bl_noncoll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_bl_noncoll[beam_n].append(popt[0])
				init_bl_noncoll_full[beam_n].append(popt_f[0])

				ytofit = b_inten_interp_noncoll[beam_n][:,i_slot]*1.0e-11
				popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
				popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
				tau_inten_noncoll[beam_n].append(3600.0*1.0/popt[1])
				tau_inten_noncoll_full[beam_n].append(3600.0*1.0/popt_f[1])
				init_inten_noncoll[beam_n].append(popt[0]*1.0e11)
				init_inten_noncoll_full[beam_n].append(popt_f[0]*1.0e11)



		## if doCycle and self.doCyclePlots:
		##     debug("#runSBandCycle : Making Cycle plots")
		##     self.makeCyclePlots(dict_intervals_two_beams, filln, t_ref)
		## copy the file to a local variable to avoid opening file
		if doCycle:
			self.filln_CycleDict           = dict_intervals_two_beams


		dict_save = {
		'eh_interp_coll':eh_interp_coll,
		'ev_interp_coll':ev_interp_coll,
		'eh_interp_raw_coll':eh_interp_raw_coll,
		'ev_interp_raw_coll':ev_interp_raw_coll,
		'b_inten_interp_coll':b_inten_interp_coll,
		'bl_interp_m_coll':bl_interp_m_coll,
		'slots_filled_coll':slots_filled_coll,

		'eh_interp_noncoll':eh_interp_noncoll,
		'ev_interp_noncoll':ev_interp_noncoll,
		'eh_interp_raw_noncoll':eh_interp_raw_noncoll,
		'ev_interp_raw_noncoll':ev_interp_raw_noncoll,
		'b_inten_interp_noncoll':b_inten_interp_noncoll,
		'bl_interp_m_noncoll':bl_interp_m_noncoll,
		'slots_filled_noncoll':slots_filled_noncoll,
		'time_range':time_range,
		'intensity_lifetime':lifetime,
		'tau_emit_h_coll'       : tau_emit_h_coll,
		'tau_emit_v_coll'       : tau_emit_v_coll,
		'tau_emit_h_noncoll'    : tau_emit_h_noncoll,
		'tau_emit_v_noncoll'    : tau_emit_v_noncoll,
		'init_emit_h_coll'      : init_emit_h_coll,
		'init_emit_v_coll'      : init_emit_v_coll,
		'init_emit_h_noncoll'   : init_emit_h_noncoll,
		'init_emit_v_noncoll'   : init_emit_v_noncoll,
		'tau_bl_coll'           : tau_bl_coll,
		'tau_bl_noncoll'        : tau_bl_noncoll,
		'init_bl_coll'          : init_bl_coll,
		'init_bl_noncoll'       : init_bl_noncoll,
		'tau_inten_coll'        : tau_inten_coll,
		'tau_inten_noncoll'     : tau_inten_noncoll,
		'init_inten_coll'       : init_inten_coll,
		'init_inten_noncoll'    : init_inten_noncoll,

		'tau_emit_h_coll_full'       : tau_emit_h_coll_full,
		'tau_emit_v_coll_full'       : tau_emit_v_coll_full,
		'tau_emit_h_noncoll_full'    : tau_emit_h_noncoll_full,
		'tau_emit_v_noncoll_full'    : tau_emit_v_noncoll_full,
		'init_emit_h_coll_full'      : init_emit_h_coll_full,
		'init_emit_v_coll_full'      : init_emit_v_coll_full,
		'init_emit_h_noncoll_full'   : init_emit_h_noncoll_full,
		'init_emit_v_noncoll_full'   : init_emit_v_noncoll_full,
		'tau_bl_coll_full'           : tau_bl_coll_full,
		'tau_bl_noncoll_full'        : tau_bl_noncoll_full,
		'init_bl_coll_full'          : init_bl_coll_full,
		'init_bl_noncoll_full'       : init_bl_noncoll_full,
		'tau_inten_coll_full'        : tau_inten_coll_full,
		'tau_inten_noncoll_full'     : tau_inten_noncoll_full,
		'init_inten_coll_full'       : init_inten_coll_full,
		'init_inten_noncoll_full'    : init_inten_noncoll_full,

		}

		if doSB:
			self.filln_StableBeamsDict     = dict_save

		if self.saveDict:
			#### Save dict for cycle data
			if doCycle:
				filename = self.fill_dir+self.Cycle_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('Saving dictionary for Cycle Data of fill {} into {}'.format(filln, filename ))
				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("Dictionary Cycle pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(self.filln_CycleDict, fid)
					else:
						warn("Dictionary Cycle pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(self.filln_CycleDict, fid)

			if doSB:
				filename = self.fill_dir+self.SB_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('Saving dictionary for SB Data of fill {} into {}'.format(filln, filename ))
				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("Dictionary SB pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(self.filln_StableBeamsDict, fid)
					else:
						warn("Dictionary SB pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(self.filln_StableBeamsDict, fid)


		if self.savePandas:
			if doCycle:
				filename = self.fill_dir+self.Cycle_filename.replace('.pkl.gz', '_df.pkl.gz')
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('Saving Pandas for Cycle Data of fill {} into {}'.format(filln, filename ))


				## Create Cycle DF

				# ## === Getting info
				# b1_inj_t_start      = self.filln_CycleDict['beam_1']['Injection']['t_start']
				# b1_inj_t_end        = self.filln_CycleDict['beam_1']['Injection']['t_end']
				# b1_inj_filled_slots = self.filln_CycleDict['beam_1']['Injection']['filled_slots']

				# b1_inj_at_start_time     	= self.filln_CycleDict['beam_1']['Injection']['at_start']['time_meas']
				# b1_inj_at_start_blength     = self.filln_CycleDict['beam_1']['Injection']['at_start']['blength']
				# b1_inj_at_start_brightness  = self.filln_CycleDict['beam_1']['Injection']['at_start']['brightness']
				# b1_inj_at_start_intensity   = self.filln_CycleDict['beam_1']['Injection']['at_start']['intensity']
				# b1_inj_at_start_emith     	= self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv']
				# b1_inj_at_start_emitv     	= self.filln_CycleDict['beam_1']['Injection']['at_start']['emith']


				# b1_inj_at_end_time     		= self.filln_CycleDict['beam_1']['Injection']['at_end']['time_meas']
				# b1_inj_at_end_blength     	= self.filln_CycleDict['beam_1']['Injection']['at_end']['blength']
				# b1_inj_at_end_brightness 	= self.filln_CycleDict['beam_1']['Injection']['at_end']['brightness']
				# b1_inj_at_end_intensity   	= self.filln_CycleDict['beam_1']['Injection']['at_end']['intensity']
				# b1_inj_at_end_emith     	= self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv']
				# b1_inj_at_end_emitv     	= self.filln_CycleDict['beam_1']['Injection']['at_end']['emith']


				

				# b2_inj_t_start      = self.filln_CycleDict['beam_2']['Injection']['t_start']
				# b2_inj_t_end        = self.filln_CycleDict['beam_2']['Injection']['t_end']
				# b2_inj_filled_slots = self.filln_CycleDict['beam_2']['Injection']['filled_slots']

				# b2_inj_at_start_time     	= self.filln_CycleDict['beam_2']['Injection']['at_start']['time_meas']
				# b2_inj_at_start_blength     = self.filln_CycleDict['beam_2']['Injection']['at_start']['blength']
				# b2_inj_at_start_brightness  = self.filln_CycleDict['beam_2']['Injection']['at_start']['brightness']
				# b2_inj_at_start_intensity   = self.filln_CycleDict['beam_2']['Injection']['at_start']['intensity']
				# b2_inj_at_start_emith     	= self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv']
				# b2_inj_at_start_emitv     	= self.filln_CycleDict['beam_2']['Injection']['at_start']['emith']

				# b2_inj_at_end_time     		= self.filln_CycleDict['beam_2']['Injection']['at_end']['time_meas']
				# b2_inj_at_end_blength     	= self.filln_CycleDict['beam_2']['Injection']['at_end']['blength']
				# b2_inj_at_end_brightness 	= self.filln_CycleDict['beam_2']['Injection']['at_end']['brightness']
				# b2_inj_at_end_intensity   	= self.filln_CycleDict['beam_2']['Injection']['at_end']['intensity']
				# b2_inj_at_end_emith     	= self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv']
				# b2_inj_at_end_emitv     	= self.filln_CycleDict['beam_2']['Injection']['at_end']['emith']
				

				# b1_he_t_start       		= self.filln_CycleDict['beam_1']['he_before_SB']['t_start']
				# b1_he_t_end         		= self.filln_CycleDict['beam_1']['he_before_SB']['t_end']
				# b1_he_filled_slots  		= self.filln_CycleDict['beam_1']['he_before_SB']['filled_slots']

				# b1_he_at_start_time     	= self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['time_meas']
				# b1_he_at_start_blength      = self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['blength']
				# b1_he_at_start_brightness  	= self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['brightness']
				# b1_he_at_start_intensity   	= self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['intensity']
				# b1_he_at_start_emith     	= self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv']
				# b1_he_at_start_emitv     	= self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith']


				# b1_he_at_end_time     		= self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['time_meas']
				# b1_he_at_end_blength     	= self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['blength']
				# b1_he_at_end_brightness 	= self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['brightness']
				# b1_he_at_end_intensity   	= self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['intensity']
				# b1_he_at_end_emith     		= self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv']
				# b1_he_at_end_emitv     		= self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith']

				

				# b2_he_t_start       		= self.filln_CycleDict['beam_2']['he_before_SB']['t_start']
				# b2_he_t_end         		= self.filln_CycleDict['beam_2']['he_before_SB']['t_end']
				# b2_he_filled_slots  		= self.filln_CycleDict['beam_2']['he_before_SB']['filled_slots']
				# b2_he_at_start_time     	= self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['time_meas']
				# b2_he_at_start_blength      = self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['blength']
				# b2_he_at_start_brightness  	= self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['brightness']
				# b2_he_at_start_intensity   	= self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['intensity']
				# b2_he_at_start_emith     	= self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv']
				# b2_he_at_start_emitv     	= self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith']


				# b2_he_at_end_time     		= self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['time_meas']
				# b2_he_at_end_blength     	= self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['blength']
				# b2_he_at_end_brightness 	= self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['brightness']
				# b2_he_at_end_intensity   	= self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['intensity']
				# b2_he_at_end_emith     		= self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv']
				# b2_he_at_end_emitv     		= self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith']

				######## INJECTION
				## beam 1 - injection start
				df_b1_inj_start = pd.DataFrame()
				df_b1_inj_start['time_meas']  = np.nan; df_b1_inj_start['time_meas'] = df_b1_inj_start['time_meas'].astype(object)
				df_b1_inj_start['blength']    = np.nan; df_b1_inj_start['blength']   = df_b1_inj_start['blength'].astype(object)
				df_b1_inj_start['brightness'] = np.nan; df_b1_inj_start['brightness']= df_b1_inj_start['brightness'].astype(object)
				df_b1_inj_start['intensity']  = np.nan; df_b1_inj_start['intensity'] = df_b1_inj_start['intensity'].astype(object)
				df_b1_inj_start['emitv']      = np.nan; df_b1_inj_start['emitv']     = df_b1_inj_start['emitv'].astype(object)
				df_b1_inj_start['emith']      = np.nan; df_b1_inj_start['emith']     = df_b1_inj_start['emith'].astype(object)
				df_b1_inj_start['filled_slots']      = np.nan; df_b1_inj_start['filled_slots']     = df_b1_inj_start['filled_slots'].astype(object)
				
				df_b1_inj_start.set_value(0, 'time_meas', self.filln_CycleDict['beam_1']['Injection']['at_start']['time_meas'])
				df_b1_inj_start.set_value(0, 'blength', self.filln_CycleDict['beam_1']['Injection']['at_start']['blength'])   
				df_b1_inj_start.set_value(0, 'brightness', self.filln_CycleDict['beam_1']['Injection']['at_start']['brightness'])
				df_b1_inj_start.set_value(0, 'intensity', self.filln_CycleDict['beam_1']['Injection']['at_start']['intensity']) 
				df_b1_inj_start.set_value(0, 'emitv', self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv'])     
				df_b1_inj_start.set_value(0, 'emith', self.filln_CycleDict['beam_1']['Injection']['at_start']['emith'])
				df_b1_inj_start.set_value(0, 'emith', self.filln_CycleDict['beam_1']['Injection']['filled_slots'])




				# df_b1_inj_start['time_meas']    	  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_start']['time_meas'])
				# df_b1_inj_start['blength']     		  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_start']['blength'])
				# df_b1_inj_start['brightness']         		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_start']['brightness'])
				# df_b1_inj_start['intensity']       	  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_start']['intensity'])
				# df_b1_inj_start['emitv']       		  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_start']['emitv'])
				# df_b1_inj_start['emith']       		  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_start']['emith'])
	
				df_b1_inj_start.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_1']['Injection']['t_end']*len(df_b1_inj_start))
				# df_b1_inj_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_inj_start['timestamp_end']))
				df_b1_inj_start.insert(0, 'timestamp_start', self.filln_CycleDict['beam_1']['Injection']['t_start']*len(df_b1_inj_start))
				# df_b1_inj_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_inj_start['timestamp_start']))
				# df_b1_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_start['time_meas']))
				df_b1_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b1_inj_start))
				df_b1_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_start))
				df_b1_inj_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_start))
				df_b1_inj_start.insert(0, 'fill'           , [filln]*len(df_b1_inj_start))
				# df_b1_inj_start['filled_slots']            = self.filln_CycleDict['beam_1']['Injection']['filled_slots']*len(df_b1_inj_start)


				## beam 1 - injection end
				df_b1_inj_end = pd.DataFrame()

				df_b1_inj_end['time_meas']  = np.nan ; df_b1_inj_end['time_meas']  = df_b1_inj_end['time_meas'].astype(object)  
				df_b1_inj_end['blength']    = np.nan ; df_b1_inj_end['blength']    = df_b1_inj_end['blength'].astype(object)    
				df_b1_inj_end['brightness'] = np.nan ; df_b1_inj_end['brightness'] = df_b1_inj_end['brightness'].astype(object) 
				df_b1_inj_end['intensity']  = np.nan ; df_b1_inj_end['intensity']  = df_b1_inj_end['intensity'].astype(object)  
				df_b1_inj_end['emitv']      = np.nan ; df_b1_inj_end['emitv']      = df_b1_inj_end['emitv'].astype(object)      
				df_b1_inj_end['emith']      = np.nan ; df_b1_inj_end['emith']      = df_b1_inj_end['emith'].astype(object)   
				df_b1_inj_end['filled_slots']      = np.nan ; df_b1_inj_end['filled_slots']      = df_b1_inj_end['filled_slots'].astype(object)   

				df_b1_inj_end.set_value(0, 'time_meas', self.filln_CycleDict['beam_1']['Injection']['at_end']['time_meas'])
				df_b1_inj_end.set_value(0, 'blength', self.filln_CycleDict['beam_1']['Injection']['at_end']['blength'])   
				df_b1_inj_end.set_value(0, 'brightness', self.filln_CycleDict['beam_1']['Injection']['at_end']['brightness'])
				df_b1_inj_end.set_value(0, 'intensity', self.filln_CycleDict['beam_1']['Injection']['at_end']['intensity']) 
				df_b1_inj_end.set_value(0, 'emitv', self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv'])     
				df_b1_inj_end.set_value(0, 'emith', self.filln_CycleDict['beam_1']['Injection']['at_end']['emith'])        
				df_b1_inj_end.set_value(0, 'filled_slots', self.filln_CycleDict['beam_1']['Injection']['filled_slots'])        

				# df_b1_inj_end['time_meas']    	  			= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_end']['time_meas'])
				# df_b1_inj_end['blength']     		  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_end']['blength'])
				# df_b1_inj_end['brightness']         		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_end']['brightness'])
				# df_b1_inj_end['intensity']       	  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_end']['intensity'])
				# df_b1_inj_end['emitv']       		  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_end']['emitv'])
				# df_b1_inj_end['emith']       		  		= pd.Series(self.filln_CycleDict['beam_1']['Injection']['at_end']['emith'])
				


				df_b1_inj_end.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_1']['Injection']['t_end']*len(df_b1_inj_end))
				# df_b1_inj_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_inj_end['timestamp_end']))
				df_b1_inj_end.insert(0, 'timestamp_start', self.filln_CycleDict['beam_1']['Injection']['t_start']*len(df_b1_inj_end))
				# df_b1_inj_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_inj_end['timestamp_start']))
				# df_b1_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_end['time_meas']))
				df_b1_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b1_inj_end))
				df_b1_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_end))
				df_b1_inj_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_end))
				df_b1_inj_end.insert(0, 'fill'           , [filln]*len(df_b1_inj_end))
				# df_b1_inj_end['filled_slots']            = self.filln_CycleDict['beam_1']['Injection']['filled_slots']*len(df_b1_inj_end)



				## beam 2 - injection start
				df_b2_inj_start = pd.DataFrame()

				df_b2_inj_start['time_meas']  = np.nan ; df_b2_inj_start['time_meas']  = df_b2_inj_start['time_meas'].astype(object)     
				df_b2_inj_start['blength']    = np.nan ; df_b2_inj_start['blength']    = df_b2_inj_start['blength'].astype(object)       
				df_b2_inj_start['brightness'] = np.nan ; df_b2_inj_start['brightness'] = df_b2_inj_start['brightness'].astype(object)    
				df_b2_inj_start['intensity']  = np.nan ; df_b2_inj_start['intensity']  = df_b2_inj_start['intensity'].astype(object)     
				df_b2_inj_start['emitv']      = np.nan ; df_b2_inj_start['emitv']      = df_b2_inj_start['emitv'].astype(object)         
				df_b2_inj_start['emith']      = np.nan ; df_b2_inj_start['emith']      = df_b2_inj_start['emith'].astype(object)         
				df_b2_inj_start['filled_slots']      = np.nan ; df_b2_inj_start['filled_slots']      = df_b2_inj_start['filled_slots'].astype(object)         

				df_b2_inj_start.set_value(0, 'time_meas', self.filln_CycleDict['beam_2']['Injection']['at_start']['time_meas'])
				df_b2_inj_start.set_value(0, 'blength', self.filln_CycleDict['beam_2']['Injection']['at_start']['blength'])   
				df_b2_inj_start.set_value(0, 'brightness', self.filln_CycleDict['beam_2']['Injection']['at_start']['brightness'])
				df_b2_inj_start.set_value(0, 'intensity', self.filln_CycleDict['beam_2']['Injection']['at_start']['intensity']) 
				df_b2_inj_start.set_value(0, 'emitv', self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv'])     
				df_b2_inj_start.set_value(0, 'emith', self.filln_CycleDict['beam_2']['Injection']['at_start']['emith'])        
				df_b2_inj_start.set_value(0, 'filled_slots', self.filln_CycleDict['beam_2']['Injection']['filled_slots'])        


				# df_b2_inj_start['time_meas']    	  = pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_start']['time_meas'])
				# df_b2_inj_start['blength']     		  = pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_start']['blength'])
				# df_b2_inj_start['brightness']         = pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_start']['brightness'])
				# df_b2_inj_start['intensity']       	  = pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_start']['intensity'])
				# df_b2_inj_start['emitv']       		  = pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_start']['emitv'])
				# df_b2_inj_start['emith']       		  = pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_start']['emith'])

				df_b2_inj_start.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_2']['Injection']['t_end']*len(df_b2_inj_start))
				# df_b2_inj_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_inj_start['timestamp_end']))
				df_b2_inj_start.insert(0, 'timestamp_start', self.filln_CycleDict['beam_2']['Injection']['t_start']*len(df_b2_inj_start))
				# df_b2_inj_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_inj_start['timestamp_start']))
				# df_b2_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_start['time_meas']))
				df_b2_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b2_inj_start))
				df_b2_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_start))
				df_b2_inj_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_start))
				df_b2_inj_start.insert(0, 'fill'           , [filln]*len(df_b2_inj_start))
				# df_b1_inj_start['filled_slots']            = self.filln_CycleDict['beam_2']['Injection']['filled_slots']*len(df_b2_inj_start)


				## beam 2 - injection end
				df_b2_inj_end = pd.DataFrame()
				df_b2_inj_end['time_meas']  = np.nan ; df_b2_inj_end['time_meas']   = df_b2_inj_end['time_meas'].astype(object)           
				df_b2_inj_end['blength']    = np.nan ; df_b2_inj_end['blength']     = df_b2_inj_end['blength'].astype(object)             
				df_b2_inj_end['brightness'] = np.nan ; df_b2_inj_end['brightness']  = df_b2_inj_end['brightness'].astype(object)          
				df_b2_inj_end['intensity']  = np.nan ; df_b2_inj_end['intensity']   = df_b2_inj_end['intensity'].astype(object)           
				df_b2_inj_end['emitv']      = np.nan ; df_b2_inj_end['emitv']       = df_b2_inj_end['emitv'].astype(object)               
				df_b2_inj_end['emith']      = np.nan ; df_b2_inj_end['emith']       = df_b2_inj_end['emith'].astype(object)      
				df_b2_inj_end['filled_slots']      = np.nan ; df_b2_inj_end['filled_slots']       = df_b2_inj_end['filled_slots'].astype(object)      

				df_b2_inj_start.set_value(0, 'time_meas', self.filln_CycleDict['beam_2']['Injection']['at_end']['time_meas'])
				df_b2_inj_start.set_value(0, 'blength', self.filln_CycleDict['beam_2']['Injection']['at_end']['blength'])   
				df_b2_inj_start.set_value(0, 'brightness', self.filln_CycleDict['beam_2']['Injection']['at_end']['brightness'])
				df_b2_inj_start.set_value(0, 'intensity', self.filln_CycleDict['beam_2']['Injection']['at_end']['intensity']) 
				df_b2_inj_start.set_value(0, 'emitv', self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv'])     
				df_b2_inj_start.set_value(0, 'emith', self.filln_CycleDict['beam_2']['Injection']['at_end']['emith'])                 
				df_b2_inj_start.set_value(0, 'filled_slots', self.filln_CycleDict['beam_2']['Injection']['filled_slots'])                 



				# df_b2_inj_end['time_meas']    	  		= pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_end']['time_meas'])
				# df_b2_inj_end['blength']     		  	= pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_end']['blength'])
				# df_b2_inj_end['brightness']         	= pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_end']['brightness'])
				# df_b2_inj_end['intensity']       	  	= pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_end']['intensity'])
				# df_b2_inj_end['emitv']       		  	= pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_end']['emitv'])
				# df_b2_inj_end['emith']       		  	= pd.Series(self.filln_CycleDict['beam_2']['Injection']['at_end']['emith'])
				df_b2_inj_end.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_2']['Injection']['t_end']*len(df_b2_inj_end))
				# df_b2_inj_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_inj_end['timestamp_end']))
				df_b2_inj_end.insert(0, 'timestamp_start', self.filln_CycleDict['beam_2']['Injection']['t_start']*len(df_b2_inj_end))
				# df_b2_inj_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_inj_end['timestamp_start']))
				# df_b2_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_end['time_meas']))
				df_b2_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b2_inj_end))
				df_b2_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_end))
				df_b2_inj_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_end))
				df_b2_inj_end.insert(0, 'fill'           , [filln]*len(df_b2_inj_end))
				# df_b2_inj_end['filled_slots']            = self.filln_CycleDict['beam_2']['Injection']['filled_slots']*len(df_b1_inj_end)




				######## FLATTOP
				## beam 1 - FLATTOP start
				df_b1_he_start = pd.DataFrame()
				df_b1_he_start['time_meas']  = np.nan ; df_b1_he_start['time_meas']  = df_b1_he_start['time_meas'].astype(object) 
				df_b1_he_start['blength']    = np.nan ; df_b1_he_start['blength']    = df_b1_he_start['blength'].astype(object)   
				df_b1_he_start['brightness'] = np.nan ; df_b1_he_start['brightness'] = df_b1_he_start['brightness'].astype(object)
				df_b1_he_start['intensity']  = np.nan ; df_b1_he_start['intensity']  = df_b1_he_start['intensity'].astype(object) 
				df_b1_he_start['emitv']      = np.nan ; df_b1_he_start['emitv']      = df_b1_he_start['emitv'].astype(object)     
				df_b1_he_start['emith']      = np.nan ; df_b1_he_start['emith']      = df_b1_he_start['emith'].astype(object)  
				df_b1_he_start['filled_slots']      = np.nan ; df_b1_he_start['filled_slots']      = df_b1_he_start['filled_slots'].astype(object)  

				df_b1_he_start.set_value(0, 'time_meas', self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['time_meas'])
				df_b1_he_start.set_value(0, 'blength', self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['blength'])   
				df_b1_he_start.set_value(0, 'brightness', self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['brightness'])
				df_b1_he_start.set_value(0, 'intensity', self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['intensity']) 
				df_b1_he_start.set_value(0, 'emitv', self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv'])     
				df_b1_he_start.set_value(0, 'emith', self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith'])				   
				df_b1_he_start.set_value(0, 'filled_slots', self.filln_CycleDict['beam_1']['he_before_SB']['filled_slots'])				   


				# df_b1_he_start['time_meas']    	  		= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['time_meas'])
				# df_b1_he_start['blength']     		  	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['blength'])
				# df_b1_he_start['brightness']         	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['brightness'])
				# df_b1_he_start['intensity']       	  	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['intensity'])
				# df_b1_he_start['emitv']       		  	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emitv'])
				# df_b1_he_start['emith']       		  	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_start']['emith'])
				df_b1_he_start.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_1']['he_before_SB']['t_end']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_start['timestamp_end']))
				df_b1_he_start.insert(0, 'timestamp_start', self.filln_CycleDict['beam_1']['he_before_SB']['t_start']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_start['timestamp_start']))
				# df_b1_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
				df_b1_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'fill'           , [filln]*len(df_b1_he_start))
				# df_b1_he_start['filled_slots']            = self.filln_CycleDict['beam_1']['he_before_SB']['filled_slots']*len(df_b1_he_start)


				## beam 1 - FLATTOP end
				df_b1_he_end = pd.DataFrame()
				df_b1_he_end['time_meas']  = np.nan ; df_b1_he_end['time_meas']   = df_b1_he_end['time_meas'].astype(object)       
				df_b1_he_end['blength']    = np.nan ; df_b1_he_end['blength']     = df_b1_he_end['blength'].astype(object)         
				df_b1_he_end['brightness'] = np.nan ; df_b1_he_end['brightness']  = df_b1_he_end['brightness'].astype(object)      
				df_b1_he_end['intensity']  = np.nan ; df_b1_he_end['intensity']   = df_b1_he_end['intensity'].astype(object)       
				df_b1_he_end['emitv']      = np.nan ; df_b1_he_end['emitv']       = df_b1_he_end['emitv'].astype(object)           
				df_b1_he_end['emith']      = np.nan ; df_b1_he_end['emith']       = df_b1_he_end['emith'].astype(object)     
				df_b1_he_end['filled_slots']      = np.nan ; df_b1_he_end['filled_slots']       = df_b1_he_end['filled_slots'].astype(object)     

				df_b1_he_end.set_value(0, 'time_meas', self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['time_meas'])
				df_b1_he_end.set_value(0, 'blength', self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['blength'])   
				df_b1_he_end.set_value(0, 'brightness', self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['brightness'])
				df_b1_he_end.set_value(0, 'intensity', self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['intensity']) 
				df_b1_he_end.set_value(0, 'emitv', self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv'])     
				df_b1_he_end.set_value(0, 'emith', self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith'])      
				df_b1_he_end.set_value(0, 'filled_slots', self.filln_CycleDict['beam_1']['he_before_SB']['filled_slots'])      



				# df_b1_he_end['time_meas']    	  		= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['time_meas'])
				# df_b1_he_end['blength']     			= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['blength'])
				# df_b1_he_end['brightness']         		= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['brightness'])
				# df_b1_he_end['intensity']       	  	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['intensity'])
				# df_b1_he_end['emitv']       		  	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emitv'])
				# df_b1_he_end['emith']       		  	= pd.Series(self.filln_CycleDict['beam_1']['he_before_SB']['at_end']['emith'])
				df_b1_he_end.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_1']['he_before_SB']['t_end']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_end['timestamp_end']))
				df_b1_he_end.insert(0, 'timestamp_start', self.filln_CycleDict['beam_1']['he_before_SB']['t_start']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_end['timestamp_start']))
				# df_b1_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_end['time_meas']))
				df_b1_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'fill'           , [filln]*len(df_b1_he_end))
				# df_b1_he_end['filled_slots']            = self.filln_CycleDict['beam_1']['he_before_SB']['filled_slots']*len(df_b1_he_end)

				## beam 2- FLATTOP start
				df_b2_he_start = pd.DataFrame()

				df_b2_he_start['time_meas']  = np.nan ;	df_b2_he_start['time_meas']    	 = df_b2_he_start['time_meas'].astype(object)         	
				df_b2_he_start['blength']   = np.nan ;	df_b2_he_start['blength']     	 = df_b2_he_start['blength'].astype(object)          	
				df_b2_he_start['brightness'] = np.nan ;	df_b2_he_start['brightness']     = df_b2_he_start['brightness'].astype(object)         
				df_b2_he_start['intensity']  = np.nan ;	df_b2_he_start['intensity']      = df_b2_he_start['intensity'].astype(object)          
				df_b2_he_start['emitv']     = np.nan ;	df_b2_he_start['emitv']       	 = df_b2_he_start['emitv'].astype(object)            	
				df_b2_he_start['emith']     = np.nan ;	df_b2_he_start['emith']       	 = df_b2_he_start['emith'].astype(object)       
				df_b2_he_start['filled_slots']     = np.nan ;	df_b2_he_start['filled_slots']       	 = df_b2_he_start['filled_slots'].astype(object)       


				df_b2_he_start.set_value(0, 'time_meas', self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['time_meas'])
				df_b2_he_start.set_value(0, 'blength', self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['blength'])   
				df_b2_he_start.set_value(0, 'brightness', self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['brightness'])
				df_b2_he_start.set_value(0, 'intensity', self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['intensity']) 
				df_b2_he_start.set_value(0, 'emitv', self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv'])     
				df_b2_he_start.set_value(0, 'emith', self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith'])       	
				df_b2_he_start.set_value(0, 'filled_slots', self.filln_CycleDict['beam_2']['he_before_SB']['filled_slots'])       	

				# df_b2_he_start['time_meas']    	  		= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['time_meas'])
				# df_b2_he_start['blength']     		  	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['blength'])
				# df_b2_he_start['brightness']         	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['brightness'])
				# df_b2_he_start['intensity']       	  	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['intensity'])
				# df_b2_he_start['emitv']       		  	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emitv'])
				# df_b2_he_start['emith']       		  	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_start']['emith'])
				df_b2_he_start.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_2']['he_before_SB']['t_end']*len(df_b2_he_start))
				# df_b2_he_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_he_start['timestamp_end']))
				df_b2_he_start.insert(0, 'timestamp_start', self.filln_CycleDict['beam_2']['he_before_SB']['t_start']*len(df_b2_he_start))
				# df_b2_he_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_he_start['timestamp_start']))
				# df_b2_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_start['time_meas']))
				df_b2_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'fill'           , [filln]*len(df_b2_he_start))
				# df_b2_he_start['filled_slots']            = self.filln_CycleDict['beam_2']['he_before_SB']['filled_slots']*len(df_b2_he_start)


				## beam 2 - FLATTOP end
				df_b2_he_end = pd.DataFrame()
				df_b2_he_end['time_meas'] = np.nan; df_b2_he_end['time_meas']  = df_b2_he_end['time_meas'].astype(object)        
				df_b2_he_end['blength']   = np.nan; df_b2_he_end['blength']    = df_b2_he_end['blength'].astype(object)          
				df_b2_he_end['brightness']= np.nan; df_b2_he_end['brightness'] = df_b2_he_end['brightness'].astype(object)       
				df_b2_he_end['intensity'] = np.nan; df_b2_he_end['intensity']  = df_b2_he_end['intensity'].astype(object)        
				df_b2_he_end['emitv']     = np.nan; df_b2_he_end['emitv']      = df_b2_he_end['emitv'].astype(object)            
				df_b2_he_end['emith']     = np.nan; df_b2_he_end['emith']      = df_b2_he_end['emith'].astype(object)   
				df_b2_he_end['filled_slots']     = np.nan; df_b2_he_end['filled_slots']      = df_b2_he_end['filled_slots'].astype(object)   

				df_b2_he_end.set_value(0, 'time_meas', self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['time_meas'])
				df_b2_he_end.set_value(0, 'blength', self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['blength'])   
				df_b2_he_end.set_value(0, 'brightness', self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['brightness'])
				df_b2_he_end.set_value(0, 'intensity', self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['intensity']) 
				df_b2_he_end.set_value(0, 'emitv', self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv'])     
				df_b2_he_end.set_value(0, 'emith', self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith'])          
				df_b2_he_end.set_value(0, 'filled_slots', self.filln_CycleDict['beam_2']['he_before_SB']['filled_slots'])          



				# df_b2_he_end['time_meas']    	  		= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['time_meas'])
				# df_b2_he_end['blength']     			= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['blength'])
				# df_b2_he_end['brightness']         		= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['brightness'])
				# df_b2_he_end['intensity']       	  	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['intensity'])
				# df_b2_he_end['emitv']       		  	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emitv'])
				# df_b2_he_end['emith']       		  	= pd.Series(self.filln_CycleDict['beam_2']['he_before_SB']['at_end']['emith'])
				df_b2_he_end.insert(0, 'timestamp_end'  , self.filln_CycleDict['beam_2']['he_before_SB']['t_end']*len(df_b2_he_end))
				# df_b2_he_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_he_end['timestamp_end']))
				df_b2_he_end.insert(0, 'timestamp_start', self.filln_CycleDict['beam_2']['he_before_SB']['t_start']*len(df_b2_he_end))
				# df_b2_he_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_he_end['timestamp_start']))
				# df_b2_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_end['time_meas']))
				df_b2_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'fill'           , [filln]*len(df_b2_he_end))
				# df_b2_he_end['filled_slots']            = self.filln_CycleDict['beam_2']['he_before_SB']['filled_slots']*len(df_b2_he_end)




				# ######## HE before SB
				# ## beam 1 - heection start
				# df_b1_he_start = pd.DataFrame.from_dict(b1_he_at_start, orient='columns')

				# df_b1_he_start.insert(0, 'timestamp_end'  , [b1_he_t_end]*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_start['timestamp_end']))
				# df_b1_he_start.insert(0, 'timestamp_start', [b1_he_t_start]*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_start['timestamp_start']))
				# df_b1_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
				# df_b1_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'fill'           , [filln]*len(df_b1_he_start))
				# df_b1_he_start['filled_slots']            = [b1_he_filled_slots]*len(df_b1_he_start)

				# ## beam 1 - heection end
				# df_b1_he_end = pd.DataFrame.from_dict(b1_he_at_end, orient='columns')

				# df_b1_he_end.insert(0, 'timestamp_end'  , [b1_he_t_end]*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_end['timestamp_end']))
				# df_b1_he_end.insert(0, 'timestamp_start', [b1_he_t_start]*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_end['timestamp_start']))
				# df_b1_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_end['time_meas']))
				# df_b1_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'fill'           , [filln]*len(df_b1_he_end))
				# df_b1_he_end['filled_slots']            = [b1_he_filled_slots]*len(df_b1_he_end)



				# df_b1_inj_start = pd.DataFrame.from_dict(b1_inj_at_start, orient='columns')

				# df_b1_inj_start.insert(0, 'timestamp_end'  , [b1_inj_t_end]*len(df_b1_inj_start))
				# df_b1_inj_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_inj_start['timestamp_end']))
				# df_b1_inj_start.insert(0, 'timestamp_start', [b1_inj_t_start]*len(df_b1_inj_start))
				# df_b1_inj_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_inj_start['timestamp_start']))
				# df_b1_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_start['time_meas']))
				# df_b1_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b1_inj_start))
				# df_b1_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_start))
				# df_b1_inj_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_start))
				# df_b1_inj_start.insert(0, 'fill'           , [filln]*len(df_b1_inj_start))
				# df_b1_inj_start['filled_slots']            = [b1_inj_filled_slots]*len(df_b1_inj_start)

				# ## beam 1 - injection end
				# df_b1_inj_end = pd.DataFrame.from_dict(b1_inj_at_end, orient='columns')

				# df_b1_inj_end.insert(0, 'timestamp_end'  , [b1_inj_t_end]*len(df_b1_inj_end))
				# df_b1_inj_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_inj_end['timestamp_end']))
				# df_b1_inj_end.insert(0, 'timestamp_start', [b1_inj_t_start]*len(df_b1_inj_end))
				# df_b1_inj_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_inj_end['timestamp_start']))
				# df_b1_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_end['time_meas']))
				# df_b1_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b1_inj_end))
				# df_b1_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_end))
				# df_b1_inj_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_end))
				# df_b1_inj_end.insert(0, 'fill'           , [filln]*len(df_b1_inj_end))
				# df_b1_inj_end['filled_slots']            = [b1_inj_filled_slots]*len(df_b1_inj_end)


				# ## beam 2 - injection start
				# df_b2_inj_start = pd.DataFrame.from_dict(b2_inj_at_start, orient='columns')

				# df_b2_inj_start.insert(0, 'timestamp_end'  , [b2_inj_t_end]*len(df_b2_inj_start))
				# df_b2_inj_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_inj_start['timestamp_end']))
				# df_b2_inj_start.insert(0, 'timestamp_start', [b2_inj_t_start]*len(df_b2_inj_start))
				# df_b2_inj_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_inj_start['timestamp_start']))
				# df_b2_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_start['time_meas']))
				# df_b2_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b2_inj_start))
				# df_b2_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_start))
				# df_b2_inj_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_start))
				# df_b2_inj_start.insert(0, 'fill'           , [filln]*len(df_b2_inj_start))
				# df_b2_inj_start['filled_slots']            = [b2_inj_filled_slots]*len(df_b2_inj_start)

				# ## beam 2 - injection end
				# df_b2_inj_end = pd.DataFrame.from_dict(b2_inj_at_end, orient='columns')

				# df_b2_inj_end.insert(0, 'timestamp_end'  , [b2_inj_t_end]*len(df_b2_inj_end))
				# df_b2_inj_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_inj_end['timestamp_end']))
				# df_b2_inj_end.insert(0, 'timestamp_start', [b2_inj_t_start]*len(df_b2_inj_end))
				# df_b2_inj_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_inj_end['timestamp_start']))
				# df_b2_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_end['time_meas']))
				# df_b2_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b2_inj_end))
				# df_b2_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_end))
				# df_b2_inj_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_end))
				# df_b2_inj_end.insert(0, 'fill'           , [filln]*len(df_b2_inj_end))
				# df_b2_inj_end['filled_slots']            = [b2_inj_filled_slots]*len(df_b2_inj_end)


				# ######## HE before SB
				# ## beam 1 - heection start
				# df_b1_he_start = pd.DataFrame.from_dict(b1_he_at_start, orient='columns')

				# df_b1_he_start.insert(0, 'timestamp_end'  , [b1_he_t_end]*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_start['timestamp_end']))
				# df_b1_he_start.insert(0, 'timestamp_start', [b1_he_t_start]*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_start['timestamp_start']))
				# df_b1_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
				# df_b1_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_start))
				# df_b1_he_start.insert(0, 'fill'           , [filln]*len(df_b1_he_start))
				# df_b1_he_start['filled_slots']            = [b1_he_filled_slots]*len(df_b1_he_start)

				# ## beam 1 - heection end
				# df_b1_he_end = pd.DataFrame.from_dict(b1_he_at_end, orient='columns')

				# df_b1_he_end.insert(0, 'timestamp_end'  , [b1_he_t_end]*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_end['timestamp_end']))
				# df_b1_he_end.insert(0, 'timestamp_start', [b1_he_t_start]*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_end['timestamp_start']))
				# df_b1_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_end['time_meas']))
				# df_b1_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_end))
				# df_b1_he_end.insert(0, 'fill'           , [filln]*len(df_b1_he_end))
				# df_b1_he_end['filled_slots']            = [b1_he_filled_slots]*len(df_b1_he_end)


				# ## beam 2 - heection start
				# df_b2_he_start = pd.DataFrame.from_dict(b2_he_at_start, orient='columns')

				# df_b2_he_start.insert(0, 'timestamp_end'  , [b2_he_t_end]*len(df_b2_he_start))
				# df_b2_he_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_he_start['timestamp_end']))
				# df_b2_he_start.insert(0, 'timestamp_start', [b2_he_t_start]*len(df_b2_he_start))
				# df_b2_he_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_he_start['timestamp_start']))
				# df_b2_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_start['time_meas']))
				# df_b2_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b2_he_start))
				# df_b2_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_start))
				# df_b2_he_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_start))
				# df_b2_he_start.insert(0, 'fill'           , [filln]*len(df_b2_he_start))
				# df_b2_he_start['filled_slots']            = [b2_he_filled_slots]*len(df_b2_he_start)

				# ## beam 2 - heection end
				# df_b2_he_end = pd.DataFrame.from_dict(b2_he_at_end, orient='columns')

				# df_b2_he_end.insert(0, 'timestamp_end'  , [b2_he_t_end]*len(df_b2_he_end))
				# df_b2_he_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_he_end['timestamp_end']))
				# df_b2_he_end.insert(0, 'timestamp_start', [b2_he_t_start]*len(df_b2_he_end))
				# df_b2_he_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_he_end['timestamp_start']))
				# df_b2_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_end['time_meas']))
				# df_b2_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b2_he_end))
				# df_b2_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_end))
				# df_b2_he_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_end))
				# df_b2_he_end.insert(0, 'fill'           , [filln]*len(df_b2_he_end))
				# df_b2_he_end['filled_slots']            = [b2_he_filled_slots]*len(df_b2_he_end)

				total_cycle = df_b1_inj_start.append(df_b1_inj_end).append(df_b2_inj_start).append(df_b2_inj_end).append(df_b1_he_start).append(df_b1_he_end).append(df_b2_he_start).append(df_b2_he_end)

				total_cycle = total_cycle.set_index(['fill','beam','cycle'], drop=False)


				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("Pandas Cycle pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(total_cycle, fid)
					else:
						warn("Pandas Cycle pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(total_cycle, fid)

			if doSB:
				filename = self.fill_dir+self.SB_filename.replace('.pkl.gz', '_df.pkl.gz')
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('Saving Pandas for SB Data of fill {} into {}'.format(filln, filename ))

				#### === STABLE BEAMS Get info
				## Beam 1
				df_b1 = pd.DataFrame()
				df_b1['bunch_intensity_coll']       = pd.Series(self.filln_StableBeamsDict['b_inten_interp_coll'][1].tolist())
				df_b1['bunch_intensity_noncoll']    = pd.Series(self.filln_StableBeamsDict['b_inten_interp_noncoll'][1].tolist())

				df_b1['eh_interp_coll']             = pd.Series(self.filln_StableBeamsDict['eh_interp_coll'][1].tolist())
				df_b1['eh_interp_noncoll']          = pd.Series(self.filln_StableBeamsDict['eh_interp_noncoll'][1].tolist())

				df_b1['eh_raw_coll']                = pd.Series(self.filln_StableBeamsDict['eh_interp_raw_coll'][1].tolist())
				df_b1['eh_raw_noncoll']             = pd.Series(self.filln_StableBeamsDict['eh_interp_raw_noncoll'][1].tolist())

				df_b1['ev_interp_coll']             = pd.Series(self.filln_StableBeamsDict['ev_interp_coll'][1].tolist())
				df_b1['ev_interp_noncoll']          = pd.Series(self.filln_StableBeamsDict['ev_interp_noncoll'][1].tolist())

				df_b1['ev_raw_coll']                = pd.Series(self.filln_StableBeamsDict['ev_interp_raw_coll'][1].tolist())
				df_b1['ev_raw_noncoll']             = pd.Series(self.filln_StableBeamsDict['ev_interp_raw_noncoll'][1].tolist())

				df_b1['bunch_length_coll']          = pd.Series(self.filln_StableBeamsDict['bl_interp_m_coll'][1].tolist())
				df_b1['bunch_length_noncoll']       = pd.Series(self.filln_StableBeamsDict['bl_interp_m_noncoll'][1].tolist())

				df_b1['intensity_lifetime']         = pd.Series(self.filln_StableBeamsDict['intensity_lifetime'][1].tolist())

				df_b1['tau_emit_h_coll_full']               = pd.Series(self.filln_StableBeamsDict['tau_emit_h_coll_full'][1])
				df_b1['tau_emit_v_coll_full']               = pd.Series(self.filln_StableBeamsDict['tau_emit_v_coll_full'][1])
				df_b1['tau_emit_h_noncoll_full']            = pd.Series(self.filln_StableBeamsDict['tau_emit_h_noncoll_full'][1])
				df_b1['tau_emit_v_noncoll_full']            = pd.Series(self.filln_StableBeamsDict['tau_emit_v_noncoll_full'][1])
				df_b1['init_emit_h_coll_full']              = pd.Series(self.filln_StableBeamsDict['init_emit_h_coll_full'][1])
				df_b1['init_emit_v_coll_full']              = pd.Series(self.filln_StableBeamsDict['init_emit_v_coll_full'][1])
				df_b1['init_emit_h_noncoll_full']           = pd.Series(self.filln_StableBeamsDict['init_emit_h_noncoll_full'][1])
				df_b1['init_emit_v_noncoll_full']           = pd.Series(self.filln_StableBeamsDict['init_emit_v_noncoll_full'][1])
				df_b1['tau_bl_coll_full']                   = pd.Series(self.filln_StableBeamsDict['tau_bl_coll_full'][1])
				df_b1['tau_bl_noncoll_full']                = pd.Series(self.filln_StableBeamsDict['tau_bl_noncoll_full'][1])
				df_b1['init_bl_coll_full']                  = pd.Series(self.filln_StableBeamsDict['init_bl_coll_full'][1])
				df_b1['init_bl_noncoll_full']               = pd.Series(self.filln_StableBeamsDict['init_bl_noncoll_full'][1])
				df_b1['tau_inten_coll_full']                = pd.Series(self.filln_StableBeamsDict['tau_inten_coll_full'][1])
				df_b1['tau_inten_noncoll_full']             = pd.Series(self.filln_StableBeamsDict['tau_inten_noncoll_full'][1])
				df_b1['init_inten_coll_full']               = pd.Series(self.filln_StableBeamsDict['init_inten_coll_full'][1])
				df_b1['init_inten_noncoll_full']            = pd.Series(self.filln_StableBeamsDict['init_inten_noncoll_full'][1])

				df_b1['tau_emit_h_coll']               = pd.Series(self.filln_StableBeamsDict['tau_emit_h_coll'][1])
				df_b1['tau_emit_v_coll']               = pd.Series(self.filln_StableBeamsDict['tau_emit_v_coll'][1])
				df_b1['tau_emit_h_noncoll']            = pd.Series(self.filln_StableBeamsDict['tau_emit_h_noncoll'][1])
				df_b1['tau_emit_v_noncoll']            = pd.Series(self.filln_StableBeamsDict['tau_emit_v_noncoll'][1])
				df_b1['init_emit_h_coll']              = pd.Series(self.filln_StableBeamsDict['init_emit_h_coll'][1])
				df_b1['init_emit_v_coll']              = pd.Series(self.filln_StableBeamsDict['init_emit_v_coll'][1])
				df_b1['init_emit_h_noncoll']           = pd.Series(self.filln_StableBeamsDict['init_emit_h_noncoll'][1])
				df_b1['init_emit_v_noncoll']           = pd.Series(self.filln_StableBeamsDict['init_emit_v_noncoll'][1])
				df_b1['tau_bl_coll']                   = pd.Series(self.filln_StableBeamsDict['tau_bl_coll'][1])
				df_b1['tau_bl_noncoll']                = pd.Series(self.filln_StableBeamsDict['tau_bl_noncoll'][1])
				df_b1['init_bl_coll']                  = pd.Series(self.filln_StableBeamsDict['init_bl_coll'][1])
				df_b1['init_bl_noncoll']               = pd.Series(self.filln_StableBeamsDict['init_bl_noncoll'][1])
				df_b1['tau_inten_coll']                = pd.Series(self.filln_StableBeamsDict['tau_inten_coll'][1])
				df_b1['tau_inten_noncoll']             = pd.Series(self.filln_StableBeamsDict['tau_inten_noncoll'][1])
				df_b1['init_inten_coll']               = pd.Series(self.filln_StableBeamsDict['init_inten_coll'][1])
				df_b1['init_inten_noncoll']            = pd.Series(self.filln_StableBeamsDict['init_inten_noncoll'][1])


				df_b1['slots_filled_coll']          = [self.filln_StableBeamsDict['slots_filled_coll'][1].tolist()] * len(df_b1)
				df_b1['slots_filled_noncoll']       = [self.filln_StableBeamsDict['slots_filled_noncoll'][1].tolist()]*len(df_b1)


				df_b1.insert(0, 'timestamp',        self.filln_StableBeamsDict['time_range'])
				df_b1.insert(0, 'date',             self.convertToLocalTime(df_b1['timestamp']))
				df_b1.insert(0, 'beam',             ["beam_1"]*len(df_b1))
				df_b1.insert(0, 'fill',             [filln]*len(df_b1))


				## Beam 2
				df_b2 = pd.DataFrame()
				df_b2['bunch_intensity_coll']       = pd.Series(self.filln_StableBeamsDict['b_inten_interp_coll'][2].tolist())
				df_b2['bunch_intensity_noncoll']    = pd.Series(self.filln_StableBeamsDict['b_inten_interp_noncoll'][2].tolist())

				df_b2['eh_interp_coll']             = pd.Series(self.filln_StableBeamsDict['eh_interp_coll'][2].tolist())
				df_b2['eh_interp_noncoll']          = pd.Series(self.filln_StableBeamsDict['eh_interp_noncoll'][2].tolist())

				df_b2['eh_raw_coll']                = pd.Series(self.filln_StableBeamsDict['eh_interp_raw_coll'][2].tolist())
				df_b2['eh_raw_noncoll']             = pd.Series(self.filln_StableBeamsDict['eh_interp_raw_noncoll'][2].tolist())

				df_b2['ev_interp_coll']             = pd.Series(self.filln_StableBeamsDict['ev_interp_coll'][2].tolist())
				df_b2['ev_interp_noncoll']          = pd.Series(self.filln_StableBeamsDict['ev_interp_noncoll'][2].tolist())

				df_b2['ev_raw_coll']                = pd.Series(self.filln_StableBeamsDict['ev_interp_raw_coll'][2].tolist())
				df_b2['ev_raw_noncoll']             = pd.Series(self.filln_StableBeamsDict['ev_interp_raw_noncoll'][2].tolist())

				df_b2['bunch_length_coll']          = pd.Series(self.filln_StableBeamsDict['bl_interp_m_coll'][2].tolist())
				df_b2['bunch_length_noncoll']       = pd.Series(self.filln_StableBeamsDict['bl_interp_m_noncoll'][2].tolist())

				df_b2['slots_filled_coll']          = [self.filln_StableBeamsDict['slots_filled_coll'][2].tolist()] * len(df_b2)
				df_b2['slots_filled_noncoll']       = [self.filln_StableBeamsDict['slots_filled_noncoll'][2].tolist()]*len(df_b2)

				df_b2['intensity_lifetime']         = pd.Series(self.filln_StableBeamsDict['intensity_lifetime'][2].tolist())

				df_b2['tau_emit_h_coll_full']               = pd.Series(self.filln_StableBeamsDict['tau_emit_h_coll_full'][2])
				df_b2['tau_emit_v_coll_full']               = pd.Series(self.filln_StableBeamsDict['tau_emit_v_coll_full'][2])
				df_b2['tau_emit_h_noncoll_full']            = pd.Series(self.filln_StableBeamsDict['tau_emit_h_noncoll_full'][2])
				df_b2['tau_emit_v_noncoll_full']            = pd.Series(self.filln_StableBeamsDict['tau_emit_v_noncoll_full'][2])
				df_b2['init_emit_h_coll_full']              = pd.Series(self.filln_StableBeamsDict['init_emit_h_coll_full'][2])
				df_b2['init_emit_v_coll_full']              = pd.Series(self.filln_StableBeamsDict['init_emit_v_coll_full'][2])
				df_b2['init_emit_h_noncoll_full']           = pd.Series(self.filln_StableBeamsDict['init_emit_h_noncoll_full'][2])
				df_b2['init_emit_v_noncoll_full']           = pd.Series(self.filln_StableBeamsDict['init_emit_v_noncoll_full'][2])
				df_b2['tau_bl_coll_full']                   = pd.Series(self.filln_StableBeamsDict['tau_bl_coll_full'][2])
				df_b2['tau_bl_noncoll_full']                = pd.Series(self.filln_StableBeamsDict['tau_bl_noncoll_full'][2])
				df_b2['init_bl_coll_full']                  = pd.Series(self.filln_StableBeamsDict['init_bl_coll_full'][2])
				df_b2['init_bl_noncoll_full']               = pd.Series(self.filln_StableBeamsDict['init_bl_noncoll_full'][2])
				df_b2['tau_inten_coll_full']                = pd.Series(self.filln_StableBeamsDict['tau_inten_coll_full'][2])
				df_b2['tau_inten_noncoll_full']             = pd.Series(self.filln_StableBeamsDict['tau_inten_noncoll_full'][2])
				df_b2['init_inten_coll_full']               = pd.Series(self.filln_StableBeamsDict['init_inten_coll_full'][2])
				df_b2['init_inten_noncoll_full']            = pd.Series(self.filln_StableBeamsDict['init_inten_noncoll_full'][2])

				df_b2['tau_emit_h_coll']               = pd.Series(self.filln_StableBeamsDict['tau_emit_h_coll'][2])
				df_b2['tau_emit_v_coll']               = pd.Series(self.filln_StableBeamsDict['tau_emit_v_coll'][2])
				df_b2['tau_emit_h_noncoll']            = pd.Series(self.filln_StableBeamsDict['tau_emit_h_noncoll'][2])
				df_b2['tau_emit_v_noncoll']            = pd.Series(self.filln_StableBeamsDict['tau_emit_v_noncoll'][2])
				df_b2['init_emit_h_coll']              = pd.Series(self.filln_StableBeamsDict['init_emit_h_coll'][2])
				df_b2['init_emit_v_coll']              = pd.Series(self.filln_StableBeamsDict['init_emit_v_coll'][2])
				df_b2['init_emit_h_noncoll']           = pd.Series(self.filln_StableBeamsDict['init_emit_h_noncoll'][2])
				df_b2['init_emit_v_noncoll']           = pd.Series(self.filln_StableBeamsDict['init_emit_v_noncoll'][2])
				df_b2['tau_bl_coll']                   = pd.Series(self.filln_StableBeamsDict['tau_bl_coll'][2])
				df_b2['tau_bl_noncoll']                = pd.Series(self.filln_StableBeamsDict['tau_bl_noncoll'][2])
				df_b2['init_bl_coll']                  = pd.Series(self.filln_StableBeamsDict['init_bl_coll'][2])
				df_b2['init_bl_noncoll']               = pd.Series(self.filln_StableBeamsDict['init_bl_noncoll'][2])
				df_b2['tau_inten_coll']                = pd.Series(self.filln_StableBeamsDict['tau_inten_coll'][2])
				df_b2['tau_inten_noncoll']             = pd.Series(self.filln_StableBeamsDict['tau_inten_noncoll'][2])
				df_b2['init_inten_coll']               = pd.Series(self.filln_StableBeamsDict['init_inten_coll'][2])
				df_b2['init_inten_noncoll']            = pd.Series(self.filln_StableBeamsDict['init_inten_noncoll'][2])

				df_b2.insert(0, 'timestamp',         self.filln_StableBeamsDict['time_range'])
				df_b2.insert(0, 'date',              self.convertToLocalTime(df_b2['timestamp']))
				df_b2.insert(0, 'beam',              ["beam_2"]*len(df_b2))
				df_b2.insert(0, 'fill',              [filln]*len(df_b2))

				## append the two df
				total_stable = df_b1.append(df_b2)
				## setting additional index to fill and beam
				total_stable = total_stable.set_index(['fill','beam'], drop=False)

				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("Pandas SB pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(total_stable, fid)
					else:
						warn("Pandas SB pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(total_stable, fid)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runCycleModel(self, filln):
		'''
		Runs and generates the pickles for the IBS model.
		Inputs : filln : fill number
		Returns: None
		'''
		self.filln_CycleModelDict.clear()
		self.filln_CycleModelInj2SBDict.clear()

		if len(self.filln_CycleDict) > 0:
			#get stuff
			debug("# runCycleModel : Cycle Dictionary is filled for Fill [{}].".format(filln))
		else:
			##populate it from file
			filename = self.fill_dir+self.Cycle_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))

			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')

			debug("# runCycleModel : Cycle Analysis has NOT ran for this fill, loading the Cycle dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename, 'rb') as fid:
				   self.filln_CycleDict = pickle.load(fid)


			gamma = {'Injection': self.gammaFB, 'he_before_SB': self.gammaFT}
			VRF = {'Injection': self.VRF_FB, 'he_before_SB': self.VRF_FT}
			tauSRxy_s = {'Injection': self.tauSRxy_FB, 'he_before_SB': self.tauSRxy_FT}

			info("#runCycleModel : Running Cycle model for Fill {}".format(filln))
			dict_model = {}
			for beam_n in [1, 2]:
				dict_model['beam_{}'.format(beam_n)] = {}
				for interval in ['Injection', 'he_before_SB']:
					dict_model['beam_{}'.format(beam_n)][interval] = {}
					dict_model['beam_{}'.format(beam_n)][interval]['at_start']={}
					for param in ['time_meas', 'emith', 'blength', 'emitv']:
						dict_model['beam_{}'.format(beam_n)][interval]['at_start'][param] = self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start'][param]

					emit_h_IBS_end, IBSx, bl_IBS_end, IBSl, ey_IBS = IBSmodel(IBSON=1, gamma=gamma[interval],
						bunch_intensity_p = np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['intensity']),
						ex_norm_m         = np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['emith'])*1.0e-06,
						ey_norm_m         = np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['emitv'])*1.0e-06,
						bl_4sigma_s       = np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['blength']),
						VRF_V             = VRF[interval],
						dt_s              = (np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_end']['time_meas'])-np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['time_meas'])))

					dict_model['beam_{}'.format(beam_n)][interval]['at_end']               = {}
					dict_model['beam_{}'.format(beam_n)][interval]['at_end']['emith']      = emit_h_IBS_end*1.0e06
					dict_model['beam_{}'.format(beam_n)][interval]['at_end']['emitv']      = ey_IBS*1.0e06
					dict_model['beam_{}'.format(beam_n)][interval]['at_end']['blength']    = bl_IBS_end
					dict_model['beam_{}'.format(beam_n)][interval]['at_end']['time_meas']  = self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_end']['time_meas']


			self.filln_CycleModelDict = dict_model
			dict_inj2sb = {}
			### Now that I have the dict_mdoel and the self.filln_CycleDict I can run Inj2SB

			for beam_n in [1, 2]:
				dict_inj2sb['beam_{}'.format(beam_n)] = {}
				for interval in ['he_before_SB']:
					dict_inj2sb['beam_{}'.format(beam_n)][interval] = {}
					dict_inj2sb['beam_{}'.format(beam_n)][interval]['at_start'] = {}
					for param in ['time_meas', 'emith', 'blength', 'emitv']:
						dict_inj2sb['beam_{}'.format(beam_n)][interval]['at_start'][param] = dict_model['beam_{}'.format(beam_n)]['Injection']['at_end'][param]

					emit_h_IBS_end, IBSx, bl_IBS_end, IBSl, ey_IBS = IBSmodel(IBSON=1, gamma=gamma[interval],
						bunch_intensity_p=np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['intensity']),
						ex_norm_m=np.array(dict_model['beam_{}'.format(beam_n)]['Injection']['at_end']['emith'])*1.0e-06,
						ey_norm_m=np.array(dict_model['beam_{}'.format(beam_n)]['Injection']['at_end']['emitv'])*1.0e-06,
						bl_4sigma_s=np.array(dict_model['beam_{}'.format(beam_n)]['Injection']['at_end']['blength']),
						VRF_V = VRF[interval],
						dt_s = (np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_end']['time_meas'])-np.array(self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_start']['time_meas'])))

				dict_inj2sb['beam_{}'.format(beam_n)][interval]['at_end']               = {}
				dict_inj2sb['beam_{}'.format(beam_n)][interval]['at_end']['emith']      = emit_h_IBS_end*1.0e06
				dict_inj2sb['beam_{}'.format(beam_n)][interval]['at_end']['emitv']      = ey_IBS*1.0e06
				dict_inj2sb['beam_{}'.format(beam_n)][interval]['at_end']['blength']    = bl_IBS_end
				dict_inj2sb['beam_{}'.format(beam_n)][interval]['at_end']['time_meas']  = self.filln_CycleDict['beam_{}'.format(beam_n)][interval]['at_end']['time_meas']

			self.filln_CycleModelInj2SBDict = dict_inj2sb

			if self.saveDict:
				filename = self.fill_dir+self.Cycle_model_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('# runCycleModel : Saving dictionary for Cycle Model of fill {} into {}'.format(filln, filename ))
				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("# runCycleModel : Dictionary Cycle Model pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(dict_model, fid)
					else:
						warn("# runCycleModel : Dictionary Cycle Model pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(dict_model, fid)


				filename = self.fill_dir+self.Cycle_model_filename.replace('.pkl.gz', '_Inj2SB.pkl.gz')
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('# runCycleModel : Saving dictionary for Cycle Model Inj2SB of fill {} into {}'.format(filln, filename ))
				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("# runCycleModel : Dictionary Cycle Model Inj2SB pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(dict_inj2sb, fid)
					else:
						warn("# runCycleModel : Dictionary Cycle Model Inj2SB pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(dict_inj2sb, fid)


			if self.savePandas:
				df_b1_inj_start = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_1']['Injection']['at_start'], orient='columns')
				df_b1_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_start['time_meas']))
				df_b1_inj_start.insert(0, 'model'          , ['model']*len(df_b1_inj_start))
				df_b1_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b1_inj_start))
				df_b1_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_start))
				df_b1_inj_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_start))
				df_b1_inj_start.insert(0, 'fill'           , [filln]*len(df_b1_inj_start))

				df_b1_inj_end = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_1']['Injection']['at_end'], orient='columns')
				df_b1_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_start['time_meas']))
				df_b1_inj_end.insert(0, 'model'          , ['model']*len(df_b1_inj_end))
				df_b1_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b1_inj_end))
				df_b1_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_end))
				df_b1_inj_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_end))
				df_b1_inj_end.insert(0, 'fill'           , [filln]*len(df_b1_inj_end))

				df_b2_inj_start = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_2']['Injection']['at_start'], orient='columns')
				df_b2_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_start['time_meas']))
				df_b2_inj_start.insert(0, 'model'          , ['model']*len(df_b2_inj_start))
				df_b2_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b2_inj_start))
				df_b2_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_start))
				df_b2_inj_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_start))
				df_b2_inj_start.insert(0, 'fill'           , [filln]*len(df_b2_inj_start))


				df_b2_inj_end = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_2']['Injection']['at_end'], orient='columns')
				df_b2_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_end['time_meas']))
				df_b2_inj_end.insert(0, 'model'          , ['model']*len(df_b2_inj_end))
				df_b2_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b2_inj_end))
				df_b2_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_end))
				df_b2_inj_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_end))
				df_b2_inj_end.insert(0, 'fill'           , [filln]*len(df_b2_inj_end))
				# -----
				df_b1_he_start = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_1']['he_before_SB']['at_start'], orient='columns')
				df_b1_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
				df_b1_he_start.insert(0, 'model'          , ['model']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'fill'           , [filln]*len(df_b1_he_start))

				df_b1_he_end = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_1']['he_before_SB']['at_end'], orient='columns')
				df_b1_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
				df_b1_he_end.insert(0, 'model'          , ['model']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'fill'           , [filln]*len(df_b1_he_end))

				df_b2_he_start = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_2']['he_before_SB']['at_start'], orient='columns')
				df_b2_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_start['time_meas']))
				df_b2_he_start.insert(0, 'model'          , ['model']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'fill'           , [filln]*len(df_b2_he_start))


				df_b2_he_end = pd.DataFrame.from_dict(self.filln_CycleModelDict['beam_2']['he_before_SB']['at_end'], orient='columns')
				df_b2_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_end['time_meas']))
				df_b2_he_end.insert(0, 'model'          , ['model']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'fill'           , [filln]*len(df_b2_he_end))

				cycle_model_total = df_b1_inj_start.append(df_b1_inj_end).append(df_b2_inj_start).append(df_b2_inj_end).append(df_b1_he_start).append(df_b1_he_end).append(df_b2_he_start).append(df_b2_he_end)
				cycle_model_total = cycle_model_total.set_index(['fill','beam','cycle'], drop=False)

				filename = self.fill_dir+self.Cycle_model_filename.replace('.pkl.gz', '_df.pkl.gz')
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('# runCycleModel : Saving Pandas for Cycle Model of fill {} into {}'.format(filln, filename ))
				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("# runCycleModel : Pandas Cycle Model pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(cycle_model_total, fid)
					else:
						warn("# runCycleModel : Pandas Cycle Model pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(cycle_model_total, fid)

				# doing the same for Inj2SB
				df_b1_he_start = pd.DataFrame.from_dict(self.filln_CycleModelInj2SBDict['beam_1']['he_before_SB']['at_start'], orient='columns')
				df_b1_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
				df_b1_he_start.insert(0, 'model'          , ['model']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_start))
				df_b1_he_start.insert(0, 'fill'           , [filln]*len(df_b1_he_start))

				df_b1_he_end = pd.DataFrame.from_dict(self.filln_CycleModelInj2SBDict['beam_1']['he_before_SB']['at_end'], orient='columns')
				df_b1_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
				df_b1_he_end.insert(0, 'model'          , ['model']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_end))
				df_b1_he_end.insert(0, 'fill'           , [filln]*len(df_b1_he_end))

				df_b2_he_start = pd.DataFrame.from_dict(self.filln_CycleModelInj2SBDict['beam_2']['he_before_SB']['at_start'], orient='columns')
				df_b2_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_start['time_meas']))
				df_b2_he_start.insert(0, 'model'          , ['model']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_start))
				df_b2_he_start.insert(0, 'fill'           , [filln]*len(df_b2_he_start))


				df_b2_he_end = pd.DataFrame.from_dict(self.filln_CycleModelInj2SBDict['beam_2']['he_before_SB']['at_end'], orient='columns')
				df_b2_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_end['time_meas']))
				df_b2_he_end.insert(0, 'model'          , ['model']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_end))
				df_b2_he_end.insert(0, 'fill'           , [filln]*len(df_b2_he_end))

				cycle_model_inj2sb_total_df = df_b1_he_start.append(df_b1_he_end).append(df_b2_he_start).append(df_b2_he_end)
				cycle_model_inj2sb_total_df = cycle_model_inj2sb_total_df.set_index(['fill','beam','cycle'], drop=False)

				filename = self.fill_dir+self.Cycle_model_filename.replace('.pkl.gz', '_Inj2SB_df.pkl.gz')
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')
				info('# runCycleModel : Saving Pandas for Cycle Model Inj2SB of fill {} into {}'.format(filln, filename ))
				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("# runCycleModel : Pandas Cycle Model Inj2SB pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(cycle_model_inj2sb_total_df, fid)
					else:
						warn("# runCycleModel : Pandas Cycle Model Inj2SB pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(cycle_model_inj2sb_total_df, fid)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runSBFits(self, filln):
		'''
		Runs the loop to create SB Fits to be used by the SB Model
		Inputs  : filln = Fill number
		Returns : None
		'''
		self.filln_SBFitsDict.clear()

		# Check that I have the SB info
		if len(self.filln_StableBeamsDict)>0:
			debug("# runSBFits : SB Analysis has ran for this fill, loading the SB dictionary")
		else:
			##populate it from file
			filename = self.fill_dir+self.SB_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			debug("# runSBFits : SB Analysis has NOT ran for this fill, loading the SB dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_StableBeamsDict = pickle.load(fid)

		if not self.skipMassi:
			# Check that I have the meas lumi
			if len(self.filln_LumiMeasDict)>0:
				debug("# runSBFits : Measured Lumi has ran for this fill, loading the Measured Lumi dictionary")
			else:
				##populate it from file
				filename = self.fill_dir+self.Massi_filename
				filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>', '')
				debug("# runSBFits : Measured Lumi Analysis has NOT ran for this fill, loading the Measured Lumi dictionary from pickle file [{}].".format(filename))
				with gzip.open(filename) as fid:
					   self.filln_LumiMeasDict = pickle.load(fid)

		#Check that I have the calc lumi
		if len(self.filln_LumiCalcDict)>0:
			debug("# runSBFits : Calculated Lumi has ran for this fill, loading the Measured Lumi dictionary")
		else:
			##populate it from file
			filename = self.fill_dir+self.Lumi_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			debug("# runSBFits : Calculated Lumi Analysis has NOT ran for this fill, loading the Calculated Lumi dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_LumiCalcDict = pickle.load(fid)

		# Now I should have everything
		trange = self.filln_StableBeamsDict['time_range']
		mask_fit = (trange-trange[0])<self.t_fit_length
		t_tot_emit_fit_length = trange[-1]-trange[0]
		experiment = {1:'ATLAS', 2:'CMS'}


		dict_fits={}
		for beam_n in [1, 2]:
			dict_fits['beam_{}'.format(beam_n)]={}
			dict_fits[experiment[beam_n]]={}
			for family in ['coll', 'noncoll']:
				for plane in ['h', 'v']:
					dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}'.format(plane, family)] = []

					dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}'.format(plane, family)] = []
					dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}'.format(family)] = []
					dict_fits['beam_{}'.format(beam_n)]['init_bl_{}'.format(family)] = []
					dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}'.format(family)] = []
					dict_fits['beam_{}'.format(beam_n)]['init_inten_{}'.format(family)] = []
					dict_fits[experiment[beam_n]]['tau_lumi_meas_{}'.format(family)] = []
					dict_fits[experiment[beam_n]]['init_lumi_meas_{}'.format(family)] = []
					dict_fits[experiment[beam_n]]['tau_lumi_calc_{}'.format(family)] = []
					dict_fits[experiment[beam_n]]['init_lumi_calc_{}'.format(family)] = []

					dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}_full'.format(plane, family)] = []
					dict_fits['beam_{}'.format(beam_n)]['init_bl_{}_full'.format(family)] = []
					dict_fits['beam_{}'.format(beam_n)]['init_inten_{}_full'.format(family)] = []


					dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}_full'.format(plane, family)] = []
					dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}_full'.format(family)] = []
					dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}_full'.format(family)] = []


					dict_fits[experiment[beam_n]]['tau_lumi_meas_{}_full'.format(family)] = []
					dict_fits[experiment[beam_n]]['tau_lumi_calc_{}_full'.format(family)] = []
					slot_list = self.filln_StableBeamsDict['slots_filled_{}'.format(family)][beam_n]

					for i_slot, slot_numb in enumerate(slot_list):
						#print i_slot, slot_numb
						ytofit = self.filln_StableBeamsDict['e{}_interp_{}'.format(plane, family)][beam_n][:,i_slot]
						popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
						popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)

						dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}'.format(plane, family)].append(3600.0*1.0/popt[1])
						dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}_full'.format(plane, family)].append(3600.0*1.0/popt_f[1])
						dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}'.format(plane, family)].append(popt[0])
						dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}_full'.format(plane, family)].append(popt_f[0])

						ytofit = self.filln_StableBeamsDict['bl_interp_m_{}'.format(family)][beam_n][:,i_slot]
						popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
						popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
						dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}'.format(family)].append(3600.0*1.0/popt[1])
						dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}_full'.format(family)].append(3600.0*1.0/popt_f[1])
						dict_fits['beam_{}'.format(beam_n)]['init_bl_{}'.format(family)].append(popt[0])
						dict_fits['beam_{}'.format(beam_n)]['init_bl_{}_full'.format(family)].append(popt_f[0])

						ytofit = self.filln_StableBeamsDict['b_inten_interp_{}'.format(family)][beam_n][:,i_slot]*1.0e-11
						popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., ytofit[mask_fit])
						popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., ytofit)
						dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}'.format(family)].append(3600.0*1.0/popt[1])
						dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}_full'.format(family)].append(3600.0*1.0/popt_f[1])
						dict_fits['beam_{}'.format(beam_n)]['init_inten_{}'.format(family)].append(popt[0]*1.0e11)
						dict_fits['beam_{}'.format(beam_n)]['init_inten_{}_full'.format(family)].append(popt_f[0]*1.0e11)


						if family=='coll':
							ytofit = self.filln_LumiMeasDict[experiment[beam_n]]['bunch_lumi'][:,i_slot]
							popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., 1.0e-34*ytofit[mask_fit])
							popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., 1.0e-34*ytofit)

							dict_fits[experiment[beam_n]]['tau_lumi_meas_{}'.format(family)].append(3600.0*1.0/popt[1])
							dict_fits[experiment[beam_n]]['tau_lumi_meas_{}_full'.format(family)].append(3600.0*1.0/popt_f[1])
							dict_fits[experiment[beam_n]]['init_lumi_meas_{}'.format(family)].append(popt[0]*1.0e34)

							ytofit = self.filln_LumiCalcDict[experiment[beam_n]]['bunch_lumi'][:,i_slot]
							popt, pcov = self.curve_fit_robust(self.fitfunc, (trange[mask_fit]-trange[0])/3600., 1.0e-34*ytofit[mask_fit])
							popt_f, pcov_f = self.curve_fit_robust(self.fitfunc, (trange-trange[0])/3600., 1.0e-34*ytofit)

							dict_fits[experiment[beam_n]]['tau_lumi_calc_{}'.format(family)].append(3600.0*1./popt[1])
							dict_fits[experiment[beam_n]]['tau_lumi_calc_{}_full'.format(family)].append(3600.0*1./popt[1])
							dict_fits[experiment[beam_n]]['init_lumi_calc_{}'.format(family)].append(popt[0]*1.0e34)

					dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}'.format(plane, family)]      = np.array(dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}'.format(plane, family)])
					dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}_full'.format(plane, family)] = np.array(dict_fits['beam_{}'.format(beam_n)]['tau_emit{}_{}_full'.format(plane, family)])
					dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}'.format(plane, family)]     = np.array(dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}'.format(plane, family)])
					dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}_full'.format(plane, family)]= np.array(dict_fits['beam_{}'.format(beam_n)]['init_emit{}_{}_full'.format(plane, family)])
					dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}'.format(family)]                 = np.array(dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}'.format(family)])
					dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}_full'.format(family)]            = np.array(dict_fits['beam_{}'.format(beam_n)]['tau_bl_{}_full'.format(family)])
					dict_fits['beam_{}'.format(beam_n)]['init_bl_{}'.format(family)]                = np.array(dict_fits['beam_{}'.format(beam_n)]['init_bl_{}'.format(family)])
					dict_fits['beam_{}'.format(beam_n)]['init_bl_{}_full'.format(family)]           = np.array(dict_fits['beam_{}'.format(beam_n)]['init_bl_{}_full'.format(family)])
					dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}'.format(family)]              = np.array(dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}'.format(family)])
					dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}_full'.format(family)]         = np.array(dict_fits['beam_{}'.format(beam_n)]['tau_inten_{}_full'.format(family)])
					dict_fits['beam_{}'.format(beam_n)]['init_inten_{}'.format(family)]             = np.array(dict_fits['beam_{}'.format(beam_n)]['init_inten_{}'.format(family)])
					dict_fits['beam_{}'.format(beam_n)]['init_inten_{}_full'.format(family)]        = np.array(dict_fits['beam_{}'.format(beam_n)]['init_inten_{}_full'.format(family)])

					dict_fits[experiment[beam_n]]['tau_lumi_meas_{}'.format(family)]                = np.array(dict_fits[experiment[beam_n]]['tau_lumi_meas_{}'.format(family)])
					dict_fits[experiment[beam_n]]['tau_lumi_meas_{}_full'.format(family)]           = np.array(dict_fits[experiment[beam_n]]['tau_lumi_meas_{}_full'.format(family)])
					dict_fits[experiment[beam_n]]['init_lumi_meas_{}'.format(family)]               = np.array(dict_fits[experiment[beam_n]]['init_lumi_meas_{}'.format(family)])
					dict_fits[experiment[beam_n]]['tau_lumi_calc_{}'.format(family)]                = np.array(dict_fits[experiment[beam_n]]['tau_lumi_calc_{}'.format(family)])
					dict_fits[experiment[beam_n]]['tau_lumi_calc_{}_full'.format(family)]           = np.array(dict_fits[experiment[beam_n]]['tau_lumi_calc_{}_full'.format(family)])
					dict_fits[experiment[beam_n]]['init_lumi_calc_{}'.format(family)]               = np.array(dict_fits[experiment[beam_n]]['init_lumi_calc_{}'.format(family)])


		self.filln_SBFitsDict = dict_fits


		if self.saveDict:
			filename = self.fill_dir+self.SB_fits_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			info('Saving dictionary for SB Fits of fill {} into {}'.format(filln, filename ))
			if os.path.exists(filename):
				if self.overwriteFiles:
					warn("Dictionary SB Fits pickle for fill {} already exists! Overwritting it...".format(filln))
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(dict_fits, fid)
				else:
					warn("Dictionary SB Fits pickle for fill {} already exists! Skipping it...".format(filln))
			else:
				with gzip.open(filename, 'wb') as fid:
					pickle.dump(dict_fits, fid)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runSBModel(self, filln):
		self.filln_SBModelDict.clear()

		# Check that I have the SB info
		if len(self.filln_StableBeamsDict)>0:
			debug("# runSBModel : SB Analysis has ran for this fill, loading the SB dictionary")
		else:
			##populate it from file
			filename = self.fill_dir+self.SB_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			debug("# runSBModel : SB Analysis has NOT ran for this fill, loading the SB dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_StableBeamsDict = pickle.load(fid)

		# Check that I have the fits
		if len(self.filln_SBFitsDict)>0:
			debug("# runSBModel : SB Fits Analysis has ran for this fill, loading the SB dictionary")
		else:
			##populate it from file
			filename = self.fill_dir+self.SB_fits_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			debug("# runSBModel : SB Fits Analysis has NOT ran for this fill, loading the SB dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_SBFitsDict = pickle.load(fid)



		dict_models_settings={}
		dict_models_settings['IBSBOff'] = {
											'tau_empirical_h1_coll'     :   None,
											'tau_empirical_v1_coll'     :   None,
											'tau_empirical_h2_coll'     :   None,
											'tau_empirical_v2_coll'     :   None,

											'tau_empirical_h1_noncoll'  :   None,
											'tau_empirical_v1_noncoll'  :   None,
											'tau_empirical_h2_noncoll'  :   None,
											'tau_empirical_v2_noncoll'  :   None,
											'BOff'                      :   1,
											'blengthBU'                 :   "Model",
											'emitBU'                    :   "Model"}

		dict_models_settings['IBSLosses'] = {
											'tau_empirical_h1_coll'     :   None,
											'tau_empirical_v1_coll'     :   None,
											'tau_empirical_h2_coll'     :   None,
											'tau_empirical_v2_coll'     :   None,

											'tau_empirical_h1_noncoll'  :   None,
											'tau_empirical_v1_noncoll'  :   None,
											'tau_empirical_h2_noncoll'  :   None,
											'tau_empirical_v2_noncoll'  :   None,
											'BOff'                      :   0,
											'blengthBU'                 :   "Model",
											'emitBU'                    :   "Model"}

		dict_models_settings['EmpiricalBlowupBOff'] = {
											'tau_empirical_h1_coll'     :   self.filln_SBFitsDict['beam_1']['tau_emith_coll_full'],
											'tau_empirical_v1_coll'     :   self.filln_SBFitsDict['beam_1']['tau_emitv_coll_full'],
											'tau_empirical_h2_coll'     :   self.filln_SBFitsDict['beam_2']['tau_emith_coll_full'],
											'tau_empirical_v2_coll'     :   self.filln_SBFitsDict['beam_2']['tau_emitv_coll_full'],

											'tau_empirical_h1_noncoll'  :   self.filln_SBFitsDict['beam_1']['tau_emith_noncoll_full'],
											'tau_empirical_v1_noncoll'  :   self.filln_SBFitsDict['beam_1']['tau_emitv_noncoll_full'],
											'tau_empirical_h2_noncoll'  :   self.filln_SBFitsDict['beam_2']['tau_emith_noncoll_full'],
											'tau_empirical_v2_noncoll'  :   self.filln_SBFitsDict['beam_2']['tau_emitv_noncoll_full'],
											'BOff'                      :   1,
											'blengthBU'                 :   "Model",
											'emitBU'                    :   "EmpiricalBlowup"}



		dict_models_settings['EmpiricalBlowupLosses'] = {
											'tau_empirical_h1_coll'     :   self.filln_SBFitsDict['beam_1']['tau_emith_coll_full'],
											'tau_empirical_v1_coll'     :   self.filln_SBFitsDict['beam_1']['tau_emitv_coll_full'],
											'tau_empirical_h2_coll'     :   self.filln_SBFitsDict['beam_2']['tau_emith_coll_full'],
											'tau_empirical_v2_coll'     :   self.filln_SBFitsDict['beam_2']['tau_emitv_coll_full'],

											'tau_empirical_h1_noncoll'  :   self.filln_SBFitsDict['beam_1']['tau_emith_noncoll_full'],
											'tau_empirical_v1_noncoll'  :   self.filln_SBFitsDict['beam_1']['tau_emitv_noncoll_full'],
											'tau_empirical_h2_noncoll'  :   self.filln_SBFitsDict['beam_2']['tau_emith_noncoll_full'],
											'tau_empirical_v2_noncoll'  :   self.filln_SBFitsDict['beam_2']['tau_emitv_noncoll_full'],
											'BOff'                      :   0,
											'blengthBU'                 :   "Model",
											'emitBU'                    :   "EmpiricalBlowup"}

		dict_models_settings['EmpiricalblengthLosses'] = {
											'tau_empirical_h1_coll'     :   self.filln_SBFitsDict['beam_1']['tau_emith_coll_full'],
											'tau_empirical_v1_coll'     :   self.filln_SBFitsDict['beam_1']['tau_emitv_coll_full'],
											'tau_empirical_h2_coll'     :   self.filln_SBFitsDict['beam_2']['tau_emith_coll_full'],
											'tau_empirical_v2_coll'     :   self.filln_SBFitsDict['beam_2']['tau_emitv_coll_full'],

											'tau_empirical_h1_noncoll'  :   self.filln_SBFitsDict['beam_1']['tau_emith_noncoll_full'],
											'tau_empirical_v1_noncoll'  :   self.filln_SBFitsDict['beam_1']['tau_emitv_noncoll_full'],
											'tau_empirical_h2_noncoll'  :   self.filln_SBFitsDict['beam_2']['tau_emith_noncoll_full'],
											'tau_empirical_v2_noncoll'  :   self.filln_SBFitsDict['beam_2']['tau_emitv_noncoll_full'],
											'BOff'                      :   0,
											'blengthBU'                 :   "EmpiricalBlowup",
											'emitBU'                    :   "Model"}

		dict_models = {}
		if filln in [5450, 5439]:
			leveling = 8.e37
		else:
			leveling = 0

		dict_case = {}
		dict_case['case']           = self.cases
		dict_case['cor_fact_1h']    = self.correction_factor_1h     # 100 = uncorrected , 1.1 = 110% = +10% corr.
		dict_case['cor_fact_2h']    = self.correction_factor_2h
		dict_case['cor_fact_1v']    = self.correction_factor_1v
		dict_case['cor_fact_2v']    = self.correction_factor_2v

		phi_full_rad_ATLAS = None
		phi_full_rad_CMS   = None
		# get xing angle info
		for key in self.XingAngle.keys():
			print key, filln 
			if filln in range(key[0], key[1]): # >= key[0] and filln<=key[1]:
				print filln, key, self.XingAngle[key]
				phi_full_rad_ATLAS  = self.XingAngle[key][0]
				phi_full_rad_CMS    = self.XingAngle[key][1]
				info('# runSBModel: Setting crossing angle IP1 = {} , IP5 = {}!'.format(phi_full_rad_ATLAS, phi_full_rad_CMS))
				break
			# else:
			# 	
			# 	return
		if phi_full_rad_ATLAS is None or phi_full_rad_CMS is None:
			warn('# runSBModel: Crossing angle information for IP1/IP5 not found!')
			return

			print phi_full_rad_ATLAS

		for i_case,case in enumerate(dict_case['case']):
			info('# runSBModel: Running for case {}...'.format(case))
			cor_fact_1h = dict_case['cor_fact_1h'][i_case]
			cor_fact_2h = dict_case['cor_fact_2h'][i_case]
			cor_fact_1v = dict_case['cor_fact_1v'][i_case]
			cor_fact_2v = dict_case['cor_fact_2v'][i_case]


			for model_name in dict_models_settings.keys():
				this_model = dict_models_settings[model_name]
				family = 'coll'
				bunch_intensity_b1_init =  self.filln_StableBeamsDict['b_inten_interp_{}'.format(family)][1]
				bunch_intensity_b2_init =  self.filln_StableBeamsDict['b_inten_interp_{}'.format(family)][2]
				emith_b1_init           =  self.filln_StableBeamsDict['eh_interp_{}'.format(family)][1]*cor_fact_1h
				emith_b2_init           =  self.filln_StableBeamsDict['eh_interp_{}'.format(family)][2]*cor_fact_2h
				emitv_b1_init           =  self.filln_StableBeamsDict['ev_interp_{}'.format(family)][1]*cor_fact_1v
				emitv_b2_init           =  self.filln_StableBeamsDict['ev_interp_{}'.format(family)][2]*cor_fact_2v


				blen_b1_init            =  self.filln_StableBeamsDict['bl_interp_m_{}'.format(family)][1]
				blen_b2_init            =  self.filln_StableBeamsDict['bl_interp_m_{}'.format(family)][2]


				trange                  = self.filln_StableBeamsDict['time_range']
				tFill_s                 = (trange[-1]-trange[0])

				(tt_s, Luminosity_invm2s_model_ATLAS, Luminosity_invm2s_model_CMS, bunch_intensity_p1_mod, bunch_intensity_p2_mod,
					ex_norm_m1_mod, ex_norm_m2_mod, ey_norm_m1_mod, ey_norm_m2_mod, bl_4sigma_s1_mod,
					bl_4sigma_s2_mod, ex_norm_m1_mod_IBScorr, ex_norm_m2_mod_IBScorr, ey_norm_m1_mod_IBScorr, ey_norm_m2_mod_IBScorr) = \
					LumiModel(gamma=self.gammaFT, betastar_m=self.betastar_m, phi_full_rad_ATLAS=phi_full_rad_ATLAS,
					phi_full_rad_CMS=phi_full_rad_CMS,
					bunch_intensityin_p1=bunch_intensity_b1_init, bunch_intensityin_p2=bunch_intensity_b2_init,
					exin_norm_m1=emith_b1_init*1.0e-06, exin_norm_m2=emith_b2_init*1.0e-06,
					eyin_norm_m1=emitv_b1_init*1.0e-06, eyin_norm_m2=emitv_b2_init*1.0e-06,
					blin_4sigma_s1=blen_b1_init*4.0/clight,
					blin_4sigma_s2=blen_b2_init*4.0/clight,
					tFill_s=tFill_s,tauSRxy_s=self.tauSRxy_FT, tauSRl_s=self.tauSRl_FT, sigmaBOff_m2=self.sigmaBOff_m2, sigmaElastic_m2=self.sigma_el_m2, VRF_V=self.VRF_FT,
					IBSON=1, emitBU=this_model['emitBU'], BoffON=this_model['BOff'], blengthBU=this_model['blengthBU'], nIPs = 2., dt_s=trange[1]-trange[0],
					tau_empirical_h1=this_model['tau_empirical_h1_coll'],
					tau_empirical_v1=this_model['tau_empirical_v1_coll'],
					tau_empirical_h2=this_model['tau_empirical_h2_coll'],
					tau_empirical_v2=this_model['tau_empirical_v2_coll'],
					leveling=leveling)

				eh_interp_coll              = {1:[], 2:[]}
				ev_interp_coll              = {1:[], 2:[]}
				eh_interp_coll_IBScorr      = {1:[], 2:[]}
				ev_interp_coll_IBScorr      = {1:[], 2:[]}
				b_inten_interp_coll         = {1:[], 2:[]}
				bl_interp_m_coll            = {1:[], 2:[]}

				eh_interp_coll[1]           = np.array(ex_norm_m1_mod)*1.0e06
				eh_interp_coll[2]           = np.array(ex_norm_m2_mod)*1.0e06
				ev_interp_coll[1]           = np.array(ey_norm_m1_mod)*1.0e06
				ev_interp_coll[2]           = np.array(ey_norm_m2_mod)*1.0e06
				eh_interp_coll_IBScorr[1]   = np.array(ex_norm_m1_mod_IBScorr)*1.0e06
				eh_interp_coll_IBScorr[2]   = np.array(ex_norm_m2_mod_IBScorr)*1.0e06
				ev_interp_coll_IBScorr[1]   = np.array(ey_norm_m1_mod_IBScorr)*1.0e06
				ev_interp_coll_IBScorr[2]   = np.array(ey_norm_m2_mod_IBScorr)*1.0e06
				b_inten_interp_coll[1]      = np.array(bunch_intensity_p1_mod)
				b_inten_interp_coll[2]      = np.array(bunch_intensity_p2_mod)
				bl_interp_m_coll[1]         = np.array(bl_4sigma_s1_mod)/4.0*clight
				bl_interp_m_coll[2]         = np.array(bl_4sigma_s2_mod)/4.0*clight

				#handle non-colliding
				family = 'noncoll'
				eh_interp_noncoll           = {1:[], 2:[]}
				ev_interp_noncoll           = {1:[], 2:[]}
				eh_interp_noncoll_IBScorr   = {1:[], 2:[]}
				ev_interp_noncoll_IBScorr   = {1:[], 2:[]}
				eh_interp_raw_noncoll       = {1:[], 2:[]}
				ev_interp_raw_noncoll       = {1:[], 2:[]}
				b_inten_interp_noncoll      = {1:[], 2:[]}
				bl_interp_m_noncoll         = {1:[], 2:[]}

				for beam_n in [1, 2]:
					(tt_s, _, _, bunch_intensity_temp, _, ex_norm_m1_temp, _, ey_norm_m1_temp, _, bl_4sigma_s1_temp, _, ex_norm_m1_temp_IBScorr, _, ey_norm_m1_temp_IBScorr, _) =  LumiModel(gamma=self.gammaFT,
					betastar_m=self.betastar_m, phi_full_rad_ATLAS=phi_full_rad_ATLAS,
					phi_full_rad_CMS=phi_full_rad_CMS, bunch_intensityin_p1=self.filln_StableBeamsDict['b_inten_interp_{}'.format(family)][beam_n],
					bunch_intensityin_p2=self.filln_StableBeamsDict['b_inten_interp_{}'.format(family)][beam_n],
					exin_norm_m1=self.filln_StableBeamsDict['eh_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_2h if beam_n==2 else self.filln_StableBeamsDict['eh_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_1h,
					exin_norm_m2=self.filln_StableBeamsDict['eh_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_2h if beam_n==2 else self.filln_StableBeamsDict['eh_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_1h,
					eyin_norm_m1=self.filln_StableBeamsDict['ev_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_2v if beam_n==2 else self.filln_StableBeamsDict['ev_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_1v,
					eyin_norm_m2=self.filln_StableBeamsDict['ev_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_2v if beam_n==2 else self.filln_StableBeamsDict['ev_interp_{}'.format(family)][beam_n]*1.0e-06*cor_fact_1v,
					blin_4sigma_s1=self.filln_StableBeamsDict['bl_interp_m_{}'.format(family)][beam_n]*4.0/clight,
					blin_4sigma_s2=self.filln_StableBeamsDict['bl_interp_m_{}'.format(family)][beam_n]*4.0/clight,
					tFill_s=tFill_s,tauSRxy_s=self.tauSRxy_FT, tauSRl_s=self.tauSRl_FT, sigmaBOff_m2=self.sigmaBOff_m2, sigmaElastic_m2=0., VRF_V=self.VRF_FT,
					IBSON=1, emitBU=this_model['emitBU'], BoffON=0, blengthBU=this_model['blengthBU'], nIPs = 2., dt_s=trange[1]-trange[0],
					tau_empirical_h1=this_model['tau_empirical_h{}_noncoll'.format(beam_n)],
					tau_empirical_v1=this_model['tau_empirical_v{}_noncoll'.format(beam_n)],
					tau_empirical_h2=this_model['tau_empirical_h{}_noncoll'.format(beam_n)],
					tau_empirical_v2=this_model['tau_empirical_v{}_noncoll'.format(beam_n)], leveling=False)

					eh_interp_noncoll[beam_n] = np.array(ex_norm_m1_temp)*1.0e06
					ev_interp_noncoll[beam_n] = np.array(ey_norm_m1_temp)*1.0e06

					eh_interp_noncoll_IBScorr[beam_n] = np.array(ex_norm_m1_temp_IBScorr)*1.0e06
					ev_interp_noncoll_IBScorr[beam_n] = np.array(ey_norm_m1_temp_IBScorr)*1.0e06

					b_inten_interp_noncoll[beam_n]  = np.array(bunch_intensity_temp)
					bl_interp_m_noncoll[beam_n]     = np.array(bl_4sigma_s1_temp)/4.0*clight

				dict_save = {\
				'settings':this_model,

				'eh_interp_coll':eh_interp_coll,
				'ev_interp_coll':ev_interp_coll,
				'eh_interp_coll_IBScorr':eh_interp_coll_IBScorr,
				'ev_interp_coll_IBScorr':ev_interp_coll_IBScorr,
				'b_inten_interp_coll':b_inten_interp_coll,
				'bl_interp_m_coll':bl_interp_m_coll,
				'bunch_lumi':{'ATLAS':Luminosity_invm2s_model_ATLAS, 'CMS':Luminosity_invm2s_model_CMS},

				'eh_interp_noncoll':eh_interp_noncoll,
				'ev_interp_noncoll':ev_interp_noncoll,
				'eh_interp_noncoll_IBScorr':eh_interp_noncoll_IBScorr,
				'ev_interp_noncoll_IBScorr':ev_interp_noncoll_IBScorr,
				'b_inten_interp_noncoll':b_inten_interp_noncoll,
				'bl_interp_m_noncoll':bl_interp_m_noncoll,
				'case': dict_case,
				'time_range':trange}

				dict_models[model_name] = dict_save


			if self.saveDict:
				filename = self.fill_dir+self.SB_model_filename.replace('.pkl.gz', '_case{}.pkl.gz'.format(case))
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')

				info('# runSBModel : Saving Dictionary for SB Model of fill {} into {}'.format(filln, filename ))
				if os.path.exists(filename):
					if self.overwriteFiles:
						warn("# runSBModel : Dictionary SB Model pickle for fill {} already exists! Overwritting it...".format(filln))
						with gzip.open(filename, 'wb') as fid:
							pickle.dump(dict_models, fid)
					else:
						warn("# runSBModel : Dictionary SB Model pickle for fill {} already exists! Skipping it...".format(filln))
				else:
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(dict_models, fid)


				info('# runSBModel : Done for case {}'.format(case))

			if self.savePandas:
				pass

	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runCalculatedLuminosity(self, filln):
		'''
		Function to create the dictionary for calculated luminosity for specific fill number
		'''
		## first of all clear the calc lumi self dict
		self.filln_LumiCalcDict.clear()

		## check if there is something in the self.filln_StableBeamsDict
		if len(self.filln_StableBeamsDict) > 0:
			debug("# runCalculatedLuminosity : SB Analysis has ran for this fill, loading the SB dictionary")
			eh_interp_coll               = self.filln_StableBeamsDict['eh_interp_coll']
			ev_interp_coll               = self.filln_StableBeamsDict['ev_interp_coll']
			eh_interp_raw_coll           = self.filln_StableBeamsDict['eh_interp_raw_coll']
			ev_interp_raw_coll           = self.filln_StableBeamsDict['ev_interp_raw_coll']
			b_inten_interp_coll          = self.filln_StableBeamsDict['b_inten_interp_coll']
			bl_interp_m_coll             = self.filln_StableBeamsDict['bl_interp_m_coll']
			slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']

			eh_interp_noncoll            = self.filln_StableBeamsDict['eh_interp_noncoll']
			ev_interp_noncoll            = self.filln_StableBeamsDict['ev_interp_noncoll']
			eh_interp_raw_noncoll        = self.filln_StableBeamsDict['eh_interp_raw_noncoll']
			ev_interp_raw_noncoll        = self.filln_StableBeamsDict['ev_interp_raw_noncoll']
			b_inten_interp_noncoll       = self.filln_StableBeamsDict['b_inten_interp_noncoll']
			bl_interp_m_noncoll          = self.filln_StableBeamsDict['bl_interp_m_noncoll']
			slots_filled_noncoll         = self.filln_StableBeamsDict['slots_filled_noncoll']
			time_range                   = self.filln_StableBeamsDict['time_range']
		else:
			##populate it from file
			filename = self.fill_dir+self.SB_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			debug("# runCalculatedLuminosity : SB Analysis has NOT ran for this fill, loading the SB dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   dict_SB = pickle.load(fid)
			eh_interp_coll               = dict_SB['eh_interp_coll']
			ev_interp_coll               = dict_SB['ev_interp_coll']
			eh_interp_raw_coll           = dict_SB['eh_interp_raw_coll']
			ev_interp_raw_coll           = dict_SB['ev_interp_raw_coll']
			b_inten_interp_coll          = dict_SB['b_inten_interp_coll']
			bl_interp_m_coll             = dict_SB['bl_interp_m_coll']
			slots_filled_coll            = dict_SB['slots_filled_coll']

			eh_interp_noncoll            = dict_SB['eh_interp_noncoll']
			ev_interp_noncoll            = dict_SB['ev_interp_noncoll']
			eh_interp_raw_noncoll        = dict_SB['eh_interp_raw_noncoll']
			ev_interp_raw_noncoll        = dict_SB['ev_interp_raw_noncoll']
			b_inten_interp_noncoll       = dict_SB['b_inten_interp_noncoll']
			bl_interp_m_noncoll          = dict_SB['bl_interp_m_noncoll']
			slots_filled_noncoll         = dict_SB['slots_filled_noncoll']
			time_range                   = dict_SB['time_range']
		## Calculate beam sizes squared for B1, B2 and H, V
		sigma_h_b1 = np.sqrt(1e-6*eh_interp_coll[1]/self.gammaFT*self.betastar_m)
		sigma_v_b1 = np.sqrt(1e-6*ev_interp_coll[1]/self.gammaFT*self.betastar_m)
		sigma_h_b2 = np.sqrt(1e-6*eh_interp_coll[2]/self.gammaFT*self.betastar_m)
		sigma_v_b2 = np.sqrt(1e-6*ev_interp_coll[2]/self.gammaFT*self.betastar_m)

		## Calculate the convoluted beam sizes and the convoluted bunch length
		sigma_h_conv = np.sqrt((sigma_h_b1**2+sigma_h_b2**2)/2.)
		sigma_v_conv = np.sqrt((sigma_v_b1**2+sigma_v_b2**2)/2.)
		bl_conv = (bl_interp_m_coll[1]+bl_interp_m_coll[2])/2.

		## Calculate the reduction factor of the crossing plane and crossing angle
		FF_ATLAS = 1./np.sqrt(1.+((bl_conv/sigma_v_conv)*(self.bmodes['CrossingAngle_ATLAS'][filln]/2.))**2.)
		FF_CMS = 1./np.sqrt(1.+((bl_conv/sigma_h_conv)*(self.bmodes['CrossingAngle_CMS'][filln]/2.))**2.)

		## Net bunch by bunch luminosity
		lumi_bbb_net = self.frev*b_inten_interp_coll[1]*b_inten_interp_coll[2]/4./np.pi/(sigma_h_conv*sigma_v_conv)

		## Luminosity ATLAS/CMS bunch by bunch
		lumi_bbb_ATLAS_invm2 = lumi_bbb_net * FF_ATLAS
		lumi_bbb_CMS_invm2   = lumi_bbb_net * FF_CMS


		## Create the output dictionary that has eh/ev, intensity, length and slots
		## filled for both colliding and non colliding and save it to a pickle
		dict_save                        = {'ATLAS':{}, 'CMS':{}}
		dict_save['ATLAS']['bunch_lumi'] = lumi_bbb_ATLAS_invm2
		dict_save['CMS']['bunch_lumi']   = lumi_bbb_CMS_invm2
		dict_save['time_range']          = time_range

		## Copy these data to external dict
		debug("# runCalculatedLuminosity : Filling Lumi calc dictionary.")
		self.filln_LumiCalcDict = dict_save

		if self.saveDict:
			filename = self.fill_dir+self.Lumi_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')
			info('# runCalculatedLuminosity : Saving dictionary for Lumi Calc of fill {} into {}'.format(filln, filename ))
			if os.path.exists(filename):
				if self.overwriteFiles:
					warn("# runCalculatedLuminosity : Dictionary Lumi pickle for fill {} already exists! Overwritting it...".format(filln))
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(dict_save, fid)
				else:
					warn("# runCalculatedLuminosity : Dictionary Lumi pickle for fill {} already exists! Skipping it...".format(filln))
			else:
				with gzip.open(filename, 'wb') as fid:
					pickle.dump(dict_save, fid)

		if self.savePandas:
			filename = self.fill_dir+self.Lumi_filename.replace('.pkl.gz', '_df.pkl.gz')
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '' ).replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')

			## add a column in case it is rescaled to a period
			if self.doRescale:
				rescperiod = str(self.bmodes['rescaledPeriod'][filln])

			## Create ATLAS DF
			df_calc_ATLAS = pd.DataFrame(np.vstack(lumi_bbb_ATLAS_invm2))

			# how many slots are filled
			df_calc_ATLAS.insert(0, 'filled_slots', [np.array(lumi_bbb_ATLAS_invm2).shape[1]]*len(df_calc_ATLAS))

			## add the bunch_lumi in one cell
			df_calc_ATLAS.insert(0, 'bunch_lumi', np.array(lumi_bbb_ATLAS_invm2).tolist())

			## first add (to the leftmost) the time range
			df_calc_ATLAS.insert(0, 'timestamp', time_range)

			## then convert this to date
			df_calc_ATLAS.insert(0, 'datetime', self.convertToLocalTime(df_calc_ATLAS['timestamp']))

			## add the resc period if it is rescaled
			if self.doRescale:
				df_calc_ATLAS.insert(0, 'rescaled', [rescperiod]*len(df_calc_ATLAS))

			## then add experiment
			df_calc_ATLAS.insert(0, 'experiment', ['ATLAS']*len(df_calc_ATLAS))

			## then add fillnumber
			df_calc_ATLAS.insert(0, 'fill', [filln]*len(df_calc_ATLAS))

			## Create CMS DF
			df_calc_CMS = pd.DataFrame(np.vstack(lumi_bbb_CMS_invm2))
			# how many slots are filled
			df_calc_CMS.insert(0, 'filled_slots', [np.array(lumi_bbb_CMS_invm2).shape[1]]*len(df_calc_CMS))

			## add the bunch_lumi in one cell
			df_calc_CMS.insert(0, 'bunch_lumi', np.array(lumi_bbb_CMS_invm2).tolist())

			## first add (to the leftmost) the time range
			df_calc_CMS.insert(0, 'timestamp', time_range)

			## then convert this to date
			df_calc_CMS.insert(0, 'datetime', self.convertToLocalTime(df_calc_CMS['timestamp']))

			## add the resc period if it is rescaled
			if self.doRescale:
				df_calc_CMS.insert(0, 'rescaled', [rescperiod]*len(df_calc_CMS))

			## then add experiment
			df_calc_CMS.insert(0, 'experiment', ['CMS']*len(df_calc_CMS))

			## then add fillnumber
			df_calc_CMS.insert(0, 'fill', [filln]*len(df_calc_CMS))


			## append the two df
			total_calc_df = df_calc_ATLAS.append(df_calc_CMS)
			## set additional indexes to fill and experiment
			total_calc_df = total_calc_df.set_index(['fill', 'experiment'], drop=False)

			info('# runCalculatedLuminosity : Saving Pandas for Lumi Calc of fill {} into {}'.format(filln, filename ))
			if os.path.exists(filename):
				if self.overwriteFiles:
					warn("# runCalculatedLuminosity : Pandas Lumi pickle for fill {} already exists! Overwritting it...".format(filln))
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(total_calc_df, fid)
				else:
					warn("# runCalculatedLuminosity : Pandas Lumi pickle for fill {} already exists! Skipping it...".format(filln))
			else:
				with gzip.open(filename, 'wb') as fid:
					pickle.dump(total_calc_df, fid)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runTimberMeasuredLuminosity(self, filln):
		'''
		Get measured luminosity from Timber
		'''
		
		self.filln_LumiMeasDict.clear()
		## Get some stuff from SB dictionary
		if len(self.filln_StableBeamsDict.keys()) > 0:
			debug("# runMeasuredLuminosity : SB Analysis has ran for this fill, loading the SB dictionary")
			slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
			time_range                   = self.filln_StableBeamsDict['time_range']
		else:
			filename = self.fill_dir+self.SB_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')

			debug("# runMeasuredLuminosity : SB Analysis has NOT ran for this fill, loading the SB dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_StableBeamsDict = pickle.load(fid)
			slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
			time_range                   = self.filln_StableBeamsDict['time_range']
		
		timber_dic = {}
		timber_dic.update(tm.parse_timber_file(config.BBB_LUMI_DATA_FILE.replace('<FILLNUMBER>',str(filln)), verbose=False))
		
		self.filln_LumiMeasDict = {"ATLAS" : {'bunch_lumi': []},
									"CMS"   : {'bunch_lumi': [] } }

		for experiment in ['ATLAS', 'CMS']:
			lumi = LUMI_bbb.LUMI(timber_dic, experiment=experiment)
			
			for t in time_range:
				self.filln_LumiMeasDict[experiment]['bunch_lumi'].append(lumi.nearest_older_sample(t))

			self.filln_LumiMeasDict[experiment]['bunch_lumi'] = np.array(self.filln_LumiMeasDict[experiment]['bunch_lumi'])

		if self.saveDict:
			filename = self.fill_dir+self.Massi_filename.replace('meas', 'timber_meas')
			filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>','')
			info('# runMeasuredLuminosity : Saving dictionary for TIMBER  Meas. Lumi of fill {} into {}'.format(filln, filename ))
			if os.path.exists(filename):
				if self.overwriteFiles:
					warn("# runTimberMeasuredLuminosity : Dictionary TIMBER Meas. Lumi pickle for fill {} already exists! Overwritting it...".format(filln))
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(self.filln_LumiMeasDict, fid)
				else:
					warn("# runTimberMeasuredLuminosity : Dictionary for Meas. TIMBER Lumi pickle for fill {} already exists! Skipping it...".format(filln))
			else:
				with gzip.open(filename, 'wb') as fid:
					pickle.dump(self.filln_LumiMeasDict, fid)

		if self.savePandas:
			pass

	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runMeasuredLuminosity(self, filln):
		'''
		Function to create the dictionary for measured luminosity for specific fill number
		'''
		## first clear the measured limi dictionary
		self.filln_LumiMeasDict.clear()
		## Get some stuff from SB dictionary
		if len(self.filln_StableBeamsDict.keys()) > 0:
			debug("# runMeasuredLuminosity : SB Analysis has ran for this fill, loading the SB dictionary")
			slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
			time_range                   = self.filln_StableBeamsDict['time_range']
		else:
			filename = self.fill_dir+self.SB_filename
			filename = filename.replace('<FILLNUMBER>',str(filln))
			if self.doRescale:
				if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
					filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
				else:
					filename = filename.replace('<RESC>', '').replace("<TO>", '')
			else:
				filename = filename.replace('<RESC>', '')

			debug("# runMeasuredLuminosity : SB Analysis has NOT ran for this fill, loading the SB dictionary from pickle file [{}].".format(filename))
			with gzip.open(filename) as fid:
				   self.filln_StableBeamsDict = pickle.load(fid)
			slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']
			time_range                   = self.filln_StableBeamsDict['time_range']

		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(filln)

		slots_filled_coll = self.filln_StableBeamsDict['slots_filled_coll']
		time_range        = self.filln_StableBeamsDict['time_range']

		for experiment in ['ATLAS', 'CMS']:
			self.filln_LumiMeasDict[experiment]={}
			self.filln_LumiMeasDict[experiment]['bunch_lumi']=[]

			lumifile = config.massi_afs_path.replace('<YEAR>', str(config.massi_year))
			if experiment == 'ATLAS':
				lumifile = lumifile+config.massi_exp_folders[0]+"{}.tgz".format(filln)
			elif experiment == 'CMS':
				lumifile = lumifile+config.massi_exp_folders[1]+"{}.tgz".format(filln)
			else:
				raise IOError("# runMeasuredLuminosity : Unknown Experiment Error when running for Measured Luminosity.")

			with tarfile.open(lumifile, 'r:gz') as tarfid:
				for slot_bun in slots_filled_coll[1]:
					if debug:
						if np.mod(slot_bun, 10)==0:
							info("Experiment : {} : Bunch Slot: {} ".format(experiment, slot_bun))

					bucket = (slot_bun)*10+1

					filename_bunch = '{}/{}_lumi_{}_{}.txt'.format(filln, filln, bucket, experiment)
					fid = tarfid.extractfile(filename_bunch)
					temp_data = np.loadtxt(fid.readlines())
					t_stamps = temp_data[:,0]
					lumi_bunch = temp_data[:,2]

					self.filln_LumiMeasDict[experiment]['bunch_lumi'].append(np.interp(time_range,t_stamps, lumi_bunch)*config.massi_bunch_lumi_scale)
					fid.close()

			self.filln_LumiMeasDict[experiment]['bunch_lumi'] = np.array(self.filln_LumiMeasDict[experiment]['bunch_lumi']).T
		if self.saveDict:
			filename = self.fill_dir+self.Massi_filename
			filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>','')
			info('# runMeasuredLuminosity : Saving dictionary for Meas. Lumi of fill {} into {}'.format(filln, filename ))
			if os.path.exists(filename):
				if self.overwriteFiles:
					warn("# runMeasuredLuminosity : Dictionary Meas. Lumi pickle for fill {} already exists! Overwritting it...".format(filln))
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(self.filln_LumiMeasDict, fid)
				else:
					warn("# runMeasuredLuminosity : Dictionary for Meas. Lumi pickle for fill {} already exists! Skipping it...".format(filln))
			else:
				with gzip.open(filename, 'wb') as fid:
					pickle.dump(self.filln_LumiMeasDict, fid)

		if self.savePandas:
			df_ATLAS = pd.DataFrame(np.vstack(self.filln_LumiMeasDict['ATLAS']['bunch_lumi']))
			df_ATLAS.insert(0, 'experiment', ["ATLAS"]*len(df_ATLAS))
			df_ATLAS.insert(0, 'fill', [filln]*len(df_ATLAS))

			df_CMS   = pd.DataFrame(np.vstack(self.filln_LumiMeasDict['CMS']['bunch_lumi']))
			df_CMS.insert(0, 'experiment', ["CMS"]*len(df_CMS))
			df_CMS.insert(0, 'fill', [filln]*len(df_CMS))

			df_meas_total = df_ATLAS.append(df_CMS)
			df_meas_total = df_meas_total.set_index(['fill', 'experiment'], drop= False)

			filename = self.fill_dir+self.Massi_filename
			filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>','').replace('.pkl.gz', '_df.pkl.gz')
			info('# runMeasuredLuminosity : Saving Pandas for Meas. Lumi of fill {} into {}'.format(filln, filename ))
			if os.path.exists(filename):
				if self.overwriteFiles:
					warn("# runMeasuredLuminosity : Pandas of Meas. Lumi pickle for fill {} already exists! Overwritting it...".format(filln))
					with gzip.open(filename, 'wb') as fid:
						pickle.dump(df_meas_total, fid)
				else:
					warn("# runMeasuredLuminosity : Pandas of Meas. Lumi pickle for fill {} already exists! Skipping it...".format(filln))
			else:
				with gzip.open(filename, 'wb') as fid:
					pickle.dump(df_meas_total, fid)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def makePerformancePlotsPerFill(self, filln, doIntLifetime, doLumiTau):     ## @TODO have not checked that yet
		'''
		Function to make performance plot per fill
		Inputs : filln          : fill number
				 doIntLifetime  : Losses plot    (depends on the availability of the lifetime file)
				 doLumiTau      : Lumi Lifetime  (depends on the availability of the fits file)
		Returns: None
		'''

		## first get the times for the fill
		t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(filln)

		## Then load the necessary stuff from the SB dictionary

		eh_interp_coll               = self.filln_StableBeamsDict['eh_interp_coll']
		ev_interp_coll               = self.filln_StableBeamsDict['ev_interp_coll']
		eh_interp_raw_coll           = self.filln_StableBeamsDict['eh_interp_raw_coll']
		ev_interp_raw_coll           = self.filln_StableBeamsDict['ev_interp_raw_coll']
		b_inten_interp_coll          = self.filln_StableBeamsDict['b_inten_interp_coll']
		bl_interp_m_coll             = self.filln_StableBeamsDict['bl_interp_m_coll']
		slots_filled_coll            = self.filln_StableBeamsDict['slots_filled_coll']

		eh_interp_noncoll            = self.filln_StableBeamsDict['eh_interp_noncoll']
		ev_interp_noncoll            = self.filln_StableBeamsDict['ev_interp_noncoll']
		eh_interp_raw_noncoll        = self.filln_StableBeamsDict['eh_interp_raw_noncoll']
		ev_interp_raw_noncoll        = self.filln_StableBeamsDict['ev_interp_raw_noncoll']
		b_inten_interp_noncoll       = self.filln_StableBeamsDict['b_inten_interp_noncoll']
		bl_interp_m_noncoll          = self.filln_StableBeamsDict['bl_interp_m_noncoll']
		slots_filled_noncoll         = self.filln_StableBeamsDict['slots_filled_noncoll']
		time_range                   = self.filln_StableBeamsDict['time_range']

		## load info from calculated lumi dictionary
		lumi_bbb_ATLAS_invm2         = self.filln_LumiCalcDict['ATLAS']['bunch_lumi']
		lumi_bbb_CMS_invm2           = self.filln_LumiCalcDict['CMS']['bunch_lumi']

		## start plotting...
		pl.close('all')
		ms.mystyle_arial(self.myfontsize)

		## Figure : Emittances B1
		info("# makePerformancePlotsPerFill : Fill {} -> Making Emittances B1 plot...".format(filln))
		fig_em1 = pl.figure(1, figsize=self.fig_tuple)
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
		info("# makePerformancePlotsPerFill : Fill {} -> Making Emittances B2 plot...".format(filln))
		fig_em2 = pl.figure(2, figsize=self.fig_tuple)
		fig_em2.canvas.set_window_title('Emittances B2')
		fig_em2.set_facecolor('w')
		ax_ne2h = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
		ax_ne2v = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
		ax_ne2h_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
		ax_ne2v_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)
		fig_em2.subplots_adjust(wspace=0.5, hspace=0.5)

		## Figure : Emittances B1 Raw
		info("# makePerformancePlotsPerFill : Fill {} -> Making Emittances B1 Raw plot...".format(filln))
		fig_em1_raw = pl.figure(101, figsize=self.fig_tuple)
		fig_em1_raw.canvas.set_window_title('Emittances B1 raw')
		fig_em1_raw.set_facecolor('w')
		ax_ne1h_raw = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
		ax_ne1h_t_raw = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
		ax_ne1v_raw = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
		ax_ne1v_t_raw = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)
		fig_em1_raw.subplots_adjust(wspace=0.5, hspace=0.5)


		## Figure : Emittances B2 Raw
		info("# makePerformancePlotsPerFill : Fill {} -> Making Emittances B2 Raw plot...".format(filln))
		fig_em2_raw = pl.figure(102, figsize=self.fig_tuple)
		fig_em2_raw.canvas.set_window_title('Emittances B2 raw')
		fig_em2_raw.set_facecolor('w')
		ax_ne2h_raw = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
		ax_ne2v_raw = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
		ax_ne2h_t_raw = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
		ax_ne2v_t_raw = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)
		fig_em2_raw.subplots_adjust(wspace=0.5, hspace=0.5)

		## Figure : Bunch intensity
		info("# makePerformancePlotsPerFill : Fill {} -> Making Bunch Intensity plot...".format(filln))
		fig_int = pl.figure(3, figsize=self.fig_tuple)
		fig_int.canvas.set_window_title('Bunch intensity')
		fig_int.set_facecolor('w')
		bx_nb1 = pl.subplot(2,3,(2,3), sharex=ax_share)
		bx_nb2 = pl.subplot(2,3,(5,6), sharex=ax_share)
		bx_nb1_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=bx_nb1)
		bx_nb2_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=bx_nb2)
		fig_int.subplots_adjust(wspace=0.5, hspace=0.5)


		## Figure : Bunch Length
		info("# makePerformancePlotsPerFill : Fill {} -> Making Bunch Length plot...".format(filln))
		fig_bl = pl.figure(4, figsize=self.fig_tuple)
		fig_bl.canvas.set_window_title('Bunch length')
		fig_bl.set_facecolor('w')
		bx_bl1 = pl.subplot(2,3,(2,3), sharex=ax_share)
		bx_bl2 = pl.subplot(2,3,(5,6), sharex=ax_share)
		bx_bl1_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=bx_bl1)
		bx_bl2_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=bx_bl1)
		fig_bl.subplots_adjust(wspace=0.5, hspace=0.5)


		self.plot_mean_and_spread(ax_ne1h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne1v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_noncoll[2], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_noncoll[2], color='grey', alpha=.5)

		self.plot_mean_and_spread(ax_ne1h_t, (time_range-t_start_STABLE)/3600., eh_interp_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne1v_t, (time_range-t_start_STABLE)/3600., ev_interp_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2h_t, (time_range-t_start_STABLE)/3600., eh_interp_noncoll[2], color='grey', alpha=.5)
		self.plot_mean_and_spread(ax_ne2v_t, (time_range-t_start_STABLE)/3600., ev_interp_noncoll[2], color='grey', alpha=.5)

		self.plot_mean_and_spread(bx_nb1_t, (time_range-t_start_STABLE)/3600., b_inten_interp_noncoll[1], color='grey', alpha=.5)
		self.plot_mean_and_spread(bx_nb2_t, (time_range-t_start_STABLE)/3600., b_inten_interp_noncoll[2], color='grey', alpha=.5)

		self.plot_mean_and_spread(bx_bl1_t, (time_range-t_start_STABLE)/3600., bl_interp_m_noncoll[1]*4/clight*1e9, color='grey', alpha=.5)
		self.plot_mean_and_spread(bx_bl2_t, (time_range-t_start_STABLE)/3600., bl_interp_m_noncoll[2]*4/clight*1e9, color='grey', alpha=.5)

		self.plot_mean_and_spread(ax_ne1h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_coll[1])
		self.plot_mean_and_spread(ax_ne1v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_coll[1])
		self.plot_mean_and_spread(ax_ne2h_t_raw, (time_range-t_start_STABLE)/3600., eh_interp_raw_coll[2])
		self.plot_mean_and_spread(ax_ne2v_t_raw, (time_range-t_start_STABLE)/3600., ev_interp_raw_coll[2])

		self.plot_mean_and_spread(ax_ne1h_t, (time_range-t_start_STABLE)/3600., eh_interp_coll[1])
		self.plot_mean_and_spread(ax_ne1v_t, (time_range-t_start_STABLE)/3600., ev_interp_coll[1])
		self.plot_mean_and_spread(ax_ne2h_t, (time_range-t_start_STABLE)/3600., eh_interp_coll[2])
		self.plot_mean_and_spread(ax_ne2v_t, (time_range-t_start_STABLE)/3600., ev_interp_coll[2])

		self.plot_mean_and_spread(bx_nb1_t, (time_range-t_start_STABLE)/3600., b_inten_interp_coll[1])
		self.plot_mean_and_spread(bx_nb2_t, (time_range-t_start_STABLE)/3600., b_inten_interp_coll[2])

		self.plot_mean_and_spread(bx_bl1_t, (time_range-t_start_STABLE)/3600., bl_interp_m_coll[1]*4/clight*1e9)
		self.plot_mean_and_spread(bx_bl2_t, (time_range-t_start_STABLE)/3600., bl_interp_m_coll[2]*4/clight*1e9)

		for i_time in range(N_steps):
			colorcurr = ms.colorprog(i_prog=i_time, Nplots=N_steps)
			ax_ne1h.plot(slots_filled_coll[1], eh_interp_coll[1][i_time, :], '.', color=colorcurr)
			ax_ne1v.plot(slots_filled_coll[1], ev_interp_coll[1][i_time, :], '.', color=colorcurr)
			ax_ne2h.plot(slots_filled_coll[2], eh_interp_coll[1][i_time, :], '.', color=colorcurr)
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

			bx_nb1.plot(slots_filled_noncoll[1], b_inten_interp_noncoll[1][i_time, :], 'x', color=colorcurr)
			bx_nb2.plot(slots_filled_noncoll[2], b_inten_interp_noncoll[2][i_time, :], 'x', color=colorcurr)
			bx_bl1.plot(slots_filled_noncoll[1], bl_interp_m_noncoll[1][i_time, :]*4/clight*1e9, 'x', color=colorcurr)
			bx_bl2.plot(slots_filled_noncoll[2], bl_interp_m_noncoll[2][i_time, :]*4/clight*1e9, 'x', color=colorcurr)

			bx_nb1.plot(slots_filled_coll[1], b_inten_interp_coll[1][i_time, :], '.', color=colorcurr)
			bx_nb2.plot(slots_filled_coll[2], b_inten_interp_coll[2][i_time, :], '.', color=colorcurr)
			bx_bl1.plot(slots_filled_coll[1], bl_interp_m_coll[1][i_time, :]*4/clight*1e9, '.', color=colorcurr)
			bx_bl2.plot(slots_filled_coll[2], bl_interp_m_coll[2][i_time, :]*4/clight*1e9, '.', color=colorcurr)


		ax_ne1h_t.set_ylim(0, 10)
		ax_ne1v_t.set_ylim(0, 10)
		ax_ne2h_t.set_ylim(0, 10)
		ax_ne2v_t.set_ylim(0, 10)


		for sp in [ax_ne1h, ax_ne1v, ax_ne2h, ax_ne2v,bx_nb1, bx_nb2, bx_bl1, bx_bl2, ax_ne1h_raw, ax_ne1v_raw, ax_ne2h_raw, ax_ne2v_raw]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

		for sp in [ax_ne1h_t, ax_ne1v_t, ax_ne2h_t, ax_ne2v_t, bx_nb1_t, bx_nb2_t, bx_bl1_t, bx_bl2_t, ax_ne1h_t_raw, ax_ne1v_t_raw, ax_ne2h_t_raw, ax_ne2v_t_raw]:
			sp.grid('on')
			sp.minorticks_on()
			sp.set_xlabel('Time [h]', fontsize=14, fontweight='bold')

		ax_ne1h_t.set_ylabel('Emittance B1H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne1v_t.set_ylabel('Emittance B1V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2h_t.set_ylabel('Emittance B2H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2v_t.set_ylabel('Emittance B2V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')

		ax_ne1h_t_raw.set_ylabel('Emittance B1H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne1v_t_raw.set_ylabel('Emittance B1V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2h_t_raw.set_ylabel('Emittance B2H [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
		ax_ne2v_t_raw.set_ylabel('Emittance B2V [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')


		bx_nb1_t.set_ylabel('Intensity B1 [p/b]'  , fontsize=14, fontweight='bold')
		bx_nb2_t.set_ylabel('Intensity B2 [p/b]'  , fontsize=14, fontweight='bold')
		bx_bl1_t.set_ylabel('Bunch length B1 [ns]', fontsize=14, fontweight='bold')
		bx_bl2_t.set_ylabel('Bunch length B1 [ns]', fontsize=14, fontweight='bold')

		## ------------ Now plot for expected lumi
		info("# makePerformancePlotsPerFill : Fill {} -> Making Expected BBB Luminosities plot...".format(filln))
		fig_lumi_calc = pl.figure(5, figsize=self.fig_tuple)
		fig_lumi_calc.canvas.set_window_title('Expected bbb lumis')
		fig_lumi_calc.set_facecolor('w')
		ax_ATLAS_calc = pl.subplot(2,3,(2,3), sharex=ax_share)
		ax_CMS_calc = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=ax_ATLAS_calc)
		ax_ATLAS_calc_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=ax_ATLAS_calc)
		ax_CMS_calc_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=ax_ATLAS_calc)
		fig_lumi_calc.subplots_adjust(wspace=0.5, hspace=0.5)

		for i_time in range(0, N_steps, self.n_skip):
			colorcurr = ms.colorprog(i_prog=i_time, Nplots=N_steps)
			ax_ATLAS_calc.plot(slots_filled_coll[1], lumi_bbb_ATLAS_invm2[i_time, :], '.', color=colorcurr)
			ax_CMS_calc.plot(slots_filled_coll[1], lumi_bbb_CMS_invm2[i_time, :], '.', color=colorcurr)

		self.plot_mean_and_spread(ax_ATLAS_calc_t, (time_range-t_start_STABLE)/3600., lumi_bbb_ATLAS_invm2)
		self.plot_mean_and_spread(ax_CMS_calc_t,   (time_range-t_start_STABLE)/3600., lumi_bbb_CMS_invm2)

		## Figure for Raw Measured Luminosity for ATLAS/CMS as taken from Massi files
		if not self.skipMassi:
			info("# makePerformancePlotsPerFill : Fill {} -> Making Measured BBB Luminosities plot...".format(filln))
			fig_lumi_meas   = pl.figure(7, figsize=self.fig_tuple)
			fig_lumi_meas.canvas.set_window_title('Measured bbb lumi raw')
			fig_lumi_meas.set_facecolor('w')

			ax_ATLAS_meas   = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=ax_ATLAS_calc)
			ax_CMS_meas     = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=ax_ATLAS_calc)
			ax_ATLAS_meas_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=ax_ATLAS_calc)
			ax_CMS_meas_t   = pl.subplot(2,3,4, sharex=ax_share_t, sharey=ax_ATLAS_calc)
			fig_lumi_meas.subplots_adjust(wspace=0.5, hspace=0.5)

			self.plot_mean_and_spread(ax_ATLAS_meas_t, (time_range-t_start_STABLE)/3600., self.filln_LumiMeasDict['ATLAS']['bunch_lumi'])
			self.plot_mean_and_spread(ax_CMS_meas_t,   (time_range-t_start_STABLE)/3600., self.filln_LumiMeasDict['CMS']['bunch_lumi'])


			for i_time in range(0, N_steps, self.n_skip):
				colorcurr = ms.colorprog(i_prog=i_time, Nplots=N_steps)
				ax_ATLAS_meas.plot(slots_filled_coll[1], self.filln_LumiMeasDict['ATLAS']['bunch_lumi'][i_time, :], '.', color=colorcurr)
				ax_CMS_meas.plot(slots_filled_coll[1],   self.filln_LumiMeasDict['CMS']['bunch_lumi'][i_time, :],   '.', color=colorcurr)

				for sp in [ax_ATLAS_meas, ax_CMS_meas]:
					sp.grid('on')
					sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

				for sp in [ax_ATLAS_meas_t, ax_CMS_meas_t]:
					sp.grid('on')
					sp.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
					sp.set_xlim(0, -(t_start_STABLE-t_end_STABLE)/3600.)

				ax_ATLAS_meas_t.set_ylabel('Meas. Luminosity ATLAS [m$\mathbf{^{2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')
				ax_CMS_meas_t.set_ylabel('Meas. Luminosity CMS [m$\mathbf{^{2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')
				ax_CMS_meas.set_xlim(0, 3564)


		for sp in [ax_ATLAS_calc, ax_CMS_calc]:
			sp.grid('on')
			sp.set_xlabel("Bunch Slots [25ns]", fontsize=14, fontweight='bold')

		for sp in [ax_ATLAS_calc_t, ax_CMS_calc_t]:
			sp.grid('on')
			sp.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			sp.set_xlim(0, -(t_start_STABLE-t_end_STABLE)/3600.)

		ax_ATLAS_calc_t.set_ylabel('Calc. Luminosity ATLAS [m$\mathbf{^{2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')
		ax_CMS_calc_t.set_ylabel('Calc. Luminosity CMS [m$\mathbf{^{2}}$ s$\mathbf{^{-1}}$]', fontsize=12, fontweight='bold')


		# Lifetime can be ran without running the full lifetime stuff
		if self.skipMassi:
			dict_lifetime = {1:{}, 2: {}}
			dNdt_bbb            = {}
			dt = self.filln_StableBeamsDict['time_range'][1]-self.filln_StableBeamsDict['time_range'][0]
			tau_Np_bbb          = {}
			for beam_n in [1,2 ]:
				dNp_bbb                   = -(b_inten_interp_coll[beam_n][:-1,:]) + (b_inten_interp_coll[beam_n][1:,:])
				dNdt_bbb[beam_n]          = (np.abs(dNp_bbb)/dt)
				tau_Np_bbb[beam_n]        = -1/((dNp_bbb/dt)/b_inten_interp_coll[beam_n][:-1,:])
				dict_lifetime[beam_n]['tau_Np_bbb'] = tau_Np_bbb[beam_n]
		else:
			dict_lifetime = self.filln_LifetimeDict


		# Figure Intensity Lifetime
		info("# makePerformancePlotsPerFill : Fill {} -> Making BBB Lifetime...".format(filln))
		fig_bbb_tau = pl.figure('bbb_tau', figsize=(15,7))
		ax_b1_bbbtau = pl.subplot(211)
		ax_b2_bbbtau = pl.subplot(212)

		self.plot_mean_and_spread(ax_b1_bbbtau, (time_range[0:-1]-time_range[0])/3600., dict_lifetime[1]['tau_Np_bbb']/3600., label='Beam 1 - $\\tau_{N_{p}^{0}}$'+'={:.2f}h'.format(np.mean(dict_lifetime[1]['tau_Np_bbb']/3600., axis=1)[0]), color='b', shade=True)
		ax_b1_bbbtau.grid('on')
		ax_b1_bbbtau.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
		ax_b1_bbbtau.set_ylabel("$\mathbf{\\tau_{N_{p}}}$ [h]", fontsize=14, fontweight='bold')
		ax_b1_bbbtau.legend(loc='best')

		self.plot_mean_and_spread(ax_b2_bbbtau, (time_range[0:-1]-time_range[0])/3600., dict_lifetime[2]['tau_Np_bbb']/3600., label='Beam 2 - $\\tau_{N_{p}^{0}}$'+'={:.2f}h'.format(np.mean(dict_lifetime[2]['tau_Np_bbb']/3600., axis=1)[0]), color='r', shade=True)
		ax_b2_bbbtau.grid('on')
		ax_b2_bbbtau.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
		ax_b2_bbbtau.set_ylabel("$\mathbf{\\tau_{N_{p}}}$ [h]", fontsize=14, fontweight='bold')
		ax_b2_bbbtau.legend(loc='best')
		pl.subplots_adjust(hspace=0.5, wspace=0.5)#, hspace=0.7)


		## Figure of Total Luminosity Measured and Calculated for ATLAS/CMS (sum up bbb)
		if not self.skipMassi:
			info("# makePerformancePlotsPerFill : Fill {} -> Making Total Luminosity plot...".format(filln))
			fig_total = pl.figure(8, figsize=self.fig_tuple)
			fig_total.canvas.set_window_title('Total Luminosity')
			fig_total.set_facecolor('w')
			pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1),       color='b', linewidth=2., label="$\mathcal{L}^{meas}_{ATLAS}$")
			pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(lumi_bbb_ATLAS_invm2,              axis=1), '--', color='b', linewidth=2., label="$\mathcal{L}^{calc}_{ATLAS}$")
			pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(self.filln_LumiMeasDict['CMS']['bunch_lumi'],   axis=1),       color='r', linewidth=2., label="$\mathcal{L}^{meas}_{CMS}$")
			pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(lumi_bbb_CMS_invm2,                axis=1), '--', color='r', linewidth=2., label="$\mathcal{L}^{calc}_{CMS}$")
			pl.legend(loc='best', prop={"size":12})
			pl.xlim(-.5, None)
			pl.ylim(0, None)
			pl.ylabel('Luminosity [cm$\mathbf{^2}$ s$\mathbf{^{-1}}$]', fontweight='bold', fontsize=14) ## NK: cm??? ATLAS??
			pl.xlabel('Time [h]', fontsize=14, fontweight='bold')
			pl.grid('on')

			if doIntLifetime:
				info("# makePerformancePlotsPerFill : Fill {} -> Making BBB Intensity Lifetime plot...".format(filln))
				## Figure for Lifeime BBB
				fig_bbblifetime = pl.figure('bbb lifetime', figsize=(15,7))
				ax_lifetime5    = pl.subplot(2,2,1)
				ax_lifetime6    = pl.subplot(2,2,2)
				ax_lifetime7    = pl.subplot(2,2,3)
				ax_lifetime8    = pl.subplot(2,2,4)

				self.plot_mean_and_spread(ax_lifetime5, (time_range[0:-1]-time_range[0])/3600., self.filln_LifetimeDict[1]['life_time_Boff_bbb']/3600., label = 'Beam 1', color='b', shade=True)
				ax_lifetime5.grid('on')
				ax_lifetime5.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
				ax_lifetime5.set_ylabel('B1 $\mathbf{1\slash\\tau{N_{p}}}$ (Burn Off Corrected) [1/h]', fontsize=10, fontweight='bold')


				self.plot_mean_and_spread(ax_lifetime6, (time_range[0:-1]-time_range[0])/3600., self.filln_LifetimeDict[2]['life_time_Boff_bbb']/3600., color='r', label='Beam 2', shade=True)
				ax_lifetime6.grid('on')
				ax_lifetime6.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
				ax_lifetime6.set_ylabel('B2 $\mathbf{1\slash\\tau{N_{p}}}$ (Burn Off Corrected) [1/h]', fontsize=10, fontweight='bold')


				ax_lifetime7.plot(self.filln_LifetimeDict[2]['slots'], 3600./self.filln_LifetimeDict[1]['life_time_Boff_bbb'].T,  ls='None', marker='o', markersize=4, label='Beam 1')
				ax_lifetime7.grid('on')
				ax_lifetime7.set_xlabel('Bunch Slot [25ns]', fontsize=14, fontweight='bold')
				ax_lifetime7.set_ylabel('B1 $\mathbf{1\slash\\tau{N_{p}}}$ (Burn Off Corrected) [1/h]', fontsize=10, fontweight='bold')

				ax_lifetime8.plot(self.filln_LifetimeDict[2]['slots'], 3600./self.filln_LifetimeDict[2]['life_time_Boff_bbb'].T,  ls='None', marker='o', markersize=4, label='Beam 2')
				ax_lifetime8.grid('on')
				ax_lifetime8.set_xlabel('Bunch Slot [25ns]', fontsize=14, fontweight='bold')
				ax_lifetime8.set_ylabel('B2 $\mathbf{1\slash\\tau{N_{p}}}$ (Burn Off Corrected) [1/h]', fontsize=10, fontweight='bold')
				pl.subplots_adjust(wspace=0.4, hspace=0.4)


				# Figure BBB Losses
				info("# makePerformancePlotsPerFill : Fill {} -> Making BBB Normalized Losses plot...".format(filln))
				fig_bbblosses = pl.figure('bbb_losses', figsize=(15,7))
				ax_b1 = pl.subplot(211)
				ax_b2 = pl.subplot(212)

				self.plot_mean_and_spread(ax_b1, (time_range[0:-1]-time_range[0])/3600., self.filln_LifetimeDict[1]['losses_dndtL_bbb']*1.0e31, label='Beam 1', color='b', shade=True)
				ax_b1.axhline(80, xmin=0, xmax=1, color='black', label='$\sigma_{\mathrm{inel}}$ = 80mb')
				ax_b1.grid('on')
				ax_b1.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
				ax_b1.set_ylabel("$\mathbf{\left(\\frac{dN}{dt}\\right)\slash\mathcal{L}}$ [mb]", fontsize=14, fontweight='bold')
				ax_b1.legend(loc='best')

				self.plot_mean_and_spread(ax_b2, (time_range[0:-1]-time_range[0])/3600., self.filln_LifetimeDict[2]['losses_dndtL_bbb']*1.0e31, label='Beam 2', color='r', shade=True)
				ax_b2.axhline(80, xmin=0, xmax=1, color='black', label='$\sigma_{\mathrm{inel}}$ = 80mb')
				ax_b2.grid('on')
				ax_b2.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
				ax_b2.set_ylabel("$\mathbf{\left(\\frac{dN}{dt}\\right)\slash\mathcal{L}}$ [mb]", fontsize=14, fontweight='bold')
				ax_b2.legend(loc='best')
				pl.subplots_adjust(hspace=0.5)#, hspace=0.7)


			# let's make the tau lumi plot here:
			if doLumiTau:

				# for the full
				info("# makePerformancePlotsPerFill : Fill {} -> Making Luminosity Lifetime Total plot...".format(filln))
				fig_tauLumi_tot = pl.figure('tauLumiTotal', figsize=(15,7))
				tau_tot_AT  = pl.subplot(211)
				tau_tot_AT.plot(slots_filled_coll[1], self.filln_SBFitsDict['ATLAS']['tau_lumi_calc_coll_full']/3600., c='g', marker='o', markersize=4, ls='None', label='ATLAS - Total Calculated')
				tau_tot_AT.plot(slots_filled_coll[1], self.filln_SBFitsDict['ATLAS']['tau_lumi_meas_coll_full']/3600., c='k', marker='o', markersize=4, ls='None', label='ATLAS - Total Measured')
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
				tau_tot_CMS.plot(slots_filled_coll[1], self.filln_SBFitsDict['CMS']['tau_lumi_calc_coll_full']/3600, c='g', marker='o', markersize=4, ls='None',  label='CMS - Total Calculated')#ls='dotted',
				tau_tot_CMS.plot(slots_filled_coll[1], self.filln_SBFitsDict['CMS']['tau_lumi_meas_coll_full']/3600, c='k', marker='o', markersize=4, ls='None',  label='CMS - Total Measured')  #ls='dotted',
				tau_tot_CMS.grid('on')
				tau_tot_CMS.set_ylabel('$\mathbf{\\tau_{\mathcal{L}}}$ [h]', fontsize=14, fontweight='bold')
				tau_tot_CMS.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
				# tau_tot_CMS.text(0.5, 1.0, "CMS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
				# tau_tot_CMS.text(0.1, 0.9, "CMS", horizontalalignment='center', verticalalignment='top', transform=tau_tot_CMS.transAxes,
				#              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)
				tau_tot_CMS.legend(loc='best', numpoints=1)
				pl.subplots_adjust(hspace=0.5)

				# Only for the fitted time
				info("# makePerformancePlotsPerFill : Fill {} -> Making Luminosity Lifetime (Fit) plot...".format(filln))
				fig_tauLumi_fit = pl.figure('tauLumiFit', figsize=(15,7))
				tau_fit_AT  = pl.subplot(211)
				tau_fit_AT.plot(slots_filled_coll[1], self.filln_SBFitsDict['ATLAS']['tau_lumi_calc_coll']/3600., c='g', marker='o', markersize=4, ls='None', label='ATLAS - {}h Calculated'.format(int(self.t_fit_length/3600.)))
				tau_fit_AT.plot(slots_filled_coll[1], self.filln_SBFitsDict['ATLAS']['tau_lumi_meas_coll']/3600., c='k', marker='o', markersize=4, ls='None', label='ATLAS - {}h Measured'.format(int(self.t_fit_length/3600.)))
				tau_fit_AT.grid('on')
				tau_fit_AT.set_ylabel('$\mathbf{\\tau_{\mathcal{L}}}$ [h]', fontsize=14, fontweight='bold')
				tau_fit_AT.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
				# tau_fit_AT.text(0.5, 1.0, "ATLAS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
				# tau_fit_AT.text(0.1, 0.9, "ATLAS", horizontalalignment='center', verticalalignment='top', transform=tau_fit_AT.transAxes,
				#              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)
				tau_fit_AT.legend(loc='best', numpoints=1)

				tau_fit_CMS = pl.subplot(212)
				tau_fit_CMS.plot(slots_filled_coll[1], self.filln_SBFitsDict['CMS']['tau_lumi_calc_coll']/3600., c='g', marker='o', markersize=4, ls='None', label='CMS - {}h Calculated'.format(int(self.t_fit_length/3600.)))
				tau_fit_CMS.plot(slots_filled_coll[1], self.filln_SBFitsDict['CMS']['tau_lumi_meas_coll']/3600., c='k', marker='o', markersize=4, ls='None', label='CMS - {}h Measured'.format(int(self.t_fit_length/3600.)))
				tau_fit_CMS.grid('on')
				# tau_fit_CMS.set_ylabel('$\mathbf{\\tau_{\mathcal{L}}}$ [h]', fontsize=14, fontweight='bold')
				tau_fit_CMS.set_xlabel('Bunch Slots [25ns]', fontsize=14, fontweight='bold')
				# tau_fit_CMS.text(0.5, 1.0, "CMS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
				# tau_fit_CMS.text(0.1, 0.9, "CMS", horizontalalignment='center', verticalalignment='top', transform=tau_fit_CMS.transAxes,
				#              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7), fontweight='bold', fontsize=12)
				tau_fit_CMS.legend(loc='best', numpoints=1)
				pl.subplots_adjust(hspace=0.5, wspace=0.5)

		print 'Skip massi ' , self.skipMassi
		if not self.skipMassi:
			figlist = [fig_total, fig_int, fig_em1, fig_em2, fig_em1_raw, fig_em2_raw, fig_bl, fig_lumi_calc, fig_lumi_meas, fig_bbb_tau]
			if doIntLifetime:
				# figlist.append(fit_totlifetime)
				figlist.append(fig_bbblifetime)
				figlist.append(fig_bbblosses)
			if doLumiTau:
				figlist.append(fig_tauLumi_tot)
				figlist.append(fig_tauLumi_fit)
		else:
			figlist = [fig_int, fig_em1, fig_em2, fig_em1_raw, fig_em2_raw, fig_bl, fig_lumi_calc, fig_bbb_tau]



		for ff in figlist:
			tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
			ff.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(filln, tref_string), fontsize=self.myfontsize)




		if not self.batch:
			pl.show()

		if self.savePlots:
			# timeString = datetime.now().strftime("%Y%m%d")
			saveString = self.plotFormat
			figDic     = {  fig_em1         :'fill_{}_b1Emittances{}'.format(filln, saveString),
							fig_em2         :'fill_{}_b2Emittances{}'.format(filln, saveString),
							fig_em1_raw     :'fill_{}_b1RawEmittances{}'.format(filln, saveString),
							fig_em2_raw     :'fill_{}_b2RawEmittances{}'.format(filln, saveString),
							fig_int         :'fill_{}_bunchIntensity{}'.format(filln, saveString),
							fig_bl          :'fill_{}_bunchLength{}'.format(filln, saveString),
							fig_lumi_calc   :'fill_{}_calcLumi{}'.format(filln, saveString),
							fig_bbb_tau     :'fill_{}_bbbIntensityTau{}'.format(filln, saveString),
			}

			if not self.skipMassi:
				figDic.update({     fig_lumi_meas   :'fill_{}_measLumi{}'.format(filln, saveString),
									fig_total       :'fill_{}_totalLumi{}'.format(filln, saveString)
									})

			if doIntLifetime:
				figDic.update({ fig_bbblifetime       :'fill_{}_bbbLifetime{}'.format(filln, saveString),
								fig_bbblosses         :'fill_{}_bbbLosses{}'.format(filln, saveString)})

			if doLumiTau:
				figDic.update({ fig_tauLumi_tot       :'fill_{}_TotalLumiLifetime{}'.format(filln, saveString),
								fig_tauLumi_fit       :'fill_{}_FitLumiLifetime{}'.format(filln, saveString)
								})


			for ifig in figDic.keys():
				figname = self.plot_dir+figDic[ifig]
				figname = figname.replace('<FILLNUMBER>',str(filln))
				ifig.savefig(figname, dpi=self.plotDpi)

			if self.makePlotTarball:
				tar = tarfile.open(self.plot_dir+"performancePlots_{}.tar.gz".format(timeString), "w:gz") ##NK @TODO do bz2
				for name in glob.glob(self.plot_dir+"*{}.{}".format(timeString, plotFormat)):
					tar.add(name)
				tar.close()

		info("makePerformancePlotsPerFill : Performance plots for fill {} done.".format(filln))
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def makeCycleModelPlots(self, filln, t_start_fill, t_end_fill, t_start_STABLE, t_end_STABLE):
		'''
		Creates emittances and bunch length comparison plots between cycle data and cycle model.
		Inputs : filln          = fill number
				 t_start_fill   = timestamp of start fill
				 t_end_fill     = timestamp of end fill
				 t_start_STABLE = timestamp of start SB
				 t_end_STABLE   = timestamp of end SB
		Returns: None
		'''
		t_fill_len = t_end_fill - t_start_fill
		t_ref = t_start_fill
		t_min = 10.0*60.0
		for interval, tag in zip(['Injection', 'he_before_SB'], ["INJ", "FT"]):

			for beam_n in [1,2]:
				pl.close('all')
				info("makeCycleModelPlots : Making Emittance plots for {} {}...".format(tag, beam_n))
				# Emittance plot
				fig_emit = pl.figure('emittances', figsize=(15,7))
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


			info("makeCycleModelPlots : Making bunch length plot for {}...".format(tag))
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

			if config.savePlots:
				# timestr  = datetime.now().strftime("%Y%m%d")
				filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))
				filename = filename + "fill_{}_cycle_model_bunchLength_{}".format(filln, tag)+self.plotFormat
				fig_blen.savefig(filename, dpi=self.plotDpi)
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def makeSBModelPlots(self, filln, case):
		'''
		Makes plots for the model currently loaded in self.filln_SBModelDict
		Inputs : filln : fill number
				 case  : number for the case
		Returns: None
		'''

		model_names = self.SB_models
		tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(self.filln_StableBeamsDict['time_range'][0]))
		pl.close('all')

		fig_model_comp  = pl.figure('model comnparison',figsize=(15,7))
		axis_ATLAS      = pl.subplot(2,1,1)
		axis_CMS        = pl.subplot(2,1,2)


		for model_name in model_names:
			pl.close('all')
			tt = self.filln_SBModelDict[model_name]['time_range'][0:-1]
			axis_ATLAS.plot((tt-tt[0])/3600.,np.sum(self.filln_SBModelDict[model_name]['bunch_lumi']['ATLAS'],axis=1)*1.0e-04,'',linewidth=2.,label='{}'.format(model_name))
			axis_CMS.plot((tt-tt[0])/3600.  ,np.sum(self.filln_SBModelDict[model_name]['bunch_lumi']['CMS'],axis=1)*1.0e-04,'',linewidth=2.,label='{}'.format(model_name))




			info("# makeSBModelPlots : Making SB Model Luminosity plot for Model : {} , Case : {}".format(model_name, case))
			## --- Luminosity
			title_string = "{} - Case {}".format(model_name, case)
			fig_lumi = pl.figure('luminosity_case_{}'.format(case), figsize = (15,7))
			ax_AT = pl.subplot(211)
			self.plot_mean_and_spread(ax_AT, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., np.array(self.filln_SBModelDict[model_name]['bunch_lumi']['ATLAS']), color='g', label='ATLAS - Model', shade=True)
			self.plot_mean_and_spread(ax_AT, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., np.array(self.filln_LumiMeasDict['ATLAS']['bunch_lumi']), color='k', label='ATLAS - Measured', shade=True)
			ax_AT.set_ylim(0, None)
			ax_AT.grid('on')
			ax_AT.set_ylabel('Luminosity [$\mathbf{m^{-2} s^{-1}}$]', fontsize=14, fontweight='bold')
			ax_AT.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_AT.set_xlim(0, (self.filln_SBModelDict[model_name]['time_range'][-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600.)
			# ax_AT.text(0.5, 1.0e34, "ATLAS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
			ax_AT.legend(loc='best')

			ax_CMS = pl.subplot(212)
			self.plot_mean_and_spread(ax_CMS, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., np.array(self.filln_SBModelDict[model_name]['bunch_lumi']['CMS']), color='g', label='CMS - Model', shade=True)
			# self.plot_mean_and_spread(ax_CMS, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., np.array(self.filln_LumiMeasDict['CMS']['bunch_lumi']), color='k', label='Measured', shade=True)
			self.plot_mean_and_spread(ax_CMS, (self.filln_StableBeamsDict['time_range']-self.filln_StableBeamsDict['time_range'][0])/3600., np.array(self.filln_LumiMeasDict['CMS']['bunch_lumi']), color='k', label='CMS - Measured', shade=True)
			ax_CMS.set_ylim(0, None)
			ax_CMS.grid('on')
			ax_CMS.set_ylabel('Luminosity [$\mathbf{m^{-2} s^{-1}}$]', fontsize=14, fontweight='bold')
			ax_CMS.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_CMS.set_xlim(0, (self.filln_SBModelDict[model_name]['time_range'][-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600.)
			# ax_CMS.text(0.5, 1.0e34, "CMS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
			ax_CMS.legend(loc='best')


			fig_lumi.suptitle('Fill {}: STABLE BEAMS declared on {}\n{}'.format(filln, tref_string, title_string), fontsize=18)
			pl.subplots_adjust(top=0.85, bottom=0.1, hspace=0.3)
			if config.savePlots:
				
				filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))
				filename = filename + "fill_{}_sbmodel_{}_luminosity_case{}".format(filln, model_name, case)+self.plotFormat
				fig_lumi.savefig(filename, dpi=self.plotDpi)



			### --- Emittances colliding
			info("# makeSBModelPlots : Making SB Model Emittance Colliding plot for Model : {} , Case : {}".format(model_name, case))
			fig_emit_coll = pl.figure('emittance_coll_case_{}'.format(case), figsize=(15,7))
			title_string = "Colliding - {} - Case {}".format(model_name, case)

			ax_b1_h  = pl.subplot(221)
			self.plot_mean_and_spread(ax_b1_h, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['eh_interp_coll'][1], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_h, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['eh_interp_coll'][1], color='k', label='Measured', shade=True)
			ax_b1_h.set_ylabel('B1 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b1_h.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_h.legend(loc='best')
			ax_b1_h.grid('on')

			ax_b1_v  = pl.subplot(223)
			self.plot_mean_and_spread(ax_b1_v, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['ev_interp_coll'][1], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_v, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['ev_interp_coll'][1], color='k', label='Measured', shade=True)
			ax_b1_v.set_ylabel('B1 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b1_v.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_v.legend(loc='best')
			ax_b1_v.grid('on')

			ax_b2_h  = pl.subplot(222)
			self.plot_mean_and_spread(ax_b2_h, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['eh_interp_coll'][2], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b2_h, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['eh_interp_coll'][2], color='k', label='Measured', shade=True)
			ax_b2_h.set_ylabel('B2 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b2_h.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_h.legend(loc='best')
			ax_b2_h.grid('on')

			ax_b2_v  = pl.subplot(224)
			self.plot_mean_and_spread(ax_b2_v, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['ev_interp_coll'][2], color='g',label='Model',  shade=True)
			self.plot_mean_and_spread(ax_b2_v, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['ev_interp_coll'][2], color='k', label='Measured', shade=True)
			ax_b2_v.set_ylabel('B2 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b2_v.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_v.legend(loc='best')
			ax_b2_v.grid('on')
			fig_emit_coll.suptitle('Fill {}: STABLE BEAMS declared on {}\n{}'.format(filln, tref_string, title_string), fontsize=18)
			pl.subplots_adjust(top=0.85, bottom=0.1, hspace=0.3)
			if self.savePlots:
				filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))
				filename = filename + "fill_{}_sbmodel_{}_emittanceColliding_case{}".format(filln, model_name, case)+self.plotFormat
				fig_emit_coll.savefig(filename, dpi=self.plotDpi)

			### --- Emittances non colliding
			info("# makeSBModelPlots : Making SB Model Emittance Non-Colliding plot for Model : {} , Case : {}".format(model_name, case))
			fig_emit_noncoll = pl.figure('emittance_noncoll_case_{}'.format(case), figsize=(15,7))
			title_string = "Non-Colliding - {} - Case {}".format(model_name, case)

			ax_b1_h_non  = pl.subplot(221)
			self.plot_mean_and_spread(ax_b1_h_non, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['eh_interp_noncoll'][1], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_h_non, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['eh_interp_noncoll'][1], color='k', label='Measured', shade=True)
			ax_b1_h_non.set_ylabel('B1 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b1_h_non.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_h_non.legend(loc='best')
			ax_b1_h_non.grid('on')

			ax_b1_v_non  = pl.subplot(223)
			self.plot_mean_and_spread(ax_b1_v_non, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['ev_interp_noncoll'][1], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_v_non, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['ev_interp_noncoll'][1], color='k', label='Measured', shade=True)
			ax_b1_v_non.set_ylabel('B1 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b1_v_non.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_v_non.legend(loc='best')
			ax_b1_v_non.grid('on')

			ax_b2_h_non  = pl.subplot(222)
			self.plot_mean_and_spread(ax_b2_h_non, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['eh_interp_noncoll'][2], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b2_h_non, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['eh_interp_noncoll'][2], color='k', label='Measured', shade=True)
			ax_b2_h_non.set_ylabel('B2 $\mathbf{\epsilon_{H}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b2_h_non.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_h_non.legend(loc='best')
			ax_b2_h_non.grid('on')

			ax_b2_v_non  = pl.subplot(224)
			self.plot_mean_and_spread(ax_b2_v_non, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['ev_interp_noncoll'][2], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b2_v_non, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['ev_interp_noncoll'][2], color='k', shade=True, label='Measured')
			ax_b2_v_non.set_ylabel('B2 $\mathbf{\epsilon_{V}}$ [$\mathbf{\mu}$m]', fontsize=14, fontweight='bold')
			ax_b2_v_non.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_v_non.legend(loc='best')
			ax_b2_v_non.grid('on')
			fig_emit_noncoll.suptitle('Fill {}: STABLE BEAMS declared on {}\n{}'.format(filln, tref_string, title_string), fontsize=18)
			pl.subplots_adjust(top=0.85, bottom=0.1, hspace=0.3)
			if self.savePlots:
				filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))
				filename = filename + "fill_{}_sbmodel_{}_emittanceNonColliding_case{}".format(filln, model_name, case)+self.plotFormat
				fig_emit_noncoll.savefig(filename, dpi=self.plotDpi)


			### --- Intensity and bunch length colliding
			info("# makeSBModelPlots : Making SB Model Intensity/Blength Colliding plot for Model : {} , Case : {}".format(model_name, case))
			fig_intBlength_coll = pl.figure('intensity_blength_coll_case_{}'.format(case), figsize=(15,7))
			title_string = "Colliding - {} - Case {}".format(model_name, case)

			ax_b1_int_coll  = pl.subplot(221)
			self.plot_mean_and_spread(ax_b1_int_coll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['b_inten_interp_coll'][1], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_int_coll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['b_inten_interp_coll'][1], color='k', label='Measured', shade=True)
			ax_b1_int_coll.set_ylabel('B1 Intensity [p]', fontsize=14, fontweight='bold')
			ax_b1_int_coll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_int_coll.legend(loc='best')
			ax_b1_int_coll.grid('on')

			ax_b1_blen_coll  = pl.subplot(223)
			self.plot_mean_and_spread(ax_b1_blen_coll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['bl_interp_m_coll'][1]*4.0/clight*1.0e09, color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_blen_coll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['bl_interp_m_coll'][1]*4.0/clight*1.0e09, color='k', label='Measured',shade=True)
			ax_b1_blen_coll.set_ylabel('B1 Bunch Length [ns]', fontsize=12, fontweight='bold')
			ax_b1_blen_coll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_blen_coll.legend(loc='best')
			ax_b1_blen_coll.grid('on')

			ax_b2_int_coll  = pl.subplot(222)
			self.plot_mean_and_spread(ax_b2_int_coll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['b_inten_interp_coll'][2], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b2_int_coll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['b_inten_interp_coll'][2], color='k', label='Measured',shade=True)
			ax_b2_int_coll.set_ylabel('B2 Intensity [p]', fontsize=14, fontweight='bold')
			ax_b2_int_coll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_int_coll.legend(loc='best')
			ax_b2_int_coll.grid('on')

			ax_b2_blen_coll  = pl.subplot(224)
			self.plot_mean_and_spread(ax_b2_blen_coll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['bl_interp_m_coll'][2]*4.0/clight*1.0e09, color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b2_blen_coll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['bl_interp_m_coll'][2]*4.0/clight*1.0e09, color='k', label='Measured', shade=True)
			ax_b2_blen_coll.set_ylabel('B2 Bunch Length [ns]', fontsize=12, fontweight='bold')
			ax_b2_blen_coll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_blen_coll.legend(loc='best')
			ax_b2_blen_coll.grid('on')
			fig_intBlength_coll.suptitle('Fill {}: STABLE BEAMS declared on {}\n{}'.format(filln, tref_string, title_string), fontsize=18)
			pl.subplots_adjust(top=0.85, bottom=0.1, hspace=0.3)
			if config.savePlots:
				filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))
				filename = filename + "fill_{}_sbmodel_{}_intens_blength_Colliding_case{}".format(filln, model_name, case)+self.plotFormat
				fig_intBlength_coll.savefig(filename, dpi=self.plotDpi)


			### --- Intensity and bunch length noncolliding
			info("# makeSBModelPlots : Making SB Model Intensity/Blength Non-Colliding plot for Model : {} , Case : {}".format(model_name, case))
			fig_intBlength_noncoll = pl.figure('intensity_blength_noncoll_case_{}'.format(case), figsize=(15,7))
			title_string = "Non-Colliding - {} - Case {}".format(model_name, case)

			ax_b1_int_noncoll  = pl.subplot(221)
			self.plot_mean_and_spread(ax_b1_int_noncoll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['b_inten_interp_noncoll'][1], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_int_noncoll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['b_inten_interp_noncoll'][1], color='k', label='Measured', shade=True)
			ax_b1_int_noncoll.set_ylabel('B1 Intensity [p]', fontsize=14, fontweight='bold')
			ax_b1_int_noncoll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_int_noncoll.legend(loc='best')
			ax_b1_int_noncoll.grid('on')

			ax_b1_blen_noncoll  = pl.subplot(223)
			self.plot_mean_and_spread(ax_b1_blen_noncoll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['bl_interp_m_noncoll'][1]*4.0/clight*1.0e09, color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b1_blen_noncoll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['bl_interp_m_noncoll'][1]*4.0/clight*1.0e09, color='k', label='Measured',shade=True)
			ax_b1_blen_noncoll.set_ylabel('B1 Bunch Length [ns]', fontsize=12, fontweight='bold')
			ax_b1_blen_noncoll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b1_blen_noncoll.legend(loc='best')
			ax_b1_blen_noncoll.grid('on')

			ax_b2_int_noncoll  = pl.subplot(222)
			self.plot_mean_and_spread(ax_b2_int_noncoll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['b_inten_interp_noncoll'][2], color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b2_int_noncoll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['b_inten_interp_noncoll'][2], color='k', label='Measured',shade=True)
			ax_b2_int_noncoll.set_ylabel('B2 Intensity [p]', fontsize=14, fontweight='bold')
			ax_b2_int_noncoll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_int_noncoll.legend(loc='best')
			ax_b2_int_noncoll.grid('on')

			ax_b2_blen_noncoll  = pl.subplot(224)
			self.plot_mean_and_spread(ax_b2_blen_noncoll, (self.filln_SBModelDict[model_name]['time_range'][:-1]-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_SBModelDict[model_name]['bl_interp_m_noncoll'][2]*4.0/clight*1.0e09, color='g', label='Model', shade=True)
			self.plot_mean_and_spread(ax_b2_blen_noncoll, (self.filln_StableBeamsDict['time_range']-self.filln_SBModelDict[model_name]['time_range'][0])/3600., self.filln_StableBeamsDict['bl_interp_m_noncoll'][2]*4.0/clight*1.0e09, color='k', label='Measured',shade=True)
			ax_b2_blen_noncoll.set_ylabel('B2 Bunch Length [ns]', fontsize=12, fontweight='bold')
			ax_b2_blen_noncoll.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			ax_b2_blen_noncoll.legend(loc='best')
			ax_b2_blen_noncoll.grid('on')
			fig_intBlength_noncoll.suptitle('Fill {}: STABLE BEAMS declared on {}\n{}'.format(filln, tref_string, title_string), fontsize=18)
			pl.subplots_adjust(top=0.85, bottom=0.1, hspace=0.3)
			if self.savePlots:
				filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))
				filename = filename + "fill_{}_sbmodel_{}_intens_blength_NonColliding_case{}".format(filln, model_name, case)+self.plotFormat
				fig_intBlength_noncoll.savefig(filename, dpi=self.plotDpi)

		if len(model_names)>1:
			info("# makeSBModelPlots : Making SB Model Comparison Luminosity plot for Case : {}".format(case))
			axis_ATLAS.set_ylim(0, None)
			axis_ATLAS.grid('on')
			axis_ATLAS.set_ylabel('Luminosity [$\mathbf{m^{-2} s^{-1}}$]', fontsize=14, fontweight='bold')
			axis_ATLAS.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			axis_ATLAS.text(0.5, 1.0e34, "ATLAS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
			axis_ATLAS.legend(loc='best')

			axis_CMS.set_ylim(0, None)
			axis_CMS.grid('on')
			axis_CMS.set_ylabel('Luminosity [$\mathbf{m^{-2} s^{-1}}$]', fontsize=14, fontweight='bold')
			axis_CMS.set_xlabel('Time [h]', fontsize=14, fontweight='bold')
			axis_CMS.text(0.5, 1.0e34, "CMS", fontsize=14, bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.7))
			axis_CMS.legend(loc='best')
			pl.subplots_adjust(top=0.85, bottom=0.1, hspace=0.3)
			fig_model_comp.suptitle('Fill {}: STABLE BEAMS declared on {}\n{}'.format(filln, tref_string, title_string), fontsize=18)


			if self.savePlots:
				filename = self.plot_dir.replace("<FILLNUMBER>", str(filln))
				filename = filename + "fill_{}_sbmodel_modelComp_luminosity_case{}".format(filln, case)+self.plotFormat
				fig_model_comp.savefig(filename, dpi=self.plotDpi)

	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runForFill(self, filln):
		'''
		Runs the full analysis for one fill
		'''
		## First check if the directories for this fill exist
		self.checkDirectories(filln)
		## Then see if the files for this fill exist and see what can be done
		getMassi, self.skipMassi, getTimber, doSB, doSBFits, doSBModel, doLifetime, doLumiCalc, doCycle, doCycle_model, skip = self.checkFiles(filln)
		print '---->', getMassi, self.skipMassi, getTimber, doSB, doSBFits, doSBModel, doLifetime, doLumiCalc, doCycle, doCycle_model, skip

		if skip:
			warn("#runForFill : I was ordered to skip fill {} from checkFiles... Skipping...".format(filln))
			return

		# print getMassi, doSB, doLumiCalc, doCycle, doCycle_model

		## Check if I have forces something (or everything)
		if self.force:
			info('#runForFill : User initialized run with the [force] option on {}'.format(self.force))
			doCycle       = self.forceCycle
			doCycle_model = self.forceCycle_model
			getMassi      = self.forceMeasLumi
			doSB          = self.forceSB
			doLumiCalc    = self.forceCalcLumi
			doSBFits      = self.forceSBFits
			doSBModel     = self.forceSBModel
			doLifetime    = self.forceLifetime

		## If everything is in place and nothing has to be done, return
		if not self.doSBPlots and not self.doCyclePlots and not self.doCycleModelPlots and not self.doSBModelPlots:
			if not getMassi and not getTimber and not doSB and not doLumiCalc and not doCycle and not doCycle_model and not doSBFits and not doSBModel and not doLifetime:
				warn("#runForFill: Nothing to do for fill {}. [getMassi = {}, getTimber = {}, doSB = {}, doLumiCalc = {}, doCycle = {}, doCycle_model = {}, doSBFits = {}, doSBModel = {}, doLifetime ={},  doSBPlots = {}]".format(filln, getMassi, getTimber, doSB, doLumiCalc, doCycle, doCycle_model, doSBFits, doSBModel, doLifetime, self.doSBPlots))
				return

		## Do I have to run cycle?
		if doCycle or doSB:
			info('#runForFill : Starting loop for Cycle and/or Stable beams for fill {}'.format(filln))
			## i merge cycle and sb in the same function since the same loop is performed
			self.runCycleSB(filln, doCycle, doSB)
		else:
			info('#runForFill : No need to loop for Cycle and/or Stable beams for fill {} [doCycle = {} | doSB = {}]'.format(filln, doCycle, doSB))

		if doLumiCalc:
			info('#runForFill : Starting loop for Calculated Luminosity for fill {}'.format(filln))
			self.runCalculatedLuminosity(filln)
		else:
			info('#runForFill : No need to loop for Calculated Luminosity for fill {} [doLumiCalc = {}]'.format(filln, doLumiCalc))

		if doCycle_model:
			info('#runForFill : Starting loop for Cycle Model for fill {}'.format(filln))
			self.runCycleModel(filln)
		else:
			info('#runForFill : No need to loop for Cycle Model for fill {} [doLumiCalc = {}]'.format(filln, doCycle_model))

		if getMassi: # or getTimber
			if not self.skipMassi:
				info('#runForFill : Starting loop for Measured Luminosity for fill {}'.format(filln))
				self.runMeasuredLuminosity(filln)
			# elif getTimber:
			# 	info('#runForFill : Starting loop for TIMBER Measured Luminosity for fill {}'.format(filln))
			# 	self.runTimberMeasuredLuminosity(filln)
		else:
			info('#runForFill : No need to loop for Measured Luminosity for fill {} [getMassi = {}]'.format(filln,getMassi))

		if doSBFits and not self.skipMassi:
			info('#runForFill : Starting loop for SB Fits for fill {}'.format(filln))
			self.runSBFits(filln)
		else:
			info('#runForFill : No need to loop for SB Fits for fill {} [doSBFits = {}]'.format(filln,doSBFits))


		print 'SBMODEL / SKIP MASSI / GET TIMBER = ', doSBModel, self.skipMassi, getTimber

		if doSBModel and not self.skipMassi:
			info('#runForFill : Starting loop for SB Model for fill {}'.format(filln))
			self.runSBModel(filln)
		else:
			info('#runForFill : No need to loop for SB Model for fill {} [doSBModel = {}]'.format(filln,doSBModel))

		if doLifetime and not self.skipMassi:
			info('#runForFill : Starting loop for Lifetime for fill {}'.format(filln))
			self.runLifetime(filln)
		else:
			info('#runForFill : No need to loop for Lifetime for fill {} [doLifetime = {}]'.format(filln,doLifetime))



		## Now I should have ran for anything I need for this fill
		## I have to check if the self.dictionaries for SB & Meas. Calc. Lumis are filled, else load them from pickle

		if self.doSBPlots:
			##-- SB first
			if len(self.filln_StableBeamsDict) == 0:
				filename = self.fill_dir+self.SB_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')

				debug("#runForFill : I have not ran for SB data for fill {} . Loading it from file : {}.".format(filln, filename))
				with gzip.open(filename) as fid:
					self.filln_StableBeamsDict = pickle.load(fid)
				debug("#runForFill : I have not ran for SB data for fill {} . Loading from file -> SB data dict keys = {}".format(filln, self.filln_StableBeamsDict.keys()))
			else:
				debug("#runForFill : I have ran for SB data for fill {} . SB data dict keys = {}".format(filln, self.filln_StableBeamsDict.keys()))

			##-- Calculated Lumi second
			if len(self.filln_LumiCalcDict.keys()) == 0:
				filename = self.fill_dir+self.Lumi_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')

				debug("#runForFill : I have not ran for Lumi Calc for fill {} . Loading it from file : {}.".format(filln, filename))
				with gzip.open(filename) as fid:
					self.filln_LumiCalcDict = pickle.load(fid)
				debug("#runForFill : I have not ran for Lumi Calc data for fill {} . Loading from file -> Lumi calc dict keys = {}".format(filln, self.filln_LumiCalcDict.keys()))
			else:
				debug("#runForFill : I have ran for Lumi Calc for fill {} . Lumi calc dict keys = {}".format(filln, self.filln_LumiCalcDict.keys()))

			##-- Measured Lumi third
			if not self.skipMassi:
				if len(self.filln_LumiMeasDict.keys()) == 0:
					filename = self.fill_dir+self.Massi_filename
					filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>', '')  ## measured lumi does not get rescaled
					debug("#runForFill : I have not ran for Lumi Meas for fill {} . Loading it from file : {}.".format(filln, filename))
					with gzip.open(filename) as fid:
						self.filln_LumiMeasDict = pickle.load(fid)
					debug("#runForFill : I have not ran for Meas Calc data for fill {} . Loading from file -> Meas calc dict keys = {}".format(filln, self.filln_LumiMeasDict.keys()))
				else:
					debug("#runForFill : I have ran for Lumi Meas for fill {} . Lumi calc dict keys = {}".format(filln, self.filln_LumiMeasDict.keys()))

				debug('#runForFill : I have all the info for fill {}. Running performance plots.'.format(filln))
			# elif getTimber:
			# 	if len(self.filln_LumiMeasDict.keys()) == 0:
			# 		filename = self.fill_dir+self.Massi_filename.replace('meas', 'timber_meas')
			# 		filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>', '')  ## measured lumi does not get rescaled
			# 		debug("#runForFill : I have not ran for TIMBER Lumi Meas for fill {} . Loading it from file : {}.".format(filln, filename))
			# 		with gzip.open(filename) as fid:
			# 			self.filln_LumiMeasDict = pickle.load(fid)
			# 		debug("#runForFill : I have not ran for TIMBER Meas Calc data for fill {} . Loading from file -> Meas calc dict keys = {}".format(filln, self.filln_LumiMeasDict.keys()))
			# 	else:
			# 		debug("#runForFill : I have ran for TIMBER Lumi Meas for fill {} . Lumi calc dict keys = {}".format(filln, self.filln_LumiMeasDict.keys()))


				##-- Lifetime final
				if len(self.filln_LifetimeDict.keys()) == 0:
					filename = self.fill_dir+self.SB_filename.replace('.pkl.gz', '_lifetime.pkl.gz')
					filename = filename.replace('<FILLNUMBER>',str(filln))
					if self.doRescale:
						if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
							filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
						else:
							filename = filename.replace('<RESC>', '').replace("<TO>", '')
					else:
						filename = filename.replace('<RESC>', '')

					debug("#runForFill : I have not ran for Lifetime for fill {} . Loading it from file : {}.".format(filln, filename))
					with gzip.open(filename) as fid:
						self.filln_LifetimeDict = pickle.load(fid)
					debug("#runForFill : I have not ran for Lifetime data for fill {} . Loading from file -> lifetime dict keys = {}".format(filln, self.filln_LifetimeDict.keys()))
				else:
					debug("#runForFill : I have ran for Lifetime for fill {} . Lifetime dict keys = {}".format(filln, self.filln_LifetimeDict.keys()))

				##-- fits final
				if len(self.filln_SBFitsDict.keys()) == 0:
					filename = self.fill_dir+self.SB_fits_filename
					filename = filename.replace('<FILLNUMBER>',str(filln))
					if self.doRescale:
						if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
							filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
						else:
							filename = filename.replace('<RESC>', '').replace("<TO>", '')
					else:
						filename = filename.replace('<RESC>', '')

					debug("#runForFill : I have not ran for SB Fits for fill {} . Loading it from file : {}.".format(filln, filename))
					with gzip.open(filename) as fid:
						self.filln_SBFitsDict = pickle.load(fid)
					debug("#runForFill : I have not ran for SB Fits data for fill {} . Loading from file -> SB Fits dict keys = {}".format(filln, self.filln_SBFitsDict.keys()))
				else:
					debug("#runForFill : I have ran for SB Fits for fill {} . SB Fits dict keys = {}".format(filln, self.filln_SBFitsDict.keys()))


			if len(self.filln_LifetimeDict.keys()) > 0:
				doIntLifetime = True
			else:
				doIntLifetime = False

			if len(self.filln_SBFitsDict.keys()) > 0:
				doLumiTau = True
			else:
				doLumiTau = False


			self.makePerformancePlotsPerFill(filln, doIntLifetime, doLumiTau)
			# self.makeLifetimePlots(filln)



		if self.doCyclePlots: ## doCycle:
			debug('#runForFill : Has Cycle ran for fill {}? [doCycle = {}]'.format(filln, doCycle))
			## if the cycle has not ran but I still want to update cycle plots I need to rerun for this:
			if len(self.filln_CycleDict.keys()) == 0:
				## I need to get the info for cycle and run cycle plots
				filename = self.fill_dir+self.Cycle_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')

				debug("#runForFill : I have not ran for Cycle for fill {} . Loading it from file : {}.".format(filln, filename))
				with gzip.open(filename) as fid:
					self.filln_CycleDict = pickle.load(fid)
				## debug("#runForFill : I have not ran for SB data for fill {} . Loading from file -> SB data dict keys = {}".format(filln, self.filln_StableBeamsDict.keys()))
			else:
				debug("#runForFill : I have ran for Cycle for fill {} . Cycle dict keys = {}".format(filln, self.filln_CycleDict.keys()))

			t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(filln)
			self.makeCyclePlots(self.filln_CycleDict, filln, t_ref)

		if self.doCycleModelPlots:
			debug('#runForFill : Has Cycle Model ran for fill {}? [doCycle_model = {}]'.format(filln, doCycle_model))
			## if the cycle has not ran but I still want to update cycle plots I need to rerun for this:
			if len(self.filln_CycleDict.keys()) == 0:
				## I need to get the info for cycle and run cycle plots
				filename = self.fill_dir+self.Cycle_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')

				debug("#runForFill : I have not ran for Cycle for fill {} . Loading it from file : {}.".format(filln, filename))
				with gzip.open(filename) as fid:
					self.filln_CycleDict = pickle.load(fid)
				## debug("#runForFill : I have not ran for SB data for fill {} . Loading from file -> SB data dict keys = {}".format(filln, self.filln_StableBeamsDict.keys()))
			else:
				debug("#runForFill : I have ran for Cycle for fill {} . Cycle dict keys = {}".format(filln, self.filln_CycleDict.keys()))

			if len(self.filln_CycleModelDict.keys()) == 0:
				## I need to get the info for cycle and run cycle plots
				filename = self.fill_dir+self.Cycle_model_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')

				debug("#runForFill : I have not ran for Cycle Model for fill {} . Loading it from file : {}.".format(filln, filename))
				with gzip.open(filename) as fid:
					self.filln_CycleModelDict = pickle.load(fid)
				## debug("#runForFill : I have not ran for SB data for fill {} . Loading from file -> SB data dict keys = {}".format(filln, self.filln_StableBeamsDict.keys()))
			else:
				debug("#runForFill : I have ran for Cycle Model for fill {} . Cycle dict keys = {}".format(filln, self.filln_CycleModelDict.keys()))

			t_start_fill, t_end_fill, t_fill_len, t_ref = self.getCycleDataTimes(filln)
			t_start_STABLE, t_end_STABLE, time_range, N_steps = self.getSBDataTimes(filln)
			self.makeCycleModelPlots(filln, t_start_fill, t_end_fill, t_start_STABLE, t_end_STABLE)

		if self.doSBModelPlots:
			##-- Measured Lumi third
			if len(self.filln_LumiMeasDict.keys()) == 0:
				filename = self.fill_dir+self.Massi_filename
				filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>', '')  ## measured lumi does not get rescaled
				debug("#runForFill : I have not ran for Lumi Meas for fill {} . Loading it from file : {}.".format(filln, filename))
				with gzip.open(filename) as fid:
					self.filln_LumiMeasDict = pickle.load(fid)
				debug("#runForFill : I have not ran for Meas Calc data for fill {} . Loading from file -> Meas calc dict keys = {}".format(filln, self.filln_LumiMeasDict.keys()))
			else:
				debug("#runForFill : I have ran for Lumi Meas for fill {} . Lumi calc dict keys = {}".format(filln, self.filln_LumiMeasDict.keys()))

			debug('#runForFill : I have all the info for fill {}. Running performance plots.'.format(filln))

			# stable beams
			if len(self.filln_StableBeamsDict) == 0:
				filename = self.fill_dir+self.SB_filename
				filename = filename.replace('<FILLNUMBER>',str(filln))
				if self.doRescale:
					if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
						filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace('<RESC>', '').replace("<TO>", '')
				else:
					filename = filename.replace('<RESC>', '')

				debug("#runForFill : I have not ran for SB data for fill {} . Loading it from file : {}.".format(filln, filename))
				with gzip.open(filename) as fid:
					self.filln_StableBeamsDict = pickle.load(fid)
				debug("#runForFill : I have not ran for SB data for fill {} . Loading from file -> SB data dict keys = {}".format(filln, self.filln_StableBeamsDict.keys()))
			else:
				debug("#runForFill : I have ran for SB data for fill {} . SB data dict keys = {}".format(filln, self.filln_StableBeamsDict.keys()))
			# SB Model
			if len(self.cases) == 1:
				if len(self.filln_SBModelDict) == 0:
					filename = self.fill_dir+self.SB_model_filename.replace('.pkl.gz','_case{}.pkl.gz'.format(self.cases[0]))
					filename = filename.replace('<FILLNUMBER>',str(filln))
					if self.doRescale:
						if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
							filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
						else:
							filename = filename.replace('<RESC>', '').replace("<TO>", '')
					else:
						filename = filename.replace('<RESC>', '')

					debug("#runForFill : I have not ran for SB model for fill {} . Loading it from file : {}.".format(filln, filename))
					with gzip.open(filename) as fid:
						self.filln_SBModelDict = pickle.load(fid)
					debug("#runForFill : I have not ran for SB model for fill {} . Loading from file -> SB model dict keys = {}".format(filln, self.filln_SBModelDict.keys()))
				else:
					debug("#runForFill : I have ran for SB model for fill {} . SB model dict keys = {}".format(filln, self.filln_SBModelDict.keys()))
				self.makeSBModelPlots(filln, self.cases[0])

			else:
				for case in self.cases:
					filename = self.fill_dir+self.SB_model_filename.replace('.pkl.gz','_case{}.pkl.gz'.format(case))
					filename = filename.replace('<FILLNUMBER>',str(filln))
					if self.doRescale:
						if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
							filename = filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
						else:
							filename = filename.replace('<RESC>', '').replace("<TO>", '')
					else:
						filename = filename.replace('<RESC>', '')

					debug("#runForFill : Opening file for SB model for fill {} . Loading it from file : {}.".format(filln, filename))
					with gzip.open(filename) as fid:
						self.filln_SBModelDict = pickle.load(fid)
					debug("#runForFill : SB model for fill {} . Loading from file -> SB model dict keys = {}".format(filln, self.filln_SBModelDict.keys()))
					self.makeSBModelPlots(filln, case)
					self.filln_SBModelDict.clear()
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def runForFillList(self):
		'''
		Function to loop over all fills in filln_list
		'''
		## @TODO ADD SUBMISSION TO LSF?
		info('# runForFillList : Running loop for fills : {}'.format(self.filln_list))

		for filln in self.filln_list:
			# clear local dictionaries
			self.filln_CycleDict.clear()
			self.filln_CycleModelDict.clear()
			self.filln_CycleModelInj2SBDict.clear()
			self.filln_StableBeamsDict.clear()
			self.filln_SBFitsDict.clear()
			self.filln_SBModelDict.clear()
			self.filln_LumiCalcDict.clear()
			self.filln_LumiMeasDict.clear()
			self.filln_LifetimeDict.clear()
			# try:
			self.runForFill(filln)
			# except:
			#   os.rmdir(self.plot_dir.replace('<FILLNUMBER>',str(filln)))
			#   os.rmdir(self.fill_dir.replace('<FILLNUMBER>',str(filln)))
			#   warn("#runForFillList : Issues with Fill {}. Look at the logs.".format(filln))

		info('# runForFillList : Done for all fills.')
	## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
	def printConfig(self):
		'''
		Function to print the configuration of the object
		'''
		c = ''' Lumi Follow Up Configuration
		## --- initialization options
		debug                   = {}
		batch                   = {}
		FORMAT                  = {}
		logfile                 = {}

		## --- input data files
		fills_bmodes_file       = {}
		BASIC_DATA_FILE         = {}
		BBB_DATA_FILE           = {}

		##  --- fill & Stable beams info
		fill_list               = {}
		min_time_SB             = {}
		first_fill              = {}
		last_fill               = {}
		t_step_sec              = {}
		intensity_threshold     = {}
		enable_smoothing_BSRT   = {}
		avg_time_smoothing      = {}

		## --- BSRT Rescaling info
		doRescale               = {}
		resc_period             = {}
		resc_string             = {}

		## --- Folder and filenames
		makedirs                = {}
		overwriteFiles          = {}
		SB_dir                  = {}
		fill_dir                = {}
		plot_dir                = {}
		SB_filename             = {}
		Cycle_filename          = {}
		Cycle_model_filename    = {}
		Lumi_filename           = {}


		## --- Massi File
		Massi_filename          = {}
		Massi_fill_Database     = {}
		Massi_year              = {}
		Massi_afs_path          = {}
		massi_exp_folders       = {}

		## --- Machine Changes and parameters
		frev                    = {}
		gammaFT                 = {}
		gammaFB                 = {}
		betastar_m              = {}
		crossingAngleChange     = {}
		XingAngle               = {}

		## --- Output & Plot info
		saveDict                = {}
		savePandas              = {}
		doOnly                  = {}
		force                   = {}
		forceCycle              = {}
		forceCycle_model        = {}
		forceSB                 = {}
		forceMeasLumi           = {}
		forceCalcLumi           = {}
		savePlots               = {}

		doCyclePlots            = {}
		doCycleModelPlots       = {}
		doSBPlots               = {}
		fig_tuple               = {}
		plotFormat              = {}
		plotDpi                 = {}
		myfontsize              = {}
		n_skip                  = {}
		makePlotTarball         = {}
		'''.format(self.debug, self.batch, self.FORMAT, self.logfile, self.fills_bmodes_file, self.BASIC_DATA_FILE, self.BBB_DATA_FILE, self.filln_list, self.min_time_SB, self.first_fill, self.last_fill, self.t_step_sec, self.intensity_threshold,
self.enable_smoothing_BSRT, self.avg_time_smoothing, self.doRescale, self.resc_period, self.resc_string, self.makedirs, self.overwriteFiles, self.SB_dir, self.fill_dir, self.plot_dir, self.SB_filename, self.Cycle_filename, self.Cycle_model_filename, self.Lumi_filename,
self.Massi_filename, self.fill_yaml_database, self.fill_year, self.massi_afs_path, self.massi_exp_folders, self.frev, self.gammaFT, self.gammaFB, self.betastar_m, self.crossingAngleChange, self.XingAngle, self.saveDict, self.savePandas, self.doOnly, self.force,
self.forceCycle, self.forceCycle_model, self.forceSB, self.forceMeasLumi, self.forceCalcLumi, self.savePlots, self.doCyclePlots, self.doCycleModelPlots, self.doSBPlots, self.fig_tuple, self.plotFormat, self.plotDpi, self.myfontsize, self.n_skip, self.makePlotTarball)

		info(c)

	def makeSummaryLumi(self):  ## @TODO is not yet implemented!!
		'''
		Function to make summary pickle for filln_list
		'''

		N_step_delay = 6

		if len(self.summaryLumi.keys()) > 0:
			warn("# makeSummaryLumi : Summary Luminosity Dictionary is not empty. Clearing it...")
			self.summaryLumi.clear()


		experiments_list = ['ATLAS', 'CMS']
		self.summaryLumi = {}
		for exp in experiments_list:
			self.summaryLumi[exp+'_meas_init_lumi'] = {}
			self.summaryLumi[exp+'_meas_init_lumi']['avg'] = []
			self.summaryLumi[exp+'_meas_init_lumi']['rms'] = []
			self.summaryLumi[exp+'_meas_init_lumi']['sum'] = []
			self.summaryLumi[exp+'_n_colliding'] = []
			self.summaryLumi[exp+'_calc_init_lumi'] = {}
			self.summaryLumi[exp+'_calc_init_lumi']['avg'] = []
			self.summaryLumi[exp+'_calc_init_lumi']['rms'] = []
			self.summaryLumi[exp+'_calc_init_lumi']['sum'] = []
		self.summaryLumi['filln'] = self.filln_list

		for filln in self.filln_list:
			info("# makeSummaryLumi : Fill {}".format(filln))

			try:
				filename = self.fill_dir+self.Massi_filename
				filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>', '')
				with gzip.open(filename) as fid:
					dict_lumi_meas = pickle.load(fid)

				for exp in experiments_list:
					self.summaryLumi[exp+'_meas_init_lumi']['avg'].append(np.mean(dict_lumi_meas[exp]['bunch_lumi'][N_step_delay, :]))
					self.summaryLumi[exp+'_meas_init_lumi']['rms'].append(np.std(dict_lumi_meas[exp]['bunch_lumi'][N_step_delay, :]))
					self.summaryLumi[exp+'_meas_init_lumi']['sum'].append(sum(dict_lumi_meas[exp]['bunch_lumi'][N_step_delay, :]))
					self.summaryLumi[exp+'_n_colliding'].append(len(dict_lumi_meas[exp]['bunch_lumi'][N_step_delay, :]))

			except IOError as err:
				warn('# makeSummaryLumi : Skipped! Got: {}'.format(err))
				for exp in experiments_list:
					self.summaryLumi[exp+'_meas_init_lumi']['avg'].append(-1.)
					self.summaryLumi[exp+'_meas_init_lumi']['rms'].append(-1.)
					self.summaryLumi[exp+'_meas_init_lumi']['sum'].append(-1.)
					self.summaryLumi[exp+'_n_colliding'].append(-1)
			except IndexError as err:
				warn('# makeSummaryLumi : Skipped! Got: {}'.format(err))
				for exp in experiments_list:
					self.summaryLumi[exp+'_meas_init_lumi']['avg'].append(-1.)
					self.summaryLumi[exp+'_meas_init_lumi']['rms'].append(-1.)
					self.summaryLumi[exp+'_meas_init_lumi']['sum'].append(-1.)
					self.summaryLumi[exp+'_n_colliding'].append(-1)


			try:
				filename = self.fill_dir+self.Lumi_filename
				filename = filename.replace('<FILLNUMBER>',str(filln)).replace('<RESC>', self.resc_string)

				if self.doRescale:
					if bmodes['period'][filln] != bmodes['rescaledPeriod'][filln]:
						filename = filename.replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
					else:
						filename = filename.replace("<TO>", '')

					with gzip.open('SB_analysis/fill_%d/fill_%d_lumi_calc_nk2_CC.pkl.gz'%(filln, filln)) as fid:
						dict_lumis = pickle.load(fid)


					for exp in experiments_list:
						self.summaryLumi[exp+'_calc_init_lumi']['avg'].append(np.mean(dict_lumis[exp]['bunch_lumi'][0, :]))
						self.summaryLumi[exp+'_calc_init_lumi']['rms'].append(np.std(dict_lumis[exp]['bunch_lumi'][0, :]))
						self.summaryLumi[exp+'_calc_init_lumi']['sum'].append(sum(dict_lumis[exp]['bunch_lumi'][0, :]))



			except IOError as err:
				print 'Skipped! Got:'
				print err
				for exp in experiments_list:
					self.summaryLumi[exp+'_calc_init_lumi']['avg'].append(-1.)
					self.summaryLumi[exp+'_calc_init_lumi']['rms'].append(-1)
					self.summaryLumi[exp+'_calc_init_lumi']['sum'].append(-1.)

		## if self.saveDict:
		##     filename = self.SB_dir+"summary_pickle_lumi_"+str(self.first_fill)+"_"+str(self.last_fill)+"_<RESC>.pkl.gz"
		##     if self.doRescale
		##     +"_{}{}{}".format(str(self.resc_period[0][1]), str(self.resc_period[1][1]), str(self.resc_period[2][1]))+".pkl.gz"
		##
		##     if self.doRescale:
		##         if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
		##             filename = filename.replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
		##         else:
		##             filename = filename.replace("<TO>", '')
		##     info('# runCalculatedLuminosity : Saving dictionary for Lumi Calc of fill {} into {}'.format(filln, filename ))
		##     if os.path.exists(filename):
		##         if self.overwriteFiles:
		##             warn("# runCalculatedLuminosity : Dictionary Lumi pickle for fill {} already exists! Overwritting it...".format(filln))
		##             with gzip.open(filename, 'wb') as fid:
		##                 pickle.dump(dict_save, fid)
		##         else:
		##             warn("# runCalculatedLuminosity : Dictionary Lumi pickle for fill {} already exists! Skipping it...".format(filln))
		##     else:
		##         with gzip.open(filename, 'wb') as fid:
		##             pickle.dump(dict_save, fid)




if __name__ == '__main__':
	fl = LumiFollowUp()
	## fl.runForFill(5456)
	## fl.runForFillList()
