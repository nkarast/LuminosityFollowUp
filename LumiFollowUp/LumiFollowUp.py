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
BIN = os.path.expanduser("/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/LumiModel_FollowUp/autoScriptTesting/")
sys.path.append(BIN)

import matplotlib
matplotlib.use('Agg')  #### needed for batch jobs
print("# LumiFollowUp - Setting Matplotlib Backend to Agg.")
import matplotlib.pyplot as pl

import LHCMeasurementTools.LHC_BCT as BCT
import LHCMeasurementTools.LHC_FBCT as FBCT
import LHCMeasurementTools.LHC_Energy as Energy
import LHCMeasurementTools.LHC_BSRT as BSRT
import LHCMeasurementTools.LHC_BQM as BQM
import LHCMeasurementTools.TimberManager as tm
import LHCMeasurementTools.mystyle as ms
import LHCMeasurementTools.LHC_Lumi as LUMI
import LHCMeasurementTools.LHC_Lumi_bbb as LUMI_bbb
import BSRT_calib_rescale as BSRT_calib
import pandas as pd
import numpy as np
import pickle
import time
from operator import add
from scipy.constants import c as clight
import gzip
import tarfile
from glob import glob
from logging import *
from datetime import datetime
import argparse
import socket
import Utilities.readYamlDB as db


class LumiFollowUp(object):
    def __init__(self, debug=True, batch=True,  FORMAT='%(asctime)s %(levelname)s : %(message)s',
                loglevel = 20, logfile = None, fills_bmodes_file = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/fills_and_bmodes.pkl', ## @TODO bmodes df to be saved?
                min_time_SB = 30*60, first_fill = 5005, last_fill = 5456+1,
                t_step_sec = 10*60, intensity_threshold = 3.0e10, enable_smoothing_BSRT = True,
                avg_time_smoothing = 3.0*3600.0, periods = {'A': (5005,  5256), 'B': (5256,     5405), 'C': (5405,     5456+1)},
                doRescale = True, resc_period = [('A', 'C'), ('B','C'), ('C', 'C')], add_resc_string = '',
                BASIC_DATA_FILE = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/fill_basic_data_csvs/basic_data_fill_<FILLNUMBER>.csv',
                BBB_DATA_FILE = '/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_<FILLNUMBER>.csv',
                makedirs = True, overwriteFiles = False, SB_dir = 'SB_analysis/',
                fill_dir = "fill_<FILLNUMBER>/", plot_dir = "plots/",
                SB_filename = "fill_<FILLNUMBER><RESC>.pkl.gz",
                Cycle_filename = "fill_<FILLNUMBER>_cycle<RESC>.pkl.gz",
                Lumi_filename = "fill_<FILLNUMBER>_lumi_calc<RESC>.pkl.gz",
                Massi_filename = 'fill_<FILLNUMBER>_lumi_meas.pkl.gz',
                saveDict = True, savePandas = False, force=False, frev = 11245.5, gamma = 6927.64, betastar_m = 0.40,
                crossingAngleChange = True, XingAngle = {(5005,  5330):2*185e-6, (5330, 5456+1): 2*140e-6},
                savePlots = True, fig_tuple = (17, 10), plotFormat = ".pdf",
                plotDpi = 300, myfontsize = 16, n_skip = 1,  ## xrange step for time
                makePlotTarball = False, doOnly = False, fill=None,
                doCyclePlots=True, doSBPlots=True, doSummaryPlots = False, doPlots = False, submit=False,
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
        self.Cycle_filename          = Cycle_filename
        self.Lumi_filename           = Lumi_filename
        self.Massi_filename          = Massi_filename

        ## --- Machine Changes and parameters
        self.frev                    = frev
        self.gamma                   = gamma
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
            self.forceSB             = True
            self.forceMeasLumi       = True
            self.forceCalcLumi       = True
            warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format('all'))
            self.overwriteFiles      = True
        elif self.force == 'cycle':
            self.forceCycle          = True
            self.forceSB             = False
            self.forceMeasLumi       = False
            self.forceCalcLumi       = False
            warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
            self.overwriteFiles      = True
        elif self.force == 'sb':
            self.forceCycle          = False
            self.forceSB             = True
            self.forceMeasLumi       = False
            self.forceCalcLumi       = False
            warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
            self.overwriteFiles      = True
        elif self.force == 'massi':
            self.forceCycle          = False
            self.forceSB             = False
            self.forceMeasLumi       = True
            self.forceCalcLumi       = False
            warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
            self.overwriteFiles      = True
        elif self.force == 'lumi':
            self.forceCycle          = False
            self.forceSB             = False
            self.forceMeasLumi       = False
            self.forceCalcLumi       = True
            warn("# LumiFollowUp : Using the force argument [force = {}] forces the files to be overwritten!!!".format(self.force))
            self.overwriteFiles      = True
        else:
            self.forceCycle          = False
            self.forceSB             = False
            self.forceMeasLumi       = False
            self.forceCalcLumi       = False

        self.savePlots               = savePlots
        self.doCyclePlots            = doCyclePlots
        self.doSBPlots               = doSBPlots
        self.doSummaryPlots          = doSummaryPlots
        self.doPlots                 = doPlots
        if self.doPlots == True: ## flag to set all plotting to true
            self.doCyclePlots        = True
            self.doSBPlots           = True
            self.doSummaryPlots      = True
        elif self.doPlots == False:
            self.doCyclePlots        = False
            self.doSBPlots           = False
            self.doSummaryPlots      = False


        self.fig_tuple             = fig_tuple
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
        ##
        ## if self.debug:
        ##     warn("# init : Setting filling list to [5256, 5456]")
        ##     self.filln_list = [5256, 5456]

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


        ## --- These dictionaries here are placeholders. Per fill they will store
        ## --- info to be accessible for all functions.
        self.filln_CycleDict           = {}
        self.filln_StableBeamsDict     = {}
        self.filln_LumiCalcDict        = {}
        self.filln_LumiMeasDict        = {}

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
    def plot_mean_and_spread(self, axc, x_vect, ymat, color = 'k', alpha=1):
        '''
        Function to plot the average value and the +/- 2sigma range of a given
        vector on a given axis

        Inputs : axc : axis on which you want me to plot stuff
                 x_vect: array of x-axis values (any iterable object)
                 y_mat : array of y-axis values for which the std and mean will be Calculated
                 color : the color for the lines
                 alpha : the alpha of the lines
        Returns: None -- simply adds stuff on the given plt.axis
        '''
        avg = np.mean(ymat, axis=1)
        std = np.std(ymat, axis=1)
        axc.plot(x_vect, avg, color=color, linewidth=2, alpha=alpha)
        axc.plot(x_vect, avg-2*std, '--', color=color, linewidth=1, alpha=alpha)
        axc.plot(x_vect, avg+2*std, '--', color=color, linewidth=1, alpha=alpha)
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
        ## if not os.path.exists(self.fills_bmodes_file.split('.')[0]+"_df_{}_{}.pkl".format(str(self.first_fill), str(self.last_fill))):

        ## Get the bmodes information into a pandas dataframe
        with open(self.fills_bmodes_file, 'rb') as fid:
            dict_fill_bmodes = pickle.load(fid)
        bmodes = pd.DataFrame.from_dict(dict_fill_bmodes, orient='index') ##index is the fill number

        ## Clean up bmodes df depending on SB flag and SB duration
        ## Same clean up as the one done in 001c loop
        bmodes = bmodes[:][(bmodes['t_start_STABLE'] > 0) & (bmodes['t_stop_STABLE']-bmodes['t_start_STABLE']>= self.min_time_SB)]
        bmodes = bmodes.ix[self.first_fill:self.last_fill]

        ## Get the fill list
        filln_list = bmodes.index.values

        ## Make a column with the "period" the fill belongs to
        bmodes['period'] = np.nan
        for key in self.periods:
            ## key = 'A', 'B', 'C'
            bmodes['period'].loc[filln_list[np.logical_and(np.less(filln_list,self.periods[key][1]), np.greater_equal(filln_list,self.periods[key][0]))]]=str(key)
        ## bmodes['period'].loc[filln_list[filln_list>=5405]]='C'
        ## bmodes['period'].loc[filln_list[np.logical_and(np.less(filln_list,5405), np.greater_equal(filln_list,5256))]]='B'
        ## bmodes['period'] = bmodes['period'].fillna('A')

        if self.crossingAngleChange:
            bmodes['CrossingAngle'] = np.nan
            for key in self.XingAngle:
                bmodes['CrossingAngle'].loc[filln_list[np.logical_and(np.less(filln_list,key[1]), np.greater_equal(filln_list,key[0]))]]=self.XingAngle[key]

        ######################################################     RESCALING     ############################################################
        if self.doRescale:
            import BSRT_calib_rescale as BSRT_calib
            bmodes['rescaledPeriod'] = np.nan
            for res in self.resc_period:
                bmodes['rescaledPeriod'][bmodes['period']==res[0]]=res[1]
        else:
            import BSRT_calib as BSRT_calib
            bmodes['rescaledPeriod'] = bmodes['period'] # @TODO IS THIS NEEDED?

        ## if not os.path.exists(self.fills_bmodes_file.split('.')[0]+"_df_{}_{}.pkl".format(str(self.first_fill), str(self.last_fill))):
        ## filename = self.fills_bmodes_file.split('.')[0]+"_df_{}_{}.pkl".format(str(self.first_fill), str(self.last_fill))
        ## info("# getBmodesDF : Saving bdmodes dataframe for fills ({},{}) in file : {}".format(str(self.first_fill), str(self.last_fill), filename))
        ## with gzip.open(filename, 'wb') as fid:
        ##     pickle.dump(bmodes, fid)
        ## else:
        ##     with open(self.fills_bmodes_file.split('.')[0]+"_df_{}_{}.pkl".format(str(self.first_fill), str(self.last_fill)), 'rb') as fid:
        ##         bmodes = pickle.load(fid)

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
        pl.close('all')
        ms.mystyle_arial(self.myfontsize)
        list_figs = []

        ##preapare empty figure for histograms
        fig_emit_hist = pl.figure(1, figsize=(14,8))
        fig_emit_hist.set_facecolor('w')
        list_figs.append(fig_emit_hist)
        sp_emit_hist_list = []
        sptemp = None
        for i_sp in range(4):
            sptemp = pl.subplot(4,2,i_sp*2+1, sharex = sptemp)
            sp_emit_hist_list.append(sptemp)
        sp_inten_hist_list = []
        sptemp = None
        for i_sp in range(2):
            sptemp = pl.subplot(4,2,i_sp*2+2, sharex = sptemp)
            sp_inten_hist_list.append(sptemp)
        sp_bright_hist_list = []
        sptemp = None
        for i_sp in range(2):
            sptemp = pl.subplot(4,2,i_sp*2+6, sharex = sptemp)
            sp_bright_hist_list.append(sptemp)

        ##preapare empty figure bunch by bunch emit
        fig_emit_bbb = pl.figure(2, figsize=(14, 8))
        fig_emit_bbb.set_facecolor('w')
        list_figs.append(fig_emit_bbb)
        sp_emit_bbb_list = []
        sptemp = None
        for i_sp in xrange(4):
            sptemp = pl.subplot(4,1,i_sp+1, sharex=sptemp, sharey=sptemp)
            sp_emit_bbb_list.append(sptemp)

        ##preapare empty figure bunch by bunch intensity
        fig_inten_bbb = pl.figure(3, figsize=(14, 8))
        fig_inten_bbb.set_facecolor('w')
        list_figs.append(fig_inten_bbb)
        sp_inten_bbb_list = []
        sptemp = None
        for i_sp in range(2):
            sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
            sp_inten_bbb_list.append(sptemp)

        ##preapare empty figure bunch by bunch brightness
        fig_bright_bbb = pl.figure(4, figsize=(14, 8))
        fig_bright_bbb.set_facecolor('w')
        list_figs.append(fig_bright_bbb)
        sp_bright_bbb_list = []
        sptemp = None
        for i_sp in range(2):
            sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
            sp_bright_bbb_list.append(sptemp)


        ##preapare empty figure bunch by bunch time
        fig_time_bbb = pl.figure(6, figsize=(14, 8))
        fig_time_bbb.set_facecolor('w')
        list_figs.append(fig_time_bbb)
        sp_time_bbb_list = []
        sptemp = None
        for i_sp in range(2):
            sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
            sp_time_bbb_list.append(sptemp)

        ##preapare empty figure bunch by bunch intensity
        fig_blength_bbb = pl.figure(8, figsize=(14, 8))
        fig_blength_bbb.set_facecolor('w')
        list_figs.append(fig_blength_bbb)
        sp_blength_bbb_list = []
        sptemp = None
        for i_sp in range(2):
            sptemp = pl.subplot(2,1,i_sp+1, sharex=sptemp, sharey=sptemp)
            sp_blength_bbb_list.append(sptemp)

        for beam_n in [1, 2]:
            dict_intervals = dict_intervals_two_beams['beam_{}'.format(beam_n)]

            n_bins_emit = 50  ## @TODO PUT IT ON TOP?
            list_labels = ['Injected', 'Start Ramp', 'End Ramp', 'Start SB']

            ## emittance plots
            info("# makeCyclePlots : Making Emittance Cycle Plots for fill {}".format(filln))
            for i_plane, plane in enumerate(['h', 'v']):
                i_sp = (beam_n-1)*2+i_plane
                i_label = 0
                for interval in ['Injection', 'he_before_SB']:
                    for moment in ['at_start', 'at_end']:
                        masknan = ~np.isnan(np.array(dict_intervals[interval][moment]['emit'+plane]))
                        debug("# makeCyclePlots : In plane loop: {} {} {} {} ".format(interval, moment, np.sum(dict_intervals[interval][moment]['emit'+plane]), np.sum(~masknan)))
                        hist, bin_edges = np.histogram(np.array(dict_intervals[interval][moment]['emit'+plane])[masknan], range =(0,5), bins=n_bins_emit)

                        sp_emit_hist_list[i_sp].step(bin_edges[:-1], hist, label=list_labels[i_label]+', Avg. %.1f um'%np.mean(np.array(dict_intervals[interval][moment]['emit'+plane])[masknan]), linewidth=1)
                        sp_emit_bbb_list[i_sp].plot(np.array(dict_intervals[interval]['filled_slots'])[masknan], np.array(dict_intervals[interval][moment]['emit'+plane])[masknan], '.', label=list_labels[i_label]+', Avg. %.1f um'%np.mean(np.array(dict_intervals[interval][moment]['emit'+plane])[masknan]))

                        i_label+=1
                sp_emit_hist_list[i_sp].set_xlabel('Beam %d, Emittance %s [$\mu$m]'%(beam_n, plane))
                sp_emit_hist_list[i_sp].set_ylabel('Occurrences')
                sp_emit_hist_list[i_sp].grid('on')
                sp_emit_hist_list[i_sp].tick_params(axis='both', which='major', pad=5)
                sp_emit_hist_list[i_sp].xaxis.labelpad = 1
                sp_emit_hist_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')


                sp_emit_bbb_list[i_sp].set_ylabel('B%d, Emitt. %s [$\mu$m]'%(beam_n, plane))
                sp_emit_bbb_list[i_sp].grid('on')
                sp_emit_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
                sp_emit_bbb_list[i_sp].xaxis.labelpad = 1

            ##intensity plots
            info("# makeCyclePlots : Making Intensity Cycle Plots for fill {}".format(filln))
            i_label = 0
            i_sp = beam_n-1
            n_bins_inten = 50
            for interval in ['Injection', 'he_before_SB']:
                    for moment in ['at_start', 'at_end']:
                        sp_inten_bbb_list[i_sp].plot(np.array(dict_intervals[interval]['filled_slots'])[masknan], np.array(dict_intervals[interval][moment]['intensity'])[masknan], '.', label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(np.array(dict_intervals[interval][moment]['intensity'])[masknan])/1e11))
                        hist, bin_edges = np.histogram(np.array(dict_intervals[interval][moment]['intensity'])[masknan], range =(0.5e11,1.5e11), bins=n_bins_inten)
                        sp_inten_hist_list[i_sp].step(bin_edges[:-1], hist, label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(np.array(dict_intervals[interval][moment]['intensity'])[masknan])/1e11), linewidth=1)
                        i_label+=1

            sp_inten_bbb_list[i_sp].set_ylabel('Beam %d, Intensity [p/b]'%(beam_n))
            sp_inten_bbb_list[i_sp].grid('on')
            sp_inten_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
            sp_inten_bbb_list[i_sp].xaxis.labelpad = 1
            sp_inten_bbb_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')

            sp_inten_hist_list[i_sp].set_xlabel('Beam %d, Intensity [p/b]'%(beam_n))
            sp_inten_hist_list[i_sp].set_ylabel('Occurrences')
            sp_inten_hist_list[i_sp].grid('on')
            sp_inten_hist_list[i_sp].tick_params(axis='both', which='major', pad=5)
            sp_inten_hist_list[i_sp].xaxis.labelpad = 1
            sp_inten_hist_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')


            ##blength plots
            info("# makeCyclePlots : Making Bunch Length Cycle Plots for fill {}".format(filln))
            i_label = 0
            i_sp = beam_n-1
            for interval in ['Injection', 'he_before_SB']:
                    for moment in ['at_start', 'at_end']:
                        sp_blength_bbb_list[i_sp].plot(np.array(dict_intervals[interval]['filled_slots'])[masknan], np.array(dict_intervals[interval][moment]['blength'])[masknan], '.', label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(np.array(dict_intervals[interval][moment]['blength'])[masknan])/1e11))
                        i_label+=1

            sp_blength_bbb_list[i_sp].set_ylabel('Beam {}, b. length [p/b]'.format(beam_n))
            sp_blength_bbb_list[i_sp].grid('on')
            sp_blength_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
            sp_blength_bbb_list[i_sp].xaxis.labelpad = 1
            sp_blength_bbb_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')


            ## plot brightness
            info("# makeCyclePlots : Making Brightness Cycle Plots for fill {}".format(filln))
            i_label = 0
            i_sp = beam_n-1
            for interval in ['Injection', 'he_before_SB']:
                for moment in ['at_start', 'at_end']:
                        sp_bright_bbb_list[i_sp].plot(np.array(dict_intervals[interval]['filled_slots'])[masknan], np.array(dict_intervals[interval][moment]['brightness'])[masknan], '.', label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(np.array(dict_intervals[interval][moment]['brightness'])[masknan])/1e11))
                        hist, bin_edges = np.histogram(np.array(dict_intervals[interval][moment]['brightness'])[masknan], range =(0,1e11), bins=n_bins_inten)
                        sp_bright_hist_list[i_sp].step(bin_edges[:-1], hist, label=list_labels[i_label]+', Avg. %.2fe11'%(np.mean(np.array(dict_intervals[interval][moment]['brightness'])[masknan])/1e11), linewidth=1)
                        i_label+=1

            sp_bright_bbb_list[i_sp].set_ylabel('Beam {}, Brightness [p/$\mu$m/b]'.format(beam_n))
            sp_bright_bbb_list[i_sp].grid('on')
            sp_bright_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
            sp_bright_bbb_list[i_sp].xaxis.labelpad = 1
            sp_bright_bbb_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')

            sp_bright_hist_list[i_sp].set_xlabel('Beam {}, Brightness [p/$\mu$m]'.format(beam_n))
            sp_bright_hist_list[i_sp].set_ylabel('Occurrences')
            sp_bright_hist_list[i_sp].grid('on')
            sp_bright_hist_list[i_sp].tick_params(axis='both', which='major', pad=5)
            sp_bright_hist_list[i_sp].xaxis.labelpad = 1
            sp_bright_hist_list[i_sp].ticklabel_format(style='sci', scilimits=(0,0),axis='y')

            ## plot time
            info("# makeCyclePlots : Making Time Cycle Plots for fill {}".format(filln))
            i_label = 0
            i_sp = beam_n-1
            for interval in ['Injection', 'he_before_SB']:
                        sp_time_bbb_list[i_sp].plot(np.array(dict_intervals[interval]['filled_slots'])[masknan], (np.array(dict_intervals[interval]['at_end']['time_meas'])[masknan]-np.array(dict_intervals[interval]['at_start']['time_meas'])[masknan])/60. , '.',
                                                    label=', Avg. %.2f'%(np.mean(np.array(dict_intervals[interval]['at_end']['time_meas'])[masknan]-np.array(dict_intervals[interval]['at_start']['time_meas'])[masknan])/60.))

            sp_time_bbb_list[i_sp].set_ylabel('Beam {}, Time start to end [min]'.format(beam_n))
            sp_time_bbb_list[i_sp].grid('on')
            sp_time_bbb_list[i_sp].tick_params(axis='both', which='major', pad=5)
            sp_time_bbb_list[i_sp].xaxis.labelpad = 1


        sp_emit_bbb_list[-1].set_xlabel('25 ns slot')
        sp_emit_bbb_list[-1].set_ylim(1., 5.)

        sp_inten_bbb_list[-1].set_xlabel('25 ns slot')
        for sp in sp_emit_hist_list + sp_emit_bbb_list + sp_inten_bbb_list + sp_emit_bbb_list + sp_bright_bbb_list + sp_inten_hist_list +sp_bright_hist_list +sp_time_bbb_list:
            sp.legend(bbox_to_anchor=(1, 1),  loc='upper left', prop={'size':self.myfontsize})

        fig_emit_hist.subplots_adjust(left=.09, bottom=.07, right=.76, top=.92, wspace=1., hspace=.55)
        tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_ref))
        for fig in list_figs:
            fig.suptitle('Fill {}: started on {}'.format(filln, tref_string), fontsize=self.myfontsize)

            if fig is fig_emit_hist:
                continue
            fig.subplots_adjust(left=.05, right=.81, top=.93)

        if not self.batch:
            pl.show()

        if self.savePlots:
            timeString = datetime.now().strftime("%Y%m%d")
            saveString = timeString+self.plotFormat
            plot_dir   = self.plot_dir.replace('<FILLNUMBER>',str(filln))
            savename   = plot_dir+'fill_{}_cycle_EmittanceHist_{}'.format(filln, saveString)
            fig_emit_hist.savefig(savename, dpi=self.plotDpi)

            savename   = plot_dir+'fill_{}_cycle_bbbEmittance_{}'.format(filln, saveString)
            fig_emit_bbb.savefig(savename, dpi=self.plotDpi)

            savename   = plot_dir+'fill_{}_cycle_bbbIntensity_{}'.format(filln, saveString)
            fig_inten_bbb.savefig(savename, dpi=self.plotDpi)

            savename   = plot_dir+'fill_{}_cycle_bbbBrightness_{}'.format(filln, saveString)
            fig_bright_bbb.savefig(savename, dpi=self.plotDpi)

            savename   = plot_dir+'fill_{}_cycle_bbbTime_{}'.format(filln, saveString)
            fig_time_bbb.savefig(savename, dpi=self.plotDpi)

            savename   = plot_dir+'fill_{}_cycle_bbbBLength_{}'.format(filln, saveString)
            fig_blength_bbb.savefig(savename, dpi=self.plotDpi)
    ## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
    def checkDirectories(self, filln):
        ##############################  CHECK IF THE OUTPUT DIRECTORIES EXIST  ##############################
        ## Check if the SB_dir, fill_dir and plot dir exist:
        debug('#checkDirectories : Checking if SB, fill and plots directories exist for fill {}'.format(filln))
        if not os.path.exists(self.SB_dir):
            if self.makedirs:
                os.makedirs(self.SB_dir)
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
        getMassi   = False
        doSB       = False
        doLumiCalc = False
        doCycle    = False

        ##############################  CHECK IF THE OUTPUT FILES EXIST  ##############################
        ## Check if the massi file of the fill exists and flag it
        massi_filename = self.fill_dir+self.Massi_filename
        massi_filename = massi_filename.replace('<FILLNUMBER>',str(filln))

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
            SB_filename = filename.replace('<RESC>', '')

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
        ## Check if the lumi calc file of the fill exists and flag it
        lumi_calc_filename = self.fill_dir+self.Lumi_filename
        lumi_calc_filename = lumi_calc_filename.replace('<FILLNUMBER>',str(filln))
        if self.doRescale:
            if self.bmodes['period'][filln] != self.bmodes['rescaledPeriod'][filln]:
                lumi_calc_filename = lumi_calc_filename.replace('<RESC>', self.resc_string).replace("<TO>", str(self.bmodes['rescaledPeriod'][filln]))
            else:
                lumi_calc_filename = lumi_calc_filename.replace('<RESC>', '').replace("<TO>", '')
        else:
            filename = filename.replace('<RESC>', '')

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
            filename = filename.replace('<RESC>', '')

        if os.path.exists(cycle_filename):
            if self.overwriteFiles:
                warn("#checkFiles : Dictionary Cycle pickle [{}] for fill {} already exists! Overwritting it...".format(cycle_filename, filln))
                doCycle = True
            else:
                warn("#checkFiles : Dictionary Cycle pickle [{}] for fill {} already exists! Skipping it...".format(cycle_filename, filln))
        else:
            doCycle = True
            print doCycle
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
                print doCycle
            debug('#checkFiles : Checking if Pandas pickle of Cycle file [{}] for fill {} exists [{}]'.format(cycle_filename, filln, (not doCycle)  ))


        return getMassi, doSB, doLumiCalc, doCycle
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
        'time_range':time_range}

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

                ## === Getting info
                b1_inj_t_start      = self.filln_CycleDict['beam_1']['Injection']['t_start']
                b1_inj_t_end        = self.filln_CycleDict['beam_1']['Injection']['t_end']
                b1_inj_at_start     = self.filln_CycleDict['beam_1']['Injection']['at_start']
                b1_inj_at_end       = self.filln_CycleDict['beam_1']['Injection']['at_end']
                b1_inj_filled_slots = self.filln_CycleDict['beam_1']['Injection']['filled_slots']

                b2_inj_t_start      = self.filln_CycleDict['beam_2']['Injection']['t_start']
                b2_inj_t_end        = self.filln_CycleDict['beam_2']['Injection']['t_end']
                b2_inj_at_start     = self.filln_CycleDict['beam_2']['Injection']['at_start']
                b2_inj_at_end       = self.filln_CycleDict['beam_2']['Injection']['at_end']
                b2_inj_filled_slots = self.filln_CycleDict['beam_2']['Injection']['filled_slots']

                b1_he_t_start       = self.filln_CycleDict['beam_1']['he_before_SB']['t_start']
                b1_he_t_end         = self.filln_CycleDict['beam_1']['he_before_SB']['t_end']
                b1_he_at_start      = self.filln_CycleDict['beam_1']['he_before_SB']['at_start']
                b1_he_at_end        = self.filln_CycleDict['beam_1']['he_before_SB']['at_end']
                b1_he_filled_slots  = self.filln_CycleDict['beam_1']['he_before_SB']['filled_slots']

                b2_he_t_start       = self.filln_CycleDict['beam_2']['he_before_SB']['t_start']
                b2_he_t_end         = self.filln_CycleDict['beam_2']['he_before_SB']['t_end']
                b2_he_at_start      = self.filln_CycleDict['beam_2']['he_before_SB']['at_start']
                b2_he_at_end        = self.filln_CycleDict['beam_2']['he_before_SB']['at_end']
                b2_he_filled_slots  = self.filln_CycleDict['beam_2']['he_before_SB']['filled_slots']

                ######## INJECTION
                ## beam 1 - injection start
                df_b1_inj_start = pd.DataFrame.from_dict(b1_inj_at_start, orient='columns')

                df_b1_inj_start.insert(0, 'timestamp_end'  , [b1_inj_t_end]*len(df_b1_inj_start))
                df_b1_inj_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_inj_start['timestamp_end']))
                df_b1_inj_start.insert(0, 'timestamp_start', [b1_inj_t_start]*len(df_b1_inj_start))
                df_b1_inj_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_inj_start['timestamp_start']))
                df_b1_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_start['time_meas']))
                df_b1_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b1_inj_start))
                df_b1_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_start))
                df_b1_inj_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_start))
                df_b1_inj_start.insert(0, 'fill'           , [filln]*len(df_b1_inj_start))
                df_b1_inj_start['filled_slots']            = [b1_inj_filled_slots]*len(df_b1_inj_start)

                ## beam 1 - injection end
                df_b1_inj_end = pd.DataFrame.from_dict(b1_inj_at_end, orient='columns')

                df_b1_inj_end.insert(0, 'timestamp_end'  , [b1_inj_t_end]*len(df_b1_inj_end))
                df_b1_inj_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_inj_end['timestamp_end']))
                df_b1_inj_end.insert(0, 'timestamp_start', [b1_inj_t_start]*len(df_b1_inj_end))
                df_b1_inj_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_inj_end['timestamp_start']))
                df_b1_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_inj_end['time_meas']))
                df_b1_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b1_inj_end))
                df_b1_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b1_inj_end))
                df_b1_inj_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_inj_end))
                df_b1_inj_end.insert(0, 'fill'           , [filln]*len(df_b1_inj_end))
                df_b1_inj_end['filled_slots']            = [b1_inj_filled_slots]*len(df_b1_inj_end)


                ## beam 2 - injection start
                df_b2_inj_start = pd.DataFrame.from_dict(b2_inj_at_start, orient='columns')

                df_b2_inj_start.insert(0, 'timestamp_end'  , [b2_inj_t_end]*len(df_b2_inj_start))
                df_b2_inj_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_inj_start['timestamp_end']))
                df_b2_inj_start.insert(0, 'timestamp_start', [b2_inj_t_start]*len(df_b2_inj_start))
                df_b2_inj_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_inj_start['timestamp_start']))
                df_b2_inj_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_start['time_meas']))
                df_b2_inj_start.insert(0, 'cycleTime'      , ['injection_start']*len(df_b2_inj_start))
                df_b2_inj_start.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_start))
                df_b2_inj_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_start))
                df_b2_inj_start.insert(0, 'fill'           , [filln]*len(df_b2_inj_start))
                df_b2_inj_start['filled_slots']            = [b2_inj_filled_slots]*len(df_b2_inj_start)

                ## beam 2 - injection end
                df_b2_inj_end = pd.DataFrame.from_dict(b2_inj_at_end, orient='columns')

                df_b2_inj_end.insert(0, 'timestamp_end'  , [b2_inj_t_end]*len(df_b2_inj_end))
                df_b2_inj_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_inj_end['timestamp_end']))
                df_b2_inj_end.insert(0, 'timestamp_start', [b2_inj_t_start]*len(df_b2_inj_end))
                df_b2_inj_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_inj_end['timestamp_start']))
                df_b2_inj_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_inj_end['time_meas']))
                df_b2_inj_end.insert(0, 'cycleTime'      , ['injection_end']*len(df_b2_inj_end))
                df_b2_inj_end.insert(0, 'cycle'          , ['injection']*len(df_b2_inj_end))
                df_b2_inj_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_inj_end))
                df_b2_inj_end.insert(0, 'fill'           , [filln]*len(df_b2_inj_end))
                df_b2_inj_end['filled_slots']            = [b2_inj_filled_slots]*len(df_b2_inj_end)


                ######## HE before SB
                ## beam 1 - heection start
                df_b1_he_start = pd.DataFrame.from_dict(b1_he_at_start, orient='columns')

                df_b1_he_start.insert(0, 'timestamp_end'  , [b1_he_t_end]*len(df_b1_he_start))
                df_b1_he_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_start['timestamp_end']))
                df_b1_he_start.insert(0, 'timestamp_start', [b1_he_t_start]*len(df_b1_he_start))
                df_b1_he_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_start['timestamp_start']))
                df_b1_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_start['time_meas']))
                df_b1_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b1_he_start))
                df_b1_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_start))
                df_b1_he_start.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_start))
                df_b1_he_start.insert(0, 'fill'           , [filln]*len(df_b1_he_start))
                df_b1_he_start['filled_slots']            = [b1_he_filled_slots]*len(df_b1_he_start)

                ## beam 1 - heection end
                df_b1_he_end = pd.DataFrame.from_dict(b1_he_at_end, orient='columns')

                df_b1_he_end.insert(0, 'timestamp_end'  , [b1_he_t_end]*len(df_b1_he_end))
                df_b1_he_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b1_he_end['timestamp_end']))
                df_b1_he_end.insert(0, 'timestamp_start', [b1_he_t_start]*len(df_b1_he_end))
                df_b1_he_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b1_he_end['timestamp_start']))
                df_b1_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b1_he_end['time_meas']))
                df_b1_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b1_he_end))
                df_b1_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b1_he_end))
                df_b1_he_end.insert(0, 'beam'           , ['beam_1']*len(df_b1_he_end))
                df_b1_he_end.insert(0, 'fill'           , [filln]*len(df_b1_he_end))
                df_b1_he_end['filled_slots']            = [b1_he_filled_slots]*len(df_b1_he_end)


                ## beam 2 - heection start
                df_b2_he_start = pd.DataFrame.from_dict(b2_he_at_start, orient='columns')

                df_b2_he_start.insert(0, 'timestamp_end'  , [b2_he_t_end]*len(df_b2_he_start))
                df_b2_he_start.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_he_start['timestamp_end']))
                df_b2_he_start.insert(0, 'timestamp_start', [b2_he_t_start]*len(df_b2_he_start))
                df_b2_he_start.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_he_start['timestamp_start']))
                df_b2_he_start.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_start['time_meas']))
                df_b2_he_start.insert(0, 'cycleTime'      , ['flattop_start']*len(df_b2_he_start))
                df_b2_he_start.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_start))
                df_b2_he_start.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_start))
                df_b2_he_start.insert(0, 'fill'           , [filln]*len(df_b2_he_start))
                df_b2_he_start['filled_slots']            = [b2_he_filled_slots]*len(df_b2_he_start)

                ## beam 2 - heection end
                df_b2_he_end = pd.DataFrame.from_dict(b2_he_at_end, orient='columns')

                df_b2_he_end.insert(0, 'timestamp_end'  , [b2_he_t_end]*len(df_b2_he_end))
                df_b2_he_end.insert(0, 'datetime_end'   , self.convertToLocalTime(df_b2_he_end['timestamp_end']))
                df_b2_he_end.insert(0, 'timestamp_start', [b2_he_t_start]*len(df_b2_he_end))
                df_b2_he_end.insert(0, 'datetime_start' , self.convertToLocalTime(df_b2_he_end['timestamp_start']))
                df_b2_he_end.insert(0, 'datetime'       , self.convertToLocalTime(df_b2_he_end['time_meas']))
                df_b2_he_end.insert(0, 'cycleTime'      , ['flattop_end']*len(df_b2_he_end))
                df_b2_he_end.insert(0, 'cycle'          , ['flattop']*len(df_b2_he_end))
                df_b2_he_end.insert(0, 'beam'           , ['beam_2']*len(df_b2_he_end))
                df_b2_he_end.insert(0, 'fill'           , [filln]*len(df_b2_he_end))
                df_b2_he_end['filled_slots']            = [b2_he_filled_slots]*len(df_b2_he_end)

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
        sigma_h_b1 = np.sqrt(1e-6*eh_interp_coll[1]/self.gamma*self.betastar_m)
        sigma_v_b1 = np.sqrt(1e-6*ev_interp_coll[1]/self.gamma*self.betastar_m)
        sigma_h_b2 = np.sqrt(1e-6*eh_interp_coll[2]/self.gamma*self.betastar_m)
        sigma_v_b2 = np.sqrt(1e-6*ev_interp_coll[2]/self.gamma*self.betastar_m)

        ## Calculate the convoluted beam sizes and the convoluted bunch length
        sigma_h_conv = np.sqrt((sigma_h_b1**2+sigma_h_b2**2)/2.)
        sigma_v_conv = np.sqrt((sigma_v_b1**2+sigma_v_b2**2)/2.)
        bl_conv = (bl_interp_m_coll[1]+bl_interp_m_coll[2])/2.

        ## Calculate the reduction factor of the crossing plane and crossing angle
        FF_ATLAS = 1./np.sqrt(1.+((bl_conv/sigma_v_conv)*(self.bmodes['CrossingAngle'][filln]/2.))**2.)
        FF_CMS = 1./np.sqrt(1.+((bl_conv/sigma_h_conv)*(self.bmodes['CrossingAngle'][filln]/2.))**2.)

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
            total_calc_df = total_calc_df.set_index(['fill', 'experiment'])

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

            if experiment=='ATLAS':
                lumifile = '/afs/cern.ch/user/l/lpc/w0/2016/measurements/%s/%d.tgz'%(experiment, filln)
            elif experiment=='CMS':
                    lumifile = '/afs/cern.ch/user/l/lpc/w0/2016/measurements/%s/lumi/%d.tgz'%(experiment, filln)

            with tarfile.open(lumifile, 'r:gz') as tarfid:
                for slot_bun in slots_filled_coll[1]:
                    if debug:
                        if np.mod(slot_bun, 10)==0:
                            info("Experiment : {} : Bunch Slot: {} ".format(experiment, slot_bun))

                    bucket = (slot_bun)*10+1

                    filename_bunch = '%d/%d_lumi_%d_%s.txt'%(filln, filln, bucket, experiment)
                    fid = tarfid.extractfile(filename_bunch)
                    temp_data = np.loadtxt(fid.readlines())
                    t_stamps = temp_data[:,0]
                    lumi_bunch = temp_data[:,2]

                    self.filln_LumiMeasDict[experiment]['bunch_lumi'].append(np.interp(time_range,t_stamps, lumi_bunch)*1e34)
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
    def makePerformancePlotsPerFill(self, filln):     ## @TODO have not checked that yet
        '''
        Function to make performance plot per fill
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


        ## Figure : Emittances B2
        info("# makePerformancePlotsPerFill : Fill {} -> Making Emittances B2 plot...".format(filln))
        fig_em2 = pl.figure(2, figsize=self.fig_tuple)
        fig_em2.canvas.set_window_title('Emittances B2')
        fig_em2.set_facecolor('w')
        ax_ne2h = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
        ax_ne2v = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
        ax_ne2h_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
        ax_ne2v_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)

        ## Figure : Emittances B1 Raw
        info("# makePerformancePlotsPerFill : Fill {} -> Making Emittances B1 Raw plot...".format(filln))
        fig_em1_raw = pl.figure(101, figsize=self.fig_tuple)
        fig_em1_raw.canvas.set_window_title('Emittances B1 raw')
        fig_em1_raw.set_facecolor('w')
        ax_ne1h_raw = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
        ax_ne1h_t_raw = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
        ax_ne1v_raw = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
        ax_ne1v_t_raw = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)


        ## Figure : Emittances B2 Raw
        info("# makePerformancePlotsPerFill : Fill {} -> Making Emittances B2 Raw plot...".format(filln))
        fig_em2_raw = pl.figure(102, figsize=self.fig_tuple)
        fig_em2_raw.canvas.set_window_title('Emittances B2 raw')
        fig_em2_raw.set_facecolor('w')
        ax_ne2h_raw = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=axy_share)
        ax_ne2v_raw = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=axy_share)
        ax_ne2h_t_raw = pl.subplot(2,3,1, sharex=ax_share_t, sharey=axy_share)
        ax_ne2v_t_raw = pl.subplot(2,3,4, sharex=ax_share_t, sharey=axy_share)

        ## Figure : Bunch intensity
        info("# makePerformancePlotsPerFill : Fill {} -> Making Bunch Intensity plot...".format(filln))
        fig_int = pl.figure(3, figsize=self.fig_tuple)
        fig_int.canvas.set_window_title('Bunch intensity')
        fig_int.set_facecolor('w')
        bx_nb1 = pl.subplot(2,3,(2,3), sharex=ax_share)
        bx_nb2 = pl.subplot(2,3,(5,6), sharex=ax_share)
        bx_nb1_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=bx_nb1)
        bx_nb2_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=bx_nb2)


        ## Figure : Bunch Length
        info("# makePerformancePlotsPerFill : Fill {} -> Making Bunch Length plot...".format(filln))
        fig_bl = pl.figure(4, figsize=self.fig_tuple)
        fig_bl.canvas.set_window_title('Bunch length')
        fig_bl.set_facecolor('w')
        bx_bl1 = pl.subplot(2,3,(2,3), sharex=ax_share)
        bx_bl2 = pl.subplot(2,3,(5,6), sharex=ax_share)
        bx_bl1_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=bx_bl1)
        bx_bl2_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=bx_bl1)


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


        for sp in [ax_ne1h, ax_ne1v, ax_ne2h, ax_ne2v,bx_nb1, bx_nb2, bx_bl1, bx_bl2, ax_ne1h_raw, ax_ne1v_raw, ax_ne2h_raw, ax_ne2v_raw]:
            sp.grid('on')
            sp.set_xlabel('25 ns slot')

        for sp in [ax_ne1h_t, ax_ne1v_t, ax_ne2h_t, ax_ne2v_t, bx_nb1_t, bx_nb2_t, bx_bl1_t, bx_bl2_t, ax_ne1h_t_raw, ax_ne1v_t_raw, ax_ne2h_t_raw, ax_ne2v_t_raw]:
            sp.grid('on')
            sp.set_xlabel('Time [h]')

        ax_ne1h_t.set_ylabel('Emittance B1H [$\mu$m]')
        ax_ne1v_t.set_ylabel('Emittance B1V [$\mu$m]')
        ax_ne2h_t.set_ylabel('Emittance B2H [$\mu$m]')
        ax_ne2v_t.set_ylabel('Emittance B2V [$\mu$m]')

        ax_ne1h_t_raw.set_ylabel('Emittance B1H [$\mu$m]')
        ax_ne1v_t_raw.set_ylabel('Emittance B1V [$\mu$m]')
        ax_ne2h_t_raw.set_ylabel('Emittance B2H [$\mu$m]')
        ax_ne2v_t_raw.set_ylabel('Emittance B2V [$\mu$m]')


        bx_nb1_t.set_ylabel('Intensity B1 [p/b]')
        bx_nb2_t.set_ylabel('Intensity B2 [p/b]')
        bx_bl1_t.set_ylabel('Bunch length B1 [ns]')
        bx_bl2_t.set_ylabel('Bunch length B1 [ns]')

        ## ------------ Now plot for expected lumi
        info("# makePerformancePlotsPerFill : Fill {} -> Making Expected BBB Luminosities plot...".format(filln))
        fig_lumi_calc = pl.figure(5, figsize=self.fig_tuple)
        fig_lumi_calc.canvas.set_window_title('Expected bbb lumis')
        fig_lumi_calc.set_facecolor('w')
        ax_ATLAS_calc = pl.subplot(2,3,(2,3), sharex=ax_share)
        ax_CMS_calc = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=ax_ATLAS_calc)
        ax_ATLAS_calc_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=ax_ATLAS_calc)
        ax_CMS_calc_t = pl.subplot(2,3,4, sharex=ax_share_t, sharey=ax_ATLAS_calc)

        for i_time in range(0, N_steps, self.n_skip):
            colorcurr = ms.colorprog(i_prog=i_time, Nplots=N_steps)
            ax_ATLAS_calc.plot(slots_filled_coll[1], lumi_bbb_ATLAS_invm2[i_time, :], '.', color=colorcurr)
            ax_CMS_calc.plot(slots_filled_coll[1], lumi_bbb_CMS_invm2[i_time, :], '.', color=colorcurr)

        self.plot_mean_and_spread(ax_ATLAS_calc_t, (time_range-t_start_STABLE)/3600., lumi_bbb_ATLAS_invm2)
        self.plot_mean_and_spread(ax_CMS_calc_t,   (time_range-t_start_STABLE)/3600., lumi_bbb_CMS_invm2)

        ## Figure for Raw Measured Luminosity for ATLAS/CMS as taken from Massi files
        info("# makePerformancePlotsPerFill : Fill {} -> Making Measured BBB Luminosities plot...".format(filln))
        fig_lumi_meas   = pl.figure(7, figsize=self.fig_tuple)
        fig_lumi_meas.canvas.set_window_title('Measured bbb lumi raw')
        fig_lumi_meas.set_facecolor('w')

        ax_ATLAS_meas   = pl.subplot(2,3,(2,3), sharex=ax_share, sharey=ax_ATLAS_calc)
        ax_CMS_meas     = pl.subplot(2,3,(5,6), sharex=ax_share, sharey=ax_ATLAS_calc)
        ax_ATLAS_meas_t = pl.subplot(2,3,1, sharex=ax_share_t, sharey=ax_ATLAS_calc)
        ax_CMS_meas_t   = pl.subplot(2,3,4, sharex=ax_share_t, sharey=ax_ATLAS_calc)

        self.plot_mean_and_spread(ax_ATLAS_meas_t, (time_range-t_start_STABLE)/3600., self.filln_LumiMeasDict['ATLAS']['bunch_lumi'])
        self.plot_mean_and_spread(ax_CMS_meas_t,   (time_range-t_start_STABLE)/3600., self.filln_LumiMeasDict['CMS']['bunch_lumi'])


        for i_time in range(0, N_steps, self.n_skip):
            colorcurr = ms.colorprog(i_prog=i_time, Nplots=N_steps)
            ax_ATLAS_meas.plot(slots_filled_coll[1], self.filln_LumiMeasDict['ATLAS']['bunch_lumi'][i_time, :], '.', color=colorcurr)
            ax_CMS_meas.plot(slots_filled_coll[1],   self.filln_LumiMeasDict['CMS']['bunch_lumi'][i_time, :],   '.', color=colorcurr)

        for sp in [ax_ATLAS_calc, ax_CMS_calc, ax_ATLAS_meas, ax_CMS_meas]:
            sp.grid('on')
            sp.set_xlabel('25 ns slot')

        for sp in [ax_ATLAS_calc_t, ax_CMS_calc_t, ax_ATLAS_meas_t, ax_CMS_meas_t]:
            sp.grid('on')
            sp.set_xlabel('Time [h]')
            sp.set_xlim(0, -(t_start_STABLE-t_end_STABLE)/3600.)

        ax_ATLAS_calc_t.set_ylabel('Luminosity ATLAS [m$^2$ s$^{-1}$]')
        ax_CMS_calc_t.set_ylabel('Luminosity CMS [m$^2$ s$^{-1}$]')
        ax_ATLAS_meas_t.set_ylabel('Luminosity ATLAS [m$^2$ s$^{-1}$]')
        ax_CMS_meas_t.set_ylabel('Luminosity CMS [m$^2$ s$^{-1}$]')
        ax_CMS_meas.set_xlim(0, 3564)

        ## Figure of Total Luminosity Measured and Calculated for ATLAS/CMS (sum up bbb)
        info("# makePerformancePlotsPerFill : Fill {} -> Making Total Luminosity plot...".format(filln))
        fig_total = pl.figure(8, figsize=self.fig_tuple)
        fig_total.canvas.set_window_title('Total Luminosity')
        fig_total.set_facecolor('w')
        pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(self.filln_LumiMeasDict['ATLAS']['bunch_lumi'], axis=1),       color='b', linewidth=2., label="ATLAS Meas.")
        pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(lumi_bbb_ATLAS_invm2,              axis=1), '--', color='b', linewidth=2., label="ATLAS Calc.")
        pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(self.filln_LumiMeasDict['CMS']['bunch_lumi'],   axis=1),       color='r', linewidth=2., label="CMS Meas.")
        pl.plot((time_range-t_start_STABLE)/3600., 1e-4*np.sum(lumi_bbb_CMS_invm2,                axis=1), '--', color='r', linewidth=2., label="CMS Calc.")
        pl.legend(loc='best', prop={"size":12})
        pl.xlim(-.5, None)
        pl.ylim(0, None)
        pl.ylabel('Luminosity ATLAS [cm$^2$ s$^{-1}$]') ## NK: cm??? ATLAS??
        pl.xlabel('Time [h]')
        pl.grid()

        figlist = [fig_total, fig_int, fig_em1, fig_em2, fig_bl, fig_lumi_calc, fig_lumi_meas]
        for ff in figlist:
            tref_string = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(t_start_STABLE))
            ff.suptitle('Fill {}: STABLE BEAMS declared on {}'.format(filln, tref_string), fontsize=self.myfontsize)

        if not self.batch:
            pl.show()

        if self.savePlots:
            timeString = datetime.now().strftime("%Y%m%d")
            saveString = timeString+self.plotFormat
            figDic     = {  fig_em1         :'fill_{}_b1Emittances_{}'.format(filln, saveString),
                            fig_em2         :'fill_{}_b2Emittances_{}'.format(filln, saveString),
                            fig_em1_raw     :'fill_{}_b1RawEmittances_{}'.format(filln, saveString),
                            fig_em2_raw     :'fill_{}_b2RawEmittances_{}'.format(filln, saveString),
                            fig_int         :'fill_{}_bunchIntensity_{}'.format(filln, saveString),
                            fig_bl          :'fill_{}_bunchLength_{}'.format(filln, saveString),
                            fig_lumi_calc   :'fill_{}_calcLumi_{}'.format(filln, saveString),
                            fig_lumi_meas   :'fill_{}_measLumi_{}'.format(filln, saveString),
                            fig_total       :'fill_{}_totalLumi_{}'.format(filln, saveString)
            }
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
    def runForFill(self, filln):
        '''
        Runs the full analysis for one fill
        '''
        ## First check if the directories for this fill exist
        self.checkDirectories(filln)
        ## Then see if the files for this fill exist and see what can be done
        getMassi, doSB, doLumiCalc, doCycle = self.checkFiles(filln)

        print getMassi, doSB, doLumiCalc, doCycle

        ## Check if I have forces something (or everything)
        if self.force:
            info('#runForFill : User initialized run with the [force] option on {}'.format(self.force))
            doCycle     = self.forceCycle
            getMassi    = self.forceMeasLumi
            doSB        = self.forceSB
            doLumiCalc  = self.forceCalcLumi

        ## If everything is in place and nothing has to be done, return
        if not self.doSBPlots and not self.doCyclePlots:
            if not getMassi and not doSB and not doLumiCalc and not doCycle:
                warn("#runForFill: Nothing to do for fill {}. [getMassi = {}, doSB = {}, doLumiCalc = {}, doCycle = {}, doSBPlots = {}]".format(filln, getMassi, doSB, doLumiCalc, doCycle, self.doSBPlots))
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


        if getMassi:
            info('#runForFill : Starting loop for Measured Luminosity for fill {}'.format(filln))
            self.runMeasuredLuminosity(filln)
        else:
            info('#runForFill : No need to loop for Measured Luminosity for fill {} [getMassi = {}]'.format(filln,getMassi))


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

            ##-- Measured Lumi final
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
            self.makePerformancePlotsPerFill(filln)



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
    ## - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
    def runForFillList(self):
        '''
        Function to loop over all fills in filln_list
        '''
        ## @TODO ADD SUBMISSION TO LSF?
        info('# runForFillList : Running loop for fills : {}'.format(self.filln_list))

        for filln in self.filln_list:
            self.runForFill(filln)

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
        Lumi_filename           = {}


        ## --- Massi File
        Massi_filename          = {}
        Massi_fill_Database     = {}
        Massi_year              = {}
        Massi_afs_path          = {}
        massi_exp_folders       = {}

        ## --- Machine Changes and parameters
        frev                    = {}
        gamma                   = {}
        betastar_m              = {}
        crossingAngleChange     = {}
        XingAngle               = {}

        ## --- Output & Plot info
        saveDict                = {}
        savePandas              = {}
        doOnly                  = {}
        force                   = {}
        forceCycle              = {}
        forceSB                 = {}
        forceMeasLumi           = {}
        forceCalcLumi           = {}
        savePlots               = {}

        doCyclePlots            = {}
        doSBPlots               = {}
        fig_tuple               = {}
        plotFormat              = {}
        plotDpi                 = {}
        myfontsize              = {}
        n_skip                  = {}
        makePlotTarball         = {}
        '''.format(self.debug, self.batch, self.FORMAT, self.logfile, self.fills_bmodes_file, self.BASIC_DATA_FILE, self.BBB_DATA_FILE, self.filln_list, self.min_time_SB, self.first_fill, self.last_fill, self.t_step_sec, self.intensity_threshold,
self.enable_smoothing_BSRT, self.avg_time_smoothing, self.doRescale, self.resc_period, self.resc_string, self.makedirs, self.overwriteFiles, self.SB_dir, self.fill_dir, self.plot_dir, self.SB_filename, self.Cycle_filename, self.Lumi_filename,
self.Massi_filename, self.fill_yaml_database, self.fill_year, self.massi_afs_path, self.massi_exp_folders, self.frev, self.gamma, self.betastar_m, self.crossingAngleChange, self.XingAngle, self.saveDict, self.savePandas, self.doOnly, self.force,
self.forceCycle, self.forceSB, self.forceMeasLumi, self.forceCalcLumi, self.savePlots, self.doCyclePlots, self.doSBPlots, self.fig_tuple, self.plotFormat, self.plotDpi, self.myfontsize, self.n_skip, self.makePlotTarball)

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
