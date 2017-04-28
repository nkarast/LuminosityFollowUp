#!/usr/bin/env python
#   submit_template.py  -- Template for python executable for HTCondor submit
#
#   Author : Nikos Karastathis (nkarast <at> cern <dot> ch)
#   Version : 0.2beta (20/04/2017)



import sys, os
# For LHC Measurement Tools
LHCMeasToolsDir = "/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/"
sys.path.append(LHCMeasToolsDir)
# For BSRT rescale
BSRTRescDir = "/afs/cern.ch/work/l/lumimod/private/LHC_2016_25ns_beforeTS1/LumiModel_FollowUp/"
sys.path.append(BSRTRescDir)
# for utilities and databases
LocalModules = os.path.expanduser("%WD")
sys.path.append(LocalModules)

# cwd
cwd = os.getcwd()
sys.path.append(cwd)

from logging import *
###################### lumimod configuration file
import config
import LumiFollowUp.LumiFollowUp as LumiFollowUp


if __name__ == '__main__':

    fl = LumiFollowUp.LumiFollowUp(debug=%DEBUG, batch=True, FORMAT=config.FORMAT, loglevel=%LOGLEVEL, logfile=None, fills_bmodes_file=config.fills_bmodes_file,
                                   min_time_SB=config.min_time_SB, first_fill=config.first_fill, last_fill=config.last_fill, t_step_sec=config.t_step_sec,
                                   intensity_threshold=config.intensity_threshold, enable_smoothing_BSRT=config.enable_smoothing_BSRT,
                                   avg_time_smoothing=config.avg_time_smoothing, periods=config.periods, doRescale=config.doRescale,
                                   resc_period=config.resc_period, add_resc_string=config.add_resc_string, BASIC_DATA_FILE=config.BASIC_DATA_FILE,
                                   BBB_DATA_FILE=config.BBB_DATA_FILE, makedirs=config.makedirs, overwriteFiles=config.overwriteFiles,
                                   SB_dir=config.stableBeams_folder, fill_dir=config.fill_dir, plot_dir=config.plot_dir,
                                   SB_filename=config.SB_filename, Cycle_filename=config.Cycle_filename, Lumi_filename=config.Lumi_filename,
                                   Massi_filename=config.Massi_filename, saveDict=config.saveDict, savePandas=config.savePandas,
                                   #machine parameters
                                   frev=config.frev, gamma=config.gamma, betastar_m=config.betastar_m, crossingAngleChange=config.crossingAngleChange,
                                   XingAngle=config.XingAngle,
                                   # plots
                                   savePlots=config.savePlots, fig_tuple=config.fig_tuple, plotFormat=config.plotFormat,
                                   plotDpi=config.plotDpi, myfontsize=config.myfontsize, n_skip=config.n_skip, doCyclePlots=config.doCyclePlots,
                                   doSBPlots=config.doSBPlots, doSummaryPlots=config.doSummaryPlots, doPlots=config.doAllPlots,
                                   #
                                   force=%FORCE, doOnly=%DOONLY, makePlotTarball=config.makePlotTarball,
                                   #
                                   fill = %FILL, submit = True, fill_yaml_database = config.massi_file_database, fill_year=config.massi_year,
                                   massi_afs_path=config.massi_afs_path, massi_exp_folders=config.massi_exp_folders)
    fl.printConfig()
    fl.runForFillList()
