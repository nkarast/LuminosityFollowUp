#   lumimod.py  -- Driver script for Luminosity Follow-Up
#
#   ``Too lazy to write json, too module-dependent to write yaml``
#   Author : Nikos Karastathis (nkarast <at> cern <dot> ch)
#   Version : 0.2beta (20/04/2017)



import sys, os
# Get current working directory
cwd = os.getcwd()
BIN = os.path.expanduser(cwd)
sys.path.append(BIN)
import pandas as pd
import numpy as np
import pickle
import time
from logging import *
from datetime import datetime
import argparse
###################### lumimod configuration file
import config
import LumiFollowUp

################################################################################
#
#   Function to get filln_list from bmodes file
#
def getFillListFromBmodes(bmodes_filename, first_fill, last_fill, min_time_SB):
    '''
    External repetion of bmodes function to get fill list available outside the class
    Inputs : bmodes_filename : filename of the bmodes file
             first_fill      : which is the first fill
             last_fill       : which is the last fill
             min_time_SB     : what is the min time required in Stable Beams
    Returns: filln_list      : a list with the fill numbers
    '''
    # Get the bmodes information into a pandas dataframe
    with open(bmodes_filename, 'rb') as fid:
        dict_fill_bmodes = pickle.load(fid)
    bmodes = pd.DataFrame.from_dict(dict_fill_bmodes, orient='index') #index is the fill number

    # Clean up bmodes df depending on SB flag and SB duration
    bmodes = bmodes[:][(bmodes['t_start_STABLE'] > 0) & (bmodes['t_stop_STABLE']-bmodes['t_start_STABLE']>= min_time_SB)]
    bmodes = bmodes.ix[first_fill:last_fill]
    # Get the fill list
    return bmodes.index.values

################################################################################
#
#   Initialize Logger Function
#
def init_logger(FORMAT, logfile, loglevel):
        '''
        Function to initialize the logger from logging module
        Inputs : FORMAT   : Format of the logging output
                 logfile  : Logfile to populate with the log info or None (for STDOUT)
                 loglevel : Log level : 10 = debug, 20 = info, 30 = warn, 40 = error, 50 = fatal
        Returns: None
        '''
        basicConfig(format=FORMAT, filename=logfile, level=loglevel)

################################################################################
#
#   This would be my main function and driver routine
#   This will have argparse for user communication with the script and will call
#   Chunk looper
def main():
    # --- Create an argparser for command line arguments
    # --- The user can control : force/doOnly a specific step (cycle, sb, lumi, massi, all)
    #                            fill_number or fill_number list
    #                            loglevel
    #                            break the filln_list into smaller chunks
    #                            submit to HTCondor
    parser = argparse.ArgumentParser(description='Run the Cycle/SB/Calc Lumi/Measured Lumi steps of the Lumi Follow up.')
    parser.add_argument("-n", "--fill",      help='Run only for the specified fill(s)', nargs='+')
    parser.add_argument("-v", "--verbose",   help='Enable verbose', action="store_true")
    parser.add_argument("-ll", "--loglevel", help='Set log level', type=int)
    parser.add_argument("-f", "--force",     help='Force a specific step [\'all\', \'cycle\', \'sb\', \'lumi\', \'massi\']', type=str)
    parser.add_argument("-d", "--doOnly",    help='Do only a specific step [\'cycle\', \'sb\', \'lumi\', \'massi\']', type=str, default=None)
    parser.add_argument("-b", "--split",     help='Break the loop in fill numbers in batches of specified number', type=int, default=None)
    parser.add_argument("-s", "--submit",    help='Submit the fill loop to HTCondor', action="store_true")
    parser.add_argument("-q", "--queue",     help='Specified the HTCondor queue for the jobs to be submitted', type=str, default=None)
    args = parser.parse_args()
    # placeholders for the argparser
    debug       = False
    fill        = None
    batch       = True
    loglevel    = 0
    force       = False
    doOnly      = False
    loopStep    = None
    doSubmit    = False
    whichQueue  = None

    # read the parser into placeholders
    if args.verbose: #args['verbose']:
        debug = True
    if args.submit: #['submit']:
        doSubmit = True
    if args.split is not None:
        loopStep = int(args.split)
    if args.queue is not None:
        whichQueue = str(args.queue)
    if args.fill is not None:
        fill = [int(x) for x in args.fill]
    if args.loglevel is not None:
        loglevel = int(args.loglevel)
    if args.force is not None:
        if args.force == 'all':
            force = True
        elif args.force == 'cycle' or args.force == 'sb' or args.force == 'lumi' or args.force == 'massi':
            force = str(args.force)
        else:
            raise ValueError("# main : Argument parser - Unrecognised force option {}".format(args.force))
    if args.doOnly is not None:
        if args.doOnly == 'cycle' or args.doOnly == 'sb' or args.doOnly == 'lumi' or args.doOnly == 'massi':
            doOnly = str(args.doOnly)
        else:
            raise ValueError("# main : Argument parser - Unrecognised force option {}".format(args.doOnly))


    ######################## ----  DONE WITH PARSER ---- ########################

    # Initialize logger
    if loglevel == 0 :  # i.e. not set by cmd line
        if debug: # if set by cmd line
            loglevel = 10
        else:
            loglevel = 20
    init_logger(FORMAT=config.FORMAT, logfile=None, loglevel=loglevel)


    # --- Get the full fill list
    if fill is None:  # get it from bmodes
        filln_list = getFillListFromBmodes(config.fills_bmodes_file, config.first_fill, config.last_fill, config.min_time_SB)
    else:  # get it the user defined one
        filln_list = fill

    # print debug, fill, batch, loglevel, force, doOnly, loopStep, doSubmit, whichQueue
    # print filln_list

    info('''# lumimod : Running script with the following configuration:
         debug      = {}
         batch      = {}
         loglevel   = {}
         force      = {}
         doOnly     = {}
         loopStep   = {}
         filln_list = {}
         doSumbit   = {}'''.format(debug, batch, loglevel, force, doOnly, loopStep, filln_list, doSubmit))



    # Do I have to submit the job in HTCondor? If I do have to submit it then
    # I check if you tell me to tokkenize the filln_list into chunks.
    #
    # If I do not have to submit the job, then I do not have any need of
    # to tokkenize the filln_list since I'll run it interactively
    if doSubmit:
        # first check if I have to break the job into chunks:
        chunks = []
        if loopStep is not None:
            chunks = np.array(np.array_split(filln_list, len(filln_list)/loopStep))
            chunks = [a.tolist() for a in chunks]  # this is now a list of lists
        else:
            # otherwise put into chunks the whole fill_list
            chunks.append(filln_list)

        # Now loop over chunks and create submit jobs
        for chunk in chunks:
            print 'Chunk: ', chunk

            # first define an index for this chunk:
            indx = "{}_{}".format(str(chunk[0]), str(chunk[-1]))
            # create a folder for the job
            folder_name = str(config.working_folder+"lumimod_JOB_{}_{}".format(indx , datetime.now().strftime("%Y%m%d")))
            print type(folder_name)
            info("# lumimod: Creating Job Submission Folder: {}".format(folder_name) )
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            else:
                raise IOError("# lumimod : Creation of directory {} failed. Directory already exists. Do your clean-up!".format(folder_name))

            # filenames
            sub_filename = "lumimod_JOB_{}_{}.sub".format(indx , datetime.now().strftime("%Y%m%d"))
            out_filename = folder_name+"/lumimod_JOB_{}_{}.out".format(indx , datetime.now().strftime("%Y%m%d"))
            err_filename = folder_name+"/lumimod_JOB_{}_{}.err".format(indx , datetime.now().strftime("%Y%m%d"))
            log_filename = folder_name+"/lumimod_JOB_{}_{}.log".format(indx , datetime.now().strftime("%Y%m%d"))

            # check if template file exists:
            if not os.path.exists(config.working_folder+"/submit_template.py"):
                raise IOError("# lumimod : Template of executable for HTCondor job submission not found.")

            # create the executable from template data:
            exe_filename = "lumimod_JOB_{}_{}.py".format(indx , datetime.now().strftime("%Y%m%d"))
            info("# lumimod: Opening Template File [{}]".format(config.working_folder+"/submit_template.py"))
            templateFile = open(config.working_folder+"/submit_template.py", 'r')
            templateData = templateFile.read()
            templateData = templateData.replace("%DEBUG", str(debug)).replace("%LOGLEVEL", str(loglevel)).replace("%FILL", "{}".format(fill))
            if force != False and force!=True:
                templateData = templateData.replace("%FORCE", "'{}'".format(str(force)))
            else:
                templateData = templateData.replace("%FORCE", "{}".format(str(force)))

            if doOnly != False and doOnly!=True:
                templateData = templateData.replace("%DOONLY", "'{}'".format(str(doOnly)))
            else:
                templateData = templateData.replace("%DOONLY", "{}".format(str(doOnly)))

            info("# lumimod: Creating Executatble [{}]".format(exe_filename))
            exe_file = open(exe_filename, 'w')
            exe_file.write(templateData)
            exe_file.close()
            templateFile.close()


            # create *.sub file:
            sub_file = open(sub_filename, 'w')
            info("# lumimod: Creating submit file [{}]".format(sub_filename))
            submit_string = config.htcondor_sub
            submit_string = submit_string.replace("<EXE>", exe_filename).replace("<OUT>", out_filename).replace("<ERR>", err_filename).replace("<LOG>", log_filename)
            sub_file.write(submit_string)
            sub_file.close()

            # submit the job to condor
            submit_command      = "condor_submit {}".format(sub_filename)
            mvsub_command       = "mv {} {}".format(sub_filename, folder_name+"/"+sub_filename)
            mvpy_command        = "mv {} {}".format(exe_filename, folder_name+"/"+exe_filename)
            mvconfig_command    = "cp config.py {}".format(folder_name+"/config.py")
            mvlumi_command      = "cp LumiFollowUp.py {}".format(folder_name+"/LumiFollowUp.py")

            info("# lumimod: Moving submit script [{}] in Job folder [{}]".format(sub_filename, folder_name+"/"+sub_filename))
            os.system(mvsub_command)
            info("# lumimod: Moving executable [{}] in Job folder [{}]".format(exe_filename, folder_name+"/"+exe_filename))
            os.system(mvpy_command)
            info("# lumimod: Copying config.py file in Job folder [{}]".format(folder_name))
            os.system(mvconfig_command)
            info("# lumimod: Copying LumiFollowUp.py file in Job folder [{}]".format(folder_name))
            os.system(mvlumi_command)

            info("# lumimod: Changing directory to submit Job [->{}]".format(folder_name))
            os.chdir(folder_name)
            info("# lumimod: Submitting Job to HTCondor for submit script [{}]".format(sub_filename))
            os.system(submit_command)

            info("# lumimod: Changing directory back to main folder [->{}]".format(cwd))
            os.chdir(cwd)


            info('# lumimod: Done for fill(s) : [{}]'.format(chunk))


    else:
        # Create the LumiFollowUp object
        fl = LumiFollowUp.LumiFollowUp(debug=debug, batch=batch, FORMAT=config.FORMAT, loglevel=loglevel, logfile=None, fills_bmodes_file=config.fills_bmodes_file,
                                       min_time_SB=config.min_time_SB, first_fill=config.first_fill, last_fill=config.last_fill, t_step_sec=config.t_step_sec,
                                       intensity_threshold=config.intensity_threshold, enable_smoothing_BSRT=config.enable_smoothing_BSRT,
                                       avg_time_smoothing=config.avg_time_smoothing, periods=config.periods, doRescale=config.doRescale,
                                       resc_period=config.resc_period, add_resc_string=config.add_resc_string, BASIC_DATA_FILE=config.BASIC_DATA_FILE,
                                       BBB_DATA_FILE=config.BBB_DATA_FILE, makedirs=config.makedirs, overwriteFiles=config.overwriteFiles,
                                       SB_dir=config.stableBeams_folder, fill_dir=config.fill_dir, plot_dir=config.plot_dir,
                                       SB_filename=config.SB_filename, Cycle_filename=config.Cycle_filename, Lumi_filename=config.Lumi_filename,
                                       Massi_filename=config.Massi_filename, saveDict=config.saveDict,
                                       #machine parameters
                                       frev=config.frev, gamma=config.gamma, betastar_m=config.betastar_m, crossingAngleChange=config.crossingAngleChange,
                                       XingAngle=config.XingAngle,
                                       # plots
                                       savePlots=config.savePlots, fig_tuple=config.fig_tuple, plotFormat=config.plotFormat,
                                       plotDpi=config.plotDpi, myfontsize=config.myfontsize, n_skip=config.n_skip, doCyclePlots=config.doCyclePlots,
                                       doSBPlots=config.doSBPlots, doSummaryPlots=config.doSummaryPlots, doPlots=config.doAllPlots,
                                       #
                                       force=force, doOnly=doOnly, makePlotTarball=config.makePlotTarball,
                                       #
                                       fill = filln_list)
        fl.printConfig()
        fl.runForFillList()




    # chunks = []
    # if loopStep is not None:
    #     chunks = np.array(np.array_split(filln_list, len(filln_list)/loopStep))
    #     chunks = [a.tolist() for a in chunks]  # this is now a list of lists
    #
    #     if doSubmit:
    #         # do a loop on chunks
    #         # every chunk on different ht condor job
    #     else:
    #         #
    #         # run chunk looper
    # else:
    #     if doSubmit:
    #         # submit the full list in one ht condor job
    #     else:
    #         # run on the fill list interactively
    #
    # print '-- done --'

#####################################
#
#   I  need the chunk looper
#
# def chunkLooper(filln_list, doOnly, force, ...):
    # '''
    # all arguments of lumifollowup must be passed?
    # '''
    # lf = LumiFollowUp(debug=debug, batch=batch, loglevel = loglevel, logfile = None, fills_bmodes_file = fills_bmodes_file, #'../fills_and_bmodes.pkl',
    #                         min_time_SB = min_time_SB, first_fill = first_fill, last_fill = last_fill,
    #                         t_step_sec = 10*60, intensity_threshold = 3.0e10, enable_smoothing_BSRT = True,
    #                         avg_time_smoothing = 3.0*3600.0, doRescale = True,
    #                         resc_period = [('A', 'C'), ('B','C'), ('C', 'C')], add_resc_string = '',
    #
    #                         makedirs = True, overwriteFiles = False,
    #                         saveDict = True, force=force,
    #                         savePlots = True, overwritePlots = None,
    #                         makePlotTarball = False, doOnly = doOnly, fill=list(filln_list), doSubmit = None, submitQueue = None, loopStep = None)
    #






################################################################################
if __name__ == '__main__':
    main()
