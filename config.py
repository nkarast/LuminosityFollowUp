#   config.py  -- Configuration file for the lumimod Luminosity Follow-Up script
#
#   ``Too lazy to write json, too module-dependent to write yaml``
#   Author : Nikos Karastathis (nkarast <at> cern <dot> ch)
#   Version : 0.3 (28/04/2017)


# --- Fills and periods ---
first_fill              = 5005
last_fill               = 5456 +1 # use + 1 if only used in bmodes
periods                 = {
                            'A': (first_fill,  5256),
                            'B': (5256,     5405),
                            'C': (5405,     last_fill)
                          }


# --- Folder layout, folder Options and naming conventions ---
input_folder            = "/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/"
working_folder          = input_folder
data_folder             = input_folder+"dataFiles/"
stableBeams_folder      = working_folder+"SB_analysis/"
fill_dir                = "fill_<FILLNUMBER>/"
plot_dir                = "plots/"
SB_filename             = "fill_<FILLNUMBER><RESC>.pkl.gz"
Cycle_filename          = "fill_<FILLNUMBER>_cycle<RESC>.pkl.gz"
Lumi_filename           = "fill_<FILLNUMBER>_lumi_calc<RESC>.pkl.gz"
makedirs                = True  # Create the directories if they do not exist
overwriteFiles          = False
saveDict                = True
savePandas              = True


# --- Massi Files ---
Massi_filename          = 'fill_<FILLNUMBER>_lumi_meas.pkl.gz'
massi_file_database     = working_folder+"Utilities/fill_db.yaml"
massi_year              = 2016
massi_afs_path          = '/afs/cern.ch/user/l/lpc/w0/<YEAR>/measurements/'
massi_exp_folders       = ['ATLAS/lumi/', 'CMS/lumi/']
massi_bunch_lumi_scale  = 1.0e34


# --- Basic Input Data Files ---
BASIC_DATA_FILE         = data_folder+'fill_basic_data_csvs/basic_data_fill_<FILLNUMBER>.csv'
BBB_DATA_FILE           = data_folder+'fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_<FILLNUMBER>.csv'
fills_bmodes_file       = data_folder+'fills_and_bmodes.pkl'


# --- BSRT Related Configuration ---
enable_smoothing_BSRT   = True
avg_time_smoothing      = 3.0*3600.0
doRescale               = False
resc_period             = [('A', 'A'), ('B','C'), ('C', 'C')]
                        # (From, To), (From, To), (From, To)
add_resc_string         = ''


# --- Stable Beams Inputs ---
min_time_SB             = 30*60     # minimum required time in SB to consider fill
t_step_sec              = 10*60     # time step for SB
intensity_threshold     = 3.0e10    # intensity threshold


# --- Machine Parameters ---
frev                    = 11245.5   # revolution frequency in Hz
gamma                   = 6927.64   # relativistic gamma # for 6.5 TeV
betastar_m              = 0.40      # beta function at IP used for lumi calculation
crossingAngleChange     = True      # has the crossing angle changed between fills?
XingAngle               = {         # dictionary for fill ranges and full crossing angle values
                            (first_fill,    5330)       : 2*185e-6,
                            (5330,          last_fill)  : 2*140e-6
                          }


# --- Plotting parameters ---
doAllPlots              = True

doCyclePlots            = True
doSBPlots               = False #True
doSummaryPlots          = True

savePlots               = True
makePlotTarball         = False
fig_tuple               = (15, 7)   # figure size
plotFormat              = ".png"    # savefig extension
plotDpi                 = 300       # savefig dpi
myfontsize              = 16        # fontsize for plots
n_skip                  = 1         # range step @ todo should be hardcoded?


# --- Misc ---
FORMAT                  = '%(asctime)s %(levelname)s : %(message)s'


# --- HT Condor Sub Script ---
htcondor_sub 			= """# Universe
universe   = vanilla

# Get env variables at submission
getenv     = True

should_transfer_files = YES

executable = <EXE>
transfer_input_files = config.py
output     = <OUT>
error      = <ERR>
log        = <LOG>


queue
"""
