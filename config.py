#   config.py  -- Configuration file for the lumimod Luminosity Follow-Up script
#
#   ``Too lazy to write json, too module-dependent to write yaml``
#   Author : Nikos Karastathis (nkarast <at> cern <dot> ch)
#   Version : 0.3 (28/04/2017)

import numpy as np

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Fills and periods
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
first_fill              = 5005
last_fill               = 5711 +1 # use + 1 if only used in bmodes
periods                 = {
                            'A': (first_fill,  5256),
                            'B': (5256,     5405),
                            'C': (5405,     last_fill)
                          }

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Folder layout, folder Options and naming conventions
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
input_folder            = "/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/"
working_folder          = input_folder
data_folder             = input_folder+"Utilities/"
# data_folder             = input_folder+"dataFiles/"
stableBeams_folder      = working_folder+"SB_analysis/"
fill_dir                = "fill_<FILLNUMBER>/"
plot_dir                = "plots/"
SB_filename             = "fill_<FILLNUMBER><RESC>.pkl.gz"
SB_fits_filename        = "fill_<FILLNUMBER>_fits<RESC>.pkl.gz"
SB_model_filename       = "fill_<FILLNUMBER>_sbmodel<RESC>.pkl.gz"
SB_burnoff_filename     = "fill_<FILLNUMBER>_sigma_burnoff<RESC>.pkl.gz"
Cycle_filename          = "fill_<FILLNUMBER>_cycle<RESC>.pkl.gz"
Cycle_model_filename    = "fill_<FILLNUMBER>_cycle_model<RESC>.pkl.gz"
Lumi_filename           = "fill_<FILLNUMBER>_lumi_calc<RESC>.pkl.gz"
makedirs                = True  # Create the directories if they do not exist
overwriteFiles          = False # Overwrite existing files
saveDict                = True  # Save data in dictionary form
savePandas              = True  # Save data in Pandas DataFrame form (experimental)

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- BSRT Related Configuration
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
enable_smoothing_BSRT   = True 			# Should I try and smooth the emittance readings?
avg_time_smoothing      = 3.0*3600.0	# average smoothing time in seconds
doRescale               = False  		# Should I do a rescale of the BSRT?
resc_period             = [('A', 'A'), ('B','C'), ('C', 'C')]	# Rescale which period to which?
                        # (From, To), (From, To), (From, To)
add_resc_string         = ''			# do i need to add an extra string in the filenames?


# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Stable Beams Configuratio
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
min_time_SB             = 30*60     # minimum required time in SB to consider fill
t_step_sec              = 10*60     # time step for SB   - data are aligned for every 10 minutes
intensity_threshold     = 3.0e10    # intensity threshold

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# -- SB Model Configuration
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
t_fit_length            = 2.0*3600  # time in seconds used to fit for SB Fits
models                  = ['EmpiricalBlowupLosses', # intensities and emittances are taken from data
							'EmpiricalBlowupBOff',  # emittances taken from data, intensities from model
							'IBSBOff',  # both emittances and intensities are taken from the model
							'IBSLosses' # intensities from the data emittances from the model
						  ]
cases                   = [1]       #Naming convention for the cases to run e.g [1,2,3,4...]
correction_factor_1h    = [1.]      #
correction_factor_2h    = [1.]	  	#	Correction factors to apply to emittances of B1/B2 H/V
correction_factor_1v    = [1.]		#	1.0 = Uncorrected / 1.1 = 110% = +10% correction factor /
correction_factor_2v    = [1.]		#					  / 0.9 = 90%  = -10% correction


# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Basic Input Data Files
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
BASIC_DATA_FILE         = data_folder+'fill_basic_data_csvs/basic_data_fill_<FILLNUMBER>.csv'
BBB_DATA_FILE           = data_folder+'fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_<FILLNUMBER>.csv'
fills_bmodes_file       = data_folder+'fills_and_bmodes.pkl'

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Massi Files Configuration
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
Massi_filename          = 'fill_<FILLNUMBER>_lumi_meas.pkl.gz'
massi_file_database     = working_folder+"Utilities/fill_db.yaml"
massi_year              = 2017
massi_afs_path          = '/afs/cern.ch/user/l/lpc/w0/<YEAR>/measurements/'
massi_exp_folders       = ['ATLAS/lumi/', 'CMS/lumi/']
massi_bunch_lumi_scale  = 1.0e34






# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Machine Parameters
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
frev                    = 11245.5   # revolution frequency in Hz
gammaFT                 = 6927.64   # relativistic gamma # for 6.5 TeV
gammaFB                 = 479.6     # relativistic gamma # for 450 GeV
tauSRxy_FT              = 64.7*3600 # damping times due to synchrotron radiation at FT energy (in s) for transverse plane
tauSRxy_FB              = np.inf    # damping times due to synchrotron radiation at FB energy (in s) for transverse plane
tauSRl_FT               = 32.35*3600# damping times due to synchrotron radiation at FT energy (in s) for longitudinal plane
VRF_FT                  = 10.0e06   # RF Voltage at flat top (in V)
VRF_FB                  = 6.0e06    # RF Voltage at flat bottom (in V)
betastar_m              = 0.40      # beta function at IP used for lumi calculation
sigmaBOff_m2            = 80.0*1.0e-31 # burnoff cross-section
sigma_el_m2             = 29.7*1.0e-31 # inelastic cross-section
crossingAngleChange     = True      # has the crossing angle changed between fills?
XingAngle               = {         # dictionary for fill ranges and full crossing angle values
                            (first_fill,    5330)       : [2*185.0e-06, 2*185.0e-06 ],
                            (5330,          5600)       : [2*140.0e-06, 2*140.0e-06 ],
                            (5600,          5700)       : [2*150.0e-06, 2*150.0e-06 ],
                            (5700,     last_fill)       : [2*140.0e-06, 2*150.0e-06 ],

                          }

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Plotting parameters
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
doAllPlots              = True 	# experimental

doCyclePlots            = True 	# make the Cycle Plots
doCycleModelPlots       = True  # make the Cycle Model Plots
doSBPlots               = False  # make the SB plots
doSBModelPlots          = False  # make the SB Model Plots
doSummaryPlots          = False  # experimental

savePlots               = True
makePlotTarball         = False
fig_tuple               = (15, 7)   # figure size
plotFormat              = ".png"    # savefig extension
plotDpi                 = 300       # savefig dpi
myfontsize              = 16        # fontsize for plots
n_skip                  = 1         # range step @ todo should be hardcoded?


# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- Misc
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
FORMAT                  = '%(asctime)s %(levelname)s : %(message)s'  # Logging module format

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
#
# --- HT Condor Sub Script
#
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
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
