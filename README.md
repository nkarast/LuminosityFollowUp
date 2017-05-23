# LHC Luminosity Follow-Up
An automated procedure to run the analysis for the follow-up of luminosity during LHC runs. 



## Quick Example:
`python lumimod.py --verbose --fill 5256 5257 5259 5260 5263 --submit --split 2 --force cycle `

This makes the script to :

  * `--verbose` : activates debug mode
  * `--fill fill1 fill2 fill3...` : runs for the specified fills. If not defined the fills are taken from beam mode file.
  * `--submit` : activates the submission to HTCondor
  * `--split <number>` : splits the fill number list in chunks of <number> fills per chunk
  **NB**: This split depends on the length of the fill list. In other words some chunks may have a bit more fills.
  * `--force` : The script in general will look for the files you require in the specified folder to check if they are already generated. If a file is there the step for the creation of this file will be skipped. With the force command taking the arguments `< all | sb | cycle | lumi | massi | model | lifetime | fits >` you force the script to run all or one specified step and overwrite the already existing file.
      - `all`  : forces to run for every step 
      - `sb`   : forces the SB data files to be recreated
      - `cycle`: forces the cycle data and cycle model to run
      - `lumi` : forces to run for the calculated luminosity
      - `massi`: forces to run for measured luminosity
      - `lifetime`: forces to calculate lifetimes
      - `fits` : forces the creation of fits from SB data and Calculated and Measured Luminosity files
      - `model` : forces the Stable Beams model to run under the assumptions and the cases defined in the `config.py` file

## Code Walkthrough

**Step 1: Getting our data**
For this step we are using Gianni's script unmodified for the moment.
  * Open `000_get_fills.py` and modify the `t_start_string` and `t_stop_string`. These indicate the start and stop time for which you want to look for data. Run it and it will create the fills and bmodes files.
  * Run `001a_get_basic_data_for_fill.py` and it will create the `fill_basic_data_csv/basic_data_fill/` folder and fill it with the basic data csv files
  * Run `001c_get_bunchbybunch_data_for_fill.py` and it will create the `fill_bunchbybunch_data_fill/bunchbybunch_data_fill/` folder and fill it with BBB data.
  * Run `001d_get_extra_data_for_fill.py` it will create the `fill_extra_data_csvs/extra_data_fill/` folder and fill it with extra data for the variables below:
      - `'LHC.BOFSU:TUNE_B1_H'`
      - `'LHC.BOFSU:TUNE_B1_V'`
      - `'LHC.BOFSU:TUNE_B2_H'`
      - `'LHC.BOFSU:TUNE_B2_V'`
      - `'LHC.BOFSU:TUNE_TRIM_B1_H'`
      - `'LHC.BOFSU:TUNE_TRIM_B1_V'`
      - `'LHC.BOFSU:TUNE_TRIM_B2_H'`
      - `'LHC.BOFSU:TUNE_TRIM_B2_V'`
      - `'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_AMPL_1'`
      - `'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_AMPL_2'`
      - `'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_FREQ_1'`
      - `'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_FREQ_2'`
      - `'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_AMPL_1'`
      - `'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_AMPL_2'`
      - `'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_FREQ_1'`
      - `'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_FREQ_2'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_AMPL_1'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_AMPL_2'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_1'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_2'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_AMPL_1'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_AMPL_2'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_1'`
      - `'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_2'`
      - `'ALICE:LUMI_TOT_INST'`
      - `'ATLAS:LUMI_TOT_INST'`
      - `'CMS:LUMI_TOT_INST'`
      - `'LHCB:LUMI_TOT_INST'`
      - `'HX:BETASTAR_IP1'`
      - `'HX:BETASTAR_IP2'`
      - `'HX:BETASTAR_IP5'`
      - `'HX:BETASTAR_IP8'`
      - `'LHC.BQM.B1:NO_BUNCHES'`
      - `'LHC.BQM.B2:NO_BUNCHES'`
      - `'ADTH.SR4.B1:CLEANING_ISRUNNING'`
      - `'ADTH.SR4.B2:CLEANING_ISRUNNING'`
      - `'ADTV.SR4.B1:CLEANING_ISRUNNING'`
      - `'ADTV.SR4.B2:CLEANING_ISRUNNING'`
  * When the data are done downloading copy them in a directory (let's call it `data_folder` for now...)

**Step 2: Configuration File**
The configuration file `config.py` holds all the "global" variables and parametrisations that the main driving routine will be based on.
In the configuration file one can change:
  * **Folder structures** (input folder, working folder, data folder etc)
  * **Filename templates** (for SB, fits, SBModel, Cycle, CycleModel, CalcLumi etc)
  * **Basic options** like : 
      - makedirs : to make the necessary output directories if they are not present
      - overwriteFiles : to be able to overwrite a file if it is redone
      - saveDict : Save output data file in dictionary form (within a gzip pickle)
      - savePandas : Save output data file in Pandas DF form (within a gzip pickle
  * **BSRT Rescaling** options
  * **Massi files options** and YAML database related options
  * **SB Model Configuration** : which models, how long the fit will be, cases for correcting emittances, etc
  * **Stable Beams Requirements**: minimum time in SB, intensity threshold etc
  * **Machine parameters** : revolution frequency, gamma at FT and FB, damping times due to SR at FT/FB transverse/longitudinal, RF Voltages at FT/FB, b*, burnoff, crossing angle
  * **Plotting Parameters**: 
      - doCyclePlots, doCycleModelPlots, doSBPlots, doSBModelPlots
      - plot format (png), plotDPI, savePlots flag, etc
  * **Misc & HTCondor** Submission template for .sub file



**Step 3) Our driver: The `lumimod.py` script**

This is the main script that you want to call from the command line. It can take the following command line arguments:
  * `-v`, `--verbose`: Activates debug environment
  * `-ll <NUM>`, `--loglevel <NUM>` : changes the default loglevel (DEBUG=10, INFO=20, WARN=30, ERROR=40, FATAL=50) only output of specified level and above will be shown.
  * `-n <NUM NUM ...>` , `--force <NUM NUM ...>` : run only for the specified fill (or fills). If more than one fills are specified provide them inline separated by whitespaces (no comma, no quotes necessary)
  * `-f <STR>` , `--force <STR>` : force a specified step the STR options are:
      - `all`      : forces all step to be done
      - `cycle`    : forces cycle and cycleModel steps to be done
      - `sb`       : forces stable beams data step to be done
      - `massi`    : forces massi file step to be done
      - `lumi`     : forces calculated lumi step to be done
      - `lifetime` : forces lifetime step to be done
      - `fits`     : forces fits step on the SB data to be done
      - `model`    : forces SB Model step to be done
      N.B. The `< >` are not needed.
      N.B. `force` activates the `overwriteFiles` option!!
  * `-d <STR>`, `--doOnly <STR>` : forces only a step to be done options same as `-f`
  * `-s`, `--submit` : Submit the job in HTCondor
  * `-b <NUM>`, `--split <NUM>`  : Works only for HTCondor submission. It splits the fill list to run in chuncks of `<NUM>` and submits them to HTCondor batch system.

_Example_
I want to submit fills 5246, 5251, 5253, 5256 in HTCondor and break them by 1 fill per job. I want to force the SB model to run. I will run...
`python lumimod.py --fill 5246 5251 5253 5256 -s -b 1 -f model`

_Code Calls_
On execution `lumimod.py` will create a list of fills to run based on the fill and bmodes file generated by `Utilities/000_get_fills.py`, and will do some cleaning by removing fills that are not between `first_fill` and `last_fill` and have SB time more than the `min_SB_time` all defined in `config.py`

Then the the script will check if the Massi file database exists. If not it will run the `Utilities/createYamlDB.py` script with inputs from the `config.py` file. the necessary ones are:
* `yamldb` = filename of yaml db file to be created (defaults: `fill_db.yaml`)
* `year`   = which year are you interested in (in YYYY format)
* `afs_path` = LPC afs path (defaults : `/afs/cern.ch/user/l/lpc/w0/<YEAR>/measurements/`)
* `exp_folders` = folders under which the experiments have their Massi files (defaults: `['ATLAS/lumi' 'CMS/Lumi']`)

_Fill numbers Database_
Now you have the fill database! Every time the script is run on a fill, the code will check if the massi file existing under the experiment folder has a more recent modification date than the one in the DB. If the modification dates are the same, then the Massi files have not been updated. 
If the modification date is different then the code forces the Massi file to be downloaded and updates the modification date in the DB.
If the fill does not exist in the DB (i.e. new fill) the DB is updated.

_Loop for fills_
You can either pass a fill (or a fill list) in the `lumimod.py` script... or even nothing at all!

In case the fill argument is used the fill list is set to the one defined by the user. Otherwise, the fill list will be taken from the bmodes file after the cleaning is performed. This means that in your day to day life, you simply need to download the data and then run `python lumimod.py`!

For every one of your fills the code will check first of all that your directory structure is there. If not (and `makedirs` is defined in `config.py`) will create them. 
Then will check for the files existing under these directories. Let's assume that no force argument is passed. The code will check if there is :
  * the stable beams data
  * the measured lumi data
  * the calculated lumi data
  * the cycle data
  * the cycle model data
  * the lifetime data
  * the SB fits data
  * the SB model data for all cases defined
If all of these files exist and no Plotting is required (in the `config.py` file), the code will skip this fill.
If some of these data files do not exist the code will run for the specified step to prepare the data.

_Plotting_
If some of the plot flags are specified (in the `config.py` file) the code will produce and save (if `savePlots` flag in `config.py` is `True`) the plots below:

  * ** Stable Beams ** 
      - BBB Raw Emittances for B1/B2 (Raw = without smoothing)
      - BBB Emittances for B1/B2
      - BBB Intensity
      - BBB Bunch Length
      - Calculated Luminosity
      - Measured Luminosity
      - Total Luminosity (calculated/measured for ATLAS/CMS)
      - BBB Intensity Lifetime
      - lifetime of luminosity for the total length of fill
      - lifetime of luminosity for the fitted period of the fill 
      - burnoff corrected 1/tau for intensities
      - intensity lifetime for B1/B2
      - BBB losses normalized with Luminosity for B1/B2

  * ** Cycle **
      - BBB Emittances during cycle
      - BBB Intensities during cycle
      - BBB Bunch Length during cycle
      - BBB Brightness during cycle
      - BBB Time during cycle
      - BBB Histos for Emittance, Intensity, Brightness at Injection, Start Ramp, End Ramp, Start SB time instances

  * ** Cycle Model **
      - Emittances for B1/B2 for Flat Top and Injection
      - Bunch Length and intensities for FlatTop and Injection

  * ** Stable Beams Model**
      - Luminosity for ATLAS and CMS
      - Emittances for B1/B2 for colliding/non-colliding bunches
      - Intensities and bunch lengths for B1/B2 for colliding/non-colliding bunches
      
      N.B. The plots for the SB Model are done for each of the cases defined in `config.py` and for each of the models :
       * EmpiricalBlowupLosses : intensities and emittances are taken from data, bunch length and luminosity calculated by model
       * EmpiricalBlowupBOff  : emittances are taken from data, intensities, luminosity and bunch length from the model
       * IBSBOff              : emittances, intensities, luminosity, bunch length all calculated by the model
       * IBSLosses            : intensities are taken from the data, emittances, bunch length and luminosity calculated by the model





## Credits
This work is based on the excellent work of F. Antoniou and G. Iadarola!

