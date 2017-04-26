# LHC Luminosity Follow-Up
An automated procedure to run the analysis for the follow-up of luminosity during LHC runs. 



## Usage

The tool for the time being is based around the scripts below:

* config.py : Defines some variable input and user options for the execution of the main scripts.
* LumiFollowUp.py : Is the main module, consisting of the LumiFollowUp class. Depending on initialization the class is based on the beam modes file and the LHC measurement tools to create Cycle Data, Stable Beams Data, Calculated Luminosity (from SB data), and Measured Luminosity (Massi files) pickles.
* lumimod.py : The driving routine for the automated script. It can handle various command line arguments from user to specify fill, force steps etc. The script also is capable of submitting jobs to HTCondor.
* submit_template.py : A template for the executable to be submitted in HTCondor.

### Example:
`python lumimod.py --verbose --fill 5256 5257 5259 5260 5263 --submit --split 2 --force 'cycle' `

This makes the script to :

  * `--verbose` : activates debug mode
  * `--fill fill1,fill2,fill3...` : runs for the specified fills. If not defined the fills are taken from beam mode file.
  * `--submit` : activates the submission to HTCondor
  * `--split <number>` : splits the fill number list in chunks of <number> fills per chunk
  **NB**: This split depends on the length of the fill list. In other words some chunks may have a bit more fills.
  * `--force` : The script in general will look for the files you require in the specified folder to check if they are already generated. If a file is there, the step for the creation of this file will be skipped. With the force command taking the arguments `< all | sb | cycle | lumi | massi >` you force the script to run all or one specified step and overwrite the already existing file.


## Credits
This work is based on the excellent work of F. Antoniou and G. Iadarola!

