#
#   A tool to read the YAML database to hold the last date each massi file
#   was last modified
#
#   Author: Nikos Karastathis ( nkarast <at> cern <dot> ch )
#   Version : 1.0 Apr. 2017
#
import sys
import yaml
import glob
import os
import datetime
from logging import *
basicConfig(format='%(asctime)s %(levelname)s : %(message)s', filename=None, level=20)
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

def readYamlDB(filln, year=2016, yamldb='fill_db.yaml', afs_path='/afs/cern.ch/user/l/lpc/w0/<YEAR>/measurements/',
               exp_folders=['ATLAS/', 'CMS/lumi/']):
    '''
    Function to read the YAML database with the fill numbers and their last modified date. If the date is not the same as
    in the database a bool True is returned to trigger the run for Massi files.
    Input : filln       : Fill number
            year        : Year [default=2016] (used in the afs folder)
            yamldb      : filename of the yamldb [defalt='fill_db.yaml'] (created with the createYamlDB.py)
            afs_path    : path to the measurements folder in LPC AFS [default='/afs/cern.ch/user/l/lpc/w0/<YEAR>/measurements/']
            exp_folders : list of the experiment folders under the measurements in case they are different [default=['ATLAS/', 'CMS/lumi/'] ]

    Returns : runMassi  : boolean to signal if I have to force Massi File run
    '''

    year = year
    experiments = ["ATLAS", "CMS"]
    filln = filln
    temp_afs_path = afs_path.replace('<YEAR>', str(year))
    exp_folders = exp_folders
    info("# readYamlDB : Reading Database [{}] for Year {} and Fill {} under the path : [{}]".format(yamldb, year, filln, temp_afs_path))

    # Temporary dictionary
    current_modDate = {'ATLAS': None, 'CMS' : None}

    # Looping for both experiments
    for exp in exp_folders:
        exp_path = temp_afs_path+exp
        infile = glob.glob(exp_path+"{}*[!*lumi*][!*tmp*][!*lumiregion*]".format(filln))

        if len(infile) == 0 :
            raise IOError('# readYamlDB : File for fill {} of experiment {} does not exist!'.format(filln, exp.split("/")[0]))
        else:
            infile = infile[0]

        lastModDate = datetime.datetime.fromtimestamp(os.path.getmtime(infile))
        if 'ATLAS' in exp:
            current_modDate['ATLAS'] = lastModDate
        elif 'CMS' in exp:
            current_modDate['CMS']   = lastModDate
        else:
            raise KeyError("#readYamlDb: Unknown experiment.")


    debug("# readYamlDB : Current modified dates : {}.".format(current_modDate))
    debug("# readYamlDB : Loading YAML database : {}".format(yamldb))
    with open(yamldb, 'r') as fid:
        database = yaml.load(fid)

    # flag to run massi :
    runMassi = False
    # filename for this fill
    filln_filename = '{}.tgz'.format(filln)
    debug("# readYamlDB : Filename of this fill is {}".format(filln_filename))
    debug("# readYamlDB : ATLAS modified date in DB : {}".format(database[year]['ATLAS'][filln_filename]))
    debug("# readYamlDB : CMS   modified date in DB : {}".format(database[year]['CMS'][filln_filename]))

    # Check if the file is in the database in case of new file:
    if filln_filename in database[year]['ATLAS'] :

        if current_modDate['ATLAS'] == database[year]['ATLAS'][filln_filename]:
            info("# readYamlDB : ATLAS Massi file is up-to-date!")
        else:
            warn("# readYamlDB : ATLAS Massi file is NOT up-to-date! Updating Database...")
            database[year]['ATLAS'][filln_filename] = current_modDate['ATLAS']
            runMassi = True
    else:
        warn('# readYamlDB : The fill number {} does not exist in database for ATLAS. Is this a new fill? Updating DB and running for Massi file...'.format(filln))
        database[year]['ATLAS'][filln_filename] = current_modDate['ATLAS']
        runMassi = True


    if filln_filename in database[year]['CMS'] :
        if current_modDate['CMS'] == database[year]['CMS'][filln_filename]:
            info("# readYamlDB : CMS Massi file is up-to-date!")
        else:
            warn("# readYamlDB : CMS Massi file is NOT up-to-date! Updating Database...")
            database[year]['CMS'][filln_filename] = current_modDate['CMS']
            runMassi = True
    else:
        warn('# readYamlDB : The fill number {} does not exist in database for CMS. Is this a new fill? Updating DB and running for Massi file...'.format(filln))
        database[year]['CMS'][filln_filename] = current_modDate['CMS']
        runMassi = True


    # now update the db
    if runMassi:
        warn("# readYamlDB : ATLAS and/or CMS Massi files were updated since last DB entry. Writing Database...")
        with open('fill_db.yaml', 'w') as fid:
            yaml.dump(database, fid, default_flow_style=True)

    return runMassi

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
if __name__ == '__main__':
    if len(sys.argv) > 1:
        readYamlDB(sys.argv[1])
    else:
        readYamlDB(5256)
