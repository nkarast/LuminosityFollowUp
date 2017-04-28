#
#   A tool to create the YAML database to hold the last date each massi file
#   was last modified
#
#   Author: Nikos Karastathis ( nkarast <at> cern <dot> ch )
#   Version : 1.0 Apr. 2017

import glob
import yaml
import os
import datetime
from logging import *
basicConfig(format='%(asctime)s %(levelname)s : %(message)s', filename=None, level=20)
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *

def createYamlDB(yamldb='fill_db.yaml', year=2016, afs_path='/afs/cern.ch/user/l/lpc/w0/<YEAR>/measurements/',
               exp_folders=['ATLAS/', 'CMS/lumi/']):

    study_year = year
    afs_path = afs_path.replace('<YEAR>', str(year))
    exp_folders = exp_folders

    # Create default DB formatting
    s = """---
{:d}:
    ATLAS: <ATLAS_FILES>
    CMS: <CMS_FILES>
...
    """.format(study_year)
    dict_obj = yaml.load(s)
    info("# createYamlDB : Format of the database created : {}".format(dict_obj))

    # Loop over experiments
    for exp in exp_folders:
        exp_path = afs_path+exp
        files = glob.glob(exp_path+"*[!*lumi*][!*tmp*][!*lumiregion*]")
        filenames = [fn.split('/')[-1] for fn in files]
        lastModDate_timestamp = map(os.path.getmtime, files)
        lastModDate_datetime  = map(datetime.datetime.fromtimestamp, lastModDate_timestamp)

        if 'ATLAS' in exp:
            dict_obj[study_year]['ATLAS'] = dict(zip(filenames, lastModDate_datetime))
        if 'CMS'   in exp:
            dict_obj[study_year]['CMS'] = dict(zip(filenames, lastModDate_datetime))


    # write database
    info("# createYamlDB : Creating YAML database file [{}]".format(yamldb))
    with open(yamldb, 'w') as fid:
        yaml.dump(dict_obj, fid, default_flow_style=True)

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - *
if __name__ == '__main__':
    createYamlDB()
