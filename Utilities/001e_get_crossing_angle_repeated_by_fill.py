import sys
sys.path.append('/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/')

import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_FILL_LDB_Query_Repeat import save_variables_and_pickle
import config
import pickle
import os

csv_folder = '/eos/user/l/lumimod/2017/fill_crossing_data_csvs'
filepath =  csv_folder+'/crossing_angle_fill'

if not os.path.isdir(csv_folder):
    os.mkdir(csv_folder)

# fills_pkl_name = 'fills_and_bmvodes.pkl'
fills_pkl_name = '/eos/user/l/lumimod/2017/fills_and_bmodes.pkl'
with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

saved_pkl = csv_folder+'/saved_fills.pkl'

varlist = ['LHC.RUNCONFIG:IP1-XING-V-MURAD', 'LHC.RUNCONFIG:IP5-XING-H-MURAD']

save_variables_and_pickle(varlist=varlist, file_path_prefix=filepath,
                          save_pkl=saved_pkl, fills_dict=dict_fill_bmodes)
