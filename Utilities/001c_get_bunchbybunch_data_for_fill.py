import sys
sys.path.append('/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/')
import LHCMeasurementTools.LHC_FBCT as FBCT
import LHCMeasurementTools.LHC_BQM as BQM
import LHCMeasurementTools.LHC_BSRT as BSRT

import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_Fill_LDB_Query import save_variables_and_pickle

import pickle
import os

csv_folder = '/eos/user/l/lumimod/2017/fill_bunchbybunch_data_csvs'
filepath =  csv_folder+'/bunchbybunch_data_fill'

if not os.path.isdir(csv_folder):
    os.mkdir(csv_folder)

fills_pkl_name = '/eos/user/l/lumimod/2017/fills_and_bmodes.pkl'
with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

saved_pkl = csv_folder+'/saved_fills.pkl'

varlist = []
varlist += FBCT.variable_list()
varlist += BQM.variable_list()
varlist += BSRT.variable_list()

save_variables_and_pickle(varlist=varlist, file_path_prefix=filepath,
                          save_pkl=saved_pkl, fills_dict=dict_fill_bmodes)
