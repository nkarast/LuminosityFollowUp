import sys
sys.path.append('/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/')
import LHCMeasurementTools.LHC_Fills as Fills
from LHCMeasurementTools.LHC_Fill_LDB_Query import save_variables_and_pickle

import pickle
import os

csv_folder = 'fill_extra_data_csvs'
filepath =  csv_folder+'/extra_data_fill'

if not os.path.isdir(csv_folder):
    os.mkdir(csv_folder)

fills_pkl_name = 'fills_and_bmodes.pkl'
with open(fills_pkl_name, 'rb') as fid:
    dict_fill_bmodes = pickle.load(fid)

saved_pkl = csv_folder+'/saved_fills.pkl'

varlist = [
'LHC.BOFSU:TUNE_B1_H',
'LHC.BOFSU:TUNE_B1_V',
'LHC.BOFSU:TUNE_B2_H',
'LHC.BOFSU:TUNE_B2_V',
'LHC.BOFSU:TUNE_TRIM_B1_H',
'LHC.BOFSU:TUNE_TRIM_B1_V',
'LHC.BOFSU:TUNE_TRIM_B2_H',
'LHC.BOFSU:TUNE_TRIM_B2_V',
'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_AMPL_1',
'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_AMPL_2',
'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_FREQ_1',
'LHC.BQBBQ.CONTINUOUS.B1:EIGEN_FREQ_2',
'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_AMPL_1',
'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_AMPL_2',
'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_FREQ_1',
'LHC.BQBBQ.CONTINUOUS.B2:EIGEN_FREQ_2',
'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_AMPL_1',
'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_AMPL_2',
'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_1',
'LHC.BQBBQ.CONTINUOUS_HS.B1:EIGEN_FREQ_2',
'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_AMPL_1',
'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_AMPL_2',
'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_1',
'LHC.BQBBQ.CONTINUOUS_HS.B2:EIGEN_FREQ_2',
'ALICE:LUMI_TOT_INST',
'ATLAS:LUMI_TOT_INST',
'CMS:LUMI_TOT_INST',
'LHCB:LUMI_TOT_INST',
'HX:BETASTAR_IP1',
'HX:BETASTAR_IP2',
'HX:BETASTAR_IP5',
'HX:BETASTAR_IP8',
'LHC.BQM.B1:NO_BUNCHES',
'LHC.BQM.B2:NO_BUNCHES',
'ADTH.SR4.B1:CLEANING_ISRUNNING',
'ADTH.SR4.B2:CLEANING_ISRUNNING',
'ADTV.SR4.B1:CLEANING_ISRUNNING',
'ADTV.SR4.B2:CLEANING_ISRUNNING']


save_variables_and_pickle(varlist=varlist, file_path_prefix=filepath,
                          save_pkl=saved_pkl, fills_dict=dict_fill_bmodes)
