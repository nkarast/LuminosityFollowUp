import sys
sys.path.append('/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/')
import LHCMeasurementTools.lhc_log_db_query as lldb
import LHCMeasurementTools.TimestampHelpers as th
import LHCMeasurementTools.LHC_Fills as Fills
import config


t_start_string = '2016_08_23 00:00:00'
t_stop_string = '2016_08_31 23:00:00'

t_start = th.localtime2unixstamp(t_start_string)
t_stop = th.localtime2unixstamp(t_stop_string)

filename = 'fills_and_bmodes'
csv_name = filename + '.csv'
pkl_name = filename + '.pkl'

# Get data from database
varlist = Fills.get_varlist()
lldb.dbquery(varlist, t_start, t_stop, csv_name)

# Make pickle
Fills.make_pickle(csv_name, pkl_name, t_stop)
