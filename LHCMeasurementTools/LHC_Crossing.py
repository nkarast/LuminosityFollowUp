import numpy as np
import TimberManager as tm

class Crossing:
	def __init__(self, timber_variable, IP=0):

		if type(timber_variable) is str:
			if not (IP == 1 or IP == 5):
				raise ValueError('You need to specify which IP! (1 or 5)')
			dict_timber = tm.parse_timber_file(timber_variable, verbose=True)
			if IP == 1:
				timber_variable_XING = dict_timber[get_variable_dict['LHC.RUNCONFIG:IP1-XING-V-MURAD']]
			elif IP == 5: 
				timber_variable_XING = dict_timber[get_variable_dict['LHC.RUNCONFIG:IP5-XING-H-MURAD']]

		elif hasattr(timber_variable, '__getitem__'):
			try:
				if IP == 1:
					timber_variable_XING = timber_variable['LHC.RUNCONFIG:IP1-XING-V-MURAD']
				elif IP == 5: 
					timber_variable_XING = timber_variable['LHC.RUNCONFIG:IP5-XING-H-MURAD']
			except:
				print '# LHC_Crossing : No Crossing Angle Information! Returning -1'
				return -1

		self.t_stamps = timber_variable_XING.t_stamps
		self.xing = timber_variable_XING.values


		self.xing     = np.array(np.float_(self.xing)).ravel()
		self.t_stamps = np.array(np.float_(self.t_stamps))

	def nearest_older_sample(self, t_obs, flag_return_time=False):
		ind_min = np.argmin(np.abs(self.t_stamps - t_obs))
		if self.t_stamps[ind_min] > t_obs:
			ind_min -= 1
		if flag_return_time:    
			if ind_min == -1:
				return 0.*self.xing[ind_min], -1
			else:   
				return self.xing[ind_min], self.t_stamps[ind_min]
		else:
			if ind_min == -1:
				return 0.*self.xing[ind_min]
			else:   
				return self.xing[ind_min]
				

		
def get_variable_dict():
	var_dict = {}
	var_dict['CROSSING_ANGLE_IP1'] = 'LHC.RUNCONFIG:IP1-XING-V-MURAD'
	var_dict['CROSSING_ANGLE_IP5'] = 'LHC.RUNCONFIG:IP5-XING-H-MURAD'

	return var_dict

def variable_list():
	var_list = []
	var_list += get_variable_dict().values()

	return var_list
