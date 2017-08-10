# Test BSRT_calib_nikTest.py

import BSRT_calib_nikTest as BSRT_calib
import TimberManager as tm
import sys

def getTimberDic(filln, infile):
	'''
	Function to get the dictionary from Timber data
	Input :     filln            = fill number
				BASIC_DATA_FILE  = basic data file for timber
				BBB_DATA_FILE    = bunch by bunch data file
				debug            = debug bool to be used as verbose option
	Returns:    timber_dic       = the dictionary from timber data
	'''
	timber_dic = {}
	timber_dic.update(tm.parse_timber_file(infile.replace('<FILLNUMBER>',str(filln)), verbose=False))
	# timber_dic.update(tm.parse_timber_file(self.BBB_LUMI_DATA_FILE.replace('<FILLNUMBER>',str(filln)), verbose=self.debug))
	return timber_dic


if __name__ == '__main__':
	input_file = "/eos/user/l/lumimod/2017/fill_bunchbybunch_data_csvs/bunchbybunch_data_fill_<FILLNUMBER>.csv"

	filln = None
	if len(sys.argv) > 1 :
		filln = sys.argv[1]
	else:
		filln = 5750

	print 'Using test filln = ', filln

	timber_dictionary = getTimberDic(filln, infile = input_file)

	bsrt_calib_dict = BSRT_calib.emittance_dictionary(filln=filln, timber_dic= timber_dictionary)
	# param_list = ["betaf_h", "betaf_v", "sigma_corr_h", "sigma_corr_v", "scale_h", "scale_v", "rescale_sigma_h", "rescale_sigma_v"]
	# for beamn in [1,2]:
	# 	for param in param_list:
	# 		for energy in [450, 6500]:
	# 			print "Beam {} - {} - {} = {:.4f}".format(beamn, param, energy, bsrt_calib_dict[param][energy][beamn])
	# 		print ' '
	# 	print ' ------ '
