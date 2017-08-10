import pickle
import gzip
import pandas as pd
import numpy as np
import itertools
import os


def makeCycleDataFrame(flist, bunches_dict, outfilename, doModel=False):
	'''
	flist        : list of fill numbers
	bunches_dict : dictionary for bunches
	outfilename  : filename of output pkl
	'''
	print '--------------------------------------------------------------------------------------'
	print ''
	print '#makeCycleDataFrame : [DISCLAIMER] : '
	print '# THIS VERSION OF THE FUNCTION IS BLIND DURING RAMP FOR THE MODEL.'
	print '# The emittances at Injection and at End Ramp are the same for measurement and model'
	print '# because the evolution INJ - FB and FT - SB is calculated, NOT THE INJ-SB!!!'
	print ''
	print '--------------------------------------------------------------------------------------'

	df_exists = False
	if os.path.exists(outfilename):
		print '#makeCycleDataFrame : [WARN] Filename of output file given for dataframe exists. Checking if it is up-to-date...'
		df_exists = True
		df_old = pd.read_pickle(outfilename)
		fills_existing = np.unique(df_old['Fill'].values)
		overlap_fills = list(set(fills_existing.tolist()).intersection(flist))
		print '#makeCycleDataFrame : [WARN] The old dataframe has already the fills : [{}] dropping them.'.format(overlap_fills)
		newflist = [fil for fil in flist if fil not in overlap_fills]
		flist = newflist

	df_loop = pd.DataFrame()
	for filln in flist:

		print '#makeCycleDataFrame : Working on fill ' , filln


		# read cycle 
		with gzip.open('SB_analysis/fill_{}/fill_{}_cycle.pkl.gz'.format(filln, filln), 'rb') as fid:
			cycle = pickle.load(fid)

		# read model
		if doModel:
			with gzip.open('SB_analysis/fill_{}/fill_{}_cycle_model.pkl.gz'.format(filln, filln), 'rb') as fid:
				model = pickle.load(fid)



		bunches = None
		for key in bunches_dict.keys():
			if filln >= key[0] and filln < key[1]:
				bunches = bunches_dict[key]


		emit_meas    = []
		time_meas    = []
		cycle_meas   = []
		slots        = []
		plane_meas   = []
		beam_meas    = []
		fill_meas    = []
		bunches_meas = []
		kind         = []

		intensity_meas    = []
		bunch_length_meas = []
		brightness_meas   = []


		# this is B1H INJ
		emit_b1h_inj  = np.array(cycle['beam_1']['Injection']['at_start']['emith'])
		time_b1h_inj  = np.array(cycle['beam_1']['Injection']['at_start']['time_meas'])
		slots_b1h_inj = np.array(cycle['beam_1']['Injection']['filled_slots'])
		
		if not doModel:

			print '#makeCycleDataFrame : [WARN] While emittance is filled both H/V with different numbers, intensity, bunch length and brightness is filled twice for slots of <<Horizontal>> and <<Vertical>>!'

			intens_b1h_inj		= np.array(cycle['beam_1']['Injection']['at_start']['intensity'])
			blength_b1h_inj		= np.array(cycle['beam_1']['Injection']['at_start']['blength'])
			brightness_b1h_inj	= np.array(cycle['beam_1']['Injection']['at_start']['brightness'])

			intensity_meas.append(intens_b1h_inj)
			bunch_length_meas.append(blength_b1h_inj)
			brightness_meas.append(brightness_b1h_inj)

		emit_meas.append(emit_b1h_inj)
		time_meas.append(time_b1h_inj)
		slots.append(slots_b1h_inj)
		cycle_meas.append(['Injection']*len(emit_b1h_inj))
		plane_meas.append(['Horizontal']*len(emit_b1h_inj))
		beam_meas.append(['B1']*len(emit_b1h_inj))
		fill_meas.append([filln]*len(emit_b1h_inj))
		bunches_meas.append([bunches]*len(emit_b1h_inj))
		kind.append(['Measurement']*len(emit_b1h_inj))

		# this is B1V INJ
		emit_b1v_inj  = np.array(cycle['beam_1']['Injection']['at_start']['emitv'])
		time_b1v_inj  = np.array(cycle['beam_1']['Injection']['at_start']['time_meas'])
		slots_b1v_inj = np.array(cycle['beam_1']['Injection']['filled_slots'])
		if not doModel:
			intens_b1v_inj		= np.array(cycle['beam_1']['Injection']['at_start']['intensity'])
			blength_b1v_inj		= np.array(cycle['beam_1']['Injection']['at_start']['blength'])
			brightness_b1v_inj	= np.array(cycle['beam_1']['Injection']['at_start']['brightness'])

			intensity_meas.append(intens_b1v_inj)
			bunch_length_meas.append(blength_b1v_inj)
			brightness_meas.append(brightness_b1v_inj)

		emit_meas.append(emit_b1v_inj)
		time_meas.append(time_b1v_inj)
		slots.append(slots_b1v_inj)
		cycle_meas.append(['Injection']*len(emit_b1v_inj))
		plane_meas.append(['Vertical']*len(emit_b1v_inj))
		beam_meas.append(['B1']*len(emit_b1v_inj))
		fill_meas.append([filln]*len(emit_b1v_inj))
		bunches_meas.append([bunches]*len(emit_b1v_inj))
		kind.append(['Measurement']*len(emit_b1v_inj))

		########
		# this is B1H FB
		emit_b1h_fb  = np.array(cycle['beam_1']['Injection']['at_end']['emith'])
		time_b1h_fb  = np.array(cycle['beam_1']['Injection']['at_end']['time_meas'])
		slots_b1h_fb = np.array(cycle['beam_1']['Injection']['filled_slots'])
		if not doModel:
			intens_b1h_fb		= np.array(cycle['beam_1']['Injection']['at_end']['intensity'])
			blength_b1h_fb		= np.array(cycle['beam_1']['Injection']['at_end']['blength'])
			brightness_b1h_fb	= np.array(cycle['beam_1']['Injection']['at_end']['brightness'])

			intensity_meas.append(intens_b1h_fb)
			bunch_length_meas.append(blength_b1h_fb)
			brightness_meas.append(brightness_b1h_fb)
		emit_meas.append(emit_b1h_fb)
		time_meas.append(time_b1h_fb)
		slots.append(slots_b1h_fb)
		cycle_meas.append(['Start Ramp']*len(emit_b1h_fb))
		plane_meas.append(['Horizontal']*len(emit_b1h_fb))
		beam_meas.append(['B1']*len(emit_b1h_fb))
		fill_meas.append([filln]*len(emit_b1h_fb))
		bunches_meas.append([bunches]*len(emit_b1h_fb))
		kind.append(['Measurement']*len(emit_b1h_fb))


		# this is B1V FB
		emit_b1v_fb  = np.array(cycle['beam_1']['Injection']['at_end']['emitv'])
		time_b1v_fb  = np.array(cycle['beam_1']['Injection']['at_end']['time_meas'])
		slots_b1v_fb = np.array(cycle['beam_1']['Injection']['filled_slots'])
		if not doModel:
			intens_b1v_fb		= np.array(cycle['beam_1']['Injection']['at_end']['intensity'])
			blength_b1v_fb		= np.array(cycle['beam_1']['Injection']['at_end']['blength'])
			brightness_b1v_fb	= np.array(cycle['beam_1']['Injection']['at_end']['brightness'])

			intensity_meas.append(intens_b1v_fb)
			bunch_length_meas.append(blength_b1v_fb)
			brightness_meas.append(brightness_b1v_fb)
		emit_meas.append(emit_b1v_fb)
		time_meas.append(time_b1v_fb)
		slots.append(slots_b1v_fb)
		cycle_meas.append(['Start Ramp']*len(emit_b1v_fb))
		plane_meas.append(['Vertical']*len(emit_b1v_fb))
		beam_meas.append(['B1']*len(emit_b1v_fb))
		fill_meas.append([filln]*len(emit_b1v_fb))
		bunches_meas.append([bunches]*len(emit_b1v_fb))
		kind.append(['Measurement']*len(emit_b1v_fb))


		########
		# this is B1H FT
		emit_b1h_ft  = np.array(cycle['beam_1']['he_before_SB']['at_start']['emith'])
		time_b1h_ft  = np.array(cycle['beam_1']['he_before_SB']['at_start']['time_meas'])
		slots_b1h_ft = np.array(cycle['beam_1']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b1h_ft		= np.array(cycle['beam_1']['he_before_SB']['at_start']['intensity'])
			blength_b1h_ft		= np.array(cycle['beam_1']['he_before_SB']['at_start']['blength'])
			brightness_b1h_ft	= np.array(cycle['beam_1']['he_before_SB']['at_start']['brightness'])

			intensity_meas.append(intens_b1h_ft)
			bunch_length_meas.append(blength_b1h_ft)
			brightness_meas.append(brightness_b1h_ft)
		emit_meas.append(emit_b1h_ft)
		time_meas.append(time_b1h_ft)
		slots.append(slots_b1h_ft)
		cycle_meas.append(['End Ramp']*len(emit_b1h_ft))
		plane_meas.append(['Horizontal']*len(emit_b1h_ft))
		beam_meas.append(['B1']*len(emit_b1h_ft))
		fill_meas.append([filln]*len(emit_b1h_ft))
		bunches_meas.append([bunches]*len(emit_b1h_ft))
		kind.append(['Measurement']*len(emit_b1h_ft))

		# this is B1V FB
		emit_b1v_ft  = np.array(cycle['beam_1']['he_before_SB']['at_start']['emitv'])
		time_b1v_ft  = np.array(cycle['beam_1']['he_before_SB']['at_start']['time_meas'])
		slots_b1v_ft = np.array(cycle['beam_1']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b1v_ft		= np.array(cycle['beam_1']['he_before_SB']['at_start']['intensity'])
			blength_b1v_ft		= np.array(cycle['beam_1']['he_before_SB']['at_start']['blength'])
			brightness_b1v_ft	= np.array(cycle['beam_1']['he_before_SB']['at_start']['brightness'])

			intensity_meas.append(intens_b1v_ft)
			bunch_length_meas.append(blength_b1v_ft)
			brightness_meas.append(brightness_b1v_ft)
		emit_meas.append(emit_b1v_ft)
		time_meas.append(time_b1v_ft)
		slots.append(slots_b1v_ft)
		cycle_meas.append(['End Ramp']*len(emit_b1v_ft))
		plane_meas.append(['Vertical']*len(emit_b1h_ft))
		beam_meas.append(['B1']*len(emit_b1v_ft))
		fill_meas.append([filln]*len(emit_b1v_ft))
		bunches_meas.append([bunches]*len(emit_b1v_ft))
		kind.append(['Measurement']*len(emit_b1v_ft))


		########
		# this is B1H SB
		emit_b1h_sb  = np.array(cycle['beam_1']['he_before_SB']['at_end']['emith'])
		time_b1h_sb  = np.array(cycle['beam_1']['he_before_SB']['at_end']['time_meas'])
		slots_b1h_sb = np.array(cycle['beam_1']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b1h_sb		= np.array(cycle['beam_1']['he_before_SB']['at_end']['intensity'])
			blength_b1h_sb		= np.array(cycle['beam_1']['he_before_SB']['at_end']['blength'])
			brightness_b1h_sb	= np.array(cycle['beam_1']['he_before_SB']['at_end']['brightness'])

			intensity_meas.append(intens_b1h_sb)
			bunch_length_meas.append(blength_b1h_sb)
			brightness_meas.append(brightness_b1h_sb)
		emit_meas.append(emit_b1h_sb)
		time_meas.append(time_b1h_sb)
		slots.append(slots_b1h_sb)
		cycle_meas.append(['Start Stable']*len(emit_b1h_sb))
		plane_meas.append(['Horizontal']*len(emit_b1h_sb))
		beam_meas.append(['B1']*len(emit_b1h_sb))
		fill_meas.append([filln]*len(emit_b1h_sb))
		bunches_meas.append([bunches]*len(emit_b1h_sb))
		kind.append(['Measurement']*len(emit_b1h_sb))

		# this is B1V SB
		emit_b1v_sb  = np.array(cycle['beam_1']['he_before_SB']['at_end']['emitv'])
		time_b1v_sb  = np.array(cycle['beam_1']['he_before_SB']['at_end']['time_meas'])
		slots_b1v_sb = np.array(cycle['beam_1']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b1v_sb		= np.array(cycle['beam_1']['he_before_SB']['at_end']['intensity'])
			blength_b1v_sb		= np.array(cycle['beam_1']['he_before_SB']['at_end']['blength'])
			brightness_b1v_sb	= np.array(cycle['beam_1']['he_before_SB']['at_end']['brightness'])

			intensity_meas.append(intens_b1v_sb)
			bunch_length_meas.append(blength_b1v_sb)
			brightness_meas.append(brightness_b1v_sb)
		emit_meas.append(emit_b1v_sb)
		time_meas.append(time_b1v_sb)
		slots.append(slots_b1v_sb)
		cycle_meas.append(['Start Stable']*len(emit_b1v_sb))
		plane_meas.append(['Vertical']*len(emit_b1v_sb))
		beam_meas.append(['B1']*len(emit_b1v_sb))
		fill_meas.append([filln]*len(emit_b1v_sb))
		bunches_meas.append([bunches]*len(emit_b1v_sb))
		kind.append(['Measurement']*len(emit_b1v_sb))

		##############################
		#	NOW B2
		#
		emit_b2h_inj  = np.array(cycle['beam_2']['Injection']['at_start']['emith'])
		time_b2h_inj  = np.array(cycle['beam_2']['Injection']['at_start']['time_meas'])
		slots_b2h_inj = np.array(cycle['beam_2']['Injection']['filled_slots'])
		if not doModel:

			print '#makeCycleDataFrame : [WARN] While emittance is filled both H/V with different numbers, intensity, bunch length and brightness is filled twice for slots of <<Horizontal>> and <<Vertical>>!'

			intens_b2h_inj		= np.array(cycle['beam_2']['Injection']['at_start']['intensity'])
			blength_b2h_inj		= np.array(cycle['beam_2']['Injection']['at_start']['blength'])
			brightness_b2h_inj	= np.array(cycle['beam_2']['Injection']['at_start']['brightness'])

			intensity_meas.append(intens_b2h_inj)
			bunch_length_meas.append(blength_b2h_inj)
			brightness_meas.append(brightness_b2h_inj)

		emit_meas.append(emit_b2h_inj)
		time_meas.append(time_b2h_inj)
		slots.append(slots_b2h_inj)
		cycle_meas.append(['Injection']*len(emit_b2h_inj))
		plane_meas.append(['Horizontal']*len(emit_b2h_inj))
		beam_meas.append(['B2']*len(emit_b2h_inj))
		fill_meas.append([filln]*len(emit_b2h_inj))
		bunches_meas.append([bunches]*len(emit_b2h_inj))
		kind.append(['Measurement']*len(emit_b2h_inj))

		# this is B1V INJ
		emit_b2v_inj  = np.array(cycle['beam_2']['Injection']['at_start']['emitv'])
		time_b2v_inj  = np.array(cycle['beam_2']['Injection']['at_start']['time_meas'])
		slots_b2v_inj = np.array(cycle['beam_2']['Injection']['filled_slots'])
		if not doModel:
			intens_b2v_inj		= np.array(cycle['beam_2']['Injection']['at_start']['intensity'])
			blength_b2v_inj		= np.array(cycle['beam_2']['Injection']['at_start']['blength'])
			brightness_b2v_inj	= np.array(cycle['beam_2']['Injection']['at_start']['brightness'])

			intensity_meas.append(intens_b2v_inj)
			bunch_length_meas.append(blength_b2v_inj)
			brightness_meas.append(brightness_b2v_inj)
		emit_meas.append(emit_b2v_inj)
		time_meas.append(time_b2v_inj)
		slots.append(slots_b2v_inj)
		cycle_meas.append(['Injection']*len(emit_b2v_inj))
		plane_meas.append(['Vertical']*len(emit_b2v_inj))
		beam_meas.append(['B2']*len(emit_b2v_inj))
		fill_meas.append([filln]*len(emit_b2v_inj))
		bunches_meas.append([bunches]*len(emit_b2v_inj))
		kind.append(['Measurement']*len(emit_b2v_inj))

		########
		# this is B2H FB
		emit_b2h_fb  = np.array(cycle['beam_2']['Injection']['at_end']['emith'])
		time_b2h_fb  = np.array(cycle['beam_2']['Injection']['at_end']['time_meas'])
		slots_b2h_fb = np.array(cycle['beam_2']['Injection']['filled_slots'])
		if not doModel:
			intens_b2h_fb		= np.array(cycle['beam_2']['Injection']['at_end']['intensity'])
			blength_b2h_fb		= np.array(cycle['beam_2']['Injection']['at_end']['blength'])
			brightness_b2h_fb	= np.array(cycle['beam_2']['Injection']['at_end']['brightness'])

			intensity_meas.append(intens_b2h_fb)
			bunch_length_meas.append(blength_b2h_fb)
			brightness_meas.append(brightness_b2h_fb)
		emit_meas.append(emit_b2h_fb)
		time_meas.append(time_b2h_fb)
		slots.append(slots_b2h_fb)
		cycle_meas.append(['Start Ramp']*len(emit_b2h_fb))
		plane_meas.append(['Horizontal']*len(emit_b2h_fb))
		beam_meas.append(['B2']*len(emit_b2h_fb))
		fill_meas.append([filln]*len(emit_b2h_fb))
		bunches_meas.append([bunches]*len(emit_b2h_fb))
		kind.append(['Measurement']*len(emit_b2h_fb))


		# this is B2V FB
		emit_b2v_fb  = np.array(cycle['beam_2']['Injection']['at_end']['emitv'])
		time_b2v_fb  = np.array(cycle['beam_2']['Injection']['at_end']['time_meas'])
		slots_b2v_fb = np.array(cycle['beam_2']['Injection']['filled_slots'])
		if not doModel:
			intens_b2v_fb		= np.array(cycle['beam_2']['Injection']['at_end']['intensity'])
			blength_b2v_fb		= np.array(cycle['beam_2']['Injection']['at_end']['blength'])
			brightness_b2v_fb	= np.array(cycle['beam_2']['Injection']['at_end']['brightness'])

			intensity_meas.append(intens_b2v_fb)
			bunch_length_meas.append(blength_b2v_fb)
			brightness_meas.append(brightness_b2v_fb)
		emit_meas.append(emit_b2v_fb)
		time_meas.append(time_b2v_fb)
		slots.append(slots_b2v_fb)
		cycle_meas.append(['Start Ramp']*len(emit_b2v_fb))
		plane_meas.append(['Vertical']*len(emit_b2v_fb))
		beam_meas.append(['B2']*len(emit_b2v_fb))
		fill_meas.append([filln]*len(emit_b2v_fb))
		bunches_meas.append([bunches]*len(emit_b2v_fb))
		kind.append(['Measurement']*len(emit_b2v_fb))


		########
		# this is B2H FT
		emit_b2h_ft  = np.array(cycle['beam_2']['he_before_SB']['at_start']['emith'])
		time_b2h_ft  = np.array(cycle['beam_2']['he_before_SB']['at_start']['time_meas'])
		slots_b2h_ft = np.array(cycle['beam_2']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b2h_ft		= np.array(cycle['beam_2']['he_before_SB']['at_start']['intensity'])
			blength_b2h_ft		= np.array(cycle['beam_2']['he_before_SB']['at_start']['blength'])
			brightness_b2h_ft	= np.array(cycle['beam_2']['he_before_SB']['at_start']['brightness'])

			intensity_meas.append(intens_b2h_ft)
			bunch_length_meas.append(blength_b2h_ft)
			brightness_meas.append(brightness_b2h_ft)
		emit_meas.append(emit_b2h_ft)
		time_meas.append(time_b2h_ft)
		slots.append(slots_b2h_ft)
		cycle_meas.append(['End Ramp']*len(emit_b2h_ft))
		plane_meas.append(['Horizontal']*len(emit_b2h_ft))
		beam_meas.append(['B2']*len(emit_b2h_ft))
		fill_meas.append([filln]*len(emit_b2h_ft))
		bunches_meas.append([bunches]*len(emit_b2h_ft))
		kind.append(['Measurement']*len(emit_b2h_ft))

		# this is B2V FB
		emit_b2v_ft  = np.array(cycle['beam_2']['he_before_SB']['at_start']['emitv'])
		time_b2v_ft  = np.array(cycle['beam_2']['he_before_SB']['at_start']['time_meas'])
		slots_b2v_ft = np.array(cycle['beam_2']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b2v_ft		= np.array(cycle['beam_2']['he_before_SB']['at_start']['intensity'])
			blength_b2v_ft		= np.array(cycle['beam_2']['he_before_SB']['at_start']['blength'])
			brightness_b2v_ft	= np.array(cycle['beam_2']['he_before_SB']['at_start']['brightness'])

			intensity_meas.append(intens_b2v_ft)
			bunch_length_meas.append(blength_b2v_ft)
			brightness_meas.append(brightness_b2v_ft)
		emit_meas.append(emit_b2v_ft)
		time_meas.append(time_b2v_ft)
		slots.append(slots_b2v_ft)
		cycle_meas.append(['End Ramp']*len(emit_b2v_ft))
		plane_meas.append(['Vertical']*len(emit_b2h_ft))
		beam_meas.append(['B2']*len(emit_b2v_ft))
		fill_meas.append([filln]*len(emit_b2v_ft))
		bunches_meas.append([bunches]*len(emit_b2v_ft))
		kind.append(['Measurement']*len(emit_b2v_ft))


		########
		# this is B2H SB
		emit_b2h_sb  = np.array(cycle['beam_2']['he_before_SB']['at_end']['emith'])
		time_b2h_sb  = np.array(cycle['beam_2']['he_before_SB']['at_end']['time_meas'])
		slots_b2h_sb = np.array(cycle['beam_2']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b2h_sb		= np.array(cycle['beam_2']['he_before_SB']['at_end']['intensity'])
			blength_b2h_sb		= np.array(cycle['beam_2']['he_before_SB']['at_end']['blength'])
			brightness_b2h_sb	= np.array(cycle['beam_2']['he_before_SB']['at_end']['brightness'])

			intensity_meas.append(intens_b2h_sb)
			bunch_length_meas.append(blength_b2h_sb)
			brightness_meas.append(brightness_b2h_sb)
		emit_meas.append(emit_b2h_sb)
		time_meas.append(time_b2h_sb)
		slots.append(slots_b2h_sb)
		cycle_meas.append(['Start Stable']*len(emit_b2h_sb))
		plane_meas.append(['Horizontal']*len(emit_b2h_sb))
		beam_meas.append(['B2']*len(emit_b2h_sb))
		fill_meas.append([filln]*len(emit_b2h_sb))
		bunches_meas.append([bunches]*len(emit_b2h_sb))
		kind.append(['Measurement']*len(emit_b2h_sb))

		# this is B2V SB
		emit_b2v_sb  = np.array(cycle['beam_2']['he_before_SB']['at_end']['emitv'])
		time_b2v_sb  = np.array(cycle['beam_2']['he_before_SB']['at_end']['time_meas'])
		slots_b2v_sb = np.array(cycle['beam_2']['he_before_SB']['filled_slots'])
		if not doModel:
			intens_b2v_sb		= np.array(cycle['beam_2']['he_before_SB']['at_end']['intensity'])
			blength_b2v_sb		= np.array(cycle['beam_2']['he_before_SB']['at_end']['blength'])
			brightness_b2v_sb	= np.array(cycle['beam_2']['he_before_SB']['at_end']['brightness'])

			intensity_meas.append(intens_b2v_sb)
			bunch_length_meas.append(blength_b2v_sb)
			brightness_meas.append(brightness_b2v_sb)
		emit_meas.append(emit_b2v_sb)
		time_meas.append(time_b2v_sb)
		slots.append(slots_b2v_sb)
		cycle_meas.append(['Start Stable']*len(emit_b2v_sb))
		plane_meas.append(['Vertical']*len(emit_b2v_sb))
		beam_meas.append(['B2']*len(emit_b2v_sb))
		fill_meas.append([filln]*len(emit_b2v_sb))
		bunches_meas.append([bunches]*len(emit_b2v_sb))
		kind.append(['Measurement']*len(emit_b2v_sb))


		########################## MODEL ################################
		# this is B1H INJ
		if doModel:
			emit_b1h_inj  = np.array(model['beam_1']['Injection']['at_start']['emith'])
			time_b1h_inj  = np.array(model['beam_1']['Injection']['at_start']['time_meas'])
			slots_b1h_inj = np.array(model['beam_1']['Injection']['filled_slots'])
			emit_meas.append(emit_b1h_inj)
			time_meas.append(time_b1h_inj)
			slots.append(slots_b1h_inj)
			cycle_meas.append(['Injection']*len(emit_b1h_inj))
			plane_meas.append(['Horizontal']*len(emit_b1h_inj))
			beam_meas.append(['B1']*len(emit_b1h_inj))
			fill_meas.append([filln]*len(emit_b1h_inj))
			bunches_meas.append([bunches]*len(emit_b1h_inj))
			kind.append(['Model']*len(emit_b1h_inj))

			# this is B1V INJ
			emit_b1v_inj  = np.array(model['beam_1']['Injection']['at_start']['emitv'])
			time_b1v_inj  = np.array(model['beam_1']['Injection']['at_start']['time_meas'])
			slots_b1v_inj = np.array(model['beam_1']['Injection']['filled_slots'])
			emit_meas.append(emit_b1v_inj)
			time_meas.append(time_b1v_inj)
			slots.append(slots_b1v_inj)
			cycle_meas.append(['Injection']*len(emit_b1v_inj))
			plane_meas.append(['Vertical']*len(emit_b1v_inj))
			beam_meas.append(['B1']*len(emit_b1v_inj))
			fill_meas.append([filln]*len(emit_b1v_inj))
			bunches_meas.append([bunches]*len(emit_b1v_inj))
			kind.append(['Model']*len(emit_b1v_inj))

			########
			# this is B1H FB
			emit_b1h_fb  = np.array(model['beam_1']['Injection']['at_end']['emith'])
			time_b1h_fb  = np.array(model['beam_1']['Injection']['at_end']['time_meas'])
			slots_b1h_fb = np.array(model['beam_1']['Injection']['filled_slots'])
			emit_meas.append(emit_b1h_fb)
			time_meas.append(time_b1h_fb)
			slots.append(slots_b1h_fb)
			cycle_meas.append(['Start Ramp']*len(emit_b1h_fb))
			plane_meas.append(['Horizontal']*len(emit_b1h_fb))
			beam_meas.append(['B1']*len(emit_b1h_fb))
			fill_meas.append([filln]*len(emit_b1h_fb))
			bunches_meas.append([bunches]*len(emit_b1h_fb))
			kind.append(['Model']*len(emit_b1h_fb))


			# this is B1V FB
			emit_b1v_fb  = np.array(model['beam_1']['Injection']['at_end']['emitv'])
			time_b1v_fb  = np.array(model['beam_1']['Injection']['at_end']['time_meas'])
			slots_b1v_fb = np.array(model['beam_1']['Injection']['filled_slots'])
			emit_meas.append(emit_b1v_fb)
			time_meas.append(time_b1v_fb)
			slots.append(slots_b1v_fb)
			cycle_meas.append(['Start Ramp']*len(emit_b1v_fb))
			plane_meas.append(['Vertical']*len(emit_b1v_fb))
			beam_meas.append(['B1']*len(emit_b1v_fb))
			fill_meas.append([filln]*len(emit_b1v_fb))
			bunches_meas.append([bunches]*len(emit_b1v_fb))
			kind.append(['Model']*len(emit_b1v_fb))


			########
			# this is B1H FT
			emit_b1h_ft  = np.array(model['beam_1']['he_before_SB']['at_start']['emith'])
			time_b1h_ft  = np.array(model['beam_1']['he_before_SB']['at_start']['time_meas'])
			slots_b1h_ft = np.array(model['beam_1']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b1h_ft)
			time_meas.append(time_b1h_ft)
			slots.append(slots_b1h_ft)
			cycle_meas.append(['End Ramp']*len(emit_b1h_ft))
			plane_meas.append(['Horizontal']*len(emit_b1h_ft))
			beam_meas.append(['B1']*len(emit_b1h_ft))
			fill_meas.append([filln]*len(emit_b1h_ft))
			bunches_meas.append([bunches]*len(emit_b1h_ft))
			kind.append(['Model']*len(emit_b1h_ft))

			# this is B1V FB
			emit_b1v_ft  = np.array(model['beam_1']['he_before_SB']['at_start']['emitv'])
			time_b1v_ft  = np.array(model['beam_1']['he_before_SB']['at_start']['time_meas'])
			slots_b1v_ft = np.array(model['beam_1']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b1v_ft)
			time_meas.append(time_b1v_ft)
			slots.append(slots_b1v_ft)
			cycle_meas.append(['End Ramp']*len(emit_b1v_ft))
			plane_meas.append(['Vertical']*len(emit_b1h_ft))
			beam_meas.append(['B1']*len(emit_b1v_ft))
			fill_meas.append([filln]*len(emit_b1v_ft))
			bunches_meas.append([bunches]*len(emit_b1v_ft))
			kind.append(['Model']*len(emit_b1v_ft))


			########
			# this is B1H SB
			emit_b1h_sb  = np.array(model['beam_1']['he_before_SB']['at_end']['emith'])
			time_b1h_sb  = np.array(model['beam_1']['he_before_SB']['at_end']['time_meas'])
			slots_b1h_sb = np.array(model['beam_1']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b1h_sb)
			time_meas.append(time_b1h_sb)
			slots.append(slots_b1h_sb)
			cycle_meas.append(['Start Stable']*len(emit_b1h_sb))
			plane_meas.append(['Horizontal']*len(emit_b1h_sb))
			beam_meas.append(['B1']*len(emit_b1h_sb))
			fill_meas.append([filln]*len(emit_b1h_sb))
			bunches_meas.append([bunches]*len(emit_b1h_sb))
			kind.append(['Model']*len(emit_b1h_sb))

			# this is B1V SB
			emit_b1v_sb  = np.array(model['beam_1']['he_before_SB']['at_end']['emitv'])
			time_b1v_sb  = np.array(model['beam_1']['he_before_SB']['at_end']['time_meas'])
			slots_b1v_sb = np.array(model['beam_1']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b1v_sb)
			time_meas.append(time_b1v_sb)
			slots.append(slots_b1v_sb)
			cycle_meas.append(['Start Stable']*len(emit_b1v_sb))
			plane_meas.append(['Vertical']*len(emit_b1v_sb))
			beam_meas.append(['B1']*len(emit_b1v_sb))
			fill_meas.append([filln]*len(emit_b1v_sb))
			bunches_meas.append([bunches]*len(emit_b1v_sb))
			kind.append(['Model']*len(emit_b1v_sb))

			##############################
			#	NOW B2
			#
			emit_b2h_inj  = np.array(model['beam_2']['Injection']['at_start']['emith'])
			time_b2h_inj  = np.array(model['beam_2']['Injection']['at_start']['time_meas'])
			slots_b2h_inj = np.array(model['beam_2']['Injection']['filled_slots'])
			emit_meas.append(emit_b2h_inj)
			time_meas.append(time_b2h_inj)
			slots.append(slots_b2h_inj)
			cycle_meas.append(['Injection']*len(emit_b2h_inj))
			plane_meas.append(['Horizontal']*len(emit_b2h_inj))
			beam_meas.append(['B2']*len(emit_b2h_inj))
			fill_meas.append([filln]*len(emit_b2h_inj))
			bunches_meas.append([bunches]*len(emit_b2h_inj))
			kind.append(['Model']*len(emit_b2h_inj))

			# this is B1V INJ
			emit_b2v_inj  = np.array(model['beam_2']['Injection']['at_start']['emitv'])
			time_b2v_inj  = np.array(model['beam_2']['Injection']['at_start']['time_meas'])
			slots_b2v_inj = np.array(model['beam_2']['Injection']['filled_slots'])
			emit_meas.append(emit_b2v_inj)
			time_meas.append(time_b2v_inj)
			slots.append(slots_b2v_inj)
			cycle_meas.append(['Injection']*len(emit_b2v_inj))
			plane_meas.append(['Vertical']*len(emit_b2v_inj))
			beam_meas.append(['B2']*len(emit_b2v_inj))
			fill_meas.append([filln]*len(emit_b2v_inj))
			bunches_meas.append([bunches]*len(emit_b2v_inj))
			kind.append(['Model']*len(emit_b2v_inj))

			########
			# this is B2H FB
			emit_b2h_fb  = np.array(model['beam_2']['Injection']['at_end']['emith'])
			time_b2h_fb  = np.array(model['beam_2']['Injection']['at_end']['time_meas'])
			slots_b2h_fb = np.array(model['beam_2']['Injection']['filled_slots'])
			emit_meas.append(emit_b2h_fb)
			time_meas.append(time_b2h_fb)
			slots.append(slots_b2h_fb)
			cycle_meas.append(['Start Ramp']*len(emit_b2h_fb))
			plane_meas.append(['Horizontal']*len(emit_b2h_fb))
			beam_meas.append(['B2']*len(emit_b2h_fb))
			fill_meas.append([filln]*len(emit_b2h_fb))
			bunches_meas.append([bunches]*len(emit_b2h_fb))
			kind.append(['Model']*len(emit_b2h_fb))


			# this is B2V FB
			emit_b2v_fb  = np.array(model['beam_2']['Injection']['at_end']['emitv'])
			time_b2v_fb  = np.array(model['beam_2']['Injection']['at_end']['time_meas'])
			slots_b2v_fb = np.array(model['beam_2']['Injection']['filled_slots'])
			emit_meas.append(emit_b2v_fb)
			time_meas.append(time_b2v_fb)
			slots.append(slots_b2v_fb)
			cycle_meas.append(['Start Ramp']*len(emit_b2v_fb))
			plane_meas.append(['Vertical']*len(emit_b2v_fb))
			beam_meas.append(['B2']*len(emit_b2v_fb))
			fill_meas.append([filln]*len(emit_b2v_fb))
			bunches_meas.append([bunches]*len(emit_b2v_fb))
			kind.append(['Model']*len(emit_b2v_fb))


			########
			# this is B2H FT
			emit_b2h_ft  = np.array(model['beam_2']['he_before_SB']['at_start']['emith'])
			time_b2h_ft  = np.array(model['beam_2']['he_before_SB']['at_start']['time_meas'])
			slots_b2h_ft = np.array(model['beam_2']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b2h_ft)
			time_meas.append(time_b2h_ft)
			slots.append(slots_b2h_ft)
			cycle_meas.append(['End Ramp']*len(emit_b2h_ft))
			plane_meas.append(['Horizontal']*len(emit_b2h_ft))
			beam_meas.append(['B2']*len(emit_b2h_ft))
			fill_meas.append([filln]*len(emit_b2h_ft))
			bunches_meas.append([bunches]*len(emit_b2h_ft))
			kind.append(['Model']*len(emit_b2h_ft))

			# this is B2V FB
			emit_b2v_ft  = np.array(model['beam_2']['he_before_SB']['at_start']['emitv'])
			time_b2v_ft  = np.array(model['beam_2']['he_before_SB']['at_start']['time_meas'])
			slots_b2v_ft = np.array(model['beam_2']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b2v_ft)
			time_meas.append(time_b2v_ft)
			slots.append(slots_b2v_ft)
			cycle_meas.append(['End Ramp']*len(emit_b2v_ft))
			plane_meas.append(['Vertical']*len(emit_b2h_ft))
			beam_meas.append(['B2']*len(emit_b2v_ft))
			fill_meas.append([filln]*len(emit_b2v_ft))
			bunches_meas.append([bunches]*len(emit_b2v_ft))
			kind.append(['Model']*len(emit_b2v_ft))


			########
			# this is B2H SB
			emit_b2h_sb  = np.array(model['beam_2']['he_before_SB']['at_end']['emith'])
			time_b2h_sb  = np.array(model['beam_2']['he_before_SB']['at_end']['time_meas'])
			slots_b2h_sb = np.array(model['beam_2']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b2h_sb)
			time_meas.append(time_b2h_sb)
			slots.append(slots_b2h_sb)
			cycle_meas.append(['Start Stable']*len(emit_b2h_sb))
			plane_meas.append(['Horizontal']*len(emit_b2h_sb))
			beam_meas.append(['B2']*len(emit_b2h_sb))
			fill_meas.append([filln]*len(emit_b2h_sb))
			bunches_meas.append([bunches]*len(emit_b2h_sb))
			kind.append(['Model']*len(emit_b2h_sb))

			# this is B2V SB
			emit_b2v_sb  = np.array(model['beam_2']['he_before_SB']['at_end']['emitv'])
			time_b2v_sb  = np.array(model['beam_2']['he_before_SB']['at_end']['time_meas'])
			slots_b2v_sb = np.array(model['beam_2']['he_before_SB']['filled_slots'])
			emit_meas.append(emit_b2v_sb)
			time_meas.append(time_b2v_sb)
			slots.append(slots_b2v_sb)
			cycle_meas.append(['Start Stable']*len(emit_b2v_sb))
			plane_meas.append(['Vertical']*len(emit_b2v_sb))
			beam_meas.append(['B2']*len(emit_b2v_sb))
			fill_meas.append([filln]*len(emit_b2v_sb))
			bunches_meas.append([bunches]*len(emit_b2v_sb))
			kind.append(['Model']*len(emit_b2v_sb))



		#####-----------

		emit_meas     		= list(itertools.chain.from_iterable(emit_meas))
		time_meas     		= list(itertools.chain.from_iterable(time_meas))
		cycle_meas    		= list(itertools.chain.from_iterable(cycle_meas))
		plane_meas    		= list(itertools.chain.from_iterable(plane_meas))
		beam_meas     		= list(itertools.chain.from_iterable(beam_meas))
		fill_meas     		= list(itertools.chain.from_iterable(fill_meas))
		bunches_meas  		= list(itertools.chain.from_iterable(bunches_meas))
		kind          		= list(itertools.chain.from_iterable(kind))
		slots         		= list(itertools.chain.from_iterable(slots))
		if not doModel:
			intensity_meas		= list(itertools.chain.from_iterable(intensity_meas))
			bunch_length_meas 	= list(itertools.chain.from_iterable(bunch_length_meas))
			brightness_meas	 	= list(itertools.chain.from_iterable(brightness_meas))

		df_emit = pd.DataFrame()
		df_emit['Emittance']   	   = pd.Series(emit_meas, dtype='float')
		if not doModel:
			df_emit['Intensity']   = pd.Series(intensity_meas, dtype='float')
			df_emit['BunchLength'] = pd.Series(bunch_length_meas, dtype='float')
			df_emit['Brightness']  = pd.Series(brightness_meas, dtype='float')
		df_emit['Time']		 	   = pd.Series(time_meas, dtype='float')
		df_emit['Cycle']     	   = pd.Series(cycle_meas, dtype='category')
		df_emit['Plane']     	   = pd.Series(plane_meas, dtype='category')
		df_emit['Beam']      	   = pd.Series(beam_meas, dtype='category')
		df_emit['Fill']      	   = pd.Series(fill_meas, dtype='int')
		df_emit['Bunches']   	   = pd.Series(bunches_meas, dtype='int')
		df_emit['Kind']      	   = pd.Series(kind, dtype='category')
		df_emit['Slot']      	   = pd.Series(slots, dtype='float')
		
		# df_emit = df_emit.dropna(axis=0, how='any')

		df_loop = df_loop.append(df_emit, ignore_index=True)
		


	df_total = pd.DataFrame()
	if df_exists:
		df_old = pd.read_pickle(outfilename)
		df_total = df_old.append(df_loop, ignore_index=True)
	else:
		df_total = df_loop
	

	df_total.to_pickle(outfilename)
	print '#makeCycleDataFrame : Writing file [{}]'.format(outfilename)
	return df_total

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 

def makeGrowthDataFrame(dataframe, outfile_growth, outfile_single_emit, min_dt=5.0,):
	'''
	dataframe = pandas dataframe or string of pickle df file
	min_dt    = min time at injection:
	'''

	df = pd.DataFrame()
	if isinstance(dataframe, pd.DataFrame):
		df = dataframe
	elif isinstance(dataframe, str):
		df = pd.read_pickle(dataframe)
	else:
		raise IOError('#makeGrowthDataFrame: Input argument [dataframe] of no recognizable type.')


	#
	#	Check if the outfilenames given correspond to any dataframe. If yes update do not overwrite
	#
	growth_df_exists      = False
	single_emit_df_exists = False
	if os.path.exists(outfile_growth):
		print '#makeGrowthDataFrame : [WARN] Filename of output file given for growth dataframe exists. Checking if it is up-to-date...'
		growth_df_exists = True
		df_tmp_growth = pd.read_pickle(outfile_growth)
		fills_existing = np.unique(df_tmp_growth['Fill'].values)
		fills_new = np.unique(df['Fill'].values)
		overlap_fills = list(set(fills_existing.tolist()).intersection(fills_new.tolist()))
		print '#makeGrowthDataFrame : [WARN] The old growth dataframe has already the fills : [{}] dropping them.'.format(overlap_fills)
		for nfill in overlap_fills:
			df = df[df['Fill']!=nfill]

	if os.path.exists(outfile_single_emit):
		print '#makeGrowthDataFrame : [WARN] Filename of output file given for single emittances dataframe exists. Checking if it is up-to-date...'
		single_emit_df_exists = True
		df_tmp_emit = pd.read_pickle(outfile_single_emit)
		fills_existing = np.unique(df_tmp_emit['Fill'].values)
		fills_new = np.unique(df['Fill'].values)
		overlap_fills = list(set(fills_existing.tolist()).intersection(fills_new.tolist()))
		print '#makeGrowthDataFrame : [WARN] The old single emittances dataframe has already the fills : [{}] dropping them.'.format(overlap_fills)
		for nfill in overlap_fills:
			df = df[df['Fill']!=nfill]

	if len(df) == 0:
		print '#makeGrowthDataFrame : [WARN] No new fills to update DF. Exiting...'
		return None, None



	#
	#	Split Dataframe given based on beams, cycle and kind
	#
	df_b1_inj_meas = df[(df['Beam']=='B1') & (df['Cycle']=='Injection')    & (df['Kind']=='Measurement')]
	df_b1_fb_meas  = df[(df['Beam']=='B1') & (df['Cycle']=='Start Ramp')   & (df['Kind']=='Measurement')]
	df_b1_ft_meas  = df[(df['Beam']=='B1') & (df['Cycle']=='End Ramp')     & (df['Kind']=='Measurement')]
	df_b1_sb_meas  = df[(df['Beam']=='B1') & (df['Cycle']=='Start Stable') & (df['Kind']=='Measurement')]

	df_b2_inj_meas = df[(df['Beam']=='B2') & (df['Cycle']=='Injection')    & (df['Kind']=='Measurement')]
	df_b2_fb_meas  = df[(df['Beam']=='B2') & (df['Cycle']=='Start Ramp')   & (df['Kind']=='Measurement')]
	df_b2_ft_meas  = df[(df['Beam']=='B2') & (df['Cycle']=='End Ramp')     & (df['Kind']=='Measurement')]
	df_b2_sb_meas  = df[(df['Beam']=='B2') & (df['Cycle']=='Start Stable') & (df['Kind']=='Measurement')]

	df_b1_inj_meas2 = df_b1_inj_meas.copy()
	df_b2_inj_meas2 = df_b2_inj_meas.copy()

	doModel = False
	if 'Model' in np.unique(df['Kind'].values):
		df_b1_inj_model = df[(df['Beam']=='B1') & (df['Cycle']=='Injection')    & (df['Kind']=='Model')]
		df_b1_fb_model  = df[(df['Beam']=='B1') & (df['Cycle']=='Start Ramp')   & (df['Kind']=='Model')]
		df_b1_ft_model  = df[(df['Beam']=='B1') & (df['Cycle']=='End Ramp')     & (df['Kind']=='Model')]
		df_b1_sb_model  = df[(df['Beam']=='B1') & (df['Cycle']=='Start Stable') & (df['Kind']=='Model')]

		df_b2_inj_model = df[(df['Beam']=='B2') & (df['Cycle']=='Injection')    & (df['Kind']=='Model')]
		df_b2_fb_model  = df[(df['Beam']=='B2') & (df['Cycle']=='Start Ramp')   & (df['Kind']=='Model')]
		df_b2_ft_model  = df[(df['Beam']=='B2') & (df['Cycle']=='End Ramp')     & (df['Kind']=='Model')]
		df_b2_sb_model  = df[(df['Beam']=='B2') & (df['Cycle']=='Start Stable') & (df['Kind']=='Model')]
		doModel = True


	df_se_tot = pd.DataFrame()

	###################################################
	#
	# DataFrame Injection to Flat Bottom
	#
	###################################################
	# B1 
	df_b1_inj_meas['Emittance_Injection_meas']                  	= df_b1_inj_meas['Emittance'].values
	df_b1_inj_meas['Emittance_StartRamp_meas']                  	= df_b1_fb_meas['Emittance'].values
	# df_b1_inj_meas['Slot']                                    	= df_b1_inj_meas['Slot'].values
	df_b1_inj_meas['dt']                                        	= df_b1_fb_meas['Time'].values - df_b1_inj_meas['Time'].values
	t1 = df_b1_inj_meas['dt'].values
	df_b1_inj_meas['Growth_raw_meas']                               = 3600.*(df_b1_inj_meas['Emittance_StartRamp_meas'].values - df_b1_inj_meas['Emittance_Injection_meas'].values)/df_b1_inj_meas['dt'].values 
	df_b1_inj_meas['DeltaEmittance_raw']                         	= df_b1_inj_meas['Emittance_StartRamp_meas'].values - df_b1_inj_meas['Emittance_Injection_meas'].values
	df_b1_inj_meas['DeltaEmittance_raw_relative']                  	= ((df_b1_inj_meas['Emittance_StartRamp_meas'].values - df_b1_inj_meas['Emittance_Injection_meas'].values)/df_b1_inj_meas['Emittance_Injection_meas'].values)*100.
	if doModel:
		df_b1_inj_meas['Emittance_Injection_model']                 = df_b1_inj_model['Emittance'].values
		df_b1_inj_meas['Emittance_StartRamp_model']                 = df_b1_fb_model['Emittance'].values
		df_b1_inj_meas['Emittance_Injection_modelCorrected']        = df_b1_inj_meas['Emittance_Injection_meas'].values - df_b1_inj_meas['Emittance_Injection_model'].values
		df_b1_inj_meas['Emittance_StartRamp_modelCorrected']        = df_b1_inj_meas['Emittance_StartRamp_meas'].values - df_b1_inj_meas['Emittance_StartRamp_model'].values
		df_b1_inj_meas['Growth_raw_model']                          = 3600.*(df_b1_inj_meas['Emittance_StartRamp_model'].values - df_b1_inj_meas['Emittance_Injection_model'].values)/df_b1_inj_meas['dt'].values 
		df_b1_inj_meas['Growth_modelCorrected']                     = 3600.*(df_b1_inj_meas['Emittance_StartRamp_modelCorrected'].values - df_b1_inj_meas['Emittance_Injection_modelCorrected'].values)/df_b1_inj_meas['dt'].values 
		df_b1_inj_meas['DeltaEmittance_modelCorrected']             = df_b1_inj_meas['Emittance_StartRamp_modelCorrected'].values - df_b1_inj_meas['Emittance_Injection_modelCorrected'].values
		df_b1_inj_meas['DeltaEmittance_modelCorrected_relative']    = ((df_b1_inj_meas['Emittance_StartRamp_modelCorrected'].values - df_b1_inj_meas['Emittance_Injection_meas'].values)/df_b1_inj_meas['Emittance_Injection_meas'].values)*100.

	df_b1_inj_meas['Cycle'] 										= ['INJ2FB']*len(df_b1_inj_meas)
	df_b1_inj_meas                       							= df_b1_inj_meas[df_b1_inj_meas['dt']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b1_inj_meas.replace(np.inf, np.nan)
	df_b1_inj_meas.replace(-np.inf, np.nan)
	df_b1_inj_meas.drop(df_b1_inj_meas['Growth_modelCorrected'][~np.isfinite(df_b1_inj_meas['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b1_inj_meas = df_b1_inj_meas.dropna(axis=0, how='any')

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b1_inj_meas['Emittance_Injection_meas'].values
	df_tmp['Cycle']              = ['Injection']*len(df_tmp)
	df_tmp['Slot']               = df_b1_inj_meas['Slot'].values
	df_tmp['Fill']               = df_b1_inj_meas['Fill'].values
	df_tmp['Bunches']            = df_b1_inj_meas['Bunches'].values
	df_tmp['Beam']               = df_b1_inj_meas['Beam'].values
	df_tmp['Plane']              = df_b1_inj_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b1_inj_meas['Emittance_StartRamp_meas'].values
	df_tmp['Cycle']              = ['StartRamp']*len(df_tmp)
	df_tmp['Slot']               = df_b1_inj_meas['Slot'].values
	df_tmp['Fill']               = df_b1_inj_meas['Fill'].values
	df_tmp['Bunches']            = df_b1_inj_meas['Bunches'].values
	df_tmp['Beam']               = df_b1_inj_meas['Beam'].values
	df_tmp['Plane']              = df_b1_inj_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

	if doModel:
		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model'] = df_b1_inj_meas['Emittance_Injection_model'].values
		df_tmp['Cycle']              = ['Injection']*len(df_tmp)
		df_tmp['Slot']               = df_b1_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b1_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b1_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b1_inj_meas['Emittance_Injection_modelCorrected'].values
		df_tmp['Cycle']              = ['Injection']*len(df_tmp)
		df_tmp['Slot']               = df_b1_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b1_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b1_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model'] = df_b1_inj_meas['Emittance_StartRamp_model'].values
		df_tmp['Cycle']              = ['StartRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b1_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b1_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b1_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b1_inj_meas['Emittance_StartRamp_modelCorrected'].values
		df_tmp['Cycle']              = ['StartRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b1_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b1_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b1_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)


	# B2
	df_b2_inj_meas['Emittance_Injection_meas']                  	= df_b2_inj_meas['Emittance'].values
	df_b2_inj_meas['Emittance_StartRamp_meas']                  	= df_b2_fb_meas['Emittance'].values
	# df_b2_inj_meas['Slot']                                   	= df_b2_inj_meas['Slot'].values
	df_b2_inj_meas['dt']                                        	= df_b2_fb_meas['Time'].values - df_b2_inj_meas['Time'].values
	t2 = df_b2_inj_meas['dt'].values
	df_b2_inj_meas['Growth_raw_meas']                                	= 3600.*(df_b2_inj_meas['Emittance_StartRamp_meas'].values - df_b2_inj_meas['Emittance_Injection_meas'].values)/df_b2_inj_meas['dt'].values 
	df_b2_inj_meas['DeltaEmittance_raw']                        	= df_b2_inj_meas['Emittance_StartRamp_meas'].values - df_b2_inj_meas['Emittance_Injection_meas'].values
	df_b2_inj_meas['DeltaEmittance_raw_relative']                 	= ((df_b2_inj_meas['Emittance_StartRamp_meas'].values - df_b2_inj_meas['Emittance_Injection_meas'].values)/df_b2_inj_meas['Emittance_Injection_meas'].values)*100.
	if doModel:
		df_b2_inj_meas['Emittance_Injection_model']                 = df_b2_inj_model['Emittance'].values
		df_b2_inj_meas['Emittance_StartRamp_model']                 = df_b2_fb_model['Emittance'].values
		df_b2_inj_meas['Emittance_Injection_modelCorrected']        = df_b2_inj_meas['Emittance_Injection_meas'].values - df_b2_inj_meas['Emittance_Injection_model'].values
		df_b2_inj_meas['Emittance_StartRamp_modelCorrected']        = df_b2_inj_meas['Emittance_StartRamp_meas'].values - df_b2_inj_meas['Emittance_StartRamp_model'].values
		df_b2_inj_meas['Growth_raw_model']                          = 3600.*(df_b2_inj_meas['Emittance_StartRamp_model'].values - df_b2_inj_meas['Emittance_Injection_model'].values)/df_b2_inj_meas['dt'].values
		df_b2_inj_meas['Growth_modelCorrected']                     = 3600.*(df_b2_inj_meas['Emittance_StartRamp_modelCorrected'].values - df_b2_inj_meas['Emittance_Injection_modelCorrected'].values)/df_b2_inj_meas['dt'].values 
		df_b2_inj_meas['DeltaEmittance_modelCorrected']             = df_b2_inj_meas['Emittance_StartRamp_modelCorrected'].values - df_b2_inj_meas['Emittance_Injection_modelCorrected'].values
		df_b2_inj_meas['DeltaEmittance_modelCorrected_relative']    = ((df_b2_inj_meas['Emittance_StartRamp_modelCorrected'].values - df_b2_inj_meas['Emittance_Injection_meas'].values)/df_b2_inj_meas['Emittance_Injection_meas'].values)*100.

	df_b2_inj_meas['Cycle'] 										= ['INJ2FB']*len(df_b2_inj_meas)
	df_b2_inj_meas                       							= df_b2_inj_meas[df_b2_inj_meas['dt']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b2_inj_meas.replace(np.inf, np.nan)
	df_b2_inj_meas.replace(-np.inf, np.nan)
	df_b2_inj_meas.drop(df_b2_inj_meas['Growth_modelCorrected'][~np.isfinite(df_b2_inj_meas['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b2_inj_meas = df_b2_inj_meas.dropna(axis=0, how='any')

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b2_inj_meas['Emittance_Injection_meas'].values
	df_tmp['Cycle']              = ['Injection']*len(df_tmp)
	df_tmp['Slot']               = df_b2_inj_meas['Slot'].values
	df_tmp['Fill']               = df_b2_inj_meas['Fill'].values
	df_tmp['Bunches']            = df_b2_inj_meas['Bunches'].values
	df_tmp['Beam']               = df_b2_inj_meas['Beam'].values
	df_tmp['Plane']              = df_b2_inj_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b2_inj_meas['Emittance_StartRamp_meas'].values
	df_tmp['Cycle']              = ['StartRamp']*len(df_tmp)
	df_tmp['Slot']               = df_b2_inj_meas['Slot'].values
	df_tmp['Fill']               = df_b2_inj_meas['Fill'].values
	df_tmp['Bunches']            = df_b2_inj_meas['Bunches'].values
	df_tmp['Beam']               = df_b2_inj_meas['Beam'].values
	df_tmp['Plane']              = df_b2_inj_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

	if doModel:
		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model'] = df_b2_inj_meas['Emittance_Injection_model'].values
		df_tmp['Cycle']              = ['Injection']*len(df_tmp)
		df_tmp['Slot']               = df_b2_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b2_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b2_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b2_inj_meas['Emittance_Injection_modelCorrected'].values
		df_tmp['Cycle']              = ['Injection']*len(df_tmp)
		df_tmp['Slot']               = df_b2_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b2_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b2_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model'] = df_b2_inj_meas['Emittance_StartRamp_model'].values
		df_tmp['Cycle']              = ['StartRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b2_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b2_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b2_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b2_inj_meas['Emittance_StartRamp_modelCorrected'].values
		df_tmp['Cycle']              = ['StartRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b2_inj_meas['Slot'].values
		df_tmp['Fill']               = df_b2_inj_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_inj_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_inj_meas['Beam'].values
		df_tmp['Plane']              = df_b2_inj_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

	# Throwing out single cycle emittance measurement and model
	for col in ['Emittance', 'Emittance_Injection_meas', 'Emittance_StartRamp_meas', 'Emittance_Injection_model', 'Emittance_StartRamp_model', 'Emittance_Injection_modelCorrected', 'Emittance_StartRamp_modelCorrected', 'Time', 'Kind']:
		del df_b2_inj_meas[col]

	###################################################
	#
	# DataFrame Flat Bottom to Flat Top
	#
	###################################################
	# B1 
	df_b1_fb_meas['Emittance_StartRamp_meas']                  		= df_b1_fb_meas['Emittance'].values
	df_b1_fb_meas['Emittance_EndRamp_meas']                  		= df_b1_ft_meas['Emittance'].values
	df_b1_fb_meas['dt']                                        		= df_b1_ft_meas['Time'].values - df_b1_fb_meas['Time'].values
	df_b1_fb_meas['Growth_raw_meas']                                = 3600.*(df_b1_fb_meas['Emittance_EndRamp_meas'].values - df_b1_fb_meas['Emittance_StartRamp_meas'].values)/df_b1_fb_meas['dt'].values 
	df_b1_fb_meas['DeltaEmittance_raw']                        		= df_b1_fb_meas['Emittance_EndRamp_meas'].values - df_b1_fb_meas['Emittance_StartRamp_meas'].values
	df_b1_fb_meas['DeltaEmittance_raw_relative']                    = ((df_b1_fb_meas['Emittance_EndRamp_meas'].values - df_b1_fb_meas['Emittance_StartRamp_meas'].values)/df_b1_fb_meas['Emittance_StartRamp_meas'].values)*100.
	if doModel:
		df_b1_fb_meas['Emittance_StartRamp_model']                  = df_b1_fb_model['Emittance'].values
		df_b1_fb_meas['Emittance_EndRamp_model']                    = df_b1_ft_model['Emittance'].values
		df_b1_fb_meas['Emittance_StartRamp_modelCorrected']         = df_b1_fb_meas['Emittance_StartRamp_meas'].values - df_b1_fb_meas['Emittance_StartRamp_model'].values
		df_b1_fb_meas['Emittance_EndRamp_modelCorrected']           = df_b1_fb_meas['Emittance_EndRamp_meas'].values - df_b1_fb_meas['Emittance_EndRamp_model'].values
		df_b1_fb_meas['Growth_raw_model']                           = 3600.*(df_b1_fb_meas['Emittance_EndRamp_model'].values - df_b1_fb_meas['Emittance_StartRamp_model'].values)/df_b1_fb_meas['dt'].values
		df_b1_fb_meas['Growth_modelCorrected']                      = 3600.*(df_b1_fb_meas['Emittance_EndRamp_modelCorrected'].values - df_b1_fb_meas['Emittance_StartRamp_modelCorrected'].values)/df_b1_fb_meas['dt'].values 
		df_b1_fb_meas['DeltaEmittance_modelCorrected']              = df_b1_fb_meas['Emittance_EndRamp_modelCorrected'].values - df_b1_fb_meas['Emittance_StartRamp_modelCorrected'].values
		df_b1_fb_meas['DeltaEmittance_modelCorrected_relative']     = ((df_b1_fb_meas['Emittance_EndRamp_modelCorrected'].values - df_b1_fb_meas['Emittance_StartRamp_modelCorrected'].values)/df_b1_fb_meas['Emittance_StartRamp_modelCorrected'].values)*100.
 
	df_b1_fb_meas['Cycle'] 										    = ['FB2FT']*len(df_b1_fb_meas)
	#df_b1_fb_meas                       						    = df_b1_fb_meas[df_b1_fb_meas['dt']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b1_fb_meas.replace(np.inf, np.nan)
	df_b1_fb_meas.replace(-np.inf, np.nan)
	df_b1_fb_meas.drop(df_b1_fb_meas['Growth_modelCorrected'][~np.isfinite(df_b1_fb_meas['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b1_fb_meas = df_b1_fb_meas.dropna(axis=0, how='any')

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b1_fb_meas['Emittance_EndRamp_meas'].values
	df_tmp['Cycle']              = ['EndRamp']*len(df_tmp)
	df_tmp['Slot']               = df_b1_fb_meas['Slot'].values
	df_tmp['Fill']               = df_b1_fb_meas['Fill'].values
	df_tmp['Bunches']            = df_b1_fb_meas['Bunches'].values
	df_tmp['Beam']               = df_b1_fb_meas['Beam'].values
	df_tmp['Plane']              = df_b1_fb_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)


	if doModel:
		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model'] = df_b1_fb_meas['Emittance_EndRamp_model'].values
		df_tmp['Cycle']              = ['EndRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b1_fb_meas['Slot'].values
		df_tmp['Fill']               = df_b1_fb_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_fb_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_fb_meas['Beam'].values
		df_tmp['Plane']              = df_b1_fb_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b1_fb_meas['Emittance_EndRamp_modelCorrected'].values
		df_tmp['Cycle']              = ['EndRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b1_fb_meas['Slot'].values
		df_tmp['Fill']               = df_b1_fb_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_fb_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_fb_meas['Beam'].values
		df_tmp['Plane']              = df_b1_fb_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

	# Throwing out single cycle emittance measurement and model
	for col in ['Emittance', 'Emittance_EndRamp_meas', 'Emittance_EndRamp_model', 'Emittance_EndRamp_modelCorrected','Emittance_StartRamp_meas', 'Emittance_StartRamp_model', 'Emittance_StartRamp_modelCorrected', 'Time', 'Kind']:
		del df_b1_fb_meas[col]


	# B2 
	df_b2_fb_meas['Emittance_StartRamp_meas']                  		= df_b2_fb_meas['Emittance'].values
	df_b2_fb_meas['Emittance_EndRamp_meas']                  		= df_b2_ft_meas['Emittance'].values
	df_b2_fb_meas['dt']                                        		= df_b2_ft_meas['Time'].values - df_b2_fb_meas['Time'].values
	df_b2_fb_meas['Growth_raw_meas']                                = 3600.*(df_b2_fb_meas['Emittance_EndRamp_meas'].values - df_b2_fb_meas['Emittance_StartRamp_meas'].values)/df_b2_fb_meas['dt'].values 
	df_b2_fb_meas['DeltaEmittance_raw']                        		= df_b2_fb_meas['Emittance_EndRamp_meas'].values - df_b2_fb_meas['Emittance_StartRamp_meas'].values
	df_b2_fb_meas['DeltaEmittance_raw_relative']                  	= ((df_b2_fb_meas['Emittance_EndRamp_meas'].values - df_b2_fb_meas['Emittance_StartRamp_meas'].values)/df_b2_fb_meas['Emittance_StartRamp_meas'].values)*100.
	if doModel:
		df_b2_fb_meas['Emittance_StartRamp_model']                  = df_b2_fb_model['Emittance'].values
		df_b2_fb_meas['Emittance_EndRamp_model']                    = df_b2_ft_model['Emittance'].values
		df_b2_fb_meas['Emittance_StartRamp_modelCorrected']         = df_b2_fb_meas['Emittance_StartRamp_meas'].values - df_b2_fb_meas['Emittance_StartRamp_model'].values
		df_b2_fb_meas['Emittance_EndRamp_modelCorrected']           = df_b2_fb_meas['Emittance_EndRamp_meas'].values - df_b2_fb_meas['Emittance_EndRamp_model'].values
		df_b2_fb_meas['Growth_raw_model']                           = 3600.*(df_b2_fb_meas['Emittance_EndRamp_model'].values - df_b2_fb_meas['Emittance_StartRamp_model'].values)/df_b2_fb_meas['dt'].values
		df_b2_fb_meas['Growth_modelCorrected']                      = 3600.*(df_b2_fb_meas['Emittance_EndRamp_modelCorrected'].values - df_b2_fb_meas['Emittance_StartRamp_modelCorrected'].values)/df_b2_fb_meas['dt'].values 
		df_b2_fb_meas['DeltaEmittance_modelCorrected']              = df_b2_fb_meas['Emittance_EndRamp_modelCorrected'].values - df_b2_fb_meas['Emittance_StartRamp_modelCorrected'].values
		df_b2_fb_meas['DeltaEmittance_modelCorrected_relative']     = ((df_b2_fb_meas['Emittance_EndRamp_modelCorrected'].values - df_b2_fb_meas['Emittance_StartRamp_modelCorrected'].values)/df_b2_fb_meas['Emittance_StartRamp_modelCorrected'].values)*100.

	df_b2_fb_meas['Cycle'] 										    = ['FB2FT']*len(df_b2_fb_meas)
	#df_b2_fb_meas                       						    = df_b2_fb_meas[df_b2_fb_meas['dt']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b2_fb_meas.replace(np.inf, np.nan)
	df_b2_fb_meas.replace(-np.inf, np.nan)
	df_b2_fb_meas.drop(df_b2_fb_meas['Growth_modelCorrected'][~np.isfinite(df_b2_fb_meas['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b2_fb_meas = df_b2_fb_meas.dropna(axis=0, how='any')

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b2_fb_meas['Emittance_EndRamp_meas'].values
	df_tmp['Cycle']              = ['EndRamp']*len(df_tmp)
	df_tmp['Slot']               = df_b2_fb_meas['Slot'].values
	df_tmp['Fill']               = df_b2_fb_meas['Fill'].values
	df_tmp['Bunches']            = df_b2_fb_meas['Bunches'].values
	df_tmp['Beam']               = df_b2_fb_meas['Beam'].values
	df_tmp['Plane']              = df_b2_fb_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)


	if doModel:
		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model'] = df_b2_fb_meas['Emittance_EndRamp_model'].values
		df_tmp['Cycle']              = ['EndRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b2_fb_meas['Slot'].values
		df_tmp['Fill']               = df_b2_fb_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_fb_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_fb_meas['Beam'].values
		df_tmp['Plane']              = df_b2_fb_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b2_fb_meas['Emittance_EndRamp_modelCorrected'].values
		df_tmp['Cycle']              = ['EndRamp']*len(df_tmp)
		df_tmp['Slot']               = df_b2_fb_meas['Slot'].values
		df_tmp['Fill']               = df_b2_fb_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_fb_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_fb_meas['Beam'].values
		df_tmp['Plane']              = df_b2_fb_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)


	# Throwing out single cycle emittance measurement and model
	for col in ['Emittance', 'Emittance_EndRamp_meas', 'Emittance_EndRamp_model', 'Emittance_EndRamp_modelCorrected', 'Emittance_StartRamp_meas', 'Emittance_StartRamp_model', 'Emittance_StartRamp_modelCorrected', 'Time', 'Kind']:
		del df_b2_fb_meas[col]


	###################################################
	#
	# DataFrame Flat Top to Start Stable
	#
	###################################################
	# B1 
	df_b1_ft_meas['Emittance_EndRamp_meas']                  		= df_b1_ft_meas['Emittance'].values
	df_b1_ft_meas['Emittance_StartStable_meas']                		= df_b1_sb_meas['Emittance'].values
	df_b1_ft_meas['dt']                                        		= df_b1_sb_meas['Time'].values - df_b1_ft_meas['Time'].values
	df_b1_ft_meas['Growth_raw_meas']                              	= 3600.*(df_b1_ft_meas['Emittance_StartStable_meas'].values - df_b1_ft_meas['Emittance_EndRamp_meas'].values)/df_b1_ft_meas['dt'].values 
	df_b1_ft_meas['DeltaEmittance_raw']                        		= df_b1_ft_meas['Emittance_StartStable_meas'].values - df_b1_ft_meas['Emittance_EndRamp_meas'].values
	df_b1_ft_meas['DeltaEmittance_raw_relative']              		= ((df_b1_ft_meas['Emittance_StartStable_meas'].values - df_b1_ft_meas['Emittance_EndRamp_meas'].values)/df_b1_ft_meas['Emittance_EndRamp_meas'].values)*100.
	if doModel:
		df_b1_ft_meas['Emittance_EndRamp_model']                    = df_b1_ft_model['Emittance'].values
		df_b1_ft_meas['Emittance_StartStable_model']                = df_b1_sb_model['Emittance'].values
		df_b1_ft_meas['Emittance_EndRamp_modelCorrected']           = df_b1_ft_meas['Emittance_EndRamp_meas'].values - df_b1_ft_meas['Emittance_EndRamp_model'].values
		df_b1_ft_meas['Emittance_StartStable_modelCorrected']       = df_b1_ft_meas['Emittance_StartStable_meas'].values - df_b1_ft_meas['Emittance_StartStable_model'].values
		df_b1_ft_meas['Growth_raw_model']                           = 3600.*(df_b1_ft_meas['Emittance_StartStable_model'].values - df_b1_ft_meas['Emittance_EndRamp_model'].values)/df_b1_ft_meas['dt'].values
		df_b1_ft_meas['Growth_modelCorrected']                      = 3600.*(df_b1_ft_meas['Emittance_StartStable_modelCorrected'].values - df_b1_ft_meas['Emittance_EndRamp_modelCorrected'].values)/df_b1_ft_meas['dt'].values 
		df_b1_ft_meas['DeltaEmittance_modelCorrected']              = df_b1_ft_meas['Emittance_StartStable_modelCorrected'].values - df_b1_ft_meas['Emittance_EndRamp_modelCorrected'].values
		df_b1_ft_meas['DeltaEmittance_modelCorrected_relative']     = ((df_b1_ft_meas['Emittance_StartStable_modelCorrected'].values - df_b1_ft_meas['Emittance_EndRamp_modelCorrected'].values)/df_b1_ft_meas['Emittance_EndRamp_modelCorrected'].values)*100.

	df_b1_ft_meas['Cycle'] 										    = ['FT2SB']*len(df_b1_ft_meas)
	#df_b1_ft_meas                       						    = df_b1_ft_meas[df_b1_ft_meas['dt']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b1_ft_meas.replace(np.inf, np.nan)
	df_b1_ft_meas.replace(-np.inf, np.nan)
	df_b1_ft_meas.drop(df_b1_ft_meas['Growth_modelCorrected'][~np.isfinite(df_b1_ft_meas['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b1_ft_meas = df_b1_ft_meas.dropna(axis=0, how='any')

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b1_ft_meas['Emittance_StartStable_meas'].values
	df_tmp['Cycle']              = ['StartStable']*len(df_tmp)
	df_tmp['Slot']               = df_b1_ft_meas['Slot'].values
	df_tmp['Fill']               = df_b1_ft_meas['Fill'].values
	df_tmp['Bunches']            = df_b1_ft_meas['Bunches'].values
	df_tmp['Beam']               = df_b1_ft_meas['Beam'].values
	df_tmp['Plane']              = df_b1_ft_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)


	if doModel:
		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model'] = df_b1_ft_meas['Emittance_StartStable_model'].values
		df_tmp['Cycle']              = ['StartStable']*len(df_tmp)
		df_tmp['Slot']               = df_b1_ft_meas['Slot'].values
		df_tmp['Fill']               = df_b1_ft_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_ft_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_ft_meas['Beam'].values
		df_tmp['Plane']              = df_b1_ft_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b1_ft_meas['Emittance_StartStable_modelCorrected'].values
		df_tmp['Cycle']              = ['StartStable']*len(df_tmp)
		df_tmp['Slot']               = df_b1_ft_meas['Slot'].values
		df_tmp['Fill']               = df_b1_ft_meas['Fill'].values
		df_tmp['Bunches']            = df_b1_ft_meas['Bunches'].values
		df_tmp['Beam']               = df_b1_ft_meas['Beam'].values
		df_tmp['Plane']              = df_b1_ft_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)


	# Throwing out single cycle emittance measurement and model
	for col in ['Emittance', 'Emittance_StartStable_meas', 'Emittance_EndRamp_meas', 'Emittance_StartStable_model', 'Emittance_EndRamp_model', 'Emittance_StartStable_modelCorrected', 'Emittance_EndRamp_modelCorrected', 'Time', 'Kind']:
		del df_b1_ft_meas[col]


	# B2
	df_b2_ft_meas['Emittance_EndRamp_meas']                  		= df_b2_ft_meas['Emittance'].values
	df_b2_ft_meas['Emittance_StartStable_meas']                		= df_b2_sb_meas['Emittance'].values
	df_b2_ft_meas['dt']                                        		= df_b2_sb_meas['Time'].values - df_b2_ft_meas['Time'].values
	df_b2_ft_meas['Growth_raw_meas']                                = 3600.*(df_b2_ft_meas['Emittance_StartStable_meas'].values - df_b2_ft_meas['Emittance_EndRamp_meas'].values)/df_b2_ft_meas['dt'].values 
	df_b2_ft_meas['DeltaEmittance_raw']                        		= df_b2_ft_meas['Emittance_StartStable_meas'].values - df_b2_ft_meas['Emittance_EndRamp_meas'].values
	df_b2_ft_meas['DeltaEmittance_raw_relative']               		= ((df_b2_ft_meas['Emittance_StartStable_meas'].values - df_b2_ft_meas['Emittance_EndRamp_meas'].values)/df_b2_ft_meas['Emittance_EndRamp_meas'].values)*100.
	if doModel:
		df_b2_ft_meas['Emittance_EndRamp_model']                    = df_b2_ft_model['Emittance'].values
		df_b2_ft_meas['Emittance_StartStable_model']                = df_b2_sb_model['Emittance'].values
		df_b2_ft_meas['Emittance_EndRamp_modelCorrected']           = df_b2_ft_meas['Emittance_EndRamp_meas'].values - df_b2_ft_meas['Emittance_EndRamp_model'].values
		df_b2_ft_meas['Emittance_StartStable_modelCorrected']       = df_b2_ft_meas['Emittance_StartStable_meas'].values - df_b2_ft_meas['Emittance_StartStable_model'].values
		df_b2_ft_meas['Growth_raw_model']                           = 3600.*(df_b2_ft_meas['Emittance_StartStable_model'].values - df_b2_ft_meas['Emittance_EndRamp_model'].values)/df_b2_ft_meas['dt'].values
		df_b2_ft_meas['Growth_modelCorrected']                      = 3600.*(df_b2_ft_meas['Emittance_StartStable_modelCorrected'].values - df_b2_ft_meas['Emittance_EndRamp_modelCorrected'].values)/df_b2_ft_meas['dt'].values 
		df_b2_ft_meas['DeltaEmittance_modelCorrected']              = df_b2_ft_meas['Emittance_StartStable_modelCorrected'].values - df_b2_ft_meas['Emittance_EndRamp_modelCorrected'].values
		df_b2_ft_meas['DeltaEmittance_modelCorrected_relative']     = ((df_b2_ft_meas['Emittance_StartStable_modelCorrected'].values - df_b2_ft_meas['Emittance_EndRamp_modelCorrected'].values)/df_b2_ft_meas['Emittance_EndRamp_modelCorrected'].values)*100.

	df_b2_ft_meas['Cycle'] 										    = ['FT2SB']*len(df_b2_ft_meas)
	#df_b2_ft_meas                       						    = df_b2_ft_meas[df_b2_ft_meas['dt']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b2_ft_meas.replace(np.inf, np.nan)
	df_b2_ft_meas.replace(-np.inf, np.nan)
	df_b2_ft_meas.drop(df_b2_ft_meas['Growth_modelCorrected'][~np.isfinite(df_b2_ft_meas['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b2_ft_meas = df_b2_ft_meas.dropna(axis=0, how='any')

	df_tmp = pd.DataFrame()
	df_tmp['Emittance_measured'] = df_b2_ft_meas['Emittance_StartStable_meas'].values
	df_tmp['Cycle']              = ['StartStable']*len(df_tmp)
	df_tmp['Slot']               = df_b2_ft_meas['Slot'].values
	df_tmp['Fill']               = df_b2_ft_meas['Fill'].values
	df_tmp['Bunches']            = df_b2_ft_meas['Bunches'].values
	df_tmp['Beam']               = df_b2_ft_meas['Beam'].values
	df_tmp['Plane']              = df_b2_ft_meas['Plane'].values
	df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)


	if doModel:
		df_tmp = pd.DataFrame()
		df_tmp['Emittance_model']    = df_b2_ft_meas['Emittance_StartStable_model'].values
		df_tmp['Cycle']              = ['StartStable']*len(df_tmp)
		df_tmp['Slot']               = df_b2_ft_meas['Slot'].values
		df_tmp['Fill']               = df_b2_ft_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_ft_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_ft_meas['Beam'].values
		df_tmp['Plane']              = df_b2_ft_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

		df_tmp = pd.DataFrame()
		df_tmp['Emittance_modelCorrected'] = df_b2_ft_meas['Emittance_StartStable_modelCorrected'].values
		df_tmp['Cycle']              = ['StartStable']*len(df_tmp)
		df_tmp['Slot']               = df_b2_ft_meas['Slot'].values
		df_tmp['Fill']               = df_b2_ft_meas['Fill'].values
		df_tmp['Bunches']            = df_b2_ft_meas['Bunches'].values
		df_tmp['Beam']               = df_b2_ft_meas['Beam'].values
		df_tmp['Plane']              = df_b2_ft_meas['Plane'].values
		df_se_tot = df_se_tot.append(df_tmp, ignore_index=True)

			
	# Throwing out single cycle emittance measurement and model
	for col in ['Emittance', 'Emittance_StartStable_meas', 'Emittance_StartStable_model', 'Emittance_StartStable_modelCorrected', 'Emittance_EndRamp_meas', 'Emittance_EndRamp_model', 'Emittance_EndRamp_modelCorrected', 'Time', 'Kind']:
		del df_b2_ft_meas[col]

	
	###################################################
	#
	# DataFrame Injection to Start Stable
	#
	###################################################
	# B1 
	df_b1_inj_meas2['Emittance_Injection_meas']                  	    = df_b1_inj_meas2['Emittance'].values
	df_b1_inj_meas2['Emittance_StartStable_meas']                		= df_b1_sb_meas['Emittance'].values
	df_b1_inj_meas2['dt_cut']                                           = t1
	df_b1_inj_meas2['dt']                                        		= df_b1_sb_meas['Time'].values - df_b1_inj_meas2['Time'].values
	df_b1_inj_meas2['Growth_raw_meas']                                	= 3600.*(df_b1_inj_meas2['Emittance_StartStable_meas'].values - df_b1_inj_meas2['Emittance_Injection_meas'].values)/df_b1_inj_meas2['dt'].values 
	df_b1_inj_meas2['DeltaEmittance_raw']                        		= df_b1_inj_meas2['Emittance_StartStable_meas'].values - df_b1_inj_meas2['Emittance_Injection_meas'].values
	df_b1_inj_meas2['DeltaEmittance_raw_relative']                		= ((df_b1_inj_meas2['Emittance_StartStable_meas'].values - df_b1_inj_meas2['Emittance_Injection_meas'].values)/df_b1_inj_meas2['Emittance_Injection_meas'].values)*100.
	if doModel:
		df_b1_inj_meas2['Emittance_Injection_model']                    = df_b1_inj_model['Emittance'].values
		df_b1_inj_meas2['Emittance_StartStable_model']                  = df_b1_sb_model['Emittance'].values
		df_b1_inj_meas2['Emittance_Injection_modelCorrected']           = df_b1_inj_meas2['Emittance_Injection_meas'].values - df_b1_inj_meas2['Emittance_Injection_model'].values
		df_b1_inj_meas2['Emittance_StartStable_modelCorrected']         = df_b1_inj_meas2['Emittance_StartStable_meas'].values - df_b1_inj_meas2['Emittance_StartStable_model'].values
		df_b1_inj_meas2['Growth_raw_model']                             = 3600.*(df_b1_inj_meas2['Emittance_StartStable_model'].values - df_b1_inj_meas2['Emittance_Injection_model'].values)/df_b1_inj_meas2['dt'].values
		df_b1_inj_meas2['Growth_modelCorrected']                        = 3600.*(df_b1_inj_meas2['Emittance_StartStable_modelCorrected'].values - df_b1_inj_meas2['Emittance_Injection_modelCorrected'].values)/df_b1_inj_meas2['dt'].values 
		df_b1_inj_meas2['DeltaEmittance_modelCorrected']                = df_b1_inj_meas2['Emittance_StartStable_modelCorrected'].values - df_b1_inj_meas2['Emittance_Injection_modelCorrected'].values
		df_b1_inj_meas2['DeltaEmittance_modelCorrected_relative']       = ((df_b1_inj_meas2['Emittance_StartStable_modelCorrected'].values - df_b1_inj_meas2['Emittance_Injection_modelCorrected'].values)/df_b1_inj_meas2['Emittance_Injection_modelCorrected'].values)*100.

	df_b1_inj_meas2['Cycle'] 										    = ['INJ2SB']*len(df_b1_inj_meas2)
	df_b1_inj_meas2                       						        = df_b1_inj_meas2[df_b1_inj_meas2['dt_cut']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b1_inj_meas2.replace(np.inf, np.nan)
	df_b1_inj_meas2.replace(-np.inf, np.nan)
	df_b1_inj_meas2.drop(df_b1_inj_meas2['Growth_modelCorrected'][~np.isfinite(df_b1_inj_meas2['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b1_inj_meas2 = df_b1_inj_meas2.dropna(axis=0, how='any')
	# Throwing out single cycle emittance measurement and model
	for col in ['Emittance', 'Emittance_Injection_meas', 'Emittance_StartStable_meas', 'Emittance_Injection_model', 'Emittance_StartStable_model', 'Emittance_Injection_modelCorrected', 'Emittance_StartStable_modelCorrected', 'Time', 'Kind', 'dt_cut']:
		del df_b1_inj_meas2[col]


	# B2
	df_b2_inj_meas2['Emittance_Injection_meas']                  	    = df_b2_inj_meas2['Emittance'].values
	df_b2_inj_meas2['Emittance_StartStable_meas']                		= df_b2_sb_meas['Emittance'].values
	df_b2_inj_meas2['dt_cut']                                           = t2
	df_b2_inj_meas2['dt']                                        		= df_b2_sb_meas['Time'].values - df_b2_inj_meas2['Time'].values
	df_b2_inj_meas2['Growth_raw_meas']                                	= 3600.*(df_b2_inj_meas2['Emittance_StartStable_meas'].values - df_b2_inj_meas2['Emittance_Injection_meas'].values)/df_b2_inj_meas2['dt'].values 
	df_b2_inj_meas2['DeltaEmittance_raw']                        		= df_b2_inj_meas2['Emittance_StartStable_meas'].values - df_b2_inj_meas2['Emittance_Injection_meas'].values
	df_b2_inj_meas2['DeltaEmittance_raw_relative']                		= ((df_b2_inj_meas2['Emittance_StartStable_meas'].values - df_b2_inj_meas2['Emittance_Injection_meas'].values)/df_b2_inj_meas2['Emittance_Injection_meas'].values)*100.
	if doModel:
		df_b2_inj_meas2['Emittance_Injection_model']                    = df_b2_inj_model['Emittance'].values
		df_b2_inj_meas2['Emittance_StartStable_model']                  = df_b2_sb_model['Emittance'].values
		df_b2_inj_meas2['Emittance_Injection_modelCorrected']           = df_b2_inj_meas2['Emittance_Injection_meas'].values - df_b2_inj_meas2['Emittance_Injection_model'].values
		df_b2_inj_meas2['Emittance_StartStable_modelCorrected']         = df_b2_inj_meas2['Emittance_StartStable_meas'].values - df_b2_inj_meas2['Emittance_StartStable_model'].values
		df_b2_inj_meas2['Growth_raw_model']                             = 3600.*(df_b2_inj_meas2['Emittance_StartStable_model'].values - df_b2_inj_meas2['Emittance_Injection_model'].values)/df_b2_inj_meas2['dt'].values
		df_b2_inj_meas2['Growth_modelCorrected']                        = 3600.*(df_b2_inj_meas2['Emittance_StartStable_modelCorrected'].values - df_b2_inj_meas2['Emittance_Injection_modelCorrected'].values)/df_b2_inj_meas2['dt'].values 
		df_b2_inj_meas2['DeltaEmittance_modelCorrected']                = df_b2_inj_meas2['Emittance_StartStable_modelCorrected'].values - df_b2_inj_meas2['Emittance_Injection_modelCorrected'].values
		df_b2_inj_meas2['DeltaEmittance_modelCorrected_relative']       = ((df_b2_inj_meas2['Emittance_StartStable_modelCorrected'].values - df_b2_inj_meas2['Emittance_Injection_meas'].values)/df_b2_inj_meas2['Emittance_Injection_meas'].values)*100.

	df_b2_inj_meas2['Cycle'] 										    = ['INJ2SB']*len(df_b2_inj_meas2)
	df_b2_inj_meas2                       						        = df_b2_inj_meas2[df_b2_inj_meas2['dt_cut']>min_dt*60.]   # clean up for bunches that stayed at least 5 min
	df_b2_inj_meas2.replace(np.inf, np.nan)
	df_b2_inj_meas2.replace(-np.inf, np.nan)
	df_b2_inj_meas2.drop(df_b2_inj_meas2['Growth_modelCorrected'][~np.isfinite(df_b2_inj_meas2['Growth_modelCorrected'])].index.values.tolist(), inplace=True)
	df_b2_inj_meas2 = df_b2_inj_meas2.dropna(axis=0, how='any')

	# Throwing out single cycle emittance measurement and model
	for col in ['Emittance', 'Emittance_Injection_meas', 'Emittance_StartStable_meas', 'Emittance_Injection_model', 'Emittance_StartStable_model', 'Emittance_Injection_modelCorrected', 'Emittance_StartStable_modelCorrected', 'Time', 'Kind', 'dt_cut']:
		del df_b2_inj_meas2[col]

	###########################
	#
	# Combine growth in one dataframe:
	df_growth = df_b1_inj_meas.append(df_b2_inj_meas, ignore_index=True).append(df_b1_fb_meas, ignore_index=True).append(df_b2_fb_meas, ignore_index=True).append(df_b1_ft_meas, ignore_index=True).append(df_b2_ft_meas, ignore_index=True).append(df_b1_inj_meas2, ignore_index=True).append(df_b2_inj_meas2, ignore_index=True)

	# if the growth file exists, open it and update it
	if growth_df_exists:
		df_growth_old = pd.read_pickle(outfile_growth)
		df_growth_total = df_growth_old.append(df_growth, ignore_index=True)
		df_growth = df_growth_total


	# if the growth file exists, open it and update it
	if single_emit_df_exists:
		df_emit_old = pd.read_pickle(outfile_single_emit)
		df_emit_total = df_emit_old.append(df_se_tot, ignore_index=True)
		df_se_tot = df_emit_total

	df_growth.to_pickle(outfile_growth)
	print '#makeGrowthDataFrame : Writing file [{}]'.format(outfile_growth)
	df_se_tot.to_pickle(outfile_single_emit)
	print '#makeGrowthDataFrame : Writing file [{}]'.format(outfile_single_emit)
	return df_growth, df_se_tot 





# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 
# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 
if __name__ == '__main__':
	flist = [5830, 5833, 5837, 5838, 5839, 5842, 5845, 5848, 5849, 5856, 5862, 5864, 5865, 5868, 5873, 5876, 5878, 
		5880, 5883, 5885, 5887, 5919, 5920, 5930, 5934, 5942, 5950, 5952, 5954, 5958, 5960, 5962, 5965, 5966, 5971, 
		5974, 5976, 5979, 5984, 6018, 6019, 6020, 6021, 6024, 6026, 6030, 6031, 6035, 6041, 6044, 6046, 6048, 6050, 
		6052, 6053, 6054, 6055]    ### make sure that the fill is here....


	bunches_dict = { (5830, 5837) : 985, 
						 (5837, 5841) : 1225, 
						 (5842, 5844) : 1561,
						 (5845, 5849) : 1741,
						 (5849, 5850) : 2029,
						 (5856, 5857) : 2173,
						 (5862, 5866) : 2317,
						 (5867, 5880) : 2460,      
						 (5880, 5890) : 2556,
						 (5880, 5890) : 2556,
						 (5919, 5921) : 2,      
						 (5930, 5933) : 51,      
						 (5934, 5935) : 601,      
						 (5942, 5947) : 1357,      
						 (5950, 5951) : 2173,      
						 (5952, 5961) : 2317,       
						 (5962, 5963) : 2317,
						 (5963, 5985) : 2556,
						 (6018, 6020) : 601,
				   		 (6020, 6060) : 2556,       ### make sure that the fill is here....
		}

	makeCycleDataFrame(flist, bunches_dict, 'summaryEmittanceDF.pkl', doModel=True)

	gr, se = makeGrowthDataFrame('summaryEmittanceDF.pkl', 'summaryEmittanceDF_growth.pkl', 'summaryEmittanceDF_single_emit.pkl', min_dt=5.0,)
