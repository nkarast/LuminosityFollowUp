import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gzip; import pickle
import sys, os
import datetime

def main(min_fill, max_fill, sbdir='/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/SB_analysis/', blacklist=None, resc_string="", updateDF=True, outplotname=None):
	'''
	This is the main routine to get the lifetime plot
	Inputs: min_fill :	minimum fill number
			max_fill :	maximum fill number
			sbdir 	 : 	path to SB_analysis folder
			blacklist: 	list of fills (int) to skip
			resc_string	: additional string put in the SB data file name
			updateDF	: if True the DF if it exists is updated and this does not run for all fills
			outplotname	: name of the plot to be made. If none a generic one is generated.
	'''
	
	# looping over fills:
	fills = []
	b1_lifetime = []
	b2_lifetime = []

	blacklist = blacklist

	outfile_name = sbdir+"summaryIntensityLifetime.pkl.gz"

	if os.path.exists(outfile_name):
		with gzip.open(outfile_name, 'rb') as fid:
			indf = pickle.load(fid)

		if updateDF:
			fills = indf['fills'].values.tolist()
			b1_lifetime += indf['Beam 1'].values.tolist()
			b2_lifetime += indf['Beam 2'].values.tolist()
			if blacklist is None:
				blacklist = fills  # skipping all fills already in df
			else:
				blacklist+=fills

	print blacklist, type(blacklist)
	for filln in np.linspace(min_fill, max_fill, num=(max_fill-min_fill+1), endpoint=True).astype(int):

		sb_file = sbdir + "fill_{}/fill_{}_df.pkl.gz".format(filln, filln)

		if not os.path.exists(sb_file):
			continue
		
		if filln in blacklist:
			print 'Skipping ', filln
			continue
		else:
			fills.append(filln)
			print 'Working on fill : {}'.format(filln)



		
		with gzip.open(sb_file, 'rb') as fid:
			df = pickle.load(fid)

		time_range = np.array(df['timestamp'][df['beam']=='beam_1'])
		dt = time_range[1]-time_range[0]
		b1_intensity = df['bunch_intensity_coll'][df['beam']=='beam_1'].values
		b1_tmp = np.array([np.array(x) for x in b1_intensity])
		b1_intensity = b1_tmp


		b2_intensity = df['bunch_intensity_coll'][df['beam']=='beam_2'].values
		b2_tmp = np.array([np.array(x) for x in b2_intensity])
		b2_intensity = b2_tmp

		dNp_bbb                   = -(b1_intensity[:-1,:]) + (b1_intensity[1:,:])
		dNdt_bbb 		          = (np.abs(dNp_bbb)/dt)
		tau_Np_bbb  	          = -1/((dNp_bbb/dt)/b1_intensity[:-1,:])
		b1_lifetime.append(np.mean(tau_Np_bbb/3600, axis=1)[0])

		dNp_bbb                   = -(b2_intensity[:-1,:]) + (b2_intensity[1:,:])
		dNdt_bbb 		          = (np.abs(dNp_bbb)/dt)
		tau_Np_bbb  	          = -1/((dNp_bbb/dt)/b2_intensity[:-1,:])
		b2_lifetime.append(np.mean(tau_Np_bbb/3600, axis=1)[0])


	
	print fills
	print b1_lifetime
	print b2_lifetime


	outdf = pd.DataFrame()
	outdf['fills'] = fills
	outdf['Beam 1'] = b1_lifetime
	outdf['Beam 2'] = b2_lifetime

	with gzip.open(outfile_name, 'wb') as fid:
		pickle.dump(outdf, fid)

	if outplotname is None:
		outplotname = sbdir+"/summaryPlots/summaryLifetime_{}_{}_{}.pdf".format(min_fill, max_fill, datetime.datetime.now().strftime("%Y%m%d"))


	plot(fills, b1_lifetime, b2_lifetime, outplotname)

# -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * 

def plot(fills, b1_lifetime, b2_lifetime, outplotname):
	'''
	Makes the plot of the data
	Inputs : fills  : list or np.array of the x -axis
			 b1_lifetime:	list or np.array of the y-axis for beam 1
			 b2_lifetime:	list or np.array of the y-axis for beam 2
			 outplotname:	name of the output plot. if None one generic is made

	'''
	fig = plt.figure('lifetime_summary', figsize=(15,7))
	ax  = plt.subplot(111)
	ax.plot(fills, b1_lifetime, ls=None, marker='o', color='b', markersize=8, label='Beam 1')
	ax.plot(fills, b2_lifetime, ls=None, marker='d', color='r', markersize=8, label='Beam 2')

	ax.set_xlabel('Fill Number', fontsize=16, fontweight='bold')
	ax.set_ylabel('Lifetime [h]', fontsize=16, fontweight='bold')
	ax.set_title('Summary Intensity Lifetime', fontsize=18, fontweight='bold')
	ax.grid('on')
	ax.legend(loc='best', fontsize=16)

	plt.tight_layout()
	plt.savefig(outplotname, dpi=300)
	plt.show()


# -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * -  - * 

if __name__ == '__main__':
	'''
	This executes the script as a single command. It can take one argument which is the last (max) fill.
	'''

	if sys.argv > 1:
		max_fill = int(sys.argv[1])
	else:
		max_fill = 5750

	blacklist = []
	main(min_fill=5698, max_fill=max_fill, sbdir='/afs/cern.ch/work/l/lumimod/a2017_luminosity_followup/SB_analysis/', blacklist = blacklist, updateDF=True, resc_string = "")
