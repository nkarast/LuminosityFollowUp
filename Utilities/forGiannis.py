import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import seaborn as sns
from makeEmittanceDF import *

####################################################################################
#
#	THE FILL LIST & TOTAL INTENSITY : 
#
#			PLEASE MAKE SURE THAT YOUR FILL IS HERE & I WILL TAKE CARE OF THE REST...
#
flist = [5837, 5838, 5839, 5842, 5845, 5848, 5849, 5856, 5862, 5864, 5865, 5868, 5873, 5876, 5878, 5880, 5883, 5885, 5887, 5965, 5966, 5971, 5974, 5976, 5979, 
		5984, 6020, 6021, 6024, 6026, 6030, 6031, 6035, 6041, 6044, 6046, 6048, 6050, 6052, 6053, 6054, 6055, 6057]


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
				   		 (6020, 6100) : 2556, 
		}

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 

################################################
#
#	THIS IS TO PREPARE THE INPUT DATAFRAMES
#
def makeInput(flist, bunches_dict):
	makeCycleDataFrame(flist, bunches_dict, 'summaryEmittanceDF.pkl', doModel=True)
	gr, se = makeGrowthDataFrame('summaryEmittanceDF.pkl', 'summaryEmittanceDF_growth.pkl', 'summaryEmittanceDF_single_emit.pkl', min_dt=5.0)

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 

################################################
#
#	THIS IS TO PLOT EMITTANCES AT INJECTION
#
def plotEmittancesAtInjection(flist):
	plt.close('all')
	df = pd.read_pickle('summaryEmittanceDF.pkl')
	df2 = df[(df['Cycle']=='Injection') & (df['Fill']!=5919) & (df['Fill']!=5920) & (df['Fill']!=5934) & (df['Fill']!=5930) & (df['Kind']=='Measurement') ]

	fig_emit = plt.figure(figsize=(17,8))
	ax_b1 = plt.subplot(211)
	ax_b2 = plt.subplot(212)
	sns.set_style("whitegrid")

	ax_b1 = sns.violinplot(x="Fill", y="Emittance", hue="Plane", data=df2[df2['Beam']=='B1'],  palette=['#71aaf2', '#ff8a3d'],  split=True, order=flist, ax=ax_b1)


	ax_b2 = sns.violinplot(x="Fill", y="Emittance", hue="Plane", data=df2[df2['Beam']=='B2'],  palette=['#71aaf2', '#ff8a3d'], split=True, order=flist, ax=ax_b2)

	mean_values_b1 = []
	std_values_b1 = []

	for nbunches in flist:
	     for nplane in ['Horizontal', 'Vertical']:
	         mean_values_b1.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B1')].mean())
	         std_values_b1.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B1')].std())


	mean_values_b2 = []
	std_values_b2 = []
	for nbunches in flist:
	     for nplane in ['Horizontal', 'Vertical']:
	         mean_values_b2.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B2')].mean())
	         std_values_b2.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B2')].std())


	mean_labels_b1 = [str(np.round(s, 2)) for s in mean_values_b1]
	mean_labels_b2 = [str(np.round(s, 2)) for s in mean_values_b2]
	pos = range(len(mean_labels_b1))

	ax_b1.set_ylabel('B1 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
	ax_b2.set_ylabel('B2 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
	ax_b1.set_xlabel('')
	ax_b2.set_xlabel('Fill Number', fontsize=16, fontweight='bold')
	ax_b1.set_title('Emittance at the Start of Injection', fontsize=20, fontweight='bold')


	for tick,label in zip(pos,ax_b1.get_xticklabels()):
	    ax_b1.text(pos[tick]-0.3, mean_values_b1[2*tick],   mean_labels_b1[2*tick],    horizontalalignment='center', size='xx-small', color='k')#, weight='bold')
	    ax_b1.text(pos[tick]+0.3, mean_values_b1[2*tick+1], mean_labels_b1[2*tick+1],  horizontalalignment='center', size='xx-small', color='k')#, weight='bold')
	for tick,label in zip(pos,ax_b2.get_xticklabels()):
	    ax_b2.text(pos[tick]-0.3, mean_values_b2[2*tick],   mean_labels_b2[2*tick],    horizontalalignment='center', size='xx-small', color='k')#, weight='bold')
	    ax_b2.text(pos[tick]+0.3, mean_values_b2[2*tick+1], mean_labels_b2[2*tick+1],  horizontalalignment='center', size='xx-small', color='k')#, weight='bold')
	    
	ax_b1.set_ylim(1,3)
	ax_b2.set_ylim(1,3)

	plt.tight_layout()

	fig_emit.savefig('plots_Giannis/emit_atInjection_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d")), dpi=300)

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 

########################################################
#
#	THIS IS TO PLOT EMITTANCES AT START OF STABLE BEAMS
#
def plotEmittancesAtStartStable(flist):
	plt.close('all')
	df = pd.read_pickle('summaryEmittanceDF.pkl')
	df2 = df[(df['Cycle']=='Start Stable') & (df['Fill']!=5919) & (df['Fill']!=5920) & (df['Fill']!=5934) & (df['Fill']!=5930) & (df['Kind']=='Measurement') ]

	fig_emit = plt.figure(figsize=(17,8))
	ax_b1 = plt.subplot(211)
	ax_b2 = plt.subplot(212)
	sns.set_style("whitegrid")

	ax_b1 = sns.violinplot(x="Fill", y="Emittance", hue="Plane", data=df2[df2['Beam']=='B1'],  palette=['#71aaf2', '#ff8a3d'],  split=True, order=flist, ax=ax_b1)


	ax_b2 = sns.violinplot(x="Fill", y="Emittance", hue="Plane", data=df2[df2['Beam']=='B2'],  palette=['#71aaf2', '#ff8a3d'], split=True, order=flist, ax=ax_b2)

	mean_values_b1 = []
	std_values_b1 = []

	for nbunches in flist:
	     for nplane in ['Horizontal', 'Vertical']:
	         mean_values_b1.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B1')].mean())
	         std_values_b1.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B1')].std())


	mean_values_b2 = []
	std_values_b2 = []
	for nbunches in flist:
	     for nplane in ['Horizontal', 'Vertical']:
	         mean_values_b2.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B2')].mean())
	         std_values_b2.append(df2['Emittance'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B2')].std())


	mean_labels_b1 = [str(np.round(s, 2)) for s in mean_values_b1]
	mean_labels_b2 = [str(np.round(s, 2)) for s in mean_values_b2]
	pos = range(len(mean_labels_b1))

	ax_b1.set_ylabel('B1 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
	ax_b2.set_ylabel('B2 $\mathbf{\epsilon_{n}}$ [$\mathbf{\mu}$m]', fontsize=16, fontweight='bold')
	ax_b2.set_xlabel('Fill Number', fontsize=16, fontweight='bold')
	ax_b1.set_xlabel('')
	ax_b1.set_title('Emittance at the Start of Stable Beams', fontsize=20, fontweight='bold')


	for tick,label in zip(pos,ax_b1.get_xticklabels()):
	    ax_b1.text(pos[tick]-0.3, mean_values_b1[2*tick],   mean_labels_b1[2*tick],    horizontalalignment='center', size='xx-small', color='k')
	    ax_b1.text(pos[tick]+0.3, mean_values_b1[2*tick+1], mean_labels_b1[2*tick+1],  horizontalalignment='center', size='xx-small', color='k')
	for tick,label in zip(pos,ax_b2.get_xticklabels()):
	    ax_b2.text(pos[tick]-0.3, mean_values_b2[2*tick],   mean_labels_b2[2*tick],    horizontalalignment='center', size='xx-small', color='k')
	    ax_b2.text(pos[tick]+0.3, mean_values_b2[2*tick+1], mean_labels_b2[2*tick+1],  horizontalalignment='center', size='xx-small', color='k')
	    
	ax_b1.set_ylim(1,4)
	ax_b2.set_ylim(1,4)
	plt.tight_layout()
	fig_emit.savefig('plots_Giannis/emit_atStable_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d")), dpi=300)


# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 

########################################################
#
#	THIS IS TO PLOT RELATIVE BLOW-UP AT FLATBOTTOM
#
def plotBlowUpAtFlatBottom(flist):
	plt.close('all')
	df = pd.read_pickle('summaryEmittanceDF_growth.pkl')
	df2 = df[(df['Cycle']=='INJ2FB') & (df['Fill']!=5919) & (df['Fill']!=5920) & (df['Fill']!=5934) & (df['Fill']!=5930)]
	sns.set_style("whitegrid")
	fig_emit = plt.figure(figsize=(17,8))
	ax_b1 = plt.subplot(211)
	ax_b2 = plt.subplot(212)
	sns.set_style("whitegrid")
	ax_b1 = sns.violinplot(x="Fill", y="DeltaEmittance_raw_relative", hue="Plane", data=df2[df2['Beam']=='B1'],  palette=['#71aaf2', '#ff8a3d'], 
		split=True, order=flist, ax=ax_b1)
	ax_b2 = sns.violinplot(x="Fill", y="DeltaEmittance_raw_relative", hue="Plane", data=df2[df2['Beam']=='B2'],  palette=['#71aaf2', '#ff8a3d'], 
		split=True, order=flist, ax=ax_b2)

	mean_values_b1 = []
	for nbunches in flist:
	   for nplane in ['Horizontal', 'Vertical']:
	       mean_values_b1.append(df2['DeltaEmittance_raw_relative'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B1')].mean())
	mean_values_b2 = []
	for nbunches in flist:
	   for nplane in ['Horizontal', 'Vertical']:
	       mean_values_b2.append(df2['DeltaEmittance_raw_relative'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B2')].mean())

	mean_labels_b1 = [str(np.round(s, 2)) for s in mean_values_b1]
	mean_labels_b2 = [str(np.round(s, 2)) for s in mean_values_b2]
	pos = range(len(mean_labels_b1))
	ax_b1.set_ylabel('B1 $\mathbf{\Delta\epsilon_{n}\slash\epsilon_{n}}$ [%]', fontsize=16, fontweight='bold')
	ax_b2.set_ylabel('B2 $\mathbf{\Delta\epsilon_{n}\slash\epsilon_{n}}$ [%]', fontsize=16, fontweight='bold')
	ax_b2.set_xlabel('Fill Number', fontsize=16, fontweight='bold')
	ax_b1.set_xlabel('')
	ax_b1.set_title('Measured Relative Emittance Blow-up from Injection to Start Ramp', fontsize=20, fontweight='bold')

	for tick,label in zip(pos,ax_b1.get_xticklabels()):
		ax_b1.text(pos[tick]-0.3, mean_values_b1[2*tick],   mean_labels_b1[2*tick],    horizontalalignment='center', size='x-small', color='k')
		ax_b1.text(pos[tick]+0.3, mean_values_b1[2*tick+1], mean_labels_b1[2*tick+1],  horizontalalignment='center', size='x-small', color='k')
	for tick,label in zip(pos,ax_b2.get_xticklabels()):
		ax_b2.text(pos[tick]-0.3, mean_values_b2[2*tick],   mean_labels_b2[2*tick],    horizontalalignment='center', size='x-small', color='k')
		ax_b2.text(pos[tick]+0.3, mean_values_b2[2*tick+1], mean_labels_b2[2*tick+1],  horizontalalignment='center', size='x-small', color='k')

	ax_b1.set_ylim(-10,100)
	ax_b2.set_ylim(-10,100)
	plt.tight_layout()
	fig_emit.savefig('plots_Giannis/blowUp_atFlatBottom_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d")), dpi=300)

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 

################################################
#
#	THIS IS TO PLOT RELATIVE BLOW-UP AT RAMP
#
def plotBlowUpAtRamp(flist):
	plt.close('all')
	df = pd.read_pickle('summaryEmittanceDF_growth.pkl')
	df2 = df[(df['Cycle']=='FB2FT') & (df['Fill']!=5919) & (df['Fill']!=5920) & (df['Fill']!=5934) & (df['Fill']!=5930)]
	fig_emit = plt.figure(figsize=(17,8))
	ax_b1 = plt.subplot(211)
	ax_b2 = plt.subplot(212)
	sns.set_style("whitegrid")
	ax_b1 = sns.violinplot(x="Fill", y="DeltaEmittance_raw_relative", hue="Plane", data=df2[df2['Beam']=='B1'],  palette=['#71aaf2', '#ff8a3d'], 
		split=True, order=flist, ax=ax_b1)
	ax_b2 = sns.violinplot(x="Fill", y="DeltaEmittance_raw_relative", hue="Plane", data=df2[df2['Beam']=='B2'],  palette=['#71aaf2', '#ff8a3d'], 
		split=True, order=flist, ax=ax_b2)

	mean_values_b1 = []
	for nbunches in flist:
	   for nplane in ['Horizontal', 'Vertical']:
	       mean_values_b1.append(df2['DeltaEmittance_raw_relative'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B1')].mean())
	mean_values_b2 = []
	for nbunches in flist:
	   for nplane in ['Horizontal', 'Vertical']:
	       mean_values_b2.append(df2['DeltaEmittance_raw_relative'][(df2['Fill']==nbunches) & (df2['Plane']==nplane) & (df2['Beam']=='B2')].mean())

	mean_labels_b1 = [str(np.round(s, 2)) for s in mean_values_b1]
	mean_labels_b2 = [str(np.round(s, 2)) for s in mean_values_b2]
	pos = range(len(mean_labels_b1))
	ax_b1.set_ylabel('B1 $\mathbf{\Delta\epsilon_{n}\slash\epsilon_{n}}$ [%]', fontsize=16, fontweight='bold')
	ax_b2.set_ylabel('B2 $\mathbf{\Delta\epsilon_{n}\slash\epsilon_{n}}$ [%]', fontsize=16, fontweight='bold')
	ax_b2.set_xlabel('Fill Number', fontsize=16, fontweight='bold')
	ax_b1.set_title('Measured Relative Emittance Blow-up from Injection to Start Ramp', fontsize=20, fontweight='bold')
	ax_b1.set_xlabel('')

	for tick,label in zip(pos,ax_b1.get_xticklabels()):
		ax_b1.text(pos[tick]-0.3, mean_values_b1[2*tick],   mean_labels_b1[2*tick],    horizontalalignment='center', size='x-small', color='k')
		ax_b1.text(pos[tick]+0.3, mean_values_b1[2*tick+1], mean_labels_b1[2*tick+1],  horizontalalignment='center', size='x-small', color='k')
	for tick,label in zip(pos,ax_b2.get_xticklabels()):
		ax_b2.text(pos[tick]-0.3, mean_values_b2[2*tick],   mean_labels_b2[2*tick],    horizontalalignment='center', size='x-small', color='k')
		ax_b2.text(pos[tick]+0.3, mean_values_b2[2*tick+1], mean_labels_b2[2*tick+1],  horizontalalignment='center', size='x-small', color='k')

	ax_b1.set_ylim(-10,100)
	ax_b2.set_ylim(-10,100)
	plt.tight_layout()
	fig_emit.savefig('plots_Giannis/blowUp_atRamp_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d")), dpi=300)

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 

########################################################
#
#	THIS IS TO PLOT GROWTH MEASURED/MODEL AT INJECTION
#
def plotGrowthAtFlatBottom():
	plt.close('all')
	## keep only INJ2FB
	df = pd.read_pickle('summaryEmittanceDF_growth.pkl')
	df = df[(df['Cycle']=='INJ2FB') & (df['Fill']!=5919) & (df['Fill']!=5920) & (df['Fill']!=5934) & (df['Fill']!=5930)]


	df_b1h   = df[(df['Beam']=='B1')&(df['Plane']=='Horizontal')]
	df_b1hgm = df_b1h.groupby('Fill').aggregate(np.mean)
	df_b1hgs = df_b1h.groupby('Fill').aggregate(np.std)
	fills = df_b1hgm.index.values
	b1h_mean_meas  = df_b1hgm['Growth_raw_meas'].values
	b1h_mean_model = df_b1hgm['Growth_raw_model'].values
	b1h_std_meas   = df_b1hgs['Growth_raw_meas'].values
	b1h_std_model  = df_b1hgs['Growth_raw_model'].values

	df_b1v   = df[(df['Beam']=='B1')&(df['Plane']=='Vertical')]
	df_b1vgm = df_b1v.groupby('Fill').aggregate(np.mean)
	df_b1vgs = df_b1v.groupby('Fill').aggregate(np.std)
	b1v_mean_meas  = df_b1vgm['Growth_raw_meas'].values
	b1v_mean_model = df_b1vgm['Growth_raw_model'].values
	b1v_std_meas   = df_b1vgs['Growth_raw_meas'].values
	b1v_std_model  = df_b1vgs['Growth_raw_model'].values

	df_b2h   = df[(df['Beam']=='B2')&(df['Plane']=='Horizontal')]
	df_b2hgm = df_b2h.groupby('Fill').aggregate(np.mean)
	df_b2hgs = df_b2h.groupby('Fill').aggregate(np.std)
	b2h_mean_meas  = df_b2hgm['Growth_raw_meas'].values
	b2h_mean_model = df_b2hgm['Growth_raw_model'].values
	b2h_std_meas   = df_b2hgs['Growth_raw_meas'].values
	b2h_std_model  = df_b2hgs['Growth_raw_model'].values

	df_b2v   = df[(df['Beam']=='B2')&(df['Plane']=='Vertical')]
	df_b2vgm = df_b2v.groupby('Fill').aggregate(np.mean)
	df_b2vgs = df_b2v.groupby('Fill').aggregate(np.std)
	b2v_mean_meas  = df_b2vgm['Growth_raw_meas'].values
	b2v_mean_model = df_b2vgm['Growth_raw_model'].values
	b2v_std_meas   = df_b2vgs['Growth_raw_meas'].values
	b2v_std_model  = df_b2vgs['Growth_raw_model'].values

	fig = plt.figure(1, figsize=(15,7))
	ax_1h = plt.subplot(211)
	ax_1v = plt.subplot(212)
	ax_1h.errorbar(fills, y = b1h_mean_meas, yerr = b1h_std_meas,   ls='None', fmt='--o', capsize=0, label='Measured')
	ax_1h.errorbar(fills, y = b1h_mean_model, yerr = b1h_std_model, ls='None', fmt='--o', capsize=0, label='Model')

	ax_1v.errorbar(fills, y = b1v_mean_meas,  yerr = b1v_std_meas,  ls='None', fmt='--o', capsize=0, label='Measured')
	ax_1v.errorbar(fills, y = b1v_mean_model, yerr = b1v_std_model, ls='None', fmt='--o', capsize=0, label='Model')

	ax_1h.set_ylim(-0.5, 2.5)
	ax_1v.set_ylim(-0.5, 2.5)
	ax_1h.set_ylabel('B1H Emit. Growth [$\mathbf{\mu}$m/h]', fontsize=12, fontweight='bold')
	ax_1v.set_ylabel('B1V Emit. Growth [$\mathbf{\mu}$m/h]', fontsize=12, fontweight='bold')
	ax_1h.set_xlabel('')
	ax_1v.set_xlabel('Fill Number' , fontsize=16, fontweight='bold')
	ax_1h.grid('on')
	ax_1v.grid('on')
	ax_1h.legend()
	ax_1v.legend()
	plt.tight_layout()

	fig2 = plt.figure(2, figsize=(15,7))
	ax_2h = plt.subplot(211)
	ax_2v = plt.subplot(212)
	ax_2h.errorbar(fills, y = b2h_mean_meas,  yerr = b2h_std_meas,  ls='None', fmt='--o', capsize=0, label='Measured')
	ax_2h.errorbar(fills, y = b2h_mean_model, yerr = b2h_std_model, ls='None', fmt='--o', capsize=0, label='Model')
	ax_2v.errorbar(fills, y = b2v_mean_meas,  yerr = b2v_std_meas,  ls='None', fmt='--o', capsize=0, label='Measured')
	ax_2v.errorbar(fills, y = b2v_mean_model, yerr = b2v_std_model, ls='None', fmt='--o', capsize=0, label='Model')

	ax_2h.set_ylim(-0.5, 2.5)
	ax_2v.set_ylim(-0.5, 2.5)
	ax_2h.set_ylabel('B2H Emit. Growth [$\mathbf{\mu}$m/h]', fontsize=12, fontweight='bold')
	ax_2v.set_ylabel('B2V Emit. Growth [$\mathbf{\mu}$m/h]', fontsize=12, fontweight='bold')
	ax_2h.set_xlabel('')
	ax_2v.set_xlabel('Fill Number' , fontsize=16, fontweight='bold')
	ax_2h.grid('on')
	ax_2v.grid('on')
	ax_2h.legend()
	ax_2v.legend()
	plt.tight_layout()

	print ' EMITTANCE GROWTH AT INJECTION <MEAN VALUES> '
	print '---------------------------------------------'
	print 'MEASURED : B1H : ', np.nanmean(b1h_mean_meas)
	print 'MEASURED : B1V : ', np.nanmean(b1v_mean_meas)
	print 'MEASURED : B2H : ', np.nanmean(b2h_mean_meas)
	print 'MEASURED : B2V : ', np.nanmean(b2v_mean_meas)
	print '---------------------------------------------'
	print 'Model    : B1H : ', np.nanmean(b1h_mean_model)
	print 'Model    : B1V : ', np.nanmean(b1v_mean_model)
	print 'Model    : B2H : ', np.nanmean(b2h_mean_model)
	print 'Model    : B2V : ', np.nanmean(b2v_mean_model)


	fig.savefig('plots_Giannis/growth_measModelFB_B1_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d")), dpi=300)
	fig2.savefig('plots_Giannis/growth_measModelFB_B2_{}.png'.format(datetime.datetime.now().strftime("%Y%m%d")), dpi=300)

# - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * - - * 



if __name__ == '__main__':
	print '------------------ ------------------------------------'
	print '   Welcome to the Nikos\' Magnificent Opera House  '
	print '------------------ ------------------------------------'
	print ' Please lay back and relax while we play the violin...'

	print ''
	print 'I hope you have decided about which sonnet you want to listen to'
	print 'I will start by writing the sheet music first...'
	makeInput(flist, bunches_dict)	

	print '\n------------------ ------------------------------------\n'
	print 'First let me start with allegretto dolce e con affetto (injection)'
	plotEmittancesAtInjection(flist)
	print '\n------------------ ------------------------------------\n'
	print 'Then the crescendo (ramp)'
	plotEmittancesAtStartStable(flist)
	print '\n------------------ ------------------------------------\n'
	print 'Let\'s go back to tempo primo (blowup injection)'
	plotBlowUpAtFlatBottom(flist)
	print '\n------------------ ------------------------------------\n'
	print 'Now, rallentando (blowup ramp)'
	plotBlowUpAtRamp(flist)
	print '\n------------------ ------------------------------------\n'
	print 'Let\'s consider what we had been listening to...'
	plotGrowthAtFlatBottom()
	print '\n------------------ ------------------------------------\n'
	print 'Finito!'




