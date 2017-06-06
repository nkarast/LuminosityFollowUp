#
#	makePDF.py -- Makes a pdf of the plots 
#

import glob
import sys, os
BIN = os.path.expanduser("/afs/cern.ch/user/l/lumimod/lumimod/a2017_luminosity_followup/")
sys.path.append(BIN)
import config
import datetime
import collections



def main(min_fill, max_fill, bunch_def_dict, outfile_name):

	fill_list = []

	for fill_dir in glob.glob(config.stableBeams_folder+"/fill_*"):
		fill_list.append(int(fill_dir.split('/')[-1].split('_')[1]))

	sb_plots = collections.OrderedDict()
	sb_plots["SB B1 Emittances"] 			= "fill_<FILLNUMBER>_b1Emittances{}".format(config.plotFormat)
	sb_plots["SB B2 Emittances"] 			= "fill_<FILLNUMBER>_b2Emittances{}".format(config.plotFormat)
	sb_plots["SB Bunch Intensity"] 			= "fill_<FILLNUMBER>_bunchIntensity{}".format(config.plotFormat)
	sb_plots["SB Bunch Length"] 			= "fill_<FILLNUMBER>_bunchLength{}".format(config.plotFormat)
	sb_plots["SB Calculated Luminosity"] 	= "fill_<FILLNUMBER>_calcLumi{}".format(config.plotFormat)
	sb_plots["SB Measured Luminosity"] 		= "fill_<FILLNUMBER>_measLumi{}".format(config.plotFormat)
	sb_plots["SB Luminosity"] 				= "fill_<FILLNUMBER>_totalLumi{}".format(config.plotFormat)
	sb_plots["SB Luminosity Lifetime (fit)"] 		= "fill_<FILLNUMBER>_TotalLumiLifetime{}".format(config.plotFormat)
	sb_plots["SB Luminosity Lifetime (fit range)"] 	= "fill_<FILLNUMBER>_FitLumiLifetime{}".format(config.plotFormat)
	sb_plots["SB Intensity Lifetime"] 		= "fill_<FILLNUMBER>_bbbIntensityTau{}".format(config.plotFormat)
	sb_plots["SB Normalized Losses"] 		= "fill_<FILLNUMBER>_bbbLosses{}".format(config.plotFormat)


	cycle_plots = collections.OrderedDict()
	cycle_plots["Cycle Emittances"] 	= "fill_<FILLNUMBER>_cycle_emittancesbbb{}".format(config.plotFormat)
	cycle_plots["Cycle Intensity"] 		= "fill_<FILLNUMBER>_cycle_intensitiesbbb{}".format(config.plotFormat)
	cycle_plots["Cycle Bunch Length"] 	= "fill_<FILLNUMBER>_cycle_blengthbbb{}".format(config.plotFormat)
	cycle_plots["Cycle Brightness"] 	= "fill_<FILLNUMBER>_cycle_brightnessbbb{}".format(config.plotFormat)
	cycle_plots["Cycle Measurement Time"] = "fill_<FILLNUMBER>_cycle_timebbb{}".format(config.plotFormat)
	cycle_plots["Cycle Histograms"] 	= "fill_<FILLNUMBER>_cycle_histos{}".format(config.plotFormat)



	masked_fills = [x for x in fill_list if x > min_fill and x < max_fill]
	print 'Running for fills : ', masked_fills


	start_tex = '''\documentclass{beamer}
	\usepackage[latin1]{inputenc}
	\usepackage{graphicx}
	\usetheme{Warsaw}
	\\title[Lumi Follow-Up]{Fills %s}
	\\author{Nikos Karastathis}
	\institute{CERN/BE-ABP-HSI}
	\date{%s}
	\\begin{document}

	\\begin{frame}
	\\titlepage
	\end{frame}

	\\begin{frame}{Overview}
	\\tableofcontents
	\end{frame}
	'''%(masked_fills, datetime.datetime.now().strftime("%Y/%m/%d"))



	end_tex = "\end{document}"

	def makeImageFrame(fileContentString, fill, title, content):
		localContentString = """
		\\begin{frame}{Fill %s : %s}
		\includegraphics[width=0.9\\paperwidth]{%s}
		\end{frame}
		"""%(fill, title, content)

		newString = "\n".join((fileContentString, localContentString))
		return newString

	def makeFillFrame(fileContentString, fill):
		localContentString = """
		\\begin{frame}{}
		FILL %s
		\end{frame}
		"""%(fill)
		newString = "\n".join((fileContentString, localContentString))
		return newString

	def makeSection(fileContentString, filln):
		localContentString = """
		\section{%s}
		"""%(bunch_def_dict[filln])
		newString = "\n".join((fileContentString, localContentString))
		return newString


	tex_document = start_tex

	for fill in masked_fills:

		if fill in bunch_def_dict.keys():
			makeSection(tex_document, fill)

		for key in sb_plots:
			plot = config.stableBeams_folder+config.fill_dir.replace("<FILLNUMBER>", str(fill))+config.plot_dir+sb_plots[key].replace("<FILLNUMBER>", str(fill))
			print plot
			if os.path.exists(plot):
				print ' Plot exists : ', plot
				tex_document = makeImageFrame(tex_document, fill, key, plot)

		for key in cycle_plots:
			plot = config.stableBeams_folder+config.fill_dir.replace("<FILLNUMBER>", str(fill))+config.plot_dir+cycle_plots[key].replace("<FILLNUMBER>", str(fill))
			if os.path.exists(plot):
				tex_document = makeImageFrame(tex_document, fill, key, plot)



	newTexDocument = "\n".join((tex_document, end_tex))

	outfile = open(outfile_name, 'w')
	outfile.write(newTexDocument)
	outfile.close()




if __name__ == '__main__':
	
	min_fill =  5698
	max_fill = 	5750 


	bunch_def_dict = {  5704 : "Multi_12b_8_8_8_4bpi_3inj_2500ns",
						5710 : "Multi_12b_8_8_2_4bpi_3inj_2500ns",
						5717 : "25ns_75b_63_30_30_12bpi_9inj",
						5718 : "25ns_75b_63_30_30_12bpi_9inj",
						5719 : "25ns_75b_63_30_30_12bpi_9inj",
						5722 : "25ns_336b_336_210_252_12bpi_28inj",
						5730 : "25ns_315b_303_228_240_48bpi_11inj",
						5737 : "25ns_345b_303_228_240_48bpi_11inj",

						5746 : "25ns_601b_589_522_540_48bpi15inj_bcms",
						5749 : "25ns_601b_589_522_540_48bpi15inj_bcms",
						5750 : "25ns_601b_589_522_540_48bpi15inj_bcms",

						}

	outfile_name = "perfPlots_{}_{}_{}.tex".format(min_fill, max_fill, datetime.datetime.now().strftime("%Y%m%d"))

	main(min_fill, max_fill, buncH_def_dict, outfile_name)
