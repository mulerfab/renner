#! /usr/bin/python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Plotting of FC factors from Renner-like output files')
	parser.add_argument('inputfiles',metavar='INPUT',nargs='+',type=str,help='input file(s) for combined plotting')
	parser.add_argument('--outputfile','-o',metavar='OUPUT',type=str,help='plot output file')
	parser.add_argument('--start','-s',type=float,help='start value of x-axis')
	parser.add_argument('--end','-e',type=float,help='end value of x-axis')
	parser.add_argument('--bottom','-b',type=float,help='start value of y-axis')
	parser.add_argument('--top','-t',type=float,help='end value of y-axis')
	parser.add_argument('--multi-color','-c',action='store_true',help='plot each data row with different color (four colors available)')
	parser.add_argument('--envelope','-v',nargs='?',type=float,const=10.0,help='plot envelope of all signals with given line width (default 10.0)')
	
	args = parser.parse_args()
	
	data_rows = []
	for f in args.inputfiles:
		if not os.path.isfile(f):
			raise Exception("File does not exist: {}".format(f))
		
		with open(f,'r') as fh:
			cont = []
			for line in fh:
				if line.strip()[0] == '#':
					continue
				
				try:
					numbers = [float(i.strip()) for i in line.strip().split() if i]
				except ValueError:
					raise Exception("Unexpacted format of input in {}. Couldn't convert floats in line:\n{}".format(f,line))
				
				cont.append(numbers)
		
		data_rows.append(np.array(cont))
	
	fig = plt.figure(figsize=(4, 4))		# width and height in inch
	ax = fig.add_axes([0.10, 0.10, 0.84, 0.84])	# [left, bottom, width, height] in % of width and height
	colorkeys = ['b','r','k','g']
	ncolors = len(colorkeys) if args.multi_color else 1
	c = 0
	val_max = data_max = -1.0e-100
	val_min = 1.0e100
	for dat in data_rows:
		val_max = np.max(dat[:,0]) if np.max(dat[:,0]) > val_max else val_max
		val_min = np.min(dat[:,0]) if np.min(dat[:,0]) < val_min else val_min
		data_max = np.max(dat[:,1]) if np.max(dat[:,1]) > data_max else data_max
		selected_data = dat[dat[:,1] > 0.0001*data_max]
		print("row {}: plotting {} relevant signals".format(c+1,len(selected_data)))
		ax.stem(selected_data[:,0], selected_data[:,1], linefmt=colorkeys[c%ncolors]+'-', markerfmt=' ', basefmt=' ')
		if args.envelope:
			gamma = args.envelope
			env_min = args.start if args.start else val_min
			env_max = args.end if args.end else val_max
			x = np.linspace(env_min,env_max,1000)
			env = np.zeros_like(x)
			for item in selected_data:
				env += ( item[1] * gamma**2/((x - item[0])**2 + gamma**2) )
			ax.plot(x,env,'k-')
		c += 1
	
	plot_start = args.start if args.start else 0.0
	plot_end = args.end if args.end else np.max(dat[:,0])
	plot_bottom = args.bottom if args.bottom else 0.0
	plot_top = args.top if args.top else np.max(dat[:,1])
	
	ax.set_xlim(plot_start, plot_end)
	ax.set_ylim(plot_bottom, plot_top)
#	ax.set_aspect(1)
	
	if args.outputfile:
		if os.path.isfile(args.outputfile):
			overwrite = input("Output file {} already exists. Overwrite? (y/n)> ".format(args.outputfile))
			if not overwrite.lower().strip() in ['y','yes','1']:
				print("\naborting.")
				exit()
	
		plt.savefig(args.outputfile)
	
	plt.show()
