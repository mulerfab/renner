"""
		renner -- Franck-Condon Simulation Class
		
		VERISON 0.3
		2019/8/23
		by famu
"""


from RennerUtils import RennerError,loadIntegrals
from Hamiltonian import Hamiltonian
import os
import yaml
from datetime import datetime
import numpy as np
import itertools
import matplotlib.pyplot as plt


class FCSim:
	def __init__(self,input_file,verbose,numprocs=4):
		if not os.path.isfile(input_file):
			raise RennerError("Input file '{}' does not exist!".format(input_file))
		
		with open(input_file,'r') as fh:
			self.inputs = yaml.safe_load(fh)
		
		self.verbose = verbose
		
		Eigenvectors = {}
		Eigenvalues = {}
		basis_sets = {}
		try:
			# request diagonalization of all given hamiltonians
			# (each hamiltonian represents an independent dimension)
			print()
			print("Producing Hamiltonian Matrices for all given dimensions:")
			for dim in self.inputs["dimensions"]:
				H = Hamiltonian(dim,self.verbose)
				Eigenvalues[dim["ref"]],Eigenvectors[dim["ref"]] = H.diagonalize(numprocs)
				basis_sets[dim["ref"]] = dim["basis"]
				if self.verbose > 1:
					print("lowest Eigenvalues: ",end='')
					print(Eigenvalues[dim["ref"]][:min(10,len(Eigenvalues[dim["ref"]]))].round(decimals=3))
				print()
			
			# produce specified FC simulations, i.e. all overlaps of the eigenstates of the given hamiltonians
			print("Calculating Overlap of Eigenstates:")
			fcsim_dict = self.inputs["fcsim"]
			FCF = {}
			dE = {}
			for fcref,fcsim_part in fcsim_dict["components"].items():
				if not len(fcsim_part) == 2:
					raise RennerError("Each component of a RT-FC simulation is the overlap of the eigenstates of exactly two diabatic Hamiltonians!")
				
				refs = list(fcsim_part.keys())
				ref1,ref2 = refs
				print("component {}: GS is {}, ES is {}".format(fcref,ref1,ref2))
				
				if isinstance(fcsim_part[ref1],int):
					vib_level1 = fcsim_part[ref1]
				else:
					vib_level1 = 0		# this part could be extended to capture a range of GS levels (temperature effects)
				
				WFC1 = Eigenvectors[ref1]	# index 1 denotes the GS
				WFC2 = Eigenvectors[ref2]	# and index 2 the ES
				E1 = Eigenvalues[ref1]
				E2 = Eigenvalues[ref2]
				
				bas_dict1 = basis_sets[ref1]
				bas_dict2 = basis_sets[ref2]
				
				with loadIntegrals(bas_dict1,numprocs) as integrals:
					FCF[fcref] = []
					dE[fcref] = []
					for vib_level2 in range(int(fcsim_part[ref2]*len(WFC2))):
						S = integrals.overlapElem(WFC1[:,vib_level1],WFC2[:,vib_level2],bas_dict2)
						FCF[fcref].append(S*S)
						dE[fcref].append(E2[vib_level2] - E1[vib_level1])
			
			# combine FCSimulations of certain dimensions if requested
			if "join" in fcsim_dict:
				print()
				print("Combining requested FC simulations to one dimension:")
				for comb in fcsim_dict["join"]:
					if not isinstance(comb,list):
						raise RennerError("To join several FC Simulations of one dimension, a list of references of at least two must be given!")
					
					print(" -> ",end='')
					FCFtmp = []			# sum the user requested FC simulations
					dEtmp = []
					for fcref in comb:
						print("{} ".format(fcref),end='')
						FCFtmp += FCF[fcref]
						dEtmp += dE[fcref]
						del FCF[fcref]		# and remove them from the dict
						del dE[fcref]
					
					print()
					FCF[comb[0]] = FCFtmp		# keep only the sum and name it like the first summand
					dE[comb[0]] = dEtmp
			
		except KeyError as err:
			raise RennerError("Input missing: {}".format(err))
		
		
		# create the complete FC simulation
		print()
		print("Create final FC simulation")
		FCF_tot = []						# direct product of all FC factors
		for fac in itertools.product(*(FCF.values())):
			FCF_tot.append(np.array(fac).prod())
		
		dE_tot = []						# direct sum of all energy differences
		for e in itertools.product(*(dE.values())):
			dE_tot.append(np.array(e).sum())
		
		if "leveling" in fcsim_dict:
			dE_shift = fcsim_dict["leveling"]
			print("leveling to {}".format(dE_shift))
			dE_tot_np = np.array(dE_tot)
			dE_min = dE_tot_np.min()
			dE_tot_np += (dE_shift - dE_min)
			dE_tot = dE_tot_np.tolist()
		
		self.FC_sim = sorted(list(zip(dE_tot,FCF_tot)),key=lambda i: i[0])	# shuffle all together and sort them by dE
		
	
	
	def writeFile(self):
		print()
		print("Creating output file")
		if not hasattr(self,"FC_sim"):
			raise RennerError("Error while producing Franck-Condon simulation!")
		
		fcsim_dict = self.inputs["fcsim"]
		if "comments" in self.inputs:
			comments = self.inputs["comments"]
		else:
			comments = {}
		
		now = datetime.now()
		time_stamp = str(now.strftime('%Y-%m-%d %H:%M:%S'))
		
		try:
			FCF_file = fcsim_dict["file"]
		except KeyError as err:
			raise RennerError("Input missing! {}".format(err))
		
		with open(FCF_file,'w') as fh:
			fh.write("# Renner v0.3 -- Date " + time_stamp + "\n#\n")
			for k,v in comments.items():
				fh.write("#  {}:  {}\n".format(k,v))
			fh.write("#\n")
			fh.write("#{:^9}  {:^8}\n".format('dE','FCF'))
			for e,fcf in self.FC_sim:
				fh.write(" {:9.2f}  {:8.5f}\n".format(e,fcf))
		
		print("written {} lines to file '{}'".format(len(self.FC_sim),FCF_file))
	
	
	def createPlot(self):
		print("Creating plot")
		
		try:
			plot_dict = self.inputs["fcsim"]["plot"]
			plot_file = plot_dict["file"]
			plot_start = plot_dict["start"]
			plot_end = plot_dict["end"]
		except KeyError as err:
			raise RennerError("Input missing! {}".format(err))
		
		data = np.array(self.FC_sim)
		data_max = np.max(data[:,1])
		selected_data = data[data[:,1] > 0.001*data_max]
		print("plotting {} relevant signals".format(len(selected_data)))
		plt.stem(selected_data[:,0], selected_data[:,1], linefmt='b-')
		plt.xlim(plot_start, plot_end)
		plt.savefig(plot_file)


