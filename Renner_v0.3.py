#! /usr/bin/python3

"""
		renner
		
		VERISON 0.3
		2019/8/23
		by famu
"""

from RennerUtils import RennerError
from Integrals.IntegralErrors import IntegralError
from Hamiltonian import Hamiltonian
from FCSim import FCSim
import os
import yaml
import numpy as np
import argparse
from datetime import datetime



def convergenceCheck(input_file,state_ref,verbose=0,numprocs=4):
	print("Convergence Check:")
	if not os.path.isfile(input_file):
		raise RennerError("Input file '{}' does not exist!".format(input_file))
	
	with open(input_file,'r') as fh:
		inputs = yaml.safe_load(fh)
	
	try:
		for d in inputs["dimensions"]:
			if d["ref"] == state_ref:
				dim_dict = d
				break
		
		for c in inputs["fcsim"]["components"].values():
			if state_ref in c:
				bs_frac = c[state_ref]
				break
		
	except KeyError as err:
		raise RennerError("Input missing! {}".format(err))
	
	# calculate eigenvalues with user defined amount of basis functions
	with Hamiltonian(dim_dict,verbose) as H:
		evals,evecs = H.diagonalize(numprocs)
		zpve = H.zpve
	
	num_evals = len(evals)
	num_relevant_evals = int(bs_frac*num_evals)
	relevant_evals = evals[:num_relevant_evals]
	print("Check if the lowest {} Eigenvalues ({:5.1f}%) are consistent upon basis set increase...".format(num_relevant_evals,float(100.0*bs_frac)))
	
	# increase number of basis functions
	diff = 1000.0
	dim_dict["basis"]["file"] = ""	# ignore basis set file, if given
	counter = 0
	while diff > 0.1*zpve and counter < 10:
		dim_dict["basis"]["num"] += 2
		with Hamiltonian(dim_dict,verbose) as H:
			evals_new,evecs = H.diagonalize(numprocs)
		
		num_new = len(evals_new)
		relevant_evals_new = evals_new[:num_relevant_evals]
		
		diff = np.linalg.norm(relevant_evals-relevant_evals_new)
		print("Num: {}  New BS size: {}  Difference upon increase: {}".format(dim_dict["basis"]["num"],num_new,diff))
		
		relevant_evals = relevant_evals_new
		counter += 1
	
	if counter == 1:
		print("Basis set is big enough.")
	else:
		print("Increase basis set! Use at least num={}".format(dim_dict["basis"]["num"]-2))
	
	if counter <= 10:
		print("Converged Eigenvalues:")
	else:
		print("Best you can get:")
	
	print(relevant_evals_new)
	print()


"""
	Main Part
"""
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Calculation of Franck-Condon factors for Renner-Teller systems: Nummetrical diagonalization of (almost) arbitrary diabatic Hamiltonians')
	parser.add_argument('inputfile',metavar='INPUT',help='input file for the potential generation')
	parser.add_argument('--num-procs','-p',metavar='NUM',type=int,default=4,help='number of threads for parallel computation (default: 4)')
	parser.add_argument('--check-conv','-c',action='store_true',help='check for basis set convergence')
	parser.add_argument('--ref-state','-r',metavar='REF',type=str,default="",help='reference state for basis set convergence test; has to be given in input file')
	parser.add_argument('--verbose','-v',metavar='LEVEL',type=int,default=0,help='verbose level (0 means off)')
	
	args = parser.parse_args()
	
	try:
		if args.check_conv:
			convergenceCheck(args.inputfile,args.ref_state,0,args.num_procs)
			exit()
		
		fcsim = FCSim(args.inputfile,args.verbose,args.num_procs)
		fcsim.writeFile()
		fcsim.createPlot()
		del fcsim
	except RennerError as rerr:
		print("Error while producing the FC simulation:")
		print(rerr)
	except IntegralError as ierr:
		print("Error while calculating integrals:")
		print(ierr)
	
	exit()

