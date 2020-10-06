#! /usr/bin/python3

import numpy as np
import yaml
from RennerUtils import RennerError
from Hamiltonian import Hamiltonian
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Eigenstates and energy eigenvalues of diabatic Hamiltonian')
	parser.add_argument('inputfile',metavar='INPUT',help='Renner input file')
	parser.add_argument('ref',help='reference state')
	parser.add_argument('--frac','-f',type=float,default=0.1,help='fraction of Eigenvalues/states that is printed (default: 0.1)')
	parser.add_argument('--num','-n',type=int,default=0,help='number of Eigenvalues/states that are printed (default: 0, not used)')
	parser.add_argument('--num-procs','-p',metavar='NUM',type=int,default=4,help='number of threads for parallel computation (default: 4)')
	parser.add_argument('--verbose','-v',metavar='LEVEL',type=int,default=0,help='verbose level (0 means off)')
	
	args = parser.parse_args()
	
	with open(args.inputfile,'r') as fh:
		inputs = yaml.safe_load(fh)
	
	try:
		for d in inputs["dimensions"]:
			if d["ref"] == args.ref:
				dim = d
				break
	except KeyError as err:
		print("Input missing! {}".format(err))
	
	with Hamiltonian(dim,args.verbose) as H:
		eigvals,eigvecs = H.diagonalize(args.num_procs)
	
	print("Energy Eigenvalues:")
	if args.num != 0:
		if args.num > len(eigvals):
			n = len(eigvals)
		else:
			n = args.num
	else:
		if args.frac <= 1.0 and args.frac > 0.0:
			n = int(len(eigvals) * args.frac)
		else:
			n = len(eigvals)
	
	print(eigvals[:n])
