"""
		renner -- Diabatic Hamiltonian Class: Creation and Diagonalization
		
		VERISON 0.3
		2019/8/23
		by famu
"""

from RennerUtils import RennerError,loadIntegrals
import numpy as np


class Hamiltonian:
	def __init__(self,input_dict,verbose):
		self.input_dict = input_dict
		self.verbose = verbose
	
	def __enter__(self):
		return self
	
	def __exit__(self, type, value, traceback):
		if isinstance(value, RennerError):
			print(value)
			return True
	
	def setup(self, numprocs=4):
		if not "basis" in self.input_dict or not "hamiltonian" in self.input_dict:
			raise RennerError("It is necessary to define both a basis and a hamiltonian for each dimension!")
		
		bas_dict = self.input_dict["basis"]
		ham_dict = self.input_dict["hamiltonian"]
		
		# create an instance of the requested integral library and provide the necessary basis set parameters
		#integrals = loadIntegrals(bas_dict,numprocs)
		
		# plug together the diabatic hamiltonian
		try:
			self.N_states = ham_dict["states"]
			N_elem_max = self.N_states * (self.N_states + 1) / 2
			
			self.zpve = ham_dict["zpve"]
			
			kinetic_factor = ham_dict["kinetic"]["factor"]
			
			pot_dict = ham_dict["potential"]
			modelmat = pot_dict["modelmatrix"]
			
			# create composite matrices
			composite_mats = {}
			try:
				for i,row in enumerate(modelmat):
					for j,elem in enumerate(row):
						if isinstance(elem,list):
							if isinstance(elem[0],float) or isinstance(elem[0],int):
								fac = elem[0]
								sym = elem[1]
						else:
							fac = 1
							sym = elem
						
						if sym in [None,0,0.0,'']:
							continue
						
						if not sym in composite_mats:
							composite_mats[sym] = np.zeros((self.N_states,self.N_states))
						
						composite_mats[sym][i,j] = fac
				
			except (TypeError,IndexError) as err:
				raise RennerError("Model matrix is not well defined! {}".format(err))
			
			if self.verbose > 1:
				print("Composite matrices for Hamiltonian {}:".format(self.input_dict["ref"]))
			for k,v in composite_mats.items():
				# symmetrize composite matrices for user-friendlyness
				composite_mats[k] += v.T - np.diag(v.diagonal())
				if self.verbose > 1:
					print("{}:".format(k))
					print(composite_mats[k])
			
			poly = pot_dict["polynomials"]
			if "couplings" in pot_dict:
				coup = pot_dict["couplings"]
			else:
				coup = {}
			
			# consistency check
			if len(poly) + len(coup) > N_elem_max or len(composite_mats) > N_elem_max:
				raise RennerError("There are too much different matrix elements defined!")
			
		except KeyError as err:
			raise RennerError("Input missing: {}".format(err))
		
		# find out which monomials and coupling elements are needed
		mono_exp = [0]				# always request the metric
		for sym,coeffs in poly.items():
			mono_exp += list(coeffs.keys())
		mono_exp = set(mono_exp)
		
		coup_exp = []
		for sym,coeffs in coup.items():
			coup_exp += list(coeffs.keys())
		coup_exp = set(coup_exp)
		
		# request the respective matrices
		if self.verbose > 0:
			print("Calculating matrix elements:")
		with loadIntegrals(bas_dict,numprocs) as integrals:
		#try:
			if self.verbose > 0:
				print("Diagonal: ", end='', flush=True)
			Rmats = {}
			for n in mono_exp:
				if self.verbose > 0:
					print("{} ".format(n), end='', flush=True)
				Rmats[n] = getattr(integrals,"R{}mat".format(n))()
		#	Rmats = {n:getattr(integrals,"R{}mat".format(n))() for n in mono_exp}
			if self.verbose > 0:
				print("\nCoupling: ", end='', flush=True)
			Pmats = {}
			for n in coup_exp:
				if self.verbose > 0:
					print("{} ".format(n), end='', flush=True)
				Pmats[n] = getattr(integrals,"P{}mat".format(n))()
		#	Pmats = {n:getattr(integrals,"P{}mat".format(n))() for n in coup_exp}
			if self.verbose > 0:
				print("\nKinetic")
			Tmat = getattr(integrals,"Tmat")()
		#except AttributeError as err:
		#	raise RennerError("Requested matrix element is not availalbe! {}".format(err))
		
		# get rid of integral object (and make sure, matrices are saved for further use if requested)
		#del integrals
		
		# save metric
		self.metric = Rmats[0]
		
		# set up the whole hamiltonian, i.e. the direct product of diab. hamiltonian and basis expansion
		if self.verbose > 0:
			print("Setting up basis expansion for {} states and {} basis functions".format(self.N_states,len(self.metric)))
		HmatParts = []
		HmatParts.append(np.kron(np.identity(self.N_states),-kinetic_factor * Tmat))
		for sym,coeffs in poly.items():
			RR = np.array([Rmats[i] for i in coeffs.keys()])
			HmatParts.append(np.kron(composite_mats[sym],np.tensordot(np.array(list(coeffs.values())),RR,axes=1)))
		for sym,coeffs in coup.items():
			PP = np.array([Pmats[i] for i in coeffs.keys()])
			HmatParts.append(np.kron(composite_mats[sym],np.tensordot(np.array(list(coeffs.values())),PP,axes=1)))
		
		self.Hmat = np.sum(np.array(HmatParts),axis=0)
	
	
	def diagonalize(self,numprocs=4):
		if hasattr(self,"Heigvals") and hasattr(self,"Wmat"):
			return (self.Heigvals,self.Wmat)
		
		if not hasattr(self,"Hmat"):
			self.setup(numprocs)
		
		# check if metric is the identity matrix, otherwise do symmetric orthonormalization
		dimx,dimy = self.metric.shape
		if dimx != dimy:
			raise RennerError("Metric of Hamiltonian is not square!")
		
		if np.linalg.norm(np.identity(dimx)-self.metric) > 1.0e-10:
			print("Basis is not orthonormal. Using symmetric orthonormalization before diagonalization.")
			
			Seigvals,Umat = np.linalg.eigh(self.metric)
			if np.min(Seigvals) < 1.0e-10:
				print("Basis is about to be linearly dependent!")
				print("Metric eigenvalues:")
				print(Seigvals)
			SmatInvSqrtDiag = np.diag(1.0/np.sqrt(Seigvals))
			if self.N_states > 1:
				print("Waring: the developer is not sure if it is a good idea to use a not orthonormal basis to calculate couplings between several RT states! The code might be wrong here...")
			self.Cmat = np.kron(np.identity(self.N_states),Umat.dot(SmatInvSqrtDiag).dot(np.transpose(Umat)))
			#self.Amat = self.Cmat.dot(np.transpose(self.Cmat))
			HmatOrth = np.transpose(self.Cmat).dot(self.Hmat).dot(self.Cmat)
		else:
			HmatOrth = self.Hmat
			self.Cmat = np.identity(dimx*self.N_states)
			#self.Amat = np.identity(dimx*self.N_states)
		
		# compute the eigenvalues and eigenvectors
		if self.verbose > 0:
			print("Diagonalizing the {0}x{0}-Hamiltonian... ".format(len(HmatOrth)), end='', flush=True)
		self.Heigvals,WmatOrth = np.linalg.eigh(HmatOrth)
		self.Wmat = self.Cmat.dot(WmatOrth)
		if self.verbose > 0:
			print("done.", flush=True)
		
		return (self.zpve*self.Heigvals,self.Wmat)

