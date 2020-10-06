"""
	Class Definition for the Integral Factory
	of the

		#####################################
		#  2D Isotropic Harmonic Oscillator #
		#####################################

	in Polar Coordinates
"""

from Integrals.IntegralErrors import IntegralError
from multiprocessing import Pool
import math
import numpy as np
from scipy.special import poch
import os


"""
	Nomenclature:
	
	A   ... prefactor
	T   ... <phi_nl|nabla^2|phi_mk>
	Z1  ... <phi_nl|r^2|phi_mk>
	Z2  ... <phi_nl|exp(2 I theta)|phi_mk>
	Z3  ... <phi_nl|r*exp(I theta)|phi_mk>
	
	where phi_nl is a isotropic harmonic oscillator wave function in 2D in
	polar coordinates with its quantum numbers n (principal) and l (angular)
"""

class Isotropic2D:
	def __init__(self,bs_dict,numprocs=4):
		self.numprocs = numprocs
		
		try:
			self.nmax = bs_dict["num"]
		except KeyError as err:
			raise IntegralError("Input missing! {}".format(err))
		
		# generate valid quantum numbers for isotropic 2D HO
		# first a 1D list of tuples of two where n is the principal QN and l is the angular QN
		#   n = 0,1,2,3,...  and   l = -n,-n+2,...,n
		qn1D = [(n,l) for n in range(self.nmax+1) for l in range(-n,n+1,2)]
		self.N = len(qn1D)
		
		# then produce a 2D list of lists of four QN
		# this list is the 1D (flattened) representation of the later matrices
		self.qn2D = [list(i + j) for i in qn1D for j in qn1D]
		
		self.cs = min(2000,int(self.N**2/numprocs))        # rather crude determination of the chunksize, should be overhauled
		
		self.__avail_mats = {}
		if "file" in bs_dict:
			self.file = bs_dict["file"]
			
			if os.path.isfile(self.file):
				print("Loading isotropic 2D HO integrals")
				from_disk = np.load(self.file)
				self.__avail_mats = dict(from_disk)
	
	def __enter__(self):
		return self
	
	def __exit__(self, type, value, traceback):
		if hasattr(self,"file"):
			print("Saving isotropic 2D HO integrals")
			np.savez_compressed(self.file,**self.__avail_mats)
		
		if isinstance(value, AttributeError):
			print("Requested matrix element is not availalbe! {}".format(value))
			return True
	
	
	##############################
	# Matrix Element Definitions #
	##############################
	
	def A(self,p,q,k,G):
		pochp = poch(q+1,abs(k))
		pochq = poch(p+1,abs(k+G))
		return math.sqrt(min(pochp,pochq)/max(pochp,pochq))
	
	def Telem(self,args):
		n,l,m,k = args
		
		nm = max(n,m)
		
		if n < 0 or m < 0:
			return 0.0
		
		if abs(l) > n or abs(k) > m:
			return 0.0
		
		if not (l-n) % 2 == 0 or not (m-k) % 2 == 0:
			return 0.0
		
		if l == k:
			if n == m:
				return -(n+1)
			if abs(n-m) == 2:
				return -0.5 * math.sqrt(nm*nm - k*k)
		
		return 0.0
	
	def Z1elem(self,args):
		n,l,m,k = args
		
		nm = max(n,m)
		
		if n < 0 or m < 0:
			return 0.0
		
		if abs(l) > n or abs(k) > m:
			return 0.0
		
		if not (l-n) % 2 == 0 or not (m-k) % 2 == 0:
			return 0.0
		
		if l == k:
			if n == m:
				return (n+1)
			if abs(n-m) == 2:
				return -0.5 * math.sqrt(nm*nm - k*k)
		
		return 0.0
	
	def Z2elem(self,args):
		n,l,m,k = args
		
		p = (n - abs(k+2))/2
		q = (m - abs(k))/2
		
		if n < 0 or m < 0:
			return 0.0
		
		if abs(l) > n or abs(k) > m:
			return 0.0
		
		if not (l-n) % 2 == 0 or not (m-k) % 2 == 0:
			return 0.0
		
		if l == (k+2):
			Apqk = self.A(p,q,k,2)
			if m == n:
				if k == -1:
					return 1.0
				else:
					return -Apqk * max(p,q)
			if (m < n and (k+1) > 0) or (m > n and (k+1) < 0):
				return Apqk * abs(k+1)
		
		return 0.0
	
	def Z3elem(self,args):
		n,l,m,k = args
		
		A = np.sign(k-0.1)		# shift of 0.1 is necessary to have -1 for 0 as input
		
		if n < 0 or m < 0:
			return 0.0
		
		if abs(l) > n or abs(k) > m:
			return 0.0
		
		if not (l-n) % 2 == 0 or not (m-k) % 2 == 0:
			return 0.0
		
		if l == (k-1):
			if m == (n+1):
				return A * math.sqrt((m+k)/2)
			if m == (n-1):
				return -A * math.sqrt((n-l)/2)
		
		return 0.0
	
	
	####################
	# for testing only #
	####################
	
	def Z22elem(self,args):
		n,l,m,k = args
		
		p = (n - abs(k+4))/2
		q = (m - abs(k))/2
		ak = abs(k)
		ak4 = abs(k+4)
	
		if n < 0 or m < 0:
			return 0.0
		
		if abs(l) > n or abs(k) > m:
			return 0.0
		
		if not (l-n) % 2 == 0 or not (m-k) % 2 == 0:
			return 0.0
		
		if l == (k+4):
			if k == -2:
				if q == p:
					return 1.0
			
			Apqk = self.A(p,q,k,4)
			if k == -1:
				if q <= p:
					return Apqk * (ak+1)
				if q == (p+1):
					return -Apqk * q
			if k == -3:
				if p <= q:
					return Apqk * (ak4+1)
				if p == (q+1):
					return -Apqk * p
			
			d = abs(p-q) + 1
			if k >= 0:
				if q <= p:
					return Apqk * (d * (ak+1) * (ak+2) - 2*q*(ak+2))
				if q == (p+1):
					return -Apqk * 2 * q * (ak+2)
				if q == (p+2):
					return Apqk * (q-1) * q
			if k <= -4:
				if p <= q:
					return Apqk * (d * (ak4+1) * (ak4+2) - 2*p*(ak4+2))
				if p == (q+1):
					return -Apqk * 2 * p * (ak4+2)
				if p == (q+2):
					return Apqk * (p-1) * p
		
		return 0.0
	
	
	###################
	# Matrix creation #
	###################
	
	def Tmat(self):
		if not "Tmat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["Tmat"] = np.array(pool.map(self.Telem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["Tmat"]
	
	def R0mat(self):
		if not "R0mat" in self.__avail_mats:
			self.__avail_mats["R0mat"] = np.identity(self.N)
		
		return self.__avail_mats["R0mat"]
	
		
	def R2mat(self):
		if not "R2mat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["R2mat"] = np.array(pool.map(self.Z1elem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["R2mat"]
	
	def R4mat(self):
		if not "R4mat" in self.__avail_mats:
			r2 = self.R2mat()
			self.__avail_mats["R4mat"] = r2.dot(r2)
		
		return self.__avail_mats["R4mat"]
	
	def R6mat(self):
		if not "R6mat" in self.__avail_mats:
			r2 = self.R2mat()
			r4 = self.R4mat()
			self.__avail_mats["R6mat"] = r4.dot(r2)
		
		return self.__avail_mats["R6mat"]
	
	def R8mat(self):
		if not "R8mat" in self.__avail_mats:
			r2 = self.R2mat()
			r6 = self.R6mat()
			self.__avail_mats["R8mat"] = r6.dot(r2)
		
		return self.__avail_mats["R8mat"]
	
	def E2mat(self):
		if not "E2mat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["E2mat"] = np.array(pool.map(self.Z2elem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["E2mat"]
	
	def E4mat(self):
		if not "E4mat" in self.__avail_mats:
			e2 = self.E2mat()
			self.__avail_mats["E4mat"] = e2.dot(e2)
		
		return self.__avail_mats["E4mat"]
	
	def P1mat(self):
		if not "P1mat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["P1mat"] = np.array(pool.map(self.Z3elem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["P1mat"]
		
	def P2mat(self):
		if not "P2mat" in self.__avail_mats:
			r2 = self.R2mat()
			e2 = self.E2mat()
			self.__avail_mats["P2mat"] = r2.dot(e2)
		
		return self.__avail_mats["P2mat"]
	
	def P4mat(self):
		if not "P4mat" in self.__avail_mats:
			r4 = self.R4mat()
			e4 = self.E4mat()
			self.__avail_mats["P4mat"] = r4.dot(e4)
		
		return self.__avail_mats["P4mat"]
	
	
	"""
		Overlap Matrix Element for 2D isotropic HO
		
			  d_ d_ B_
			  \  \  \ 
		<mu|nu> = /_ /_ /_ c_mu,j^I c_j,nu^K
			  I  K  j
		
		|mu>, |nu>	... coupled 2D non-BO nuclear WFs
		d_mu, d_nu	... number of mixed states in H_diab for |mu>/|nu>
		B		... number of 2D-HO functions in basis expansion
				    Attention: has to be equal for both expansions!
		c_mu^I, c_nu^K	... I-th/K-th part of the expansion coefficient vector c_mu/c_nu
				    -> Number of elements in c_mu: d_mu * B
				    -> Number of elements in c_nu: d_nu * B
	"""
	def overlapElem(self,c_mu,c_nu,bs2_dict):
		try:
			nmax_bs2 = bs2_dict["num"]
		except KeyError as err:
			raise IntegralError("Input missing! {}".format(err))
		
		dim_mu = len(c_mu)
		dim_nu = len(c_nu)
		
		dim_bs1 = self.N
		dim_bs2 = int((nmax_bs2+1)*(nmax_bs2+2)/2)
		
		if dim_mu % dim_bs1 != 0:
			raise IntegralError("Dimension of the first wave function vector is inconsistent with given basis set size! dim(WFV)={} dim(BS)={}".format(dim_mu,dim_bs1))
		
		if dim_nu % dim_bs2 != 0:
			raise IntegralError("Dimension of the second wave function vector is inconsistent with given basis set size! dim(WFV)={} dim(BS)={}".format(dim_nu,dim_bs2))
		
		if dim_bs1 != dim_bs2:
			raise IntegralError("Different number of basis functions was used to create the first and second wave function vector! Unable to compute overlap. dim(BS1)={} dim(BS2)={}".format(dim_bs1,dim_bs2))
		
		num_basis_functions = dim_bs1
		d_mu = dim_mu // dim_bs1
		d_nu = dim_nu // dim_bs2
		
		C_mu_j = c_mu.reshape((d_mu,num_basis_functions)).transpose()
		C_nu_j = c_nu.reshape((d_nu,num_basis_functions)).transpose()
		
		state_overlap = np.transpose(C_mu_j).dot(C_nu_j)
		
		return state_overlap.sum()		# some kind of normalization is maybe needed here
							# either 1/num_elements or 1/sqrt(d_mu) * 1/sqrt(d_nu)

