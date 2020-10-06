"""
	Class Definition for the Integral Factory
	of the

		########################################
		#  1D Harmonic Oscillator Ground State #
		########################################

	aka Gauss Function
"""

from Integrals.IntegralErrors import IntegralError
from multiprocessing import Pool
import math
import numpy as np
import os

"""
	Nomenclature:
	
	S   ... <chi_pa|chi_qb>
	T   ... <chi_pa|nabla^2|chi_qb>
	Vn  ... <chi_pa|x^n|chi_qb>
	A   ... C.Ct = Ct.C = S^(-1)  (inverse of overlap matrix)
	
	where chi_pa is a normalized Gauss-type function centred at p with the exponent a. The
	basis {chi_pa} is NOT orthogonal.
"""


class Gauss1D:
	def __init__(self,bs_dict,numprocs=4):
		self.numprocs = numprocs
		self.start,self.end,self.num = self.__unpack_param(bs_dict)
		self.__last_bs2_start = 0
		self.__last_bs2_end = 0
		self.__last_bs2_num = -1
		
		qn1D = self.__get_qn1D(self.start,self.end,self.num)
		self.N = len(qn1D)
		self.qn2D = [list(i + j) for i in qn1D for j in qn1D]
		
		self.cs = min(2000,int(self.N**2/self.numprocs))        # rather crude determination of the chunksize, should be overhauled
		
		self.__avail_mats = {}
		if "file" in bs_dict:
			self.file = bs_dict["file"]
			
			if os.path.isfile(self.file):
				print("Loading gauss type 1D integrals")
				from_disk = np.load(self.file)
				self.__avail_mats = dict(from_disk)
		
		# this basis set is not orthonormal, thus the resolution of the identity needs the inverse of the metric: A = S^(-1)
		Seigvals,Umat = np.linalg.eigh(self.R0mat())
		if np.min(Seigvals) < 1.0e-10:
			print("Basis set is about to be linearly dependent!")
			print("Metric eigenvalues:")
			print(Seigvals)
		SmatInvDiag = np.diag(1.0/Seigvals)
		self.Amat = Umat.dot(SmatInvDiag).dot(np.transpose(Umat))
	
	
	def __enter__(self):
		return self
	
	
	def __exit__(self, type, value, traceback):
		if hasattr(self,"file"):
			print("Saving gauss type 1D integrals")
			np.savez_compressed(self.file,**self.__avail_mats)
		
		if isinstance(value, AttributeError):
			print("Requested matrix element is not availalbe! {}".format(value))
			return True
	
	
	def __unpack_param(self,bs_dict):
		try:
			start = bs_dict["start"]
			end = bs_dict["end"]
			num = bs_dict["num"]
		except KeyError as err:
			raise IntegralError("Input missing! {}".format(err))
		
		return (start,end,num)
	
	
	def __get_qn1D(self,start,end,num):
		# calculate a distant-dependent exponent, gauss functions intersect at 90% of their height
		dist = (abs(start)+abs(end))/(num-1.0)
		bs_exp = 0.8428841253 / (dist**2)
		
		qn1D = [(p,bs_exp) for p in np.linspace(start,end,num)]
		
		return qn1D
	
	
	##############################
	# Matrix Element Definitions #
	##############################
	
	def Selem(self,args):
		p,a,q,b = args
		
		assert a > 0 and b > 0, "Selem: Exponents are not greater than zero."
		
		ab = a*b
		k = a + b
		d = p - q
		#r = (a*p + b*q) / k
		
		return math.sqrt(2.0/k) * (ab)**0.25 * math.exp(-ab/(2.0*k) * d*d)
	
	
	def Telem(self,args):
		p,a,q,b = args
		
		assert a > 0 and b > 0, "Telem: Exponents are not greater than zero."
		
		ab = a*b
		k = a + b
		d = p - q
		#r = (a*p + b*q) / k
		
		return math.sqrt(2.0/k) * (ab**2.25)/(k*k) * (d*d - 1.0/a - 1.0/b) * math.exp(-ab/(2.0*k) * d*d)
	
	
	def V1elem(self,args):
		p,a,q,b = args
		
		assert a > 0 and b > 0, "V1elem: Exponents are not greater than zero."
		
		ab = a*b
		k = a + b
		d = p - q
		r = (a*p + b*q) / k
		
		return math.sqrt(2.0/k) * (ab)**0.25 * r * math.exp(-ab/(2.0*k) * d*d)
	
	
	def V2elem(self,args):
		p,a,q,b = args
		
		assert a > 0 and b > 0, "V2elem: Exponents are not greater than zero."
		
		ab = a*b
		k = a + b
		d = p - q
		r = (a*p + b*q) / k
		
		return math.sqrt(2.0/k) * (ab)**0.25 * ( r*r + 1.0/k )  * math.exp(-ab/(2.0*k) * d*d)
	
	
	def V4elem(self,args):
		p,a,q,b = args
		
		assert a > 0 and b > 0, "V4elem: Exponents are not greater than zero."
		
		ab = a*b
		k = a + b
		d = p - q
		r = (a*p + b*q) / k
		
		return math.sqrt(2.0/k) * (ab)**0.25 * ( r**4 + 6.0/k * r*r + 3.0/(k*k) )  * math.exp(-ab/(2.0*k) * d*d)
	
	
	def V7elem(self,args):
		p,a,q,b = args
		
		assert a > 0 and b > 0, "V7elem: Exponents are not greater than zero."
		
		ab = a*b
		k = a + b
		d = p - q
		r = (a*p + b*q) / k
		
		return math.sqrt(2.0/k) * (ab)**0.25 * ( r**7 + 21.0/k * r**5 + 105.0/(k*k) * r**3 + 105.0/(k**3) * r )  * math.exp(-ab/(2.0*k) * d*d)
	
	
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
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["R0mat"] = np.array(pool.map(self.Selem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["R0mat"]
	
	def R1mat(self):
		if not "R1mat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["R1mat"] = np.array(pool.map(self.V1elem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["R1mat"]
	
	def R2mat(self):
		if not "R2mat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["R2mat"] = np.array(pool.map(self.V2elem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["R2mat"]
	
	def R3mat(self):
		if not "R3mat" in self.__avail_mats:
			r1 = self.R1mat()
			r2 = self.R2mat()
			self.__avail_mats["R3mat"] = r2.dot(self.Amat).dot(r1)
		
		return self.__avail_mats["R3mat"]
	
	def R4mat(self):
		if not "R4mat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["R4mat"] = np.array(pool.map(self.V4elem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["R4mat"]
	
	def R5mat(self):
		if not "R5mat" in self.__avail_mats:
			r1 = self.R1mat()
			r4 = self.R4mat()
			self.__avail_mats["R5mat"] = r4.dot(self.Amat).dot(r1)
		
		return self.__avail_mats["R5mat"]
	
	def R6mat(self):
		if not "R6mat" in self.__avail_mats:
			r2 = self.R2mat()
			r4 = self.R4mat()
			self.__avail_mats["R6mat"] = r4.dot(self.Amat).dot(r2)
		
		return self.__avail_mats["R6mat"]
	
	def R7mat(self):
		if not "R7mat" in self.__avail_mats:
			with Pool(processes = self.numprocs) as pool:
				self.__avail_mats["R7mat"] = np.array(pool.map(self.V7elem, tuple(self.qn2D),chunksize=self.cs)).reshape((self.N,self.N))
		
		return self.__avail_mats["R7mat"]
	
	def R8mat(self):
		if not "R8mat" in self.__avail_mats:
			r1 = self.R1mat()
			r7 = self.R7mat()
			self.__avail_mats["R8mat"] = r7.dot(self.Amat).dot(r1)
		
		return self.__avail_mats["R8mat"]
	
	
	########################################################
	# Overlap Matrix for two different 1D Gauss basis sets #
	########################################################
	
	def __overlapGauss(self,bs2_dict):
		bs1qn1D = self.__get_qn1D(self.start,self.end,self.num)
		bs2qn1D = self.__get_qn1D(*(self.__unpack_param(bs2_dict)))
		
		N1 = len(bs1qn1D)
		N2 = len(bs2qn1D)
		
		qn2D = [list(i + j) for i in bs1qn1D for j in bs2qn1D]
		
		cs = min(2000,int(N1*N2/self.numprocs))        # rather crude determination of the chunksize, should be overhauled
		with Pool(processes = self.numprocs) as pool:
			self.__Qmat = np.array(pool.map(self.Selem, tuple(qn2D),chunksize=cs)).reshape((N1,N2))
		
		return self.__Qmat
	
	
	def overlapElem(self,c_mu,c_nu,bs2_dict):
		bs2_start,bs2_end,bs2_num = self.__unpack_param(bs2_dict)
		if self.__last_bs2_start != bs2_start or self.__last_bs2_end != bs2_end or self.__last_bs2_num != bs2_num:
			Qmat = self.__overlapGauss(bs2_dict)
			self.__last_bs2_start = bs2_start
			self.__last_bs2_end = bs2_end
			self.__last_bs2_num = bs2_num
		else:
			Qmat = self.__Qmat
		
		return np.transpose(c_mu).dot(Qmat).dot(c_nu)

