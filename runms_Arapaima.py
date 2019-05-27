#!/usr/bin/python

## Code that simulate genetic data for Perez et al. (2018) Arapaima...

## in order to use this code you have to have ms installed on your computer
## ms can be freely downloaded from:
## http://home.uchicago.edu/rhudson1/source/mksamples.html

import random
import os
import math
import shlex, subprocess
import numpy as np
from sklearn.neighbors import NearestNeighbors

def sort_min_diff(amat):
    '''this function takes in a SNP matrix with indv on rows and returns the same matrix with indvs sorted by genetic similarity.
    this problem is NP-hard, so here we use a nearest neighbors approx.  it's not perfect, but it's fast and generally performs ok.
    assumes your input matrix is a numpy array'''
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=0))
    return amat[v[1][smallest]]

def ms2nparray(xfile):
	g = list(output)
	k = [idx for idx,i in enumerate(g) if len(i) > 0 and i.startswith('//')]
	f = []
	for i in k:
	    L = g[i+4:i+110]
	    q = []
	    for i in L:
	    	i = [int(j) for j in list(i[0])]
	    	i = np.array(i, dtype=np.int8)
	        q.append(i)
	    q = np.array(q)
	    q = q.astype("int8")
	    f.append(np.array(q))
	return f



### variable declarations

#define the number of simulations
Priorsize = 2000

## nDNA sample size of Pop_Am.
nDNAPop1 = 48
## nDNA sample size of Pop_Codajas.
nDNAPop2 = 18
## nDNA sample size of Pop_ArTO.
nDNAPop3 = 40
## nDNA sample sizes (number of alleles).
nDNANsam = nDNAPop1+nDNAPop2+nDNAPop3
## create a file to store parameters and one to store the models
simModel1 = []
#simModel2 = []
simModel3 = []
simModel4 = []
parameters = file("parameters.txt","w")
models = file("models.txt","w")

### define default values for all parameters that are specific to a subset of the models
#Set founded population size ratio to 1, equal to the ancestral population.
FoundedSizeRatio = 1
#Set current population size ratio to 1, equal to the ancestral population.
GrowthRatio = 1


### One Pop Model
for i in range(Priorsize):
	### Define parameters
	## Ne prior following a uniform distribution from 1000 to 50000.Hrbek 2005 -> 150000 females
	Ne = random.uniform(1000, 300000)
	## mutation rate according to Vialle et al. (2018)
	mutrate =(1.250E-9)
	## use Ne and mutrate values to obtain theta (required by ms)
	Theta = 4*Ne*mutrate
	## divergence time prior following an uniform distribution from 100.000 to 1 million years ago
	AM_ArTO_DivTime = 0
	Cod_DivTime = 0
	## nDNA ms's command
	com=subprocess.Popen("./ms %d 300 -s 1 -t %f" % (nDNANsam, Theta), shell=True, stdout=subprocess.PIPE).stdout
	output = com.read().splitlines()
	simModel1.append(sort_min_diff(np.array(ms2nparray(output)).swapaxes(0,1).reshape(nDNANsam,-1)))

	## save parameter values and models
	parameters.write("%f\t%f\t%f\t%f\t%f\n" % (Ne, Cod_DivTime, AM_ArTO_DivTime, FoundedSizeRatio, GrowthRatio))
	models.write("1\n")

simModel1=np.array(simModel1)
simModel1=simModel1.swapaxes(1,2)
np.savez_compressed('simModel1.npz', simModel1=simModel1)
del(simModel1)

## Colonization ArTO->Am model
for i in range(Priorsize):

	### Define parameters
	## Ne prior following a uniform distribution from 1000 to 50000.Hrbek 2005 -> 150000 females
	Ne = random.uniform(1000, 300000)
	## mutation rate according to Vialle et al. (2018)
	mutrate =(1.250E-9)
	## use Ne and mutrate values to obtain the SSR theta (required by ms)
	Theta = 4*Ne*mutrate
	## divergence time prior following an uniform distribution from 100.000 to 1 million years ago
	Cod_DivTime = random.uniform(0, 200000)
	AM_ArTO_DivTime = random.uniform(Cod_DivTime, 2000000)

	## number of years per generation
	genlen = random.uniform(4,5)
	## use the DivTime in years to calculte divergence time in coalescent units (required by ms)
	coalAM_ArTO_DivTime = AM_ArTO_DivTime/(genlen*4*Ne)
	coalCod_DivTime = Cod_DivTime/(genlen*4*Ne)

	##Size of founded population
	FoundedSizeRatio = random.uniform(0.01, 0.1)
	##Ratio between the the current and the ancient population sizes
	GrowthRatio = random.uniform(0.1, 1)
	##Calculates the growth rate
	Growth = -(1/coalAM_ArTO_DivTime-coalCod_DivTime)*math.log((1/GrowthRatio)/(1/FoundedSizeRatio))

	## nDNA ms's command
	com=subprocess.Popen("./ms %d 300 -s 1 -t %f -I 3 %d %d %d -ej %f 2 1 -eg %f 1 %f -en %f 1 %f -ej %f 3 1" % (nDNANsam, Theta, nDNAPop1, nDNAPop2, nDNAPop3, coalCod_DivTime, coalCod_DivTime, Growth, coalAM_ArTO_DivTime, FoundedSizeRatio, coalAM_ArTO_DivTime), shell=True, stdout=subprocess.PIPE).stdout
	output = com.read().splitlines()
	simModel2.append(sort_min_diff(np.array(ms2nparray(output)).swapaxes(0,1).reshape(nDNANsam,-1)))

	## save parameter values
	parameters.write("%f\t%f\t%f\t%f\t%f\n" % (Ne, Cod_DivTime, AM_ArTO_DivTime, FoundedSizeRatio, GrowthRatio))
	models.write("2\n")

simModel2=np.array(simModel2)
simModel2=simModel2.swapaxes(1,2)
np.savez_compressed('simModel2.npz', simModel2=simModel2)
del(simModel2)

# Colonization Am->ArTO model
for i in range(Priorsize):

	### Define parameters
	## Ne prior following a uniform distribution from 1000 to 50000.Hrbek 2005 -> 150000 females
	Ne = random.uniform(1000, 300000)
	## mutation rate according to Vialle et al. (2018)
	mutrate =(1.250E-9)
	## use Ne and mutrate values to obtain the SSR theta (required by ms)
	Theta = 4*Ne*mutrate
	## divergence time prior following an uniform distribution from 100.000 to 1 million years ago
	Cod_DivTime = random.uniform(0, 200000)
	AM_ArTO_DivTime = random.uniform(Cod_DivTime, 2000000)

	## number of years per generation
	genlen = random.uniform(4,5)
	## use the DivTime in years to calculte divergence time in coalescent units (required by ms)
	coalAM_ArTO_DivTime = AM_ArTO_DivTime/(genlen*4*Ne)
	coalCod_DivTime = Cod_DivTime/(genlen*4*Ne)

	##Size of founded population
	FoundedSizeRatio = random.uniform(0.01, 0.1)
	##Ratio between the the current and the ancient population sizes
	GrowthRatio = random.uniform(0.1, 1)
	##Calculates the growth rate
	Growth = -(1/coalAM_ArTO_DivTime)*math.log((1/GrowthRatio)/(1/FoundedSizeRatio))

	## nDNA ms's command
	com=subprocess.Popen("./ms %d 300 -s 1 -t %f -I 3 %d %d %d -g 3 %f -ej %f 2 1 -en %f 3 %f -ej %f 3 1" % (nDNANsam, Theta, nDNAPop1, nDNAPop2, nDNAPop3, Growth, coalCod_DivTime, coalAM_ArTO_DivTime, FoundedSizeRatio, coalAM_ArTO_DivTime), shell=True, stdout=subprocess.PIPE).stdout
	output = com.read().splitlines()
	simModel3.append(sort_min_diff(np.array(ms2nparray(output)).swapaxes(0,1).reshape(nDNANsam,-1)))

	## save parameter values
	parameters.write("%f\t%f\t%f\t%f\t%f\n" % (Ne, Cod_DivTime, AM_ArTO_DivTime, FoundedSizeRatio, GrowthRatio))
	models.write("3\n")

simModel3=np.array(simModel3)
simModel3=simModel3.swapaxes(1,2)
np.savez_compressed('simModel3.npz', simModel4=simModel3)
