#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import scipy.constants as const
import sys
from astropy import units as un, constants as cons
import matplotlib.pyplot as plt
import matplotlib
import operator
import math

h = 0.7
X = 0.7
mu = 0.62
R = const.R
kb = const.Boltzmann
Time_scale = 0.978


# In[3]:


class Snapshot:
    def __init__(self,snapshotNum):
        self.num = snapshotNum
        f = h5py.File('snapshot_' + str(snapshotNum).zfill(3) + '.hdf5','r')
        self.header = f['Header']
        self.headerattrs = self.header.attrs
        self.gas = f['PartType0']
        self.dum1 = f['PartType2']
        self.dum2 = f['PartType3']
        #print("So far this is snapshot number: " + str(snapshotNum))
        if snapshotNum != 0:
            self.stars = f['PartType4']
            self.TSolarMass = np.sum(self.stars['Masses'])*(1e10/h) #in Msun
        self.BH = f['PartType5']
        self.Time = self.headerattrs['Time']*Time_scale #in Gyrs
        
    def rhos(self):  # in g/m^3
        return ((un.Msun / un.kpc ** 3).to('g/m**3') *
                np.array(list(self.gas['Density'])) * 1e10)

    def nHs(self):  # in m^-3
        return X * self.rhos() / cons.m_p.to('g').value
    
    def Temperature(self):  # in K
        internalEnergy = self.gas['InternalEnergy']  # energy per unit mass
        return (un.km ** 2 / un.s ** 2 * cons.m_p / cons.k_B).to('K').value * (2. / 3 * mu) * internalEnergy

#     def totalSolarMass(self):
#         return np.sum(self.stars['Masses'])
    
    def diffSFR(self):
        try:
            nxtsnp = Snapshot(self.num + 1)
        except FileNotFoundError:
                print("The SFR is calculated using the total solar mass of the next snapshot, there for if try to determine the SFR of the last snapshot you cant. Will return 0")
                return 0
        if (self.num ==0):
            return Snapshot(1).TSolarMass
        return nxtsnp.TSolarMass - self.TSolarMass
    
    def totalSFR(self):
        totalSFR = 0
        for i in range(len(self.gas['StarFormationRate'])):
            if self.gas['StarFormationRate'][i] > 0:
                totalSFR += self.gas['StarFormationRate'][i]
        return totalSFR
    
    
    def sfr(self):
        sfr = {}
        for i in range(len(self.gas['StarFormationRate'])):
            if self.gas['StarFormationRate'][i] > 0:
                sfr[str(self.gas['ParticleIDs'][i])] = self.gas['StarFormationRate'][i]
                print(i)   
        return sfr
                
    def sfr2(self):
        for id in self.gas["ParticleIDs"]:
            if self.gas['StarFormationRate'][id] > 0:
                print('Particle number: ' + str(id))
                print("Particle SFR: "+ str(self.gas['StarFormationRate'][id])+ ' MSun per year')
    
    def spericalRadiuses(self): # in kpc
        return np.array(self.gas['BH_Dist'])*0.7
    
    def cylindricalRadiuses(self): # in kpc
        return np.linalg.norm(np.array(self.gas['Coordinates'])[:,:2],axis = 1)*0.7
    
    
    def gasPressure(self): #in J/m^3=Pa
        n = self.nHs()
        P = 2.3 * n * self.Temperature() * cons.k_B.value
        return P

    
    def fixPressures(self,aveP,DP):
        #print(self.num)
        N = len(aveP)
        k = 0
        for j in range(N):
            if aveP[j] == 0:
                k += 1
                print("count is: " + str(k))
                i = 0
                if j == N-1:
                    aveP[j] = aveP[j-1]/2
                    DP[j] = DP[j-1]/2
                else:
                    try: 
                        while aveP[j+i] == 0:
                            i += 1
                        if j-i<0:
                            aveP[j] = np.mean(aveP[:j+i+1])
                            DP[j] = np.mean(DP[:j+i+1])
                        else: 
                            aveP[j] = np.mean(aveP[j-i:j+i+1])
                            DP[j] = np.mean(DP[j-i:j+i+1])
                    except IndexError:
                        print(i)
                        for l in range(i):
#                             if i == N
                            aveP[j+l] = np.mean(aveP[j-1:j+l+1])
                            DP[j+l] = np.mean(DP[j-1:j+l+1])
                    
            
    
    
    def fixPressuresWithZ(self,aveP,DP):
        #print(self.num)
        N = len(aveP[0])
        print("N is: " + str(N))
        for j in range(N):
            self.fixPressures(aveP[:,j],DP[:,j])
            
    
    def aveSphericalPressure(self,r0 = 0.1, rN = 1e3):
        # r0 in kpc
        # rN sn Mpc in kpc
        P = self.gasPressure()
        M = self.gas['Masses']
        radiuses = self.spericalRadiuses()
        aveP = []
        DP = []
        while r0 <= rN:
            dr = 0.1*r0
            indecies = np.where((radiuses>r0) & (radiuses<(r0+dr)))
            getter = operator.itemgetter(*list(indecies))
            if indecies[0].size == 0:
                aveP.append(0)
                DP.append(0)
            else: 
                aveP.append(np.average(getter(P),weights=getter(M)))
                DP.append(np.sqrt(np.mean((getter(P) - aveP[-1])**2)))
            r0 += dr
        self.fixPressures(aveP,DP)
        print("Finished with snapshot number: " + str(self.num))
        return (aveP,DP)
    
    def aveCylindricalPressure(self,r0 = 0.1, rN = 1e3):
        # r0 in kpc
        # rN is Mpc in kpc
        # dz in kpc
        P = self.gasPressure()
        M = self.gas['Masses']
        radiuses = self.cylindricalRadiuses()
        aveP = []
        DP = []
        while r0 <= rN:
            dr = 0.1*r0
            indecies = np.where((radiuses>r0) & (radiuses<(r0+dr)))
            getter = operator.itemgetter(*list(indecies))
            if indecies[0].size == 0:
                aveP.append(0)
                DP.append(0)
            else: 
                aveP.append(np.average(getter(P),weights=getter(M)))
                DP.append(np.sqrt(np.mean((getter(P) - aveP[-1])**2)))
            r0 += dr
        self.fixPressures(aveP,DP)
        print("Finished with snapshot number: " + str(self.num))
        return (aveP,DP)
   

    def aveCylindricalPressureWithZ(self,r0 = 0.1, rN = 50,dr = 0.1,Zmin = -5,Zmax = 5, dz = 0.01):
        # r0 in kpc
        # rN iמ kpc
        # dz in kpc
        P = self.gasPressure()
        M = self.gas['Masses']
        radiuses = self.cylindricalRadiuses()
        zeds = np.array(self.gas['Coordinates'])[:,2]
        numOfZBins = math.ceil(Zmax/dz)*2
#         print(Zmax/dz)
#         print(numOfZBins)
        aveP = []
        DP = []
        while r0 <= rN:
            Rindecies = np.where((radiuses>r0) & (radiuses<(r0*(1+dr))))
            Rgetter = operator.itemgetter(*list(Rindecies))
            if Rindecies[0].size == 0:
                print("NOT")
                aveP.append(np.zeros(numOfZBins))
                DP.append(np.zeros(numOfZBins))
            else:
                avePPos = []
                avePNeg = []
                DPPos = []
                DPNeg = []
#                 zedss = []
                z = 0
                while z<=Zmax:
                    ZindeciesPos = np.where((z<zeds) & (zeds<(z+dz)))
                    ZindeciesNeg = np.where((-z>zeds) & (zeds>-(z+dz)))
                    RZPos = np.intersect1d(ZindeciesPos[0],Rindecies[0]) # has the indicies of particles that are within the [r,r+dr] and [z,z+dz] ranges.
                    RZNeg = np.intersect1d(ZindeciesNeg[0],Rindecies[0]) # has the indicies of particles that are within the [r,r+dr] and [-z-dz,-z] ranges.
                    if RZPos.size ==0:
#                         print("NOT POS")
                        avePPos.append(0)
                        DPPos.append(0)
                    else:
                        RZgetterPos = operator.itemgetter(*list(RZPos))
#                         avePPos.append(np.log10(np.average(RZgetterPos(P),weights=RZgetterPos(M))))
#                         DPPos.append(np.log10(np.sqrt(np.mean((RZgetterPos(P) - avePPos[-1])**2))))
                        avePPos.append(np.average(RZgetterPos(P),weights=RZgetterPos(M)))
                        DPPos.append(np.sqrt(np.mean((RZgetterPos(P) - avePPos[-1])**2)))
                    if RZNeg.size == 0:
#                         print("NOT NEG")
                        avePNeg.append(0)
                        DPNeg.append(0)
                    else:
                        RZgetterNeg = operator.itemgetter(*list(RZNeg))
#                         avePNeg.append(np.log10(np.average(RZgetterNeg(P),weights=RZgetterNeg(M))))
#                         DPNeg.append(np.log10(np.sqrt(np.mean((RZgetterNeg(P) - avePNeg[-1])**2))))
                        avePNeg.append(np.average(RZgetterNeg(P),weights=RZgetterNeg(M)))
                        DPNeg.append(np.sqrt(np.mean((RZgetterNeg(P) - avePNeg[-1])**2)))
#                     zedss.append(z)
                    z += dz
                avePNeg.reverse()
                DPNeg.reverse()
                aveP.append(avePNeg+avePPos)
                DP.append(DPNeg+DPPos)
            r0 = r0*(1+dr)
#         Zrev = -zedss
#         Zrev.reverse()
        aveP = np.array(aveP)
        DP = np.array(DP)    
#         self.fixPressuresWithZ(aveP,DP)
        print("Finished with snapshot number: " + str(self.num))
        return (aveP,DP)
   

#                     print("Zindicies is: " + str(Zindecies[0]))
#                     print("Rindecies is: " + str(Rindecies[0]))
#                     print("Number of potential zeds: " + str(Zindecies[0].size))

#                     print("The number of particles both in r and z ranges is: " + str(RZ.size))


