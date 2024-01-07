import h5py
import numpy as np
import scipy.constants as const
import sys
from astropy import units as un, constants as cons
import matplotlib.pyplot as plt
import matplotlib


# h = 0.7
h = 1
X = 0.7
mu = 0.62
R = const.R
kb = const.Boltzmann
Time_scale = 0.978

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
    
    def sphericalRadiuses(self): # in kpc
        # return np.array(self.gas['BH_Dist'])/h
        return np.linalg.norm(np.array(self.gas['Coordinates'])[:,:],axis = 1)/h
    
    def cylindricalRadiuses(self): # in kpc
        return np.linalg.norm(np.array(self.gas['Coordinates'])[:,:2],axis = 1)/h
    
    def gasPressure(self): #in J/m^3=Pa
        n = self.nHs()
        P = 2.3 * n * self.Temperature() * cons.k_B.value
        return P
