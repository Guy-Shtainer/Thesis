#!/usr/bin/env python
# coding: utf-8

# In[3]:


# basic python libraries
import sys, os, time, importlib, glob, pdb
import matplotlib, pylab as pl, numpy as np
import matplotlib.pyplot as plt
import math

# useful scientific libraries
import h5py, astropy, scipy, scipy.stats

# some useful shortcuts
from astropy import units as un, constants as cons
from numpy import log10 as log
import yt
import yt.units as units

#homedir = os.getenv("HOME") + '/'
#projectdir = homedir + 'Dropbox/otherRepositories/radial_to_rotating_flows/'
#simname = 'm11_ad_25_lr_extralight_2_cooling_1'
#datadir = projectdir + 'data/' + simname + '/'

# some constants
X = 0.7  # hydrogen mass fraction
gamma = 5 / 3.  # adiabatic index
mu = 0.62  # mean molecular weight


class h5py_dic:
    """class for handling hdf5 files"""

    def __init__(self, fs, non_subhalo_inds=None):
        self.dic = {}
        self.fs = fs

    def __getitem__(self, k):
        if type(k) == type((None,)):
            particle, field = k
        else:
            particle = 'PartType0'
            field = k
        if (particle, field) not in self.dic:
            if particle in self.fs[0].keys():
                arr = self.fs[0][particle][field][...]
                for f in self.fs[1:]:
                    new_arr = f[particle][field][...]
                    arr = np.append(arr, new_arr, axis=0)
            else:
                arr = []
            self.dic[(particle, field)] = arr
            print('loaded %s, %s' % (particle, field))
        return self.dic[(particle, field)]


class Snapshot:
    """interface class for a single simulation snapshot"""
    zvec = np.array([0., 0., 1.])

    def __init__(self, fn):
        self.f = h5py.File(fn)
        self.dic = h5py_dic([self.f])
        self.iSnapshot = int(fn[-8:-5])

    def time(self):  # in Gyr
        return self.f['Header'].attrs['Time']

    def number_of_particles(self):
        return self.f['Header'].attrs['NumPart_ThisFile']

    def masses(self, iPartType=0):  # in Msun
        return 1e10 * self.dic[('PartType%d' % iPartType, 'Masses')]

    def coords(self, iPartType=0):  # in kpc
        # TODO: verify center is at (1500,1500,1500)
        #return self.dic[('PartType%d' % iPartType, 'Coordinates')] - np.array([1500, 1500, 1500])
        return self.dic[('PartType%d' % iPartType, 'Coordinates')]

    def vs(self):  # in km/s
        return self.dic[('PartType0', 'Velocities')]

    def Ts(self):  # in K
        epsilon = self.dic[('PartType0', 'InternalEnergy')][:]  # energy per unit mass
        return (un.km ** 2 / un.s ** 2 * cons.m_p / cons.k_B).to('K').value * (2. / 3 * mu) * epsilon

    def rs(self, iPartType=0):  # in kpc
        return ((self.coords(iPartType) ** 2).sum(axis=1)) ** 0.5

    def rhos(self):  # in g/cm^3
        return ((un.Msun / un.kpc ** 3).to('g/cm**3') *
                self.dic[('PartType0', 'Density')] * 1e10)

    def nHs(self):  # in cm^-3
        return X * self.rhos() / cons.m_p.to('g').value

    def cos_thetas(self):
        normed_coords = (self.coords().T / np.linalg.norm(self.coords(), axis=1)).T
        return np.dot(normed_coords, self.zvec)

    def vrs(self):
        vs = self.vs()
        coords = self.coords()
        return (vs[:, 0] * coords[:, 0] + vs[:, 1] * coords[:, 1] + vs[:, 2] * coords[:, 2]) / self.rs()

#Editable
#fn =r'C:\Users\Nadav\Dropbox\m11_ad_25_lr_extralight_2_cooling_1\snapshot_200.hdf5' #snapshot path
#snap=Snapshot(fn)
#coords = snap.coords()
#return the x,y,z coordinates seperately of a given coordinates list
def split_to_axis(coordinates):
    x_coords = []
    y_coords = []
    z_coords = []
    for i in range(len(coordinates)):
        x_coords.append(coordinates[i][0])
        y_coords.append(coordinates[i][1])
        z_coords.append(coordinates[i][2])
    return [x_coords,y_coords,z_coords]
#return the coordinates and indices of the particles who's r is (somewhat) the same
def find_same_r(coordinates,r,err):
    indices = []
    cords = []
    x,y,z = split_to_axis(coordinates)
    for i in range(len(coordinates)):
        if np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) < r + err and np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) > r - err :
            cords.append(coordinates[i])
            indices.append(i)
    return cords,indices
#return the coordinates and indices of the particles who's theta is (somewhat) the same
def find_same_theta(coordinates,theta,err):
    cords =[]
    indices = []
    x,y,z = split_to_axis(coordinates)
    for i in range(len(coordinates)):
        teta = math.acos(z[i] / np.sqrt(x[i] ** 2 + y[i] ** 2 + z[i] ** 2))
        if teta < theta + err and teta > theta - err:
            cords.append(coordinates[i])
            indices.append(i)
    return cords,indices
#calculates the phi angle of every particle in a given coordinates list
def calc_Phi(cords):
    phi = np.arctan2(cords[:,1],cords[:,0])
#    for cord in cords:
#        phi.append(math.atan2(cord[1],cord[0]))
    return phi
#return the intersections of two lists
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
#sort the first list with respect to the other
def sort_list(list1, list2):
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]
    return z
#calculate the central difference derivative of y with respect to x and "normalize" the derivative
def central_diff_der(y,x,max):
    der = []
    der.append((y[1]-y[0])/(x[1]-x[0]))
    for i in range(1,len(y)-1):
        der.append((y[i+1]-y[i-1])/(x[i+1]-x[i-1]))
    der.append((y[-1]-y[-2])/((x[-1]-x[-2])*max))
    return der
#calculates the r of a given coordinates list
def calc_r(cords):
    r =[]
    for cord in cords:
        r.append(math.sqrt(cord[0]**2 + cord[1]**2 + cord[2]**2))
    return r
def theta(cords):
    teta =[]
    for cord in cords:
        teta.append(math.acos(cord[2]/(math.sqrt(cord[0]**2 + cord[1]**2 + cord[2]**2))))
    return teta
def Vtheta(theta,phi,v):
    vtheta = []
    for i in range(len(theta)):
        vtheta.append(math.cos(theta[i])*math.cos(phi[i])*v[i,0]+math.cos(theta[i])*math.sin(phi[i])*v[i,1]-math.sin(theta[i])*v[i,2])
    return vtheta
def Vphi(theta,phi,v):
    vphi = []
    for i in range(len(theta)):
        vphi.append(-math.sin(phi[i])*v[i,0]+math.cos(phi[i])*v[i,1])
    return vphi
def vRs(Theta,v):
    vR = []
    for i in range(len(Theta)):
        vR.append(math.cos(Theta[i])*v[i,0]+math.sin(Theta[i])*v[i,1])
    return vR
def Rs(cords):
    R = []
    for cord in cords:
        R.append(math.sqrt(cord[0]**2 + cord[1]**2))
    return R
#def Mass_within(r):
#    DM_mass = snap.masses(1)[(snap.rs(1)<r)].sum()
#    GM_mass = snap.masses()[(snap.rs()<r)].sum()
#    S_mass = snap.masses(4)[(snap.rs(4)<r)].sum()
#    return DM_mass + GM_mass + S_mass
#def Vc(r):
#    G = 4.30091e-6 # in kpc*km^2/M_sun*s^2
#    return np.sqrt((G*Mass_within(r))/r)
#def for_which_r_circ(r,)
#def Rcirc(R,vphi):


# In[ ]:




