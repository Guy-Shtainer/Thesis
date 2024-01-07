#!/usr/bin/env python
# coding: utf-8

# ## Imports and constants

# In[1]:


import snapshotsClass as ga
import Scripts as sc
import Graphs as gr
from scipy.interpolate import griddata
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import multiprocessing
import psutil
import time as ti
import operator
import math

#lest test this thing
#well well this is a better thing to do !!

num_of_total_snapshots = 400
h = 1


# out_put_location = '/shared/vc150_Rs0_Mdot745_Rcirc10/output'
out_put_location = '/shared/vc150_Rs3_Mdot7470_Rcirc10/output'
os.chdir(out_put_location)

def main():
    r0 = 0.1
    rN = 1e3
    dr = 0.1
    Zmin = -5
    Zmax = 5
    dz = 0.1
    r = sc.rRange(r0, rN, dr)  # in [kpc]
    z = sc.zRange2(Zmin, Zmax, dz / 10)
    P,DP,T,DT,SFR,time = sc.spherical_P_and_T_SFRs(188,r0,rN,dr)



def parallel_analysis(k):
    """
    Since the analysis can take **a lot** of time its smart to save the analysis files.
    The name is informative of course. It contains the type of analysis and the parameters.
    It can overwrite existing files so make sure to change "k" accordingly.
    The list is: **k=0**: "S" - Spherical, **k=1**: "C" - Cylindrical, **k=2**: "CZ" - Cylindrical with Z bins.
    :param k:
    :return: Ps,DPs,Ts,DTs,SFRs,Time and maybe snapPoints
    """

    Ps = []
    DPs = []
    Ts = []
    DTs = []
    SFRs = []
    Time = []
    if k == 0 or k == 1:
        args = [(i, r0, rN, dr) for i in range(num_of_total_snapshots)]
        start_time = ti.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            if k == 0:
                results = pool.starmap(sc.spherical_P_and_T_SFRs, args)
            elif k == 1: results = pool.starmap(sc.spherical_P_and_T_SFRs, args)

            for aveP, DP, aveT, DT, SFR, time in results:
                Ps.append(aveP)  # in [Pa]
                DPs.append(DP)  # in [Pa]
                Ts.append(aveT)
                DTs.append(DT)
                SFRs.append(SFR)  # in [Msun/yr]
                Time.append(time)  # in [Gyrs]

        end_time = ti.time()
        print((end_time - start_time) / 60)
        Ps = np.array(Ps)
        DPs = np.array(DPs)
        Ts = np.array(Ts)
        DTs = np.array(DTs)
        SFRs = np.array(SFRs)
        Time = np.array(Time)
        return Ps,DPs,Ts,DTs,SFRs,Time
    elif k == 2:
        snapPoints = []
        args = [(i, r0, rN, dr, Zmin, Zmax, dz) for i in range(num_of_total_snapshots)]
        start_time = ti.time()
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(sc.cylindrical_P_and_T_SFRsWithZ_R, args)
            for aveP, DP, aveT, DT, points, SFR, time in results:
                Ps.append(aveP)  # in [Pa]
                DPs.append(DP)  # in [Pa]
                Ts.append(aveT)
                DTs.append(DT)
                snapPoints.append(points)
                SFRs.append(SFR)  # in [Msun/yr]
                Time.append(time)  # in [Gyrs]

            end_time = ti.time()
            print((end_time - start_time) / 60)
            Ps = np.array(Ps, dtype='object')
            DPs = np.array(DPs, dtype='object')
            Ts = np.array(Ts, dtype='object')
            DTs = np.array(DTs, dtype='object')
            snapPoints = np.array(snapPoints, dtype='object')
            SFRs = np.array(SFRs)
            Time = np.array(Time)
            return Ps,DPs,Ts,DTs,SFRs,Time,snapPoints



if __name__ == '__main__':
    main()


def save(k):
    # string = "[r0,rN,dr,Zmin,Zmax,dz] CZ"
    # S: "Spherical", C: "Cylindrical", CZ: "CylindricalWithZ"
    cases = ["S","C","CZ"]
    k = 0
    if k == 0 or k==1 :
        string = "[r0=" + str(r0) + ", rN=" + str(rN) + ", dr=" + str(dr) + "," + '] ' + cases[k]
        sc.save2(Ps,DPs,Ts,DTs,SFRs,Time,string)
    elif k == 2:
        string = "[r0=" + str(r0) + ", rN=" + str(rN) + ", dr=" + str(dr) + ", Zmin=" + str(Zmin) + ", Zmax=" + str(Zmax) + ", dz=" + str(dz) + '] ' + cases[k]
        sc.save(Ps,DPs,Ts,DTs,snapPoints,SFRs,Time,string)
    else: print("You didn't choose a right k so it didnt save at all!")

# ## Load
# 
# Choose the right "k" to load what you want

# In[4]:


cases = ["S","C","CZ","CZR"]
k = 0
if k ==0 or k==1:
    string = "[r0=" + str(r0) + ", rN=" + str(rN) + ", dr=" + str(dr) + "," + '] ' + cases[k] 
    Ps,DPs,Ts,DTs,SFRs,Time = sc.load2(string)
else:
    string = "[r0=" + str(r0) + ", rN=" + str(rN) + ", dr=" + str(dr) + ", Zmin=" + str(Zmin) + ", Zmax=" + str(Zmax) + ", dz=" + str(dz) + '] ' + cases[k] 
    Ps,DPs,Ts,DTs,snapPoints,tSFRs,Time = sc.load(string)


# In[112]:


# arry = [114]
arry = [187,188]
gr.plotAvePForArrSnaps(arry,Ps,r,Time,SFRs)


# In[121]:


fix,ax = plt.subplots(figsize=(17,13))
plt.plot(r,np.log10(P188))
print(P188[27:80])
print(r[27:80])


# In[127]:


fix,ax = plt.subplots(figsize=(17,13))
plt.loglog(r,P187)
print(P187[52:88])
print(r[52:88])


# In[28]:


P188,DP188,T188,DT188,SFR188,Time188 = spherical_P_and_T_SFRs(188,r0,rN,dr)


# In[29]:


P187,DP187,T187,DT187,SFR187,Time187 = spherical_P_and_T_SFRs(187,r0,rN,dr)


# In[6]:


log_P_of_t_at_r2 = np.concatenate([log_P_of_t_at_r[2:187],log_P_of_t_at_r[189:]])
# averagePressure3 = np.mean(np.power(10,log_P_of_t_at_r2))
averagePressure2 = [np.mean(np.power(10,log_P_of_t_at_r2[i*4:40+i*4])) for i in range(int((len(log_P_of_t_at_r2)-len(log_P_of_t_at_r2)%4-40)/4))]
tmp = [np.mean(np.power(10,log_P_of_t_at_r2[i*4:])) for i in range(int((len(log_P_of_t_at_r2)-len(log_P_of_t_at_r2)%4-40)/4),int((len(log_P_of_t_at_r2)-len(log_P_of_t_at_r2)%4)/4))]
averagePressure2 += tmp
averagePressure2 = np.array(averagePressure2)
r2 = np.concatenate([r[2:187],r[189:]])
Time2 = np.concatenate([Time[2:187],Time[189:]])
# print(averagePressure3)


# In[7]:


Time3 = [Time2[i*4] for i in range(int((len(log_P_of_t_at_r2)-len(log_P_of_t_at_r2)%4)/4))]


# In[5]:


log_P_of_t_at_r = sc.pressure_in_time_for_r(Ps,r,30)
# print(log_P_of_t_at_r)
# print(np.power(10,log_P_of_t_at_r))
averagePressure = np.mean(np.power(10,log_P_of_t_at_r))
print(averagePressure)


# In[54]:


fig,ax = plt.subplots(figsize =(10, 7))
plt.plot(Time2,log_P_of_t_at_r2)
plt.xlabel("Time [Gyr]",fontsize=17)
plt.ylabel("log(P(r=30kpc)) [Pa]",fontsize=17)


# In[35]:


fig,ax = plt.subplots(figsize =(10, 7))
plt.plot(Time3,np.log10(averagePressure2))
plt.title("Average pressure with 1Gyr window at r=30kpc over time",fontsize=17)
plt.xlabel("Time [Gyr]",fontsize=17)
plt.xticks(np.arange(0,max(Time3)+1,1))
plt.yticks(np.arange(min(np.log10(averagePressure2)),max(np.log10(averagePressure2)),0.02))
ax.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel("<P>(t) [Pa]",fontsize=17)


# In[8]:


fig,ax = plt.subplots(figsize =(10, 7))
Norm_P_of_t_at_r2 = []
# print(log_P_of_t_at_r2.shape)
# print(averagePressure2)
# print(np.power(10,log_P_of_t_at_r2[3*4:3*4+4]))
for i in range(len(averagePressure2)):
    tmp = averagePressure2[i:i+1]
#     print(tmp.shape)
    Norm_P_of_t_at_r2.append(np.power(10,log_P_of_t_at_r2[i*4:i*4+4])/tmp)
Norm_P_of_t_at_r2 = np.array(Norm_P_of_t_at_r2)
print(Norm_P_of_t_at_r2.shape)
Norm_P_of_t_at_r2 = Norm_P_of_t_at_r2.reshape((Norm_P_of_t_at_r2.shape[0]*4,))
plt.plot(Time2,Norm_P_of_t_at_r2-1)
plt.title("Normalized & Centered pressure at r=30kpc over time",fontsize=17)
plt.xlabel("Time [Gyr]",fontsize=17)
plt.xticks(np.arange(0,max(Time2)+1,1))
ax.tick_params(axis='both', which='major', labelsize=18)
plt.ylabel("P(t,r=30kpc)/<P> -1 [Pa]",fontsize=17)


# In[72]:


# y = log_P_of_t_at_r
y = Norm_P_of_t_at_r2-1
# y = (np.power(10,log_P_of_t_at_r2)/averagePressure2)
window = np.hamming(len(y))
y_windowed = y * window

fft = np.fft.fft(y_windowed)
psd = np.abs(fft)**2
freq = np.fft.fftfreq(len(y), d=Time[1]-Time[0])
fig,ax = plt.subplots(figsize =(10, 7))
plt.plot(freq, psd)
# print(psd)
# print(freq)
plt.xlabel('Frequency [1/Gyr]',fontsize=27)
plt.ylabel('Power spectral density',fontsize=27)
plt.xticks(np.arange(0,max(freq),1))
ax.tick_params(axis='both', which='major', labelsize=18)
# plt.axis([-10, 10, 0, 1e7])
plt.xlim(0,21)
plt.show()


# In[75]:


radii_range = [15,20,25,30,35,40,45,50]
log_P_of_t_at_r_arr = []
ave_P_t_arr = []
for R in radii_range:
    log_P_of_t_at_r = sc.pressure_in_time_for_r(Ps,r,R)
    log_P_of_t_at_r2 = np.concatenate([log_P_of_t_at_r[2:187],log_P_of_t_at_r[189:]])
    log_P_of_t_at_r_arr.append(log_P_of_t_at_r2)
    averagePressure = [np.mean(np.power(10,log_P_of_t_at_r2[i*4:40+i*4])) for i in range(int((len(log_P_of_t_at_r2)-len(log_P_of_t_at_r2)%4-40)/4))]
    tmp = [np.mean(np.power(10,log_P_of_t_at_r2[i*4:])) for i in range(int((len(log_P_of_t_at_r2)-len(log_P_of_t_at_r2)%4-40)/4),int((len(log_P_of_t_at_r2)-len(log_P_of_t_at_r2)%4)/4))]
    averagePressure += tmp
    averagePressure = np.array(averagePressure2)
    ave_P_t_arr.append(averagePressure)
# averagePressure3 = np.mean(np.power(10,log_P_of_t_at_r2))
r2 = np.concatenate([r[2:187],r[189:]])
Time2 = np.concatenate([Time[2:187],Time[189:]])
# print(averagePressure3)


# In[79]:


for i in range(len(radii_range)):
    fig,axes = plt.subplots(nrows=1,ncols=3,figsize =(7, 5))
    axes[0].plot(Time3,np.log10(ave_P_t_arr[i]))
    axes[0].title("Average pressure with 1Gyr window at r= " + str(radii_range[i]) + "kpc over time",fontsize=17)
    axes[0].xlabel("Time [Gyr]",fontsize=17)
    axes[0].xticks(np.arange(0,max(Time3)+1,1))
    axes[0].yticks(np.arange(min(np.log10(ave_P_t_arr[i])),max(np.log10(ave_P_t_arr[i])),0.02))
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].ylabel("<P>(t) [Pa]",fontsize=17)


# In[ ]:




