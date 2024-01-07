#!/usr/bin/env python
# coding: utf-8

# In[7]:


import MyAnalysis as ma
import numpy as np
import matplotlib.pyplot as plt
import os

out_put_location = '/shared/vc150_Rs0_Mdot745_Rcirc10/output'
os.chdir(out_put_location)


# In[8]:


# Ps = np.zeros(400)
# DPs = np.zeros(400)
# rs = []
# SFRs = np.zeros(400)
# Time = np.zeros(400)

Ps = []
DPs = []
rs = []
SFRs = []
Time = []
snplist = [ma.Snapshot(0),ma.Snapshot(1),ma.Snapshot(2),ma.Snapshot(3),ma.Snapshot(4)]
i = 5
# In[ ]:


# for i in range(400):
#     snp = ma.Snapshot(i)
#     [r,P,DP] = snp.avePressure()
#     Ps[i] = P
#     DPs[i] = DP
#     if len(rs) == 0:
#         rs = r
#     SFRs[i] = snp.diffSFR()
#     Time[i] = snp.time()

while i < 400:
    [r0,P0,DP0] = snplist[0].avePressure()
    [r1,P1,DP1] = snplist[0].avePressure()
    [r2,P2,DP2] = snplist[0].avePressure()
    [r3,P3,DP3] = snplist[0].avePressure()
    Ps.append([P0,P1,P2,P3])
    DPs.append([DP0,DP1,DP2,DP3])
    if len(rs) == 0:
        rs = r0
        SFR0 = snplist[1].TSolarMass
    else:
        SFR0 = snplist[1].TSolarMass - snplist[0].TSolarMass
    SFR1 = snplist[2].TSolarMass - snplist[1].TSolarMass
    SFR2 = snplist[3].TSolarMass - snplist[2].TSolarMass
    SFR3 = snplist[4].TSolarMass - snplist[3].TSolarMass
    SFRs.append([SFR0,SFR1,SFR2,SFR3])
    Time.append([snplist[0].Time,snplist[1].Time,snplist[2].Time,snplist[3].Time])
    snplist[0] = snplist[4]
    snplist[1] = ma.Snapshot(i)
    snplist[2] = ma.Snapshot(i+1)
    snplist[3] = ma.Snapshot(i+2)
    snplist[4] = ma.Snapshot(i+3)
    i += 4

# In[ ]:


plt.plot(SFRs,Time)

