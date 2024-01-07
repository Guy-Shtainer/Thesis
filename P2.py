import MyAnalysis as ma
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing

out_put_location = '/shared/vc150_Rs0_Mdot745_Rcirc10/output'
os.chdir(out_put_location)

def function_to_run(i):
    snp = ma.Snapshot(i)
    nxtsnp = ma.Snapshot(i+1)
    [aveP,DP] = snp.avePressure2()
    if i == 0:
        SFR = nxtsnp.TSolarMass
    else:
        SFR = nxtsnp.TSolarMass - snp.TSolarMass
    time = snp.Time
    return (aveP,DP,SFR,time,i)

if __name__ == '__main__':
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.imap(function_to_run, range(400))
        
        for aveP,DP,SFR,time,i in results:
            print("snapshot number " + str(i) + " is at time " + str(time))