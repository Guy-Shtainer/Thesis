import Scripts as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

num_of_total_snapshots = 400
h = 1

def plotSFR(Time,SFRs):
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    plt.plot(Time,SFRs)
    plt.xlabel("Time [Gyrs]",fontsize = 28)
    plt.ylabel("SFR [$M_\odot$ $yr^{-1}$]",fontsize = 28)
    plt.show()

def plotAvePForNSnaps(N,Ps,r,Time,SFRs):
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    dt = round(num_of_total_snapshots/(N+1))
    snapshots = ""
    for i in range(1,N+1):
        plt.loglog(r,Ps[i*dt],label=str(round(Time[i*dt],2)) + " Gyrs, " +str(round(SFRs[i*dt],3))+ " SFR")
        snapshots += str(i*dt)+ ','
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel("$\mathit{log(<P(r)>)}$ [Pa]", fontsize = 28)
    plt.title("Log of Average Pressure of snapshots " + snapshots[:-1], fontsize = 28)
    plt.legend(fontsize = 28)

def plotAvePForArrSnaps(arr,Ps,r,Time,SFRs):
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    snapshots = ""
    for n in arr:
        plt.loglog(r,Ps[n],label=str(round(Time[n],2)) + " Gyrs, " +str(round(SFRs[n],3))+ " SFR")
        snapshots += str(n)+ ','
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel("$\mathit{log(<P(r)>)}$ [Pa]", fontsize = 28)
    plt.title("Log of Average Pressure of snapshots " + snapshots[:-1], fontsize = 28)
    plt.legend(fontsize = 28)

def plotDPOverAveP_tForNSnaps(N,Ps,DPs,r,Time,SFRs,n=5):
    """
    This function returns a garph of the variation of the pressure (r'$\Delta$P(r,t)') over the average on time of n snapshots
    with equael time steps, of N snapshots. The initial snapshot (number 0) is not shown.
    
    :param N: Must be smaller than number of total snapshots. The number of snapshots you are interested to see in the graph. The snapshots are with equal time steps.
    :param n: The number of snapshots that goes into the avreage of the pressure over time, also equal time steps.
    """
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    number_of_time_steps = n
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    dt = round(num_of_total_snapshots/(N+1))
    snapshots = ""
    for i in range(1,N+1):
        plt.loglog(r,DPs[i*dt]/newP,label=str(round(Time[i*dt],2)) + " Gyrs, " +str(round(SFRs[i*dt],3))+ " SFR")
        snapshots += str(i*dt)+ ','
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel(r"log($\frac{\Delta P(r,t)}{<P(r)>}$) [Pa]", fontsize = 28)
    plt.title("Log of variation in pressure of snapshots " + snapshots[:-1], fontsize = 28)
    plt.legend(fontsize = 28)

def plotDPOverAveP_tForArrSnaps(arr,Ps,DPs,r,Time,SFRs,n=5):
    """
    This function returns a garph of the variation of the pressure (r'$\Delta$P(r,t)') over the average on time of n snapshots
    with equael time steps, of snapshots in arr. The initial snapshot (number 0) is not shown.
    
    :param N: Must be smaller than number of total snapshots. The number of snapshots you are interested to see in the graph. The snapshots are with equal time steps.
    :param n: The number of snapshots that goes into the avreage of the pressure over time, also equal time steps.
    """
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    number_of_time_steps = n
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    snapshots = ""
    for n in arr:
        plt.loglog(r,DPs[n]/newP,label=str(round(Time[n],2)) + " Gyrs, " +str(round(SFRs[n],3))+ " SFR")
        snapshots += str(n)+ ','
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel(r"log($\frac{\Delta P(r,t)}{<P(r)>}$) [Pa]", fontsize = 28)
    plt.title("Log of variation in pressure of snapshots " + snapshots[:-1], fontsize = 28)
    plt.legend(fontsize = 28)

def plotAvePOverAveP_tForNSnaps(N,Ps,r,Time,SFRs,n=5):
    """
    This function returns a garph of the average pressure (<P(r)>_m) over the average on time of n snapshots
    with equael time steps, of N snapshots. The initial snapshot (number 0) is not shown.
    
    :param N: Must be smaller than number of total snapshots. The number of snapshots you are interested to see in the graph. The snapshots are with equal time steps.
    :param n: The number of snapshots that goes into the avreage of the pressure over time, also equal time steps.
    """
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    number_of_time_steps = n
    number_of_snapshots_to_analyze = N
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    snapshots = ""
    dt = round(num_of_total_snapshots/(number_of_snapshots_to_analyze+1))
    for i in range(1,number_of_snapshots_to_analyze+1):
        plt.loglog(r,Ps[i*dt]/newP,label=str(round(Time[i*dt],2)) + " Gyrs, " +str(round(SFRs[i*dt],3))+ " SFR")
        snapshots += str(i*dt)+ ','
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize = 28)
    plt.title("Log of Average Pressure of snapshots " + snapshots[:-1], fontsize = 28)
    plt.legend(fontsize = 20)
    

def plotAvePOverAveP_tForArrSnaps(arr,Ps,r,Time,SFRs,n=5):
    """
    This function returns a garph of the average pressure (<P(r)>_m) over the average on time of n snapshots
    with equael time steps, of snapshots in arr. The initial snapshot (number 0) is not shown.
    
    :param N: Must be smaller than number of total snapshots. The number of snapshots you are interested to see in the graph. The snapshots are with equal time steps.
    :param n: The number of snapshots that goes into the avreage of the pressure over time, also equal time steps.
    """
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    number_of_time_steps = n
    number_of_snapshots_to_analyze = len(list(arr))
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    snapshots = ""
    dt = round(num_of_total_snapshots/(number_of_snapshots_to_analyze+1))
    for n in arr:
        plt.loglog(r,Ps[n]/newP,label=str(round(Time[n],3)) + " Gyrs, " +str(round(SFRs[n],3))+ " SFR")
        snapshots += str(n)+ ','
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize = 28)
    plt.title("Log of Average Pressure of snapshots " + snapshots[:-1], fontsize = 28)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize = 20)

def plotSoloAvePOverAveP_tForNSnaps(N,Ps,r,Time,SFRs,n=5):
    number_of_time_steps = n
    number_of_snapshots_to_analyze = N
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    dt = round(num_of_total_snapshots/(number_of_snapshots_to_analyze+1))
    for i in range(1,number_of_snapshots_to_analyze+1):
        fig,ax = plt.subplots(figsize =(7, 5))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.loglog(r,Ps[i*dt]/newP,label=str(round(Time[i*dt],2)) + " Gyrs, " +str(round(SFRs[i*dt],3))+ " SFR")
        plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 18)
        plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize = 18)
        plt.title("Log of Average Pressure of snapshots " + str(i*dt), fontsize = 18)
        plt.legend(fontsize = 10)

def plotSoloAvePOverAveP_tForArrSnaps(arr,Ps,r,Time,SFRs,n=5):
    number_of_time_steps = n
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    for n in arr:
        fig,ax = plt.subplots(figsize =(7, 5))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.loglog(r,Ps[n]/newP,label=str(round(Time[n],2)) + " Gyrs, " +str(round(SFRs[n],3))+ " SFR")
        plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 18)
        plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize = 18)
        plt.title("Log of Average Pressure of snapshots " + str(n), fontsize = 18)
        plt.legend(fontsize = 10)
        
def plotSoloAvePOverForArrSnaps(arr,Ps,r,Time,SFRs):
    for n in arr:
        fig,ax = plt.subplots(figsize =(7, 5))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.loglog(r,Ps[n],label=str(round(Time[n],2)) + " Gyrs, " +str(round(SFRs[n],3))+ " SFR")
        plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 18)
        plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize = 18)
        plt.title("Log of Average Pressure of snapshots " + str(n), fontsize = 18)
        plt.legend(fontsize = 10)

def plotSoloDPOverAveP_tForNSnaps(N,Ps,DPs,r,Time,SFRs,n=5):
    number_of_time_steps = n
    number_of_snapshots_to_analyze = N
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    dt = round(num_of_total_snapshots/(number_of_snapshots_to_analyze+1))
    for i in range(1,number_of_snapshots_to_analyze+1):
        fig,ax = plt.subplots(figsize =(7, 5))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.loglog(r,DPs[i*dt]/newP,label=str(round(Time[i*dt],2)) + " Gyrs, " +str(round(SFRs[i*dt],3))+ " SFR")
        plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 18)
        plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize = 18)
        plt.title("Log of Average Pressure of snapshots " + str(i*dt), fontsize = 18)
        plt.legend(fontsize = 10)

def plotSoloDPOverAveP_tForArrSnaps(arr,Ps,DPs,r,Time,SFRs,n=5):
    number_of_time_steps = n
    number_of_snapshots_to_analyze = len(list(arr))
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    dt = round(num_of_total_snapshots/(number_of_snapshots_to_analyze+1))
    for n in arr:
        fig,ax = plt.subplots(figsize =(7, 5))
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        plt.loglog(r,DPs[n]/newP,label=str(round(Time[n],2)) + " Gyrs, " +str(round(SFRs[n],3))+ " SFR")
        plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 18)
        plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize = 18)
        plt.title("Log of Average Pressure of snapshots " + str(n), fontsize = 18)
        plt.legend(fontsize = 10)

def analyzeSingleSnapshotP(num,Ps,r,Time,SFRs,n=5):
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    number_of_time_steps = n
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    plt.loglog(r,Ps[num],label=str(round(Time[num],2)) + " Gyrs, " +str(round(SFRs[num]))+ " SFR")
    plt.loglog(r,newP,label='<P(r)> over t')
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel(r"log(<P(r)>) [Pa]", fontsize = 28)
    plt.title("Log of Average Pressure of snapshot " + str(num), fontsize = 28)
    plt.legend(fontsize = 20)

def analyzeSingleSnapshotDP(num,Ps,DPs,r,Time,SFRs,n=5):
    fig,ax = plt.subplots(figsize =(17, 13))
    ax.tick_params(axis='x', labelsize=24)
    ax.tick_params(axis='y', labelsize=24)
    number_of_time_steps = n
    newP,T = sc.averagePOverTime(Ps,number_of_time_steps)
    plt.loglog(r,DPs[num],label=str(round(Time[num],2)) + " Gyrs, " +str(round(SFRs[num]))+ " SFR")
    plt.loglog(r,newP,label='<P(r)> over t')
    plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize = 28)
    plt.ylabel(r"log(<P(r)>) [Pa]", fontsize = 28)
    plt.title("Log of Average Pressure of snapshot " + str(num), fontsize = 28)
    plt.legend(fontsize = 20)

def plotPressureCZArr(arr,Ps,DPs,snapPoints,r,z):
    if isinstance(arr,int):
        arr = [arr]
    else:
        for n in arr:
            aveP = Ps[n]
            DP = DPs[n]
            points = np.array(snapPoints[n])
            points = [np.log10(points[:0]),points[:,1]]
            # log_r = np.log10(r)
            # X, Y = np.meshgrid(log_r, z)
            X, Y = np.meshgrid(r, z)
            aveP_no_zeros = [x for x in aveP if x != 0]
            DP_no_zeros = [x for x in DP if x != 0]
            aveP2 = griddata(points, np.log10(aveP_no_zeros), (X, Y), method='cubic')
#             aveP2 = griddata(points, np.log10(aveP_no_zeros), (X, Y), method='cubic')
            # DP2 = griddata(points, np.log10(DP_no_zeros), (X, Y), method='cubic')
            fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 13))
            ax0.scatter(points[:, 0], points[:, 1], c=np.log10(aveP_no_zeros))
            ax0.set(title='Scattered data')
            contour = ax1.contourf(X, Y, aveP2)
            cbar = fig.colorbar(contour)
#             ax1.set(title='Interpolated grid')
            # plt.xlabel("$\mathit{log(r)}$ [kpc]", fontsize=18)
            # plt.ylabel(r"log($\frac{<P(r,t)>}{<P(r)>}$) [Pa]", fontsize=18)
            plt.title("Log of Average Pressure of snapshots " + str(n), fontsize=18)
            plt.show()