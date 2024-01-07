import snapshotsClass as ga
import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from scipy.interpolate import Akima1DInterpolator
import operator
import math
import matplotlib.pyplot as plt
from PIL import Image
import io
import multiprocessing

def spherical_P_and_T_SFRs(i,r0 = 0.1,rN=1e3,dr=0.1):
    snp = ga.Snapshot(i)
    nxtsnp = ga.Snapshot(i + 1)
    [aveP, DP,aveT,DT] = __aveSphericalPressure(i,r0,rN,dr)
    if i == 0:
        SFR = nxtsnp.TSolarMass / (nxtsnp.Time * 1e9)  # in [Msun/yr]
    else:
        SFR = (nxtsnp.TSolarMass - snp.TSolarMass) / ((nxtsnp.Time - snp.Time) * 1e9)  # in [Msun/yr]
    time = snp.Time
    return (aveP, DP,aveT,DT, SFR, time)

def cylindricalPressuresAndSFRs(i,r0,rN,dr):
    snp = ga.Snapshot(i)
    nxtsnp = ga.Snapshot(i + 1)
    [aveP, DP,aveT,DT] = __aveCylindricalPressure(i,r0,rN,dr)
    if i == 0:
        SFR = nxtsnp.TSolarMass / (nxtsnp.Time * 1e9)  # in [Msun/yr]
    else:
        SFR = (nxtsnp.TSolarMass - snp.TSolarMass) / ((nxtsnp.Time - snp.Time) * 1e9)  # in [Msun/yr]
    time = snp.Time
    return (aveP, DP,aveT,DT, SFR, time)


def cylindrical_P_and_T_SFRsWithZ_R(i, r0=0.1, rN=1e3, dr=0.1, Zmin=-5, Zmax=5, dz=0.1):
    snp = ga.Snapshot(i)
    nxtsnp = ga.Snapshot(i + 1)
    [aveP, DP,aveT,DT, points] = __aveCylindrical_P_and_T_WithZ_R(i, r0, rN, dr, Zmin, Zmax, dz)
    if i == 0:
        SFR = nxtsnp.TSolarMass / (nxtsnp.Time * 1e9)  # in [Msun/yr]
    else:
        SFR = (nxtsnp.TSolarMass - snp.TSolarMass) / ((nxtsnp.Time - snp.Time) * 1e9)  # in [Msun/yr]
    time = snp.Time
    return (aveP, DP,aveT,DT, points, SFR, time)

def rRange(r0, rN, dr):
    r = []
    while r0 <= rN:
        r.append(r0)
        r0 = r0 * (1 + dr)
    return np.array(r)

def rRange2(r0, rN, dr):
    rs = []
    r = r0
    while r <= rN:
        rs.append(r)
        r += dr
    return np.array(rs)


def zRange(Zmax, dz):
    zPos = np.arange(0, Zmax, dz)
    zNeg = np.flip(-zPos)
    return np.concatenate((zNeg, zPos), axis=0)


def zRange2(Zmin, Zmax, dz):
    zeds = []
    z = Zmin
    while z <= Zmax:
        zeds.append(z)
        z += dz
    return np.array(zeds)


def save(Ps, DPs,Ts,DTs, snapPoints, SFRs, Time, string):
    """
    This function saves the Ps,DPs,SFRs and Time outputs and saves them accorfing to k parameter.

    :param k:  k=0: "Spherical", k=1: "Cylindrical", k=2: "CylindricalWithZ"
    """
    np.save(string + "_Ps.npy", Ps)
    np.save(string + "_DPs.npy", DPs)
    np.save(string + "_Ts.npy", Ts)
    np.save(string + "_DTs.npy", DTs)
    np.save(string + "_snapPoints.npy", snapPoints)
    np.save(string + "_SFRs.npy", SFRs)
    np.save(string + "_Time.npy", Time)

def load(string):
    """
    This function saves the Ps,DPs,SFRs and Time outputs and saves them accorfing to k parameter.

    :param k:  k=0: "Spherical", k=1: "Cylindrical", k=2: "CylindricalWithZ"
    """
    Ps = np.load(string + "_Ps.npy",allow_pickle=True)
    DPs = np.load(string + "_DPs.npy", allow_pickle=True)
    Ts = np.load(string + "_Ts.npy", allow_pickle=True)
    DTs = np.load(string + "_DTs.npy", allow_pickle=True)
    snapPoints = np.load(string + "_snapPoints.npy", allow_pickle=True)
    SFRs = np.load(string + "_SFRs.npy")
    Time = np.load(string + "_Time.npy")
    return Ps, DPs,Ts,DTs, snapPoints, SFRs, Time


def save2(Ps, DPs,Ts,DTs, SFRs, Time, string):
    """
    This function saves the Ps,DPs,SFRs and Time outputs and saves them accorfing to k parameter.

    :param k:  k=0: "Spherical", k=1: "Cylindrical", k=2: "CylindricalWithZ"
    """
    np.save(string + "_Ps.npy", Ps)
    np.save(string + "_DPs.npy", DPs)
    np.save(string + "_Ts.npy", Ts)
    np.save(string + "_DTs.npy", DTs)
    np.save(string + "_SFRs.npy", SFRs)
    np.save(string + "_Time.npy", Time)

def load2(string):
    """
    This function saves the Ps,DPs,SFRs and Time outputs and saves them accorfing to k parameter.

    :param k:  k=0: "Spherical", k=1: "Cylindrical", k=2: "CylindricalWithZ"
    """
    Ps = np.load(string + "_Ps.npy")
    DPs = np.load(string + "_DPs.npy")
    Ts = np.load(string + "_Ts.npy")
    DTs = np.load(string + "_DTs.npy")
    SFRs = np.load(string + "_SFRs.npy")
    Time = np.load(string + "_Time.npy")
    return Ps, DPs,Ts,DTs, SFRs, Time


def pressure_in_time_for_r(Ps,radii,r):
    log_P_of_t_at_r = np.array(p_and_t_in_time_for_r_helper(Ps,radii,r)) # converts it into a numpy array
    return log_P_of_t_at_r

def temperature_in_time_for_r(Ts,radii,r):
    log_T_of_t_at_r = np.array(p_and_t_in_time_for_r_helper(Ts,radii,r)) # converts it into a numpy array
    return log_T_of_t_at_r

def p_and_t_in_time_for_r_helper(arrs,radii,r=30):
    log_radii = np.log10(radii)
    log_r= np.log10(r)
    results = [] # will be the values(P or T) over time at the relavant radius (r)
    for i in range(len(arrs)):
        log_vals = np.log10(arrs[i])
        akima_interp = Akima1DInterpolator(log_radii, log_vals)
        results.append(akima_interp(log_r))
    return results

# def interpolate_pressure_at_r_for_t(Ps,radii,)


def interestingSnaps(Ps, r, minr=0, scale=1, number_of_time_steps=5):
    interesting_snapshots = []
    idx = np.searchsorted(r, minr, side='right')
    newP, T = averagePOverTime(Ps, number_of_time_steps)
    newP = np.array(newP)
    for i in range(1,len(Ps)):
        if max(np.log10(Ps[i][idx:] / newP[idx:])) > scale:
            interesting_snapshots.append(i)
    return np.array(interesting_snapshots)


def averagePOverTime(Ps, N, num_of_total_snapshots=400):
    dt = round(num_of_total_snapshots / N)
    T = np.arange(dt, num_of_total_snapshots, dt)
    newP = np.mean(Ps[dt::dt], axis=0)
    return newP, T


def __fillZerosOld(aveP, DP):
    N = len(aveP)
    k = 0
    for j in range(N):
        if aveP[j] == 0:
            k += 1
            # print("count is: " + str(k))
            i = 0
            if j == N - 1:
                aveP[j] = aveP[j - 1] / 2
                DP[j] = DP[j - 1] / 2
            else:
                try:
                    while aveP[j + i] == 0:
                        i += 1
                    if j - i < 0:
                        aveP[j] = np.mean(aveP[:j + i + 1])
                        DP[j] = np.mean(DP[:j + i + 1])
                    else:
                        aveP[j] = np.mean(aveP[j - i:j + i + 1])
                        DP[j] = np.mean(DP[j - i:j + i + 1])
                except IndexError:
                    # print(i)
                    for l in range(i):
                        # if i == N
                        aveP[j + l] = np.mean(aveP[j - 1:j + l + 1])
                        DP[j + l] = np.mean(DP[j - 1:j + l + 1])

def findLastNone(vals):
    vals = np.array(vals)
    j = 0
    while np.isnan(vals[j]) and j<=vals.size:
        j += 1
    return j

def __fillZeros(vals, r):
    NZ_indecies = np.where(np.array(vals) != 0) # indecies where the valus are not zero
    # Z_indecies = np.where(np.array(vals) == 0) # indecies where the valus are... yes zero
    getter = operator.itemgetter(*list(NZ_indecies[0]))
    log_NZ_vals = np.log10(getter(vals))
    log_NZ_radii = np.log10(getter(r))
    akima_interp = Akima1DInterpolator(log_NZ_radii,log_NZ_vals) # interpolation for vals
    vals = np.power(10,akima_interp(np.log10(r)))
    if np.isnan(vals).any():
        firstNanIndex = np.isnan(vals).argmax()
        if firstNanIndex !=0:
            m,b = np.polyfit(np.log10(r[:firstNanIndex]),akima_interp(np.log10(r))[:firstNanIndex],1)
            vals[firstNanIndex:] = np.power(10,m * np.log10(r[firstNanIndex:]) + b)
        else:
            firstNNanIndex = findLastNone(vals)
            lastNNanIndex = np.isnan(vals[firstNNanIndex:]).argmax() + firstNNanIndex
            if lastNNanIndex == firstNNanIndex:
                m,b = np.polyfit(np.log10(r[firstNNanIndex:]),akima_interp(np.log10(r))[firstNNanIndex:],1)
                vals[:firstNNanIndex] = np.power(10,m * np.log10(r[:firstNNanIndex]) + b)
            else: 
                m,b = np.polyfit(np.log10(r[firstNNanIndex:lastNNanIndex]),akima_interp(np.log10(r))[firstNNanIndex:lastNNanIndex],1)
                vals[:firstNNanIndex] = np.power(10,m * np.log10(r[:firstNNanIndex]) + b)
                vals[lastNNanIndex:] = np.power(10,m * np.log10(r[lastNNanIndex:]) + b)
    return vals

def interpolate(aveP, DP, points, r, z):
    log_r = np.log10(r)
    X, Y = np.meshgrid((list(log_r)), z)
    aveP_no_zeros = aveP[aveP != 0]
    DP_no_zeros = DP[DP != 0]
    aveP2 = griddata(points, np.log10(aveP_no_zeros), (X, Y), method='cubic')
    DP2 = griddata(points, np.log10(DP_no_zeros), (X, Y), method='cubic')
    return aveP2, DP2


def __fillZerosWithZ(vals, r):
    N = len(vals[0])
    # print("N is: " + str(N))
    args = [(vals[:,j],r) for j in range(N)]
    result = []
    if __name__ == '__main__':
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.starmap(__fillZeros, args)
            for val in results:
                result += vals
    for j in range(N):
        __fillZeros(vals[:, j], r)


def averagePoverTimeWindows(Ps,r,radii_range,time_window_size = 40, time_resolution = 4):
    # time_window_size = 40  # its number of snapshots. 1 snp ~= 25Myr so 40 ~= 1Gyr
    # time_resolution = 4  # This is the propagation speed of the window. 1 means move window only 25Myrs (1 snapshot) for every average so 4 is about 0.1Gyrs.
    Norm_of_P_per_R_over_time = []
    for R in radii_range:
        log_P_of_t_at_R = pressure_in_time_for_r(Ps, r, R)
        n = len(log_P_of_t_at_R)

        # averaging using the time windows from beginning until where window ends at the end of the array
        average_P_over_time_windows_for_R = np.array(
            [np.mean(np.power(10, log_P_of_t_at_R[i * time_resolution:time_window_size + i * time_resolution])) for i in
             range(int((n - time_window_size - n % time_resolution) / time_resolution))])

        # averaging using the tie windows from where it stopped above, now naturally shrinking the time windows to end at the end of the array.
        tmp = np.array([np.mean(np.power(10, log_P_of_t_at_R[i * time_resolution:])) for i in
                        range(int((n - time_window_size - n % time_resolution) / time_resolution),
                              int((n - n % time_resolution) / time_resolution))])

        # concatenating them
        average_P_over_time_windows_for_R = np.concatenate([average_P_over_time_windows_for_R, tmp])

        # adding the averages to a list of <P(t,r=R)>
        # average_P_over_time_windows.append(average_P_over_time_windows_for_R)

        # normalizing each pressure with the time window its in
        Norm_of_P_for_R_over_time = np.array([np.power(10, log_P_of_t_at_R)[
                                              i * time_resolution:i * time_resolution + time_resolution] /
                                              average_P_over_time_windows_for_R[i:i + 1][0] for i in
                                              range(len(average_P_over_time_windows_for_R))]).flatten()
        Norm_of_P_per_R_over_time.append(Norm_of_P_for_R_over_time)
    Norm_of_P_per_R_over_time = np.transpose(Norm_of_P_per_R_over_time)
    return Norm_of_P_per_R_over_time

def multicore_gif(x,y,min_y,max_y,Time,j):
    fig,ax = plt.subplots(figsize =(10, 7));
    plt.plot(x,y,label="T=" + str(format(Time[j],".3f")) + " Gyrs");
    plt.title("Pressure Shockwave Propagation in Time",fontsize=17);
    plt.xlabel("Radius [Kpc]",fontsize=17);
    plt.ylabel("P(r,t=" + str(format(Time[j],".3f")) + " Gyr)/<P> -1 [Pa]",fontsize=17);
    plt.legend(fontsize = 18);
    plt.ylim(min_y,max_y);

    # Save the plot image in memory using BytesIO
    buffer = io.BytesIO()
    fig.savefig(buffer, format='jpg')
    buffer.seek(0)  # Move the buffer cursor to the beginning
    im = Image.open(buffer)

    plt.close()
    print("finished with " + str(j))
    return im

def __aveSphericalPressure(i,r0,rN,dr):
    # r0 in kpc
    # rN sn Mpc in kpc
    snp = ga.Snapshot(i)
    T = snp.Temperature()
    P = snp.gasPressure()
    M = snp.gas['Masses']
    radii = snp.sphericalRadiuses()
    aveP = []
    DP = []
    aveT = []
    DT = []
    r = []
    while r0 <= rN:
        r.append(r0)
        indecies = np.where((r0 < radii) & (radii <= r0*(1+dr)))
        getter = operator.itemgetter(*list(indecies))
        if indecies[0].size == 0:
            aveP.append(0)
            DP.append(0)
            aveT.append(0)
            DT.append(0)
        else:
            aveP.append(np.average(getter(P), weights=getter(M)))
            DP.append(np.sqrt(np.mean((getter(P) - aveP[-1]) ** 2)))
            aveT.append(np.average(getter(T), weights=getter(M)))
            DT.append(np.sqrt(np.mean((getter(T) - aveT[-1]) ** 2)))
        r0 = r0*(1+dr)
    aveP = __fillZeros(aveP,r)
    DP = __fillZeros(DP,r)
    aveT = __fillZeros(aveT,r)
    DT = __fillZeros(DT,r)
    print("Finished with snapshot number: " + str(i))
    return (aveP, DP,aveT,DT)


def __aveCylindricalPressure(i, r0, rN,dr):
    # r0 in kpc
    # rN is Mpc in kpc
    # dz in kpc
    snp = ga.Snapshot(i)
    T = snp.Temperature()
    P = snp.gasPressure()
    M = snp.gas['Masses']
    radii = snp.cylindricalRadiuses()
    aveP = []
    DP = []
    aveT = []
    DT = []
    r=[]
    while r0 <= rN:
        r.append(r0)
        dr = 0.1 * r0
        indecies = np.where((r0 < radii) & (radii <= r0*(1+dr)))
        getter = operator.itemgetter(*list(indecies))
        if indecies[0].size == 0:
            aveP.append(0)
            DP.append(0)
            aveT.append(0)
            DT.append(0)
        else:
            aveP.append(np.average(getter(P), weights=getter(M)))
            DP.append(np.sqrt(np.mean((getter(P) - aveP[-1]) ** 2)))
            aveT.append(np.average(getter(T), weights=getter(M)))
            DT.append(np.sqrt(np.mean((getter(T) - aveT[-1]) ** 2)))
        r0 = r0*(1+dr)
    aveP = __fillZeros(aveP,r)
    DP = __fillZeros(DP,r)
    aveT = __fillZeros(aveT,r)
    DT = __fillZeros(DT,r)
    print("Finished with snapshot number: " + str(i))
    return (aveP, DP,aveT,DT)


def __aveCylindrical_P_and_T_WithZ_R(i, r0=0.1, rN=50, dr=0.1, Zmin=-5, Zmax=5, dz=0.01):
    # r0 in kpc
    # rN iמ kpc
    # dz in kpc
    # Zmax in kpc
    snp = ga.Snapshot(i)
    P = snp.gasPressure()
    T = snp.Temperature()
    M = snp.gas['Masses']
    radii = snp.cylindricalRadiuses()
    zeds = np.array(snp.gas['Coordinates'])[:, 2]
    aveP = []
    DP = []
    aveT = []
    DT = []
    points = []
    r = []
    while r0 <= rN:
        r.append(r0)
        Rindecies = np.where((r0 < radii) & (radii <= r0*(1+dr)))
        if Rindecies[0].size == 0:
            print("snp " + str(i) + " with radius: " + str(r0))
        else:
            avePt = []
            DPt = []
            aveTt = []
            DTt = []
            pointst = []
            z = Zmin
            while z <= Zmax:
                Zindecies = np.where((z < zeds) & (zeds <= (z + dz)))
                RZ = np.intersect1d(Zindecies[0], Rindecies[
                    0])  # has the indicies of particles that are within the [r,r+dr] and [z,z+dz] ranges.
                if RZ.size == 0:
                    avePt.append(0)
                    DPt.append(0)
                    aveTt.append(0)
                    DTt.append(0)
                else:
                    RZgetter = operator.itemgetter(*list(RZ))
                    avePt.append(np.average(RZgetter(P), weights=RZgetter(M)))
                    DPt.append(np.sqrt(np.mean((RZgetter(P) - avePt[-1]) ** 2)))
                    aveTt.append(np.average(RZgetter(T), weights=RZgetter(M)))
                    DTt.append(np.sqrt(np.mean((RZgetter(T) - aveTt[-1]) ** 2)))
                    pointst.append([r0, z])
                z += dz
            points = points + pointst
            aveP = aveP + avePt
            DP = DP + DPt
            aveT = aveT + aveTt
            DT = DT + DTt
        r0 = r0*(1+dr)
    aveP = __fillZeros(aveP, r)
    DP = __fillZeros(DP, r)
    aveT = __fillZeros(aveT, r)
    DT = __fillZeros(DT, r)
    print("Finished with snapshot number: " + str(i))
    return (aveP, DP,aveT,DT, points)
