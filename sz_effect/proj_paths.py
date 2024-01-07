import os

BASE = "/jonathan_storage/edengiladi/"
PROJNAME = "ksz_effect_project"
PROJDIR = os.path.join(BASE,PROJNAME)

PROJECTIONS = os.path.join(PROJDIR, "projections/")
ANALITIC_DATA = os.path.join(PROJECTIONS, "Analitical/data/")
ANALITIC_PLOTS = os.path.join(PROJECTIONS, "Analitical/plots/")
IDEAL_DATA = os.path.join(PROJECTIONS, "Idealized/data/")
IDEAL_PLOTS = os.path.join(PROJECTIONS, "Idealized/plots/")
COSMO_DATA = os.path.join(PROJECTIONS, "Cosmological/data/")
COSMO_PLOTS = os.path.join(PROJECTIONS, "Cosmological/plots/")

IDEAL_SNAPS = os.path.join(BASE, "Ideal_snapshots/")
SNAP71 = "snapshot_071.hdf5"
SNAP71_PATH = os.path.join(IDEAL_SNAPS, SNAP71)
SNAP49 = "snapshot_049.hdf5"
SNAP49_PATH = os.path.join(IDEAL_SNAPS, SNAP49)

SIMS_READ = os.path.join(BASE, 'shared/Sims/')
SIMS_WRITE = os.path.join(BASE, 'Sims_write/')
#"m12z_res4200", "m12w_res7100","m12r_res7100",

SIMNAMES = [ "m12m_res7100", "m12i_res7100", "m12f_res7100", "m12c_res7100", "m12b_res7100", "m12_elvis_ThelmaLouise_res4000", "m12_elvis_RomulusRemus_res4000", "m12_elvis_RomeoJuliet_res3500"]
SNAPS = ["snapdir_600", "snapdir_534", "snapdir_486"]

ALLSIMS_READ = []
ALLSNAPS_READ = []
ALLSNAPS_WRITE = []
for sim in SIMNAMES:
    ALLSIMS_READ.append(os.path.join(SIMS_READ, sim))
    for snap in SNAPS:
        ALLSNAPS_READ.append(os.path.join(SIMS_READ, sim, "output", snap))
        ALLSNAPS_WRITE.append(os.path.join(SIMS_WRITE, sim, snap))




