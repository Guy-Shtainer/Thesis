# math
import numpy as np
import math
import astropy
from astropy.coordinates import SkyCoord
import astropy.units as ua

# oparating system and times and file managing
import os
import time
import pickle

# better resource usage
import multiprocessing

# simulation file reading and analysis
import h5py 
import gizmo_analysis as gizmo
import halo_analysis as halo
import utilities as ut
import yt
import unyt as u
import re

# visuals
from PIL import Image
import ipywidgets as wg
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoLocator, MultipleLocator, MaxNLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact

#interpolation
from scipy.interpolate import griddata

# galaxy alignment and centering
import edens as ed

def _data_loading(simulation_galaxy,snapshots,output_directory):
    galaxy = simulation_galaxy.split("_res")[0]
    simulation_directory = "Sims/" + simulation_galaxy + "/output"

    ts = []
    try: 
        snapshot_times = np.load(output_directory + galaxy + '/snapshot_times.npy')
    except:
        snapshot_times = {}
    
    # loading simulation using yt
    for snapshot_num in snapshots:
        if galaxy == "m11q" or galaxy == "m12_elvis_RomeoJuliet" or galaxy == "m12_elvis_ThelmaLouise" or galaxy =="m12_elvis_RomulusRemus" or galaxy == "m12i" or galaxy == "m12b" or galaxy == "m12c" or galaxy == "m12f" or galaxy == "m12m" or galaxy == "m12r" or galaxy == "m12w" or galaxy == "m12z":
            directory_path = simulation_directory + "/snapdir_" + str(snapshot_num) + "/snapshot_" + str(snapshot_num) + ".0.hdf5"
        elif galaxy == "m11b" or galaxy == "m11d" or galaxy == "m11e" or galaxy == "m11h" or galaxy == "m11i":
            directory_path = simulation_directory + "/snapshot_" + str(snapshot_num) + ".hdf5"
        ts.append(yt.load(directory_path))
    # print(ts)
    for ds in ts:
        ad = ds.all_data()
        snapshot_num = int(ds.basename.split('.')[0].split('_')[1])
        snapshot_times[str(snapshot_num)] = round(float(ds.current_time.in_units("Gyr")), 1)        
        print(str(snapshot_times[str(snapshot_num)]) + "Gyrs for " + galaxy + " and snapshot: " + str(snapshot_num))
        # redshift = ds.parameters['Redshift']
    print("Saving snapshot_times for galaxy " + galaxy + ":")
    print(snapshot_times)
    np.save(output_directory + galaxy + '/snapshot_times.npy',snapshot_times)
    return ts,galaxy

def _sphere_creation(simulation_galaxy,snapshots,output_directory,fraction_from_Rvir = 0.3):
    ts,galaxy = _data_loading(simulation_galaxy,snapshots,output_directory)
    sps = [] # array of spheres
    simulation_directory = "Sims/" + simulation_galaxy
    
    # Ensure the directory exists, create it if necessary
    os.makedirs(output_directory + galaxy, exist_ok=True)
    main_halo_centers = {}
    main_halo_virial_radii = {}
    main_halo_stellar_masses = {}
    for i in range(len(ts)):
        ds = ts[i]
        snapshot_num = snapshots[i]
        redshift = redshift = round(ds.parameters['Redshift'],3)
        hal = halo.io.IO.read_catalogs('redshift', redshift, simulation_directory)
        redshift = round(ds.parameters['Redshift'],1)
        h = ds.cosmology.hubble_constant.value
        if galaxy == "m12_elvis_RomeoJuliet" or galaxy == "m12_elvis_ThelmaLouise" or galaxy =="m12_elvis_RomulusRemus":
            sps_list = []
            main_halo_centers_list = []
            main_halo_virial_radii_list = []
            main_halo_stellar_masses_list = []
            for j in range(2):
                tmp = []
                halo_num = str(j+1)
                if j == 0:
                    main_halo_index = hal['host.index'][0]
                    # print("main index 1: ",main_halo_index)
                if j == 1:
                    main_halo_index = hal['host2.index'][0]
                    # print("main index 2: ",main_halo_index)
                
                main_halo_center = np.array(hal['position'][main_halo_index]*h) #in kpc
                # print("halo center " + str(halo_num) + " : ", main_halo_center)
                main_halo_centers_list.append(main_halo_center)
                # np.save(output_directory + galaxy + "/"+ str(halo_num) + "_halo_center Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",main_halo_center)
                
                main_halo_virial_radius = hal['radius'][main_halo_index]
                main_halo_virial_radii_list.append(main_halo_virial_radius)
                
                # np.save(output_directory + galaxy + "/"+ str(halo_num) + "_Rvir Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",main_halo_virial_radius)
                # print("halo virial radius " + str(halo_num) + " : ", main_halo_virial_radius)
                
                main_halo_stellar_mass = hal['star.mass'][main_halo_index]
                main_halo_stellar_masses_list.append(main_halo_stellar_mass)
                # print("halo stellar mass " + str(halo_num) + " : ", main_halo_stellar_mass)
                # np.save(output_directory + galaxy + "/"+ str(halo_num) + "_stellar_mass Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",main_halo_stellar_mass)
            
                radius = 0.2*main_halo_virial_radius
                sp = ds.sphere(main_halo_center,(radius,'kpc'))
                sps_list.append(sp)
            main_halo_centers[str(snapshot_num)] = main_halo_centers_list
            main_halo_virial_radii[str(snapshot_num)] = main_halo_virial_radii_list
            main_halo_stellar_masses[str(snapshot_num)] = main_halo_stellar_masses_list
            sps.append(sps_list)
        
        else:
            main_halo_index = hal['host.index'][0]
        
            main_halo_center = np.array(hal['position'][main_halo_index]*h) #in kpc
            main_halo_centers[str(snapshot_num)] = main_halo_center
            # np.save(output_directory + galaxy + "/halo_center Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",main_halo_center)
            
            main_halo_virial_radius = hal['radius'][main_halo_index]
            main_halo_virial_radii[str(snapshot_num)] = main_halo_virial_radius
            # np.save(output_directory + galaxy + "/Rvir Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",main_halo_virial_radius)
            
            main_halo_stellar_mass = hal['star.mass'][main_halo_index]
            main_halo_stellar_masses[str(snapshot_num)] = main_halo_stellar_mass
            # np.save(output_directory + galaxy + "/stellar_mass Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",main_halo_stellar_mass)
        
            radius = 0.2*main_halo_virial_radius
            sp = ds.sphere(main_halo_center,(radius,'kpc'))
            sps.append(sp)
    np.save(output_directory + galaxy + "/halo_centers.npy",main_halo_centers)
    np.save(output_directory + galaxy + "/halo_virial_radii.npy",main_halo_virial_radii)
    np.save(output_directory + galaxy + "/halo_stellar_masses.npy",main_halo_stellar_masses)
    return sps,ts

def angular_momentum(simulation_galaxy,snapshots,output_directory,fraction_from_Rvir = 0.3):
    sps,ts = _sphere_creation(simulation_galaxy,snapshots,output_directory,fraction_from_Rvir = 0.3)
    L_disk_norms = {}
    galaxy = simulation_galaxy.split("_res")[0]
    search_args = dict(use_gas=False, use_particles=True, particle_type='PartType4') # making sure to use only star particles and mass-weighted
    if galaxy == "m12_elvis_RomeoJuliet" or galaxy == "m12_elvis_ThelmaLouise" or galaxy =="m12_elvis_RomulusRemus":
        for i in range(len(sps)):
            sp = sps[i]
            ds = ts[i]
            snapshot_num = snapshots[i]
            redshift = round(ds.parameters['Redshift'],1)
            L_disk_norms_list = []
            for j in range(2):
                search_args = dict(use_gas=False, use_particles=True, particle_type='PartType4') # making sure to use only star particles and mass-weighted
                L_disk = sp[j].quantities.angular_momentum_vector(**search_args)
                L_disk_norm = (L_disk/np.linalg.norm(L_disk)).tolist()
                L_disk_norms_list.append(L_disk_norm)
            L_disk_norms[str(snapshot_num)] = L_disk_norms_list
            # print(L_disk_norms)
    else:    
        for i in range(len(sps)):
            sp = sps[i]
            ds = ts[i]
            snapshot_num = snapshots[i]
            redshift = round(ds.parameters['Redshift'],1)
            search_args = dict(use_gas=False, use_particles=True, particle_type='PartType4') # making sure to use only star particles and mass-weighted
            L_disk = sp.quantities.angular_momentum_vector(**search_args)
            L_disk_norm = (L_disk/np.linalg.norm(L_disk)).tolist()
            L_disk_norms[str(snapshot_num)] = L_disk_norm
            # print(L_disk_norms)
    np.save(output_directory + galaxy + "/L_disk_norms.npy",L_disk_norms)
    return L_disk_norms,ts

def _save(p,property,output_directory,galaxy,direction,width,redshift,snapshot_num):
    p_pixel_value = np.array(p.frb.data[('gas', property)])
    np.save(output_directory + galaxy + "/" + direction + " " + property + "_W" + str(int(width)) + "_Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",p_pixel_value)

def plot_projection(simulation_galaxy,snapshots,output_directory,directions,property, fraction_from_Rvir = 0.3):
    galaxy = simulation_galaxy.split("_res")[0]
    
    # skipping prior calculations if possible
    try:
        L_disk_norms = np.load(output_directory + galaxy + "/L_disk_norms.npy",allow_pickle = True).item()
        
        # skipping snapshots that were projected already
        try:
            pixel_values = np.load(output_directory + galaxy + "/" + property + " pixel_values.npy",allow_pickle = True).item()
            finished_snapshot_calcs = [int(num) for num in list(pixel_values.keys())]
            # if galaxy == "m12_elvis_RomeoJuliet" or galaxy == "m12_elvis_ThelmaLouise" or galaxy =="m12_elvis_RomulusRemus":
            #     try:
            #         for snapshot_num in finished_snapshot_calcs:
            #             if 
            #             pixel_values_dict = pixel_values[str(snapshot_num)]
            #             print("manadged to try")
            #             directions_copy = directions
            #             finished_direction_calcs = [direction for direction in list(pixel_values_dict.keys())]
            #             directions = [direction for direction in directions_copy if direction not in finished_direction_calcs]
            #     except:    
            #         pixel_values_dict = {}
            # else:
            snapshots = [num for num in snapshots if num not in finished_snapshot_calcs]
            if len(snapshots) == 0:
                print("############################################# finished " + galaxy + " #############################################")
                os.makedirs(output_directory + " finished with "  + galaxy + " " + property,exist_ok = True)
                return 0
        except:
            pixel_values = {}
        ts,galaxy = _data_loading(simulation_galaxy,snapshots,output_directory)
    except FileNotFoundError: 
        L_disk_norms,ts = angular_momentum(simulation_galaxy,snapshots,output_directory,fraction_from_Rvir)
        pixel_values = {}

    # setting weight_field for integration for different properties
    if property == 'H_p0_number_density':
        weight_field = None
    elif property == 'temperature':
        weight_field = ("gas", "density")

    print('reached ' + property + '_projections for ' + galaxy)
    for i in range(len(snapshots)):
        ds = ts[i]
        snapshot_num = snapshots[i]
        redshift = round(ds.parameters['Redshift'],1)
        # try:
        #     pixel_values_dict = pixel_values[str(snapshot_num)]
        #     finished_direction_calcs = [direction for direction in list(pixel_values_dict.keys())]
        #     directions = [direction for direction in directions_copy if direction not in finished_direction_calcs]
        # except:    
        #     pixel_values_dict = {}
        pixel_values_dict = {}
        if galaxy == "m12_elvis_RomeoJuliet" or galaxy == "m12_elvis_ThelmaLouise" or galaxy =="m12_elvis_RomulusRemus":
            for direction in directions:
                pixel_values_list = []
                for j in range(2):
                    halo_num = j+1
                    north_vector = None # will stay None for FaceOn and will become L_disk_norm for EdgeOn since this is the north_vector direction
                    norm = L_disk_norms[str(snapshot_num)][j] # direction of integration
                    
                    # loading fields from memory
                    main_halo_virial_radius = np.load(output_directory + galaxy + "/halo_virial_radii.npy", allow_pickle = True).item()[str(snapshot_num)][j]
                    main_halo_center = np.load(output_directory + galaxy + "/halo_centers.npy", allow_pickle = True).item()[str(snapshot_num)][j]
                    
                    # settings for ProjectionPlot
                    width = 2*fraction_from_Rvir*main_halo_virial_radius
                    cell_size = width/1000
                    num_cells = int(width/cell_size)
                    adj_width = cell_size*num_cells
                    fields=('gas', property)
                    center = main_halo_center

                    if direction == "EdgeOn":
                        arbitrary_vector = [1,1,3]
                        norm = np.cross(norm,arbitrary_vector) # projection on Edge
                        north_vector =  L_disk_norms[str(snapshot_num)][j]
                    p = yt.ProjectionPlot(ds, norm, fields=fields,center = center, width= adj_width, weight_field = weight_field, buff_size = (num_cells+1,num_cells+1),north_vector = north_vector)
                    pixel_values_list.append(np.array(p.frb.data[('gas', property)]))
                pixel_values_dict[direction] = pixel_values_list
                
                    # _save(p,property,output_directory,galaxy,str(halo_num) + "_" + direction,width,redshift,snapshot_num)
            pixel_values[str(snapshot_num)] = pixel_values_dict
        else:    
            snapshot_num = snapshots[i]
            # print("thats string of snapsot_num: " + str(snapshot_num))
            # print(type(snapshot_num))
            north_vector = None # will stay None for FaceOn and will become L_disk_norm for EdgeOn since this is the north_vector direction
            norm = L_disk_norms[str(snapshot_num)] # direction of integration

            # loading fields from memory
            main_halo_virial_radius = np.load(output_directory + galaxy + "/halo_virial_radii.npy", allow_pickle = True).item()[str(snapshot_num)]
            main_halo_center = np.load(output_directory + galaxy + "/halo_centers.npy", allow_pickle = True).item()[str(snapshot_num)]
    
            # settings for ProjectionPlot
            width = 2*fraction_from_Rvir*main_halo_virial_radius
            cell_size = width/1000
            num_cells = int(width/cell_size)
            adj_width = cell_size*num_cells
            fields=('gas', property)
            center = main_halo_center
            # print("for galaxy: " + galaxy + " its center is: " + str(center))
            
            pixel_values_dict = {}
            for direction in directions:
                if direction == "EdgeOn":
                    arbitrary_vector = [1,1,3]
                    norm = np.cross(norm,arbitrary_vector) # projection on Edge
                    north_vector = L_disk_norms[str(snapshot_num)]
                p = yt.ProjectionPlot(ds, norm, fields=fields,center = center, width= adj_width, weight_field = weight_field, buff_size = (num_cells+1,num_cells+1),north_vector = north_vector)
                pixel_values_dict[direction] = np.array(p.frb.data[('gas', property)])
            pixel_values[str(snapshot_num)] = pixel_values_dict
                    # _save(p,property,output_directory,galaxy,str(halo_num) + "_" + direction,width,redshift,snapshot_num) 
        np.save(output_directory + galaxy + "/" + property + " pixel_values.npy",pixel_values)
    print("############################################# finished " + galaxy + " #############################################")
    os.makedirs(output_directory + " finished with "  + galaxy + " " + property,exists_ok = True)