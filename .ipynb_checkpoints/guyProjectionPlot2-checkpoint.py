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
import numba
from numba import jit, prange

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

def _data_loading(simulation_galaxy,snapshot_num,output_directory,lock):
    galaxy = simulation_galaxy.split("_res")[0]
    simulation_directory = "Sims/" + simulation_galaxy + "/output"
    
    if galaxy == "m11q" or galaxy == "m12_elvis_RomeoJuliet" or galaxy == "m12_elvis_ThelmaLouise" or galaxy =="m12_elvis_RomulusRemus" or galaxy == "m12i" or galaxy == "m12b" or galaxy == "m12c" or galaxy == "m12f" or galaxy == "m12m" or galaxy == "m12r" or galaxy == "m12w" or galaxy == "m12z":
        directory_path = simulation_directory + "/snapdir_" + str(snapshot_num) + "/snapshot_" + str(snapshot_num) + ".0.hdf5"
    elif galaxy == "m11b" or galaxy == "m11d" or galaxy == "m11e" or galaxy == "m11h" or galaxy == "m11i":
        directory_path = simulation_directory + "/snapshot_" + str(snapshot_num) + ".hdf5"
    ds = yt.load(directory_path)

    with lock:
        try: 
            snapshot_times = np.load(output_directory + galaxy + '/snapshot_times.npy',allow_pickle = True).item()
        except Exception as e:
            print('when trying to load snapshot_times for galaxy: ' + galaxy)
            print(f"An error occurred: {e}")
            snapshot_times = {}
        snapshot_times[str(snapshot_num)] = round(float(ds.current_time.in_units("Gyr")), 1)        
        print(str(snapshot_times[str(snapshot_num)]) + "Gyrs for " + galaxy + " and snapshot: " + str(snapshot_num))
        # redshift = ds.parameters['Redshift']
        print("Saving snapshot_times for galaxy " + galaxy + ":")
        print(snapshot_times)
        np.save(output_directory + galaxy + '/snapshot_times.npy',snapshot_times)
    return ds

def _sphere_creation(simulation_galaxy,snapshot_num,output_directory,lock):
    galaxy = simulation_galaxy.split("_res")[0]
    ds = _data_loading(simulation_galaxy,snapshot_num,output_directory,lock)
    simulation_directory = "Sims/" + simulation_galaxy
    
    # Ensure the directory exists, create it if necessary
    os.makedirs(output_directory + galaxy, exist_ok=True)
    sps_list = []

    redshift = redshift = round(ds.parameters['Redshift'],3)
    hal = halo.io.IO.read_catalogs('redshift', redshift, simulation_directory)
    redshift = round(ds.parameters['Redshift'],1)
    h = ds.cosmology.hubble_constant.value
    num_of_halos = 1
    if 'elvis' in galaxy:
        num_of_halos = 2
    main_halo_centers_list = []
    main_halo_virial_radii_list = []
    main_halo_stellar_masses_list = []
    for j in range(num_of_halos):
        tmp = []
        halo_num = str(j+1)
        if j == 0:
            main_halo_index = hal['host.index'][0]
        if j == 1:
            main_halo_index = hal['host2.index'][0]
        
        main_halo_center = np.array(hal['position'][main_halo_index]*h) #in kpc
        main_halo_centers_list.append(main_halo_center)
        
        main_halo_virial_radius = hal['radius'][main_halo_index]
        main_halo_virial_radii_list.append(main_halo_virial_radius)

        
        main_halo_stellar_mass = hal['star.mass'][main_halo_index]
        main_halo_stellar_masses_list.append(main_halo_stellar_mass)
    
        radius = 0.2*main_halo_virial_radius
        sp = ds.sphere(main_halo_center,(radius,'kpc'))
        sps_list.append(sp)

    with lock:
        try:
            main_halo_centers = np.load(output_directory + galaxy + "/halo_centers.npy",allow_pickle=True).item
        except:
            main_halo_centers = {}
        main_halo_centers[str(snapshot_num)] = main_halo_centers_list
        np.save(output_directory + galaxy + "/halo_centers.npy",main_halo_centers)

        try:
            main_halo_virial_radii = np.load(output_directory + galaxy + "/halo_virial_radii.npy",allow_pickle=True).item
        except:
            main_halo_virial_radii = {}
        main_halo_virial_radii[str(snapshot_num)] = main_halo_virial_radii_list
        np.save(output_directory + galaxy + "/halo_virial_radii.npy",main_halo_virial_radii)

        try:
            main_halo_stellar_masses = np.load(output_directory + galaxy + "/halo_stellar_masses.npy",allow_pickle=True).item
        except:
            main_halo_stellar_masses = {}
        main_halo_stellar_masses[str(snapshot_num)] = main_halo_stellar_masses_list
        np.save(output_directory + galaxy + "/halo_stellar_masses.npy",main_halo_stellar_masses)
    return sps_list,ds

def angular_momentum(simulation_galaxy,snapshot_num,output_directory,lock):
    sp,ds = _sphere_creation(simulation_galaxy,snapshot_num,output_directory,lock)
    galaxy = simulation_galaxy.split("_res")[0]
    search_args = dict(use_gas=False, use_particles=True, particle_type='PartType4') # making sure to use only star particles and mass-weighted
    num_of_halos = 1
    if 'elvis' in galaxy:
        num_of_halos = 2
    redshift = round(ds.parameters['Redshift'],1)
    L_disk_norms_list = []
    for j in range(num_of_halos):
        search_args = dict(use_gas=False, use_particles=True, particle_type='PartType4') # making sure to use only star particles and mass-weighted
        L_disk = sp[j].quantities.angular_momentum_vector(**search_args)
        L_disk_norm = (L_disk/np.linalg.norm(L_disk)).tolist()
        L_disk_norms_list.append(L_disk_norm)

    with lock:
        try:
            L_disk_norms = np.load(output_directory + galaxy + "/L_disk_norms.npy",allow_pickle=True).item
        except:
            L_disk_norms = {}
        L_disk_norms[str(snapshot_num)] = L_disk_norms_list
        np.save(output_directory + galaxy + "/L_disk_norms.npy",L_disk_norms)
    return L_disk_norms,ds

def _save(p,property,output_directory,galaxy,direction,width,redshift,snapshot_num):
    p_pixel_value = np.array(p.frb.data[('gas', property)])
    np.save(output_directory + galaxy + "/" + direction + " " + property + "_W" + str(int(width)) + "_Z" + str(redshift) + "_ssn" +str(snapshot_num) + ".npy",p_pixel_value)

def plot_projection(simulation_galaxy,snapshot_num,output_directory,direction,property,lock, fraction_from_Rvir = 0.3):
    galaxy = simulation_galaxy.split("_res")[0]
    file_exists = 0
    # skipping prior calculations if possible
    try:
        L_disk_norms = np.load(output_directory + galaxy + "/L_disk_norms.npy",allow_pickle = True).item()

        # skipping snapshots that were projected already
        try:
            pixel_values = np.load(output_directory + galaxy + "/" + property + " pixel_values.npy",allow_pickle = True).item()
            try:
                pixel_values_direction = pixel_values[str(snapshot_num)]
                try:
                    pixel_values_Rvir = pixel_values_direction[direction]
                    if str(fraction_from_Rvir) in pixel_values_Rvir.keys():
                        print("Already done with galaxy: " + galaxy + ", snapshot: " + str(snapshot_num) + ", direction: " + direction + ", fraction_from_Rvir: " + str(fraction_from_Rvir))
                        return 0
                except:
                    pixel_values_Rvir = {}
            except:
                pixel_values_direction = {}
                pixel_values_Rvir = {}
        except:
            pixel_values = {}
            pixel_values_direction = {}
            pixel_values_Rvir = {}

        ds = _data_loading(simulation_galaxy,snapshot_num,output_directory,lock)
    except FileNotFoundError: 
        L_disk_norms,ts = angular_momentum(simulation_galaxy,snapshot_num,output_directory,lock)
        pixel_values = {}

    # setting weight_field for integration for different properties
    if property == 'H_p0_number_density' or property == 'density':
        weight_field = None
    elif property == 'temperature':
        weight_field = ("gas", "density")
    elif property == 'velocity':
        weight_field = ('gas','H_p0_number_density')

    print('reached ' + property + '_projections for ' + galaxy)
    redshift = round(ds.parameters['Redshift'],1)
    num_of_halos = 1
    if 'elvis' in galaxy:
        num_of_halos = 2

    pixel_values_list = []
    for j in range(num_of_halos):
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

        if direction == "x" or direction == 'EdgeOn':
            arbitrary_vector = [1,1,3]
            norm = np.cross(norm,arbitrary_vector) # projection on Edge
            north_vector =  L_disk_norms[str(snapshot_num)][j]
        elif direction == "y" or direction == 'OtherEdgeOn':
            arbitrary_vector = [1,1,3]
            norm = np.cross(norm,arbitrary_vector) # projection on Edge
            norm = np.cross(norm,L_disk_norms[str(snapshot_num)][j])
            north_vector =  L_disk_norms[str(snapshot_num)][j]
        p = yt.ProjectionPlot(ds, norm, fields=fields,center = center, width= adj_width, weight_field = weight_field, buff_size = (num_cells+1,num_cells+1),north_vector = north_vector)
        pixel_values_list.append(np.array(p.frb.data[('gas', property)]))
    
    with lock:
        try:
            pixel_values = np.load(output_directory + galaxy + "/" + property + " pixel_values.npy",allow_pickle = True).item()
            try:
                pixel_values_direction = pixel_values[str(snapshot_num)]
                try:
                    pixel_values_Rvir = pixel_values_direction[direction]
                    pixel_values[str(snapshot_num)][direction][str(fraction_from_Rvir)] = pixel_values_list
                except:
                    pixel_values_Rvir[str(fraction_from_Rvir)] = pixel_values_list
                    pixel_values[str(snapshot_num)][direction] = pixel_values_Rvir
            except:
                pixel_values_Rvir[str(fraction_from_Rvir)] = pixel_values_list
                pixel_values_direction[direction] = pixel_values_Rvir
                pixel_values[str(snapshot_num)] = pixel_values_direction     
        except Exception as e:
            pixel_values_Rvir[str(fraction_from_Rvir)] = pixel_values_list
            pixel_values_direction[direction] = pixel_values_Rvir
            pixel_values[str(snapshot_num)] = pixel_values_direction 
        np.save(output_directory + galaxy + "/" + property + " pixel_values.npy",pixel_values)
        print("Finished and saved galaxy: " + galaxy + ", and snapshot: " + str(snapshot_num) + " direction: " + direction + ", fraction_from_Rvir: " + str(fraction_from_Rvir))