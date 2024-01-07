import edens as e
import analytical_funcs as a
import numpy as np
import matplotlib.pyplot as plt
from numpy import sin,cos,tan, sqrt, pi
import os
from numpy_unit import Unit, ArrayUnit
import unyt as u
from unyt import kpc, km, cm, s, Msun, gram, dimensionless


"""
This function returns all the points of distance d from a line with parameters x_0, z_0 and theta_0, 
within the positions array r. 
"""
def positions_near_line( x_0 , z_0, theta_0, r, d ):

    N_p = np.shape(r[:,0])[0] # Number of points
    
    points_x = np.array(r[:,0]).reshape((N_p,1))
    points_y = np.array(r[:,1]).reshape((N_p,1))
    points_z = np.array(r[:,2]).reshape((N_p,1))

    direction_y =  sin(theta_0) if sin(theta_0) >= 1e-12 else 0 
    direction_z =  cos(theta_0) if cos(theta_0) >= 1e-12 else 0
    direction_xyz = np.array([0,direction_y,direction_z])
  
    grid = np.array(np.meshgrid(x_0, 0, z_0)).T.reshape(-1,3)
    
    points = np.concatenate( (points_x, points_y, points_z), axis=1 )
    points = np.tile( points, grid.shape[0] ).reshape(grid.shape[0], N_p, 3)

    l_point = np.tile(grid,N_p).reshape((grid.shape[0],N_p,3))
    l_direction = np.tile(direction_xyz,N_p*grid.shape[0]).reshape((grid.shape[0],N_p,3))

    crossp = e.explicitcross(points-l_point, l_direction)
   
    D = np.linalg.norm(crossp,axis=2) / np.linalg.norm(l_direction,axis=2)

    condition = ( D <= d )
    cond3 = np.repeat(condition, 3, axis=1).reshape(condition.shape[0],condition.shape[1],3)

    #r_near_line = np.where(cond3, points, np.full_like(points, np.nan)) * r.units
    r_near_line = points[cond3].reshape((-1,3)) * r.units

    return( r_near_line , condition[0,:] , cond3[0,:] )


"""
This function gets a position array of all particles within a line's range r_near_line, and the line's direction theta_0 (see line parametrization in analytical_funcs) and returns an array of the relative distance each point "takes" in space (a "dl" array)
"""
def find_relative_part(theta_0, r_near_line):
    
    # Rotate data to the y direction
    c, s = np.cos(-(pi/2-theta_0)), np.sin(-(pi/2-theta_0))
    Rot = np.array(((1, 0, 0),(0, c, -s), (0, s, c)))
    #new_r = np.einsum("hij,kj->hik", r_near_line, Rot)
    #new_y = new_r[:,:,1]
    new_r = np.einsum("ij,kj->ik", r_near_line, Rot)
    new_y = new_r[:,1]
 
    # Sort along the lines for distance calculation
    permut = np.argsort(new_y)
    
    #sorted_new_y = np.array(list(map(lambda x, y: y[x], permut, new_y)))
    sorted_new_y = new_y[permut]
   
    # un-permutation to return the the array to original order after calculation
    unpermut = np.argsort(permut)
    
    # calculate distances between neighboring points by shifting and subtracting
    shiftleft = np.append(sorted_new_y,[0])
    shiftright = np.append([0],sorted_new_y)

    distances = shiftleft - shiftright
    distances = np.delete(distances,0)
    distances = np.delete(distances,-1)

    # the relative part of each particle is half the distance to the left + half the distance to the right
    rel_part = ( np.append(distances/2,[0])+np.append([0],distances/2) )
     
    # return to original order
    rel_part = rel_part[unpermut]

    return (rel_part * r_near_line.units)


"""
Numerical integrals
"""
def v_los_sim( v_near_line , theta_0 ):
    v_los = None
    if v_near_line.ndim == 3:
        v_los = sin(theta_0) * v_near_line[:,:,1] + cos(theta_0) * v_near_line[:,:,2]
    elif v_near_line.ndim == 2:
        v_los = sin(theta_0) * v_near_line[:,1] + cos(theta_0) * v_near_line[:,2]
    return( v_los ) 

def tau_sim( n_e_near_line , rel_part):
    return( u.sigma_thompson * np.sum(rel_part * n_e_near_line) ) 

def kSZ_sim( tau , n_e_near_line , v_los, rel_part ):
    return(  u.sigma_thompson * np.exp(-tau) * np.sum(rel_part * n_e_near_line * v_los)  / u.c) 