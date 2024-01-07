import numpy as np
from numpy import sin,cos,tan, sqrt, pi
from astropy import units as un, constants as cons


"""
" CONSTANTS
"""
kpc_to_km = 3.086e+16; cm_to_km = 1e-5; m_to_km = 1e-3
year_to_sec = 3.154e+7
SM_to_gram = 1.989e+33

T_CMB = 2.72548                            # [K]
R_circ = 15                                # [kpc]
R_vir = 300                                # [kpc]
#Mdot = 1.5                                 # [SM/yr]
Mdot = 3.8
Mdot_s = Mdot / year_to_sec                # [SM/s]
Mdot_g_s = Mdot_s * SM_to_gram             # [gr/s]
Lambda = 1.397 * 1e-23                     # [erg*cm^3/s] = [cm^5*gr/s^3]
#Lambda = 2.93e-23
sigma_T = cons.sigma_T.value               # [m^2]
c = cons.c.value                           # [m/s]

# km constants
#v_c = 200                                  # [km/s]
v_c = 220
R_circ_km = R_circ * kpc_to_km             # [km]
R_vir_km = R_vir * kpc_to_km               # [km]
Lambda_km = Lambda * cm_to_km**5           # [km^5*gr/s^3]
sigma_T_km = sigma_T * m_to_km**2          # [km^2]
c_km = c * m_to_km                         # [km/s]

n_const = 1.3 * sqrt( ( Mdot_g_s * v_c**2 ) / ( 4 * pi * Lambda_km ) ) # [km^-1.5]
C = - ( 27 / sqrt(10) ) * ( 200 / v_c ) * sqrt( Mdot ) * sqrt( Lambda / (1e-22) ) # should be length^1.5/time (r^0.5 * v)
# [s/cm] * [gr/s]^0.5 * [km^5*gr/s^3]^0.5 = [ gr * km^1.5 / s ] 

# kpc constants
v_c_kpc = v_c / kpc_to_km                  # [kpc/s]
Lambda_kpc = Lambda_km / kpc_to_km**5      # [kpc^5*gr/s^3]
n_const_kpc = n_const * kpc_to_km**1.5     # [kpc^1.5]
sigma_T_kpc = sigma_T_km / kpc_to_km**2    # [kpc^2]
C_kpc = C / kpc_to_km**1.5
c_kpc = c_km / kpc_to_km

"""
" IN PLANE INTEGRALS
"""
def n_e_inplane(y,b):
    return n_const_kpc * ( (b**2+y**2)**-0.75 + (31/24)*(R_circ**2)*((b**2+y**2)**-1.75) )
def n_e_v_dl_inplane(y,b):
    return n_const_kpc * ( C_kpc*y * ((b**2+y**2)**-1.5 + (5/18)*(R_circ**2)*((b**2+y**2)**-2.5)- (2263/1728)*R_circ**4*(b**2+y**2)**-3.5) + b*v_c_kpc* ( R_circ*(b**2+y**2)**-1.75 + (31/24) * R_circ**3 * (b**2 + y**2)**-2.75))

"""
" GENERAL INTEGRALS
"""
def parametrization(x_0,z_0,theta_0,t):
    x = x_0
    y = t * ( sin(theta_0) if sin(theta_0) >1e-12 else 0 )
    z = t * ( cos(theta_0) if cos(theta_0) >1e-12 else 0 ) + z_0
    r = sqrt ( x**2 + y**2 + z**2 )
    return ( x , y , z , r )

def n_e(t,x_0,z_0,theta_0):
    x , y , z , r = parametrization(x_0,z_0,theta_0,t)
    return ( n_const_kpc * r**-1.5 * ( 1 + ( R_circ**2 / r**2 ) * (2.75 * (x**2 + y**2) / r**2 - 35/24 ) ) )
    
def v(t,x_0,z_0,theta_0):
    x , y , z , r = parametrization(x_0,z_0,theta_0,t)
    v_r = C_kpc * r**-0.5 * ( 1 - ( R_circ**2 / r**2 ) * ( (23/12) * (x**2 + y**2) / r**2 - 65/72) )
    v_theta = - (5/9) * C * ( R_circ**2 / r**4.5 ) * z * sqrt( x**2 + y**2 )
    v_phi = v_c_kpc * ( R_circ / r**2 ) * sqrt( x**2 + y**2 ) 
    return ( [ v_r , v_theta , v_phi ] )

def dl(t,x_0,z_0,theta_0):
    x , y , z , r = parametrization(x_0,z_0,theta_0,t)
    rho = sqrt( x**2 + y**2 )
    yhat = np.array( [ y/r, y*z/( r*rho ), x/rho ] ) # [rhat, thetahat, phihat]
    zhat = np.array( [ z/r, - sqrt(x**2 + y**2)/r, 0 ] ) # [rhat, thetahat, phihat]
    dl = sin(theta_0) * yhat + cos(theta_0) * zhat
    return ( dl )

def v_dl(t,x_0,z_0,theta_0):
    v_ = v(t,x_0,z_0,theta_0)
    dl_ = dl(t,x_0,z_0,theta_0)
    return( np.dot( v_, dl_ ) )
            
def n_e_v_dl(t,x_0,z_0,theta_0):
    v_dl_ = v_dl(t,x_0,z_0,theta_0)
    n_e_ =  n_e(t,x_0,z_0,theta_0)
    return ( n_e_ * v_dl_ )
