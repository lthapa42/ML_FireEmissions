# import statement
import numpy as np
np.set_printoptions(threshold=100000)
import netCDF4 as nc
import math

# IN: temperature in K
# OUT: sat vapor pressure in hPa
# Calculates the sat vapor pressure using Murphy and Koop, 2005 eq 7, via Glenn Diskin.  
def sat_vap_press(T):
    svp = np.exp(54.842763-6763.22/T-4.21*np.log(T)+0.000367*T+np.tanh(0.0415*(T-218.8))\
                      *(53.878-1331.22/T-9.44523*np.log(T)+0.014025*T))/101325*1013.25 # hPa
    return svp

# IN: specific humidity, unitless
# OUT: water vapor pressure in hPa
def vap_press(q, p):
    MW_H2O = 18.02 # g/mol
    MW_air = 28.97 # g/mol
    r = q/(1-q) # unitless water vapor mass mixing ratio
    rv = r*MW_H2O/MW_air # unitless water vapor volume mixing ratio
    vp = p*rv/100 # hPa
    return vp

#in: vapore pressure (e_hPa), saturation vapor pressure (esat_hPa) and wind speed (u)
# out: hot dry windy index
def hot_dry_windy(e_hPa, esat_hPa, u):
    vpd = esat_hPa-e_hPa
    hdw = u*vpd
    return hdw

# Calculate Haines Index (instructions here: https://www.nwcg.gov/publications/pms437/weather/stability) using the "high" version
# IN: 700mb temp (t_700), 500mb temp (t_500), 700mb dewpoint (td_700)
#OUT: the Haines index
def haines(t_700, t_500, td_700):
    # A term = 700-500 temps in decgrees C, stability term
    A = t_700-t_500 # difference in degrees C
    #convert differences
    A_cat = np.zeros(A.shape)
    A_cat[(A<=17)] = 1 # A_cat = 1 when A<=17
    A_cat[(A>17) & (A<22)] = 2 # A_cat = 2 when A>17 or A<22
    A_cat[(A>=22)] = 3 # A_cat = 3 when A>=22

    # B term = 700 T - 700 Td (both in degrees C), moisture term
    B = (t_700 - 273) - (td_700)
    B_cat = np.zeros(B.shape)
    B_cat[(B<=14)] = 1 # B_cat = 1 when B<=14
    B_cat[(B>14) & (B<21)] = 2 # B_cat = 2 when B>14 and B<21
    B_cat[(B>=21)] = 3 # B_cat = 3 when B>=21
    
    haines = A_cat + B_cat  # calculate the Haines Index
    return haines
