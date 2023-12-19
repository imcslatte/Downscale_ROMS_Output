"""
DOWNSCALE_L0_v2.py

Created by Elias Hunter, hunter@marine.rutgers.edu, 7/29/2023 
"""
import os,glob
import numpy as np
import xroms
import xarray as xr
import pandas as pd
import xesmf as xe
import cartopy 
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import datetime
import romsutil as rutil
from  scipy.interpolate import interp1d
import netCDF4 as nc



#Set relevant inout parameters


#Set time
today=datetime.datetime.now().strftime('%Y%m%d')
today='20221102'

#Donor grid Info
# % Enter vertical characteristics for the L0 grid
L0grdfile='/home/hunter/roms/NOPP/forecast/grids/L0/NewEngland_qck_grid.nc' # can be a thredds url
#Vertical coordinate information
L0Vtransform= 2
L0Vstretching= 4
L0theta_s = 8.0
L0theta_b = 4.0
L0hc=200
L0N=50
#Donor inputfiles
datadir=f'/home/hunter/roms/NOPP/forecast/{today}/roms_L0/'# can be a thredds url
L0his=datadir+'roms_his_forNewEngland.nc' 
L0qck=datadir+'roms_qck_forNewEngland.nc' 


#Receiver Grid Info
L1grdfile='/home/hunter/roms/NOPP/forecast/grids/L1/NYBIGHT_grd_L1.nc' # can be a thredds url
L1theta_s=8.0
L1theta_b=4.0
L1Tcline=20.0
L1N=30
L1Vtransform  =2        #vertical transformation equation
L1Vstretching =4        #vertical stretching function
L1hc=200
Nbed=1
Nveg=1
NCS=0
NNS=1


(s_w_L0,C_w_L0)=rutil.stretching(L0Vstretching,L0theta_s,L0theta_b,L0hc,L0N,1)
(s_r_L0,C_r_L0)=rutil.stretching(L0Vstretching,L0theta_s,L0theta_b,L0hc,L0N,0)
(s_w_L1,C_w_L1)=rutil.stretching(L1Vstretching,L1theta_s,L1theta_b,L1hc,L1N,1)
(s_r_L1,C_r_L1)=rutil.stretching(L1Vstretching,L1theta_s,L1theta_b,L1hc,L1N,0)

#Receiver Output files
#inifile=f'/home/hunter/roms/NOPP/output/receiver_{today}_ini.nc'
inifile=f'./receiver_{today}_ini.nc'
bryfile=f'/home/hunter/roms/NOPP/output/receiver_{today}_bry.nc'
clmfile=f'/home/hunter/roms/NOPP/output/receiver_{today}_clm.nc'

#Receiver Output time dimensoons
#High resolution time dimension, maybe barotropic  
t2step=1
t2N=121
t2end=t2step*t2N
#High resolution time dimension, maybe Baroclinic
t3step=8
t3N=6
t3end=t3step*t3N

#Output file flags, if True create file
INIflag=True
CLMflag=False
BRYflag=False



############################################################################
#Main Program
############################################################################

def main():
    ########################################################################
    print('Initilaizing grids and regridder')
    ########################################################################
        #Lazy read Grid files 
    dsgrd=xr.open_dataset(L0grdfile)
    dsgrd=dsgrd.drop({'lon_rho', 'lat_rho', 'lon_u', 'lat_v', 'lat_u', 'lon_v'})
    
    cfgrd=xr.open_dataset(L1grdfile)
    cfgrd.attrs['sc_r']=L1N
    cfgrd.attrs['sc_w']=L1N+1
    cfgrd.attrs['time']=1
    cfgrd.attrs['Nbed']=1
    cfgrd.attrs['Nveg']=1
    cfgrd.attrs['NCS']=0
    cfgrd.attrs['NNS']=1
    cfgrd.attrs['t2d']=t2N
    cfgrd.attrs['t3d']=t3N
    

    #Lazy read Input model files 
    dsqckl0=xr.open_dataset(L0qck)
    dshisl0=xr.open_dataset(L0his)
    dsqckl0['s_w']=('s_w',s_w_L0)
    dsqckl0['Cs_w']=('s_w',C_w_L0)
    dsqckl0['s_rho']=('s_rho',s_r_L0)
    dsqckl0['Cs_r']=('s_rho',C_r_L0)
    dsqckl0.attrs['hc']=L0hc
    dsqckl0=xr.merge([dsqckl0,dsgrd])
    dsl0sub=dsqckl0.sel(ocean_time=dshisl0.ocean_time.values)
    
    
    
    dshisl0=xr.merge([dshisl0,dsgrd])
    dshisl0['ubar']=dsl0sub['ubar']
    dshisl0['vbar']=dsl0sub['vbar']
    dshisl0['zeta']=dsl0sub['zeta']
    dshisl0['s_w']=('s_w',s_w_L0)
    dshisl0['Cs_w']=('s_w',C_w_L0)
    dshisl0['s_r']=('s_rho',s_r_L0)
    dshisl0['Cs_r']=('s_rho',C_r_L0)
    dshisl0.attrs['hc']=L0hc
    
    
    #tmp=dsl0sub.zeta  

            
                
            
        
        
        
    ########################################################################
    #Process donwscaling files
    ########################################################################
    if INIflag:
    #Processing initilization file
        downscale_init_file(cfgrd,dsqckl0,dshisl0,dsl0sub)
    if  CLMflag:
        pass
    if  BRYflag:
        pass
def downscale_init_file(cfgrd,dsqckl0,dshisl0,dsl0sub):
    tmp=dsl0sub.rename({'mask_rho':'mask'})
    tmp=tmp.rename({'lat_rho':'lat'})
    tmp=tmp.rename({'lon_rho':'lon'})
    
    # lats = np.asarray(cfgrd.lat_rho).flatten()
    # lons = np.asarray(cfgrd.lon_rho).flatten()
    # masks = np.asarray(cfgrd.mask_rho).flatten()
    # locstream_out = True
    varnew =cfgrd[['lon_rho','lat_rho','mask_rho']]
    varnew=varnew.rename({'lat_rho':'lat'})
    varnew=varnew.rename({'lon_rho':'lon'})
    varnew=varnew.rename({'mask_rho':'mask'})
    #regridder = xe.Regridder(tmp, varnew, "bilinear", locstream_out=locstream_out)
    regridder = xe.Regridder( tmp,varnew, "bilinear",extrap_method="nearest_s2d")  
  
  
    rutil.create_init_file(inifile,cfgrd)
    dsqckl0_I=dsqckl0.isel(ocean_time=0)
    dshisl0_I=dshisl0.isel(ocean_time=0)
    
    
    print('RUNNING horizontal regridding')
    (xromqckL0,gridqckL0)=xroms.roms_dataset(dsqckl0_I,Vtransform=L0Vtransform)
    uv=rutil.uv_rot_2d(xromqckL0.ubar, xromqckL0.vbar, gridqckL0,xromqckL0.angle)
    ru=uv[0]
    rv=uv[1]
    xromqckL0=xr.merge([xromqckL0,ru,rv])
  
    (xromhisL0,gridhisL0)=xroms.roms_dataset(dshisl0_I,Vtransform=L0Vtransform)
    uv=rutil.uv_rot(xromhisL0.u, xromhisL0.v, gridhisL0,xromhisL0.angle)
    ruhis=uv[0]
    rvhis=uv[1]
    xromhisL0=xr.merge([xromhisL0,ruhis,rvhis])
  
    xromhisL0=xromhisL0.rename({'mask_rho':'mask'})
    xromqckL0=xromqckL0.rename({'mask_rho':'mask'})
    
    xromqckL1 = regridder(xromqckL0,keep_attrs=True)
    xromhisL1 = regridder(xromhisL0,keep_attrs=True)
    
    #reset masks after interpolations
    print('Preparing for vertical interpolation')
    xromqckL1['mask_rho']=cfgrd['mask_rho']
    
    newdims=['eta_rho','xi_rho'] 
    dim='points'
    coords=[np.arange(cfgrd.sizes['eta_rho']),np.arange(cfgrd.sizes['xi_rho'])]
    ind = pd.MultiIndex.from_product(coords, names=newdims)
    
    xromqckL1.coords[dim]=ind
    xromqckL1 = xromqckL1.unstack(dim)
    xromqckL1 = xromqckL1.assign_coords({"lon_rho": cfgrd.lon_rho,"lat_rho": cfgrd.lat_rho})
    
    xromhisL1.coords[dim]=ind
    xromhisL1 = xromhisL1.unstack(dim)
    xromhisL1 = xromhisL1.assign_coords({"lon_rho": cfgrd.lon_rho,"lat_rho": cfgrd.lat_rho})
    
  
    xromhisL1=xromhisL1.drop({'z_w_psi','z_rho_psi','z_rho_psi0','z_w_psi0'})
    xromhisL1['lon_psi']=cfgrd['lon_psi']
    xromhisL1['lat_psi']=cfgrd['lat_psi']
  
    xromhisL1['xi_u']=cfgrd['xi_u']
    xromhisL1['eta_v']=cfgrd['eta_v']
    xromhisL1['mask_u']=cfgrd['mask_u']
    xromhisL1['mask_v']=cfgrd['mask_v']
    xromhisL1['lon_u']=cfgrd['lon_u']
    xromhisL1['lon_v']=cfgrd['lon_v']
    xromhisL1['lat_u']=cfgrd['lat_u']
    xromhisL1['lat_v']=cfgrd['lat_v']
    xromhisL1['mask_rho']=cfgrd['mask_rho']
    xromhisL1['s_w']=('s_w',s_w_L0)
    xromhisL1['Cs_w']=('s_w',C_w_L0)
    xromhisL1['s_rho']=('s_rho',s_r_L0)
    xromhisL1['Cs_r']=('s_rho',C_r_L0)
    xromhisL1.attrs['hc']=L0hc
    
    (xromhisL1,gridhisL1)=xroms.roms_dataset(xromhisL1,Vtransform=L0Vtransform)
    
    xromhisL1_z=xromhisL1.zeta.to_dataset()
    xromhisL1_z=xr.merge([xromhisL1_z,cfgrd])
    xromhisL1_z['s_w']=('s_w',s_w_L1)
    xromhisL1_z['Cs_w']=('s_w',C_w_L1)
    xromhisL1_z['s_rho']=('s_rho',s_r_L1)
    xromhisL1_z['Cs_r']=('s_rho',C_r_L1)
    xromhisL1_z.attrs['hc']=L1hc
  
    (xromhisL1_z,gridhisL1_z)=xroms.roms_dataset(xromhisL1_z,Vtransform=L1Vtransform)
    
        
        
    dim_dict=xromhisL1.dims
    
    temp = np.empty((L1N,dim_dict['eta_rho'],dim_dict['xi_rho']))
    temp[:]=np.nan
    salt = np.empty((L1N,dim_dict['eta_rho'],dim_dict['xi_rho']))
    salt[:]=np.nan
    u_east = np.empty((L1N,dim_dict['eta_rho'],dim_dict['xi_rho']))
    u_east[:]=np.nan
    v_north = np.empty((L1N,dim_dict['eta_rho'],dim_dict['xi_rho']))
    v_north[:]=np.nan
    
    
    
    print('Running vertical interpolation, this may take a few minutes')
    
    tlat=[]
    tlon=[]
    
    for eta in range(0,dim_dict['eta_rho']):
        for xi in range(0,dim_dict['xi_rho']):
            maskflag=xromhisL1.mask_rho.isel(eta_rho=eta,xi_rho=xi).values
            if maskflag==0.0:
                continue
                
    
            tmpz=xromhisL1_z.z_rho.isel(eta_rho=eta,xi_rho=xi)
    
            
            tmp=xromhisL1.temp.isel(eta_rho=eta,xi_rho=xi)
            test=np.isnan(tmp.values).any()
            
            if test:
                tlat.append(tmp.lat_rho.values)
                tlon.append(tmp.lon_rho.values)
    
            ifun=interp1d(tmp.z_rho.values,tmp.values,kind='cubic',bounds_error=False,fill_value=(tmp.values[0],tmp.values[-1]))
            ntmp=ifun(tmpz.values)
            temp[:,eta,xi]=ntmp
            
            tmps=xromhisL1.salt.isel(eta_rho=eta,xi_rho=xi)
            ifuns=interp1d(tmp.z_rho.values,tmps.values,kind='cubic',bounds_error=False,fill_value=(tmps.values[0],tmps.values[-1]))
            ntmp=ifuns(tmpz.values)
            salt[:,eta,xi]=ntmp
            
            tmpu=xromhisL1.u_eastward.isel(eta_rho=eta,xi_rho=xi)
            ifunu=interp1d(tmp.z_rho.values,tmpu.values,kind='cubic',bounds_error=False,fill_value=(tmpu.values[0],tmpu.values[-1]))
            ntmp=ifunu(tmpz.values)
            u_east[:,eta,xi]=ntmp
            
            tmpv=xromhisL1.v_northward.isel(eta_rho=eta,xi_rho=xi)
            ifunv=interp1d(tmp.z_rho.values,tmpv.values,kind='cubic',bounds_error=False,fill_value=(tmpv.values[0],tmpv.values[-1]))
            ntmp=ifunv(tmpz.values)
            v_north[:,eta,xi]=ntmp
        
        
        
    print('Rotating velocities to receiver grid coordinate system. ')   
    xromhisL1_z['temp']=(('s_rho', 'eta_rho', 'xi_rho'),temp)
    xromhisL1_z['salt']=(('s_rho', 'eta_rho', 'xi_rho'),salt) 
    xromhisL1_z['u_eastward']=(('s_rho', 'eta_rho', 'xi_rho'),u_east) 
    xromhisL1_z['v_northward']=(('s_rho', 'eta_rho', 'xi_rho'),v_north) 
    xromhisL1_z['vbar_northward']=xromqckL1['vbar_northward']
    xromhisL1_z['ubar_eastward']=xromqckL1['ubar_eastward']
    
    
    uv=rutil.uv_rot_2d(xromhisL1_z.ubar_eastward, xromhisL1_z.vbar_northward, gridhisL1_z,xromhisL1_z.angle,reverse=True)
    ru=uv[0]
    rv=uv[1]
    xromhisL1_z=xr.merge([xromhisL1_z,ru,rv])
    
    uv=rutil.uv_rot(xromhisL1_z.u_eastward, xromhisL1_z.v_northward, gridhisL1_z,xromhisL1_z.angle,reverse=True)
    ruhis=uv[0]
    rvhis=uv[1]
    xromhisL1_z=xr.merge([xromhisL1_z,ruhis,rvhis])
    
    
    print(['Writing initialization data to file'+inifile])
    ncid = nc.Dataset(inifile, "r+", format="NETCDF4")
    
    ncid.variables['ubar'][0,:,:]=xromhisL1_z.ubar.values[:,:]
    ncid.variables['vbar'][0,:,:]=xromhisL1_z.vbar.values[:,:]
    ncid.variables['zeta'][0,:,:]=xromhisL1_z.zeta.values[:,:]
    ncid.variables['u'][0,:,:,:]=xromhisL1_z.u.values[:,:,:]
    ncid.variables['v'][0,:,:,:]=xromhisL1_z.v.values[:,:,:]
    ncid.variables['salt'][0,:,:,:]=xromhisL1_z.salt.values[:,:,:]
    ncid.variables['temp'][0,:,:,:]=xromhisL1_z.temp.values[:,:,:]
    ncid.sync()
    ncid.close()

if __name__ == "__main__":
    print('Running Downscale')
    main()
    
    print('Finished Downscale')
