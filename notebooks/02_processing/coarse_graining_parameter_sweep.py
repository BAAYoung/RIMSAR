#When coarse graining with the Lucy function, we have two parameters - the spatial coarse graining scale and the temporal coarse graining scale. This scripts calculates time-averaged
#statistical fields of the flow, such as mean, variance, second order correlations and then saves them for various values of our two parameters


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate

def spatial_coarse_grain(c,Xp):

    filtered_u_list = []
    filtered_v_list = []
    filtered_rho_list = []


    image_numbers = np.arange(0,4000,1)

    for image_number_orig in image_numbers:

        point_data = np.loadtxt('/home/bay22/github_repos/RIMSAR/data/03_processed/point_data/point_data_'+str(image_number_orig)+'.dat')

        #calibrating
        #12mm = 29.4 pixels
        #camera at 300 hz


        #density of borosilicate glass: 2.23 g/cm3
        #volume of 0.6cm radius glass sphere = 0.90 cm^3
        #mass = 0.002 kg
        mass = 0.002

        point_data *= 12/(29.4*1000)
        point_data[:,1] = 0.4-point_data[:,1]
        point_data[:,3:] *= 300
        #print(point_data)


        Xc = point_data[:,0:2]

        r = np.sqrt( (Xp[:,0:1] - Xc[:,0:1].transpose())**2 + (Xp[:,1:2] - Xc[:,1:2].transpose())**2 )/c
        #r = np.sqrt( (xx - Xc[0,0] )**2 + (zz - Xc[0,1])**2 )/c

        W = 5*( -3*r**4 + 8*r**3 - 6*r**2 + 1   )/(3.14159*c**2)

        #numerical stability parameter
        epsilon = 1e-4
        W[r>=1] = 0
        rho = np.sum(W,axis=1).reshape((nx,nz))*mass + epsilon
        u = mass*np.matmul(W,point_data[:,3:4]).reshape((nz,nx))/rho
        v = mass*np.matmul(W,point_data[:,4:5]).reshape((nz,nx))/rho

        filtered_rho_list.append(rho)
        filtered_u_list.append(u)
        filtered_v_list.append(v)
        #print(image_number_orig)

    return filtered_u_list, filtered_v_list, filtered_rho_list





dx = 0.01
dz = 0.01

xx,zz = np.meshgrid(np.arange(0,0.45,dx),np.arange(0,0.45,dz))
nx = xx.shape[1]
nz = xx.shape[0]
xp = xx.reshape((nx*nz,1))
zp = zz.reshape((nx*nz,1))

Xp = np.concatenate((xp,zp),axis=1)

#c = 0.05 #coarsegraining parameter
c_index = np.arange(0.025,0.2,0.025)

for c in c_index:

    filtered_u_list, filtered_v_list, filtered_rho_list = spatial_coarse_grain(c,Xp)

    #temporal coarse graining


    start_frame = 500
    end_frame = 3500



    for c_t in range(5,100,5):
        t_base = np.abs(np.arange(-c_t,c_t+1,1)/c_t)


        t_kernel = (5/(4*c_t))*(-3*t_base**4  + 8*t_base**3 - 6*t_base**2 + 1  )
        #print(t_kernel)
        u_mean = xx*0
        u2_mean = xx*0
        v_mean = xx*0
        v2_mean = xx*0

        mox_mean = xx*0
        moz_mean = xx*0
        rho_mean = xx*0
        rhou2_mean = xx*0
        rhov2_mean = xx*0
        rhouv_mean = xx*0

        for frame_number in range(start_frame,end_frame):
            u_filtered = xx*0
            v_filtered = xx*0
            mox_filtered = xx*0
            moz_filtered = xx*0
            rho_filtered = xx*0
            rhou2_filtered = xx*0
            rhov2_filtered = xx*0
            rhouv_filtered = xx*0
            for i in t_base:
                #print(int(frame_number + i))
                rho_filtered += filtered_rho_list[int(frame_number + i)]*t_kernel[int(i+c_t)]
                u_filtered += filtered_u_list[int(frame_number + i)]*t_kernel[int(i+c_t)]
                v_filtered += filtered_v_list[int(frame_number + i)] *t_kernel[int(i+c_t)]
                

            u_mean += u_filtered
            u2_mean += u_filtered**2

            v_mean += v_filtered
            v2_mean += v_filtered**2

            rho_mean += rho_filtered
            mox_mean += rho_filtered*u_filtered
            moz_mean += rho_filtered*v_filtered

            rhou2_mean += rho_filtered*u_filtered**2
            rhov2_mean += rho_filtered*v_filtered**2
            rhouv_mean += rho_filtered*u_filtered*v_filtered

        N = end_frame-start_frame

        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/u_mean_ct'+str(c_t)+'_c'+str(c)+'.dat',u_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/u2_mean_ct'+str(c_t)+'_c'+str(c)+'.dat',u2_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/v_mean_ct'+str(c_t)+'_c'+str(c)+'.dat',v_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/v2_mean_ct'+str(c_t)+'_c'+str(c)+'.dat',v2_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/mox_mean_ct'+str(c_t)+'.dat',mox_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/moz_mean_ct'+str(c_t)+'.dat',moz_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/rho_mean_ct'+str(c_t)+'.dat',rho_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/rhou2_mean_ct'+str(c_t)+'.dat',rhou2_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/rhov2_mean_ct'+str(c_t)+'.dat',rhov2_mean/N)
        np.savetxt('/home/bay22/github_repos/RIMSAR/data/03_processed/time_filtering/rhouv_mean_ct'+str(c_t)+'.dat',rhouv_mean/N)

    print(c)
