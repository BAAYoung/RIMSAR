import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import interpolate

def background_finder():
    root_folder = '/media/ben/Seagate Expansion Drive/Twente_backup/TwenteDay3_5/Hz300_240_18RPMbig/Acquisition_001/'
    background_mean = cv2.imread(root_folder + 'Os4-S3 Camera000001.tif',0).astype(np.float64)*0
    background_stdev = background_mean*0
    n_background = 100
    background_step_size = 10
    for i in range(0,n_background*background_step_size,background_step_size):
        n_str = str(i)
        n_str = n_str.zfill(6)
        background_img = cv2.imread(root_folder + 'Os4-S3 Camera'+n_str+'.tif',0).astype(np.float64)
        background_mean += background_img/n_background
        background_stdev += background_img**2/n_background
        print(i)
    background_stdev = np.sqrt(background_stdev - background_mean**2)
    background_action = background_stdev + 0
    #applying gaussian blur to image
    sigmaX = 100
    ksize = (15,15)
    background_action = cv2.GaussianBlur(background_action, ksize, sigmaX)

    background_threshold = 10
    background_action[background_action<background_threshold] = 0
    background_action[background_action>=background_threshold] = 1
    return background_action

def circle_extract(image_number,p_val, background_action):
    r = 3
    kernel = np.zeros((2*r+1,2*r+1),dtype = np.uint8)
    cv2.circle(kernel,(r,r),radius=r, color = 1,thickness=-1)
    sigmaX = 100
    ksize = (15,15)

    root_folder = '/media/ben/Seagate Expansion Drive/Twente_backup/TwenteDay3_5/Hz300_240_18RPMbig/Acquisition_001/'
    n_str = str(image_number)
    n_str = n_str.zfill(6)
    img = cv2.imread(root_folder + 'Os4-S3 Camera'+n_str+'.tif',0)
    blurred_img = cv2.GaussianBlur(img, ksize, sigmaX)

    imMod = img/(blurred_img+10)

    #thresholding:
    imMod *= background_action
    imMod[imMod>0.8] = 255
    imMod[imMod<=0.8] = 0
    #imMod *= 255
    imMod = 255-imMod

    imMod = np.floor(imMod).astype(np.uint8)

    imMod = cv2.morphologyEx(imMod,cv2.MORPH_OPEN,kernel)

    sigmaX = 100
    ksize = (7,7)
    imMod = cv2.GaussianBlur(imMod, ksize, sigmaX)

    #parameters 1 sets sensitivity: how strong the edges of the circle need to be
    #parameter 2 sets how many edge points need to be present for a circle to be there


    circles = cv2.HoughCircles(imMod,cv2.HOUGH_GRADIENT,1,20,
                                param1=200,param2=p_val,minRadius=5,maxRadius=30)
    
    return circles

def circle_heirarchy(image_number,background_action):

    p_vals = np.arange(45,20,-5)
    p_vals_index = p_vals*0


    circle_info_per_image = circle_extract(image_number, p_vals[0],background_action)[0,:,:]
    circle_info = circle_info_per_image + 0
    p_vals_index[0] = circle_info_per_image.shape[0]
    index = 0
    for p_val in p_vals[1:]:
        index += 1
        circle_info_per_image = circle_extract(image_number, p_val,background_action)[0,:,:]
        circle_info = np.concatenate( (circle_info, circle_info_per_image),axis=0 )
        p_vals_index[index] = circle_info_per_image.shape[0]

    p_vals_cumu_index = np.cumsum(p_vals_index)

    #removing duplicate circles:
    n_indices = p_vals.shape[0]
    for i in range(0,n_indices-1):

        r2 = (circle_info[0:p_vals_cumu_index[i],0:1] - circle_info[p_vals_cumu_index[i]:p_vals_cumu_index[i+1],0:1].transpose() )**2 + (circle_info[0:p_vals_cumu_index[i],1:2] - circle_info[p_vals_cumu_index[i]:p_vals_cumu_index[i+1],1:2].transpose() )**2
        r2 -= 0.25*circle_info[0:p_vals_cumu_index[i],2:3]**2

        
        r2[r2>0] = 0
        intersecting_circles = np.argwhere(r2)
        


        #updating index so that duplicates are removed:
        #print(intersecting_circles.shape)
        circle_info = np.delete(circle_info, (intersecting_circles[:,1]+p_vals_cumu_index[i]),axis=0 )
        #print(circle_info.shape)
        p_vals_index[i+1] -= intersecting_circles.shape[0]
        #recalculating cumulative index:
        p_vals_cumu_index = np.cumsum(p_vals_index)

    return circle_info, p_vals_cumu_index

def vector_generator(circles0,circles1):



    n_potential_matches = min(circles0.shape[0],circles1.shape[0])-1
    indij = np.zeros((n_potential_matches,5),np.int16)
    r2 = np.abs(circles0[:,0:1] - circles1[:,0:1].transpose()) + np.abs(circles0[:,1:2] - circles1[:,1:2].transpose())
    circlesorig0 = circles0+0
    circlesorig1 = circles1+0
    distance_cutoff = 20
    for i in range(0,n_potential_matches):
        indij[i,0:2] = np.unravel_index(np.argmin(r2, axis=None), r2.shape)
        indij[i,2] = r2[indij[i,0],indij[i,1]]
        indij[i,3] = circles0[indij[i,0],0] - circles1[indij[i,1],0]
        indij[i,4] = circles0[indij[i,0],1] - circles1[indij[i,1],1]

        r2[indij[i,0],:] = 3000 #some arbitrarily high number
        r2[:,indij[i,1]] = 3000
        c=np.random.rand(3,)


    indij[-1,2] = 1000
    #pruning any matched pairs larger than the radial cutoff:
    for i in range(n_potential_matches-1,-1,-1):
        if indij[i,2] > distance_cutoff:
            indij = np.delete(indij,i,0)

    indij = indij.astype(int)
    vector_points = circles0[indij[:,0],:]
    vector = circles1[indij[:,1],0:2] - circles0[indij[:,0],0:2] 
    return vector_points, vector

background_action = background_finder()

for image_number in range(0,5000,1):
    circles0, circles_t_n_likelihoods = circle_heirarchy(image_number,background_action)
    circles1, circles_t_np1_likelihoods = circle_heirarchy(image_number+1,background_action)
    vector_points, vector = vector_generator(circles0,circles1)
    
    point_data = np.concatenate((vector_points,vector),axis=1)

    #saving point_data
    np.savetxt('point_data/point_data_'+str(image_number)+'.dat',point_data)
    #print(np.concatenate((vector_points,vector),axis=1).shape)
    print(image_number)