# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 11:51:08 2019

@author: ericv
"""

#%%
#Import statements
import os
import time

import cv2
import numpy as np
import gdal

#os.chdir(os.path.dirname(os.path.dirname(sys.argv[0])) + '/ImageAnalysis')
#os.chdir(r'C:\Users\ericv\Dropbox\Python scripts\GitHub')

#os.chdir(r'C:\Users\VanBoven\Documents\GitHub\ImageAnalysis-6599c1bf5e3b3ac8b748b9fdae7d1af9d20dd997')

import vector_functions as vector_functions
import plant_count_functions as plant_count_functions
import raster_functions as raster_functions
import image_processing as ip
import detect_plants as dp
import clip_ortho_2_plot_gdal as clip_ortho_2_plot_gdal
import clip_raster2shp

#os.chdir(os.path.dirname(os.getcwd()) + '/VanBovenProcessing')
#import VanBovenProcessing.clip_ortho_2_plot_gdal

#%%
img_path_list = [r"A:\2\107\30.tif",
    ]

#Output path
out_path = r'O:\Manual_jobs\Plant_detections\107'
 
#Specify path to clip_shp
clip_shp = r"O:\Manual_jobs\Plant_detections\107\Bresseleers_batch_90.shp"


#%%
#Set variables for clustering

#Set lower and higher limits to plant size (px)
min_area = 25
max_area = 3500

#Set no_data_value
#no_data_value = 255
no_data_value = 0

#Use small block size to cluster based on colors in local neighbourhood
x_block_size = 512
y_block_size = 512

#Create iterator to process blocks of imagery one by one and be able to process subsets.
it = list(range(0,2000000, 1))

#True if you want to generate shapefile output with points and shapes of plants
vectorize_output = True
#True if you want to fit the cluster centres iteratively to every image block, false if you fit in once to a random 10% subset of the entire ortho
iterative_fit = False
#True to clip ortho to clip_shape (usefull for removing grass around the fields)
clip_ortho2shp = True
#True to create a tif file of the plant mask of the image.
tif_output = True
#True if you want to write plant count and segmentation from clustering to file
write_shp2file = True
#nr_clusters = 2 or 3 depending on desired segmentation results. 3 is better for plant count. 2 can perform better for segmentation of grown crops
nr_clusters = 3
#specify if you also want to use clustering algorithm for getting a plant count with area and shapes
perform_clustering = True
#specify if you want to use local max algorithm for gettnig a plant count with area
perform_local_max = False

crop_type = "Broccoli"

for i, img_path in enumerate(img_path_list):
    print(i+1, "/", len(img_path_list))
    #%%
    #Set variables for local maxima plant count
    
    #Determine factor for scaling parameters (for different image scales)
    xpix,ypix = ip.PixelSize(img_path)
    par_fact = 7.68e-5/(xpix*ypix)
    
    #Sigma is smoothing factor higher sigma will result in smoother image and less detected local maxima
    sigma = 8.0*par_fact
    sigma_grass = 2.0*par_fact
    neighborhood_size = 30*par_fact
    #Threshold is a parameter specifying when to classify a local maxima as plant/significant.
    #Increasing threshold will result in less detected local maxima. Treshold and sigma interact
    threshold = 2.0
    
    block_size = 3000
    #%%
    #Set initial cluster centres for clustering algorithm based on sampling in images
    
    #Broccoli:
    leaf_sun = cv2.cvtColor(np.array([[[239, 243, 233]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    leaf_sun_init = np.array(leaf_sun[0,0,1:3], dtype = np.uint8)
    leaf_cloudy = cv2.cvtColor(np.array([[[150, 182, 141]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    leaf_cloudy_init = np.array(leaf_cloudy[0,0,1:3], dtype = np.uint8)
    leaf_blueish_breed = cv2.cvtColor(np.array([[[239, 239, 231]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    leaf_blueish_breed_init = np.array(leaf_blueish_breed[0,0,1:3], dtype = np.uint8)
    soil1 = cv2.cvtColor(np.array([[[177, 179, 189]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    soil1_init = np.array(soil1[0,0,1:3], dtype = np.uint8)
    soil2 = cv2.cvtColor(np.array([[[113, 130, 138]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    soil2_init = np.array(soil2[0,0,1:3], dtype = np.uint8)
    grass_weeds = cv2.cvtColor(np.array([[[118, 210, 139]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    grass_weeds_init = np.array(grass_weeds[0,0,1:3], dtype = np.uint8)
    blue_middle = cv2.cvtColor(np.array([[[190, 174, 204]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    blue_middle_init = np.array(blue_middle[0,0,1:3], dtype = np.uint8)
    
    #Brussel sprouts
    cover_lab = cv2.cvtColor(np.array([[[165,168,148]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    cover_init = np.array(cover_lab[0,0,1:3], dtype = np.uint8)
    background_init = cv2.cvtColor(np.array([[[120,125,130]]]).astype(np.uint8), cv2.COLOR_BGR2LAB) # as sampled from tif file
    background_init = np.array(background_init[0,0,1:3])
    green_lab = cv2.cvtColor(np.array([[[87,125,89]]]).astype(np.uint8), cv2.COLOR_BGR2LAB)
    green_init = np.array(green_lab[0,0,1:3])
    
    #Create init input for clustering algorithm
    if nr_clusters == 3:
        kmeans_init = np.array([background_init, green_init, cover_init])
    elif nr_clusters == 2:
        kmeans_init = np.array([background_init, green_init])
    else:
        kmeans_init = np.array([background_init, green_init, cover_init])
# =============================================================================
#     if crop_type == "Broccoli":
# #        kmeans_init = np.array([leaf_sun_init, leaf_cloudy_init, blue_middle_init, leaf_blueish_breed_init, soil1_init, soil2_init, grass_weeds_init])
#         kmeans_init = np.array([leaf_sun_init, leaf_cloudy_init, blue_middle_init, soil1_init, soil2_init, grass_weeds_init])
#     
# =============================================================================
    #Create init input for clustering algorithm
    #kmeans_init = np.array([background_init, green_init, cover_init])
    #%%
    #Set other variables
    
    #Set distances for merging of close points, provide a list with slowely increasing values for best result
    list_of_distances = [0.05, 0.08, 0.12, 0.16, 0.22]
    #%%
    #Run script
    if __name__ == '__main__':
        if clip_ortho2shp == True:
            #ds = clip_ortho_2_plot_gdal.clip_ortho2shp_array(img_path, clip_shp)
            #clip_raster2shp.clip_ortho2plot(ortho_path=img_path, filename='temp.tif', shp_path=clip_shp, output_path=out_path)
            #ds = 
            clip_ortho_2_plot_gdal.clip_ortho2plot_gdal(img_path, clip_shp, out_path)
            ds = gdal.Open(os.path.join(out_path, 'temp.tif'))
        else:
            ds = gdal.Open(img_path)
        print(ds.GetRasterBand(1).GetBlockSize())
        #%%
        if perform_local_max == True:
            #Run local maxima plant detector
            time_begin = time.time()
    
            #Get information of image partition
            div_shape = ip.divide_image(ds, block_size, remove_size=block_size)
            ysize, xsize, yblocks, xblocks, block_size = div_shape
    
            #Detect center of plants using local minima
            xcoord, ycoord = dp.DetectLargeImage(img_path, ds, div_shape, sigma, neighborhood_size, threshold, sigma_grass)
            #Write to shapefile
            gdf_local_max = vector_functions.coords2gdf(ds, xcoord, ycoord)
    
            #Add area column to gdf_local_max
            gdf_local_max = plant_count_functions.add_random_area_column(gdf_local_max)
            #Add column to specify that the area is not a correct measurement
            gdf_local_max['correct_area'] = 0
        # =============================================================================
        #
        #     #Create array with the positions of the coordinates
        #     arr_points = np.zeros([yblocks*block_size//10,xblocks*block_size//10], dtype='uint8')
        #     arr_points[(ycoord/10).astype(int),(xcoord/10).astype(int)] = 1
        #     arr_points = cv2.dilate(arr_points,np.ones((3,3),np.uint8),iterations = 1)
        #
        #     #Detect lines using Hough Lines Transform
        #     lines = dp.HoughLinesP(arr_points, par_fact)
        #     #Write to shapefile
        #     lines = lines.reshape(lines.shape[0],4) * 10
        #     dp.WriteShapefileLines(ds, lines[:,0], lines[:,1], lines[:,2], lines[:,3], out_path)
        #
        # =============================================================================
            time_end = time.time()
            print('Total time: {}'.format(time_end-time_begin))
        #%%
        if perform_clustering == True:
            #Perform clustering
            plant_pixels, clustering_output = plant_count_functions.cluster_objects(x_block_size, y_block_size, ds, kmeans_init, iterative_fit, it, no_data_value)
    
            #Write clustering output to tif to be able to inspect
#            raster_functions.array2tif(img_path, out_path, clustering_output, name_extension = 'clustering_output')
            raster_functions.array2tif(img_path, out_path, plant_pixels, name_extension = 'clustering_output')
    
            #Save some memory
            clustering_output = None
    
        if vectorize_output == True:
            #Get contours of plants and create a df with derived characteristics.
            #At this point there is no proper classification algorithm so run_classification = False
            df = plant_count_functions.contours2shp(plant_pixels, out_path, min_area, max_area, ds, run_classification = False)
    
            #Convert df to gdf
            gdf_points, gdf_shapes = vector_functions.detected_plants2projected_shp_and_points(img_path, out_path, df, ds, write_shp2file)
    
            #Add column to specify that area is correct measurement
            gdf_points['correct_area'] = 1
            
            if perform_local_max == True:
                #Append both count dataframes
                gdf = vector_functions.append_gdfs(gdf_points, gdf_local_max)
            else:
                gdf = gdf_points.copy()
            #Merge close points
            gdf = plant_count_functions.merge_close_points(gdf, list_of_distances)
    
            #Write output to geopackage
            gdf.to_file(os.path.join(out_path, (os.path.basename(img_path)[-16:-4] + '_plant_count_batch90.gpkg')), driver = 'GPKG')