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
#Specify paths

# =============================================================================
# #Input img_path
# img_path_list = [r"D:\Old GR\c01_verdonk-Wever west-201907170749-GR.tif",
#             #r"D:\Old GR\c01_verdonk-Wever west-201907240724-GR.tif",
#             #r"D:\Old GR\c01_verdonk-Wever west-201908041528-GR.tif",
#             r"D:\Old GR\c01_verdonk-Wever west-201908221713-GR.tif",
#             r"D:\Old GR\c01_verdonk-Wever west-201908291238-GR.tif"]
#                      
# #Output path
# out_path = r'D:\800 Operational\c01_verdonk\Wever west\Season evaluation\No_fitting'
# #Specify path to clip_shp
# clip_shp = r"D:\800 Operational\c01_verdonk\Wever west\clip_shape_goed.shp"
# 
# =============================================================================
#%%
# =============================================================================
# 
# #Input img_path
# img_path_list = [r"D:\Old GR\c01_verdonk-Rijweg stalling 1-201907091137-GR.tif",
#                  r"D:\Old GR\c01_verdonk-Rijweg stalling 1-201907170849-GR.tif",
#                  #r"D:\Old GR\c01_verdonk-Rijweg stalling 1-201907230859-GR.tif",
#                  r"D:\Old GR\c01_verdonk-Rijweg stalling 1-201908041120-GR.tif"]
#                  #r"D:\Old GR\c01_verdonk-Rijweg stalling 1-201908051539-GR.tif"]
#                      
# #Output path
# out_path = r'D:\800 Operational\c01_verdonk\Wever west\Season evaluation\test\soil1' 
# #Specify path to clip_shp
# clip_shp = r"D:\800 Operational\c01_verdonk\Rijweg stalling 1\clip_shape.shp"
# 
# #img_path = img_path_list[0]
# =============================================================================

#%%

# =============================================================================
# #Input img_path
# img_path_list = [r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ91\20190527\1321\Orthomosaic\c08_biobrass-AZ91-201905271321.tif"
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ91\20190614\1505\Orthomosaic\c08_biobrass-AZ91-201906141505-GR.vrt",
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ91\20190620\1457\Orthomosaic\c08_biobrass-AZ91-201906201457-GR.vrt",
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ91\20190628\1013\Orthomosaic\c08_biobrass-AZ91-201906281013-GR.vrt",
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ91\20190705\0837\Orthomosaic\c08_biobrass-AZ91-201907050837-GR.vrt",
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ91\20190716\1410\Orthomosaic\c08_biobrass-AZ91-201907161410-GR.vrt"
#                  ]
#                      
# #Output path
# out_path = r'D:\800 Operational\c08_biobrass\AZ91\Season evaluation'
# #Specify path to clip_shp
# clip_shp = r"D:\800 Operational\c08_biobrass\AZ91\clip_shape.shp"
# =============================================================================

#%%

# =============================================================================
# #Input img_path
# img_path_list = [#r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190517\1650\Orthomosaic\c08_biobrass-AZ74-201905171650-GR.vrt",                 
#                  r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190620\1550\Orthomosaic\c08_biobrass-AZ74-201906201550-GR.vrt"]#,
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190527\1248\Orthomosaic\c08_biobrass-AZ74-201905271248-GR.vrt",
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190529\1141\Orthomosaic\c08_biobrass-AZ74-201905291141-GR.vrt",
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190604\1446\Orthomosaic\c08_biobrass-AZ74-201906041446-GR.vrt",
#                  #r"D:\VanBovenDrive\VanBoven MT\Archive\c08_biobrass\AZ74\20190614\1536\Orthomosaic\c08_biobrass-AZ74-201906141536-GR.vrt"]
#                      
# #Output path
# out_path = r'D:\800 Operational\c08_biobrass\AZ74\Season evaluation'
# #Specify path to clip_shp
# clip_shp = r"D:\800 Operational\c08_biobrass\AZ74\Clip_shape.shp"
# =============================================================================

#%%

#Input img_path
# =============================================================================
# img_path_list = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190906\0917\Orthomosaic\c07_hollandbean-Aart Maris-201909060917-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190501\0200\Orthomosaic\c07_hollandbean-Aart Maris-20190501.tif",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190513\1240\Orthomosaic\c07_hollandbean-Aart Maris-201905131240-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190522\0956\Orthomosaic\c07_hollandbean-Aart Maris-201905220956-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190527\1152\Orthomosaic\c07_hollandbean-Aart Maris-201905271152-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190603\0946\Orthomosaic\c07_hollandbean-Aart Maris-201906030946-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190617\1333\Orthomosaic\c07_hollandbean-Aart Maris-201906171333-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190624\0942\Orthomosaic\c07_hollandbean-Aart Maris-201906240942-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190703\0945\Orthomosaic\c07_hollandbean-Aart Maris-201907030945-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190708\0914\Orthomosaic\c07_hollandbean-Aart Maris-201907080914-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190724\1028\Orthomosaic\c07_hollandbean-Aart Maris-201907241028-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190806\1005\Orthomosaic\c07_hollandbean-Aart Maris-201908061005-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190823\1048\Orthomosaic\c07_hollandbean-Aart Maris-201908231048-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Aart Maris\20190829\1123\Orthomosaic\c07_hollandbean-Aart Maris-201908291123-GR.vrt"
#                  ]
#                      
# #Output path
# out_path = r'D:\800 Operational\c07_hollandbean\Season evaluation'
# 
# #Specify path to clip_shp
# clip_shp = r"D:\800 Operational\c07_hollandbean\Season evaluation\clip_shapes\Aart_Maris_clip_shp.shp"
# =============================================================================

#%%

#Input img_path
#img_path_list = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190724\0955\Orthomosaic\c07_hollandbean-Hein de Schutter-201907240955-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190806\1119\Orthomosaic\c07_hollandbean-Hein de Schutter-201908061119-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190823\1146\Orthomosaic\c07_hollandbean-Hein de Schutter-201908231146-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190829\1210\Orthomosaic\c07_hollandbean-Hein de Schutter-201908291210-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190906\1020\Orthomosaic\c07_hollandbean-Hein de Schutter-201909061020-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190617\1419\Orthomosaic\c07_hollandbean-Hein de Schutter-201906171419-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190624\1432\Orthomosaic\c07_hollandbean-Hein de Schutter-201906241432-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190703\1301\Orthomosaic\c07_hollandbean-Hein de Schutter-201907031301-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190708\1101\Orthomosaic\c07_hollandbean-Hein de Schutter-201907081101-GR.vrt",
#                 ]
img_path_list = [#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190605\1255\Orthomosaic\c07_hollandbean-Hein de Schutter-201906051255.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190708\1101\Orthomosaic\c07_hollandbean-Hein de Schutter-201907081101-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190724\0955\Orthomosaic\c07_hollandbean-Hein de Schutter-201907240955-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190806\1119\Orthomosaic\c07_hollandbean-Hein de Schutter-201908061119-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hein de Schutter\20190823\1146\Orthomosaic\c07_hollandbean-Hein de Schutter-201908231146-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190819\1156\Orthomosaic\c07_hollandbean-Hage-201908191156-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190830\1016\Orthomosaic\c07_hollandbean-Hage-201908301016-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190905\1318\Orthomosaic\c07_hollandbean-Hage-201909051318-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190503\0200\Orthomosaic\c07_hollandbean-Hage-20190503.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190513\1326\Orthomosaic\c07_hollandbean-Hage-201905131326-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190522\1113\Orthomosaic\c07_hollandbean-Hage-201905221113-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190527\1347\Orthomosaic\c07_hollandbean-Hage-201905271347-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190603\1128\Orthomosaic\c07_hollandbean-Hage-201906031128-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190619\1259\Orthomosaic\c07_hollandbean-Hage-201906191259-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190625\0840\Orthomosaic\c07_hollandbean-Hage-201906250840-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190701\1102\Orthomosaic\c07_hollandbean-Hage-201907011102-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190710\1111\Orthomosaic\c07_hollandbean-Hage-201907101111-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190724\1108\Orthomosaic\c07_hollandbean-Hage-201907241108-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190724\1201\Orthomosaic\c07_hollandbean-Hendrik de Heer-201907241201-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190802\0933\Orthomosaic\c07_hollandbean-Hendrik de Heer-201908020933-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190819\1243\Orthomosaic\c07_hollandbean-Hendrik de Heer-201908191243-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190830\0942\Orthomosaic\c07_hollandbean-Hendrik de Heer-201908300942-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190905\1401\Orthomosaic\c07_hollandbean-Hendrik de Heer-201909051401-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190503\0100\Orthomosaic\c07_hollandbean-Hendrik de Heer-201905030100.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190513\1422\Orthomosaic\c07_hollandbean-Hendrik de Heer-201905131422-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190522\1036\Orthomosaic\c07_hollandbean-Hendrik de Heer-201905221036-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190527\1311\Orthomosaic\c07_hollandbean-Hendrik de Heer-201905271311-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190603\1049\Orthomosaic\c07_hollandbean-Hendrik de Heer-201906031049-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190619\1238\Orthomosaic\c07_hollandbean-Hendrik de Heer-201906191238-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190625\0808\Orthomosaic\c07_hollandbean-Hendrik de Heer-201906250808-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190701\1024\Orthomosaic\c07_hollandbean-Hendrik de Heer-201907011024-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hendrik de Heer\20190710\1037\Orthomosaic\c07_hollandbean-Hendrik de Heer-201907101037-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190830\0729\Orthomosaic\c07_hollandbean-Joke Visser-201908300729-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190906\0802\Orthomosaic\c07_hollandbean-Joke Visser-201909060802-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190522\1245\Orthomosaic\c07_hollandbean-Joke Visser-201905221245.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190527\1514\Orthomosaic\c07_hollandbean-Joke Visser-201905271514-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190603\1020\Orthomosaic\c07_hollandbean-Joke Visser-201906031020-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190619\1208\Orthomosaic\c07_hollandbean-Joke Visser-201906191208-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190625\0739\Orthomosaic\c07_hollandbean-Joke Visser-201906250739-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190701\0933\Orthomosaic\c07_hollandbean-Joke Visser-201907010933-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190710\1007\Orthomosaic\c07_hollandbean-Joke Visser-201907101007-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190724\1431\Orthomosaic\c07_hollandbean-Joke Visser-201907241431-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190802\0829\Orthomosaic\c07_hollandbean-Joke Visser-201908020829-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Joke Visser\20190823\1004\Orthomosaic\c07_hollandbean-Joke Visser-201908231004-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190829\1048\Orthomosaic\c07_hollandbean-Osseweyer-201908291048-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190906\0841\Orthomosaic\c07_hollandbean-Osseweyer-201909060841-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190522\0915\Orthomosaic\c07_hollandbean-Osseweyer-201905220915.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190529\0830\Orthomosaic\c07_hollandbean-Osseweyer-201905290830-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190605\1202\Orthomosaic\c07_hollandbean-Osseweyer-201906051202-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190617\1212\Orthomosaic\c07_hollandbean-Osseweyer-201906171212-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190624\1346\Orthomosaic\c07_hollandbean-Osseweyer-201906241346-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190703\1025\Orthomosaic\c07_hollandbean-Osseweyer-201907031025-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190708\1008\Orthomosaic\c07_hollandbean-Osseweyer-201907081008-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190724\1241\Orthomosaic\c07_hollandbean-Osseweyer-201907241241-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190806\0922\Orthomosaic\c07_hollandbean-Osseweyer-201908060922-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Osseweyer\20190823\0926\Orthomosaic\c07_hollandbean-Osseweyer-201908230926-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190827\0803\Orthomosaic\c07_hollandbean-Jos Schelling-201908270803-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190905\1036\Orthomosaic\c07_hollandbean-Jos Schelling-201909051036-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190522\1346\Orthomosaic\c07_hollandbean-Jos Schelling-201905221346.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190529\0920\Orthomosaic\c07_hollandbean-Jos Schelling-201905290920-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190603\1318\Orthomosaic\c07_hollandbean-Jos Schelling-201906031318-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190618\1027\Orthomosaic\c07_hollandbean-Jos Schelling-201906181027-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190625\1102\Orthomosaic\c07_hollandbean-Jos Schelling-201906251102-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190701\1300\Orthomosaic\c07_hollandbean-Jos Schelling-201907011300-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190709\1020\Orthomosaic\c07_hollandbean-Jos Schelling-201907091020-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190724\1542\Orthomosaic\c07_hollandbean-Jos Schelling-201907241542-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Jos Schelling\20190819\1407\Orthomosaic\c07_hollandbean-Jos Schelling-201908191407-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190830\1050\Orthomosaic\c07_hollandbean-Mellisant-201908301050-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190905\1226\Orthomosaic\c07_hollandbean-Mellisant-201909051226-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190522\1153\Orthomosaic\c07_hollandbean-Mellisant-201905221153.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190527\1437\Orthomosaic\c07_hollandbean-Mellisant-201905271437-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190603\1216\Orthomosaic\c07_hollandbean-Mellisant-201906031216-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190619\1311\Orthomosaic\c07_hollandbean-Mellisant-201906191311-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190625\0935\Orthomosaic\c07_hollandbean-Mellisant-201906250935-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190701\1144\Orthomosaic\c07_hollandbean-Mellisant-201907011144-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190710\1152\Orthomosaic\c07_hollandbean-Mellisant-201907101152-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190724\1131\Orthomosaic\c07_hollandbean-Mellisant-201907241131-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190802\0958\Orthomosaic\c07_hollandbean-Mellisant-201908020958-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Mellisant\20190819\1242\Orthomosaic\c07_hollandbean-Mellisant-201908191242-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190827\1050\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201908271050-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190905\1134\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201909051134-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190527\1542\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201905271542.tif",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190603\1305\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201906031305-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190618\0944\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201906180944-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190625\1017\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201906251017-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190701\1229\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201907011229-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190709\0932\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201907090932-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190724\1459\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201907241459-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190802\1136\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201908021136-GR.vrt",
#r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\NoviFarm 1 8ha\20190819\1017\Orthomosaic\c07_hollandbean-NoviFarm 1 8ha-201908191017-GR.vrt"
]
                     
#Output path
out_path = r'D:\800 Operational\c07_hollandbean\Season evaluation\Counts'

#Specify path to clip_shp
clip_shp = r"D:\800 Operational\c07_hollandbean\Season evaluation\clip_shapes\Aart_Maris_klein.shp"

#%%

# =============================================================================
# #Input img_path
# img_path_list = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190819\1156\Orthomosaic\c07_hollandbean-Hage-201908191156-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190503\0200\Orthomosaic\c07_hollandbean-Hage-20190503.tif",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190513\1326\Orthomosaic\c07_hollandbean-Hage-201905131326-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190522\1113\Orthomosaic\c07_hollandbean-Hage-201905221113-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190527\1347\Orthomosaic\c07_hollandbean-Hage-201905271347-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190603\1128\Orthomosaic\c07_hollandbean-Hage-201906031128-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190619\1259\Orthomosaic\c07_hollandbean-Hage-201906191259-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190625\0840\Orthomosaic\c07_hollandbean-Hage-201906250840-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190701\1102\Orthomosaic\c07_hollandbean-Hage-201907011102-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190710\1111\Orthomosaic\c07_hollandbean-Hage-201907101111-GR.vrt",
# r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190724\1108\Orthomosaic\c07_hollandbean-Hage-201907241108-GR.vrt",
#                  ]
#                      
# #Output path
# out_path = r'D:\800 Operational\c07_hollandbean\Season evaluation'
# 
# #Specify path to clip_shp
# clip_shp = r"D:\800 Operational\c07_hollandbean\Season evaluation\clip_shapes\Hage_clip_shp.shp"
# 
# =============================================================================

#%%
# img_path_list = [r"D:\VanBovenDrive\VanBoven MT\Archive\c07_hollandbean\Hage\20190819\1156\Orthomosaic\c07_hollandbean-Hage-201908191156-GR.vrt",

#Output path
out_path = r'D:\800 Operational\c07_hollandbean\Season evaluation'
 
#Specify path to clip_shp
clip_shp = r"D:\800 Operational\c07_hollandbean\Season evaluation\clip_shapes\Hage_clip_shp.shp"


#%%
#Set variables for clustering

#Set lower and higher limits to plant size (px)
min_area = 16
max_area = 16000000000000

#Set no_data_value
no_data_value = 255

#Use small block size to cluster based on colors in local neighbourhood
x_block_size = 512
y_block_size = 512

#Create iterator to process blocks of imagery one by one and be able to process subsets.
it = list(range(0,2000000, 1))

#True if you want to generate shapefile output with points and shapes of plants
vectorize_output = False
#True if you want to fit the cluster centres iteratively to every image block, false if you fit in once to a random 10% subset of the entire ortho
iterative_fit = False
#True to clip ortho to clip_shape (usefull for removing grass around the fields)
clip_ortho2shp = False
#True to create a tif file of the plant mask of the image.
tif_output = True
#True if you want to write plant count and segmentation from clustering to file
write_shp2file = False
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
            gdf.to_file(os.path.join(out_path, (os.path.basename(img_path)[-16:-4] + '_plant_count.gpkg')), driver = 'GPKG')