import rasterio

import sys

import os

import cv2

import numpy as np

import pandas as pd

from qgis.core import *

qgs = QgsApplication([], False)

QgsApplication.setPrefixPath("C:\Program Files\QGIS 3.28.4\bin", True)

qgs.initQgis()

sys.path.append('C:\Program Files\QGIS 3.28.4\apps\qgis-ltr\python\plugins')

import processing

from processing.core.Processing import Processing

Processing.initialize()

# IMPORTANT STUFFF INFACT EXTREMELY CRUCIAL

# tiff file and marked area should be in EPSG 3857 

# dissolve and fix geometry for marked areas

# make convex hull for marked_area_raster_images

# expression for filter should use marked_area_raster_images

# create temp files folder

print('starting')

print('--------------------------')

tile_size = 1024

tiff_path = r"C:\Users\AhmadWaseem\Desktop\bhalli\PD-Seg Work\datasets for training - 20Z\tiff files\2017\2017 reproject.tif"

grid_file_path =  r"C:\Users\AhmadWaseem\Desktop\bhalli\PD-Seg Work\tiling code\2017 temp files\grid_file.csv"

marked_area_raster_images_path = r"C:\Users\AhmadWaseem\Desktop\bhalli\PD-Seg Work\datasets for training - 20Z\markings\2017\marked_area_raster_images_wo_new_markings\marked_area_raster_images.shp"

marked_area_marking_images_path = r"C:\Users\AhmadWaseem\Desktop\bhalli\PD-Seg Work\datasets for training - 20Z\markings\2017\marked_area_marking_images_wo_new_markings\marked_area_marking_images.shp"

filterd_grid_file_path = r'C:\Users\AhmadWaseem\Desktop\bhalli\PD-Seg Work\tiling code\2017 temp files\filterd_grid_file.csv'

raster_images_folder_path = r'C:\Users\AhmadWaseem\Desktop\bhalli\PD-Seg Work\datasets for training - 20Z\TS 1024 wo new markings\raster images\2017'

marking_images_folder_path = r'C:\Users\AhmadWaseem\Desktop\bhalli\PD-Seg Work\datasets for training - 20Z\TS 1024 wo new markings\marking images\2017'

expression_for_filter = "\"marked_area_raster_images_area\" > '0'"

non_cliped_raster = rasterio.open(tiff_path)

def clip_raster(tiff_path,marked_area_raster_images_path):

    print('clipping raster')

    cliped_raster = processing.run("gdal:cliprasterbymasklayer", {'INPUT':f'{tiff_path}',
    'MASK':f'{marked_area_raster_images_path}','SOURCE_CRS':None,'TARGET_CRS':None,
    'TARGET_EXTENT':None,'NODATA':None,'ALPHA_BAND':False,'CROP_TO_CUTLINE':False,'KEEP_RESOLUTION':False,'SET_RESOLUTION':False,'X_RESOLUTION':None,'Y_RESOLUTION':None,
    'MULTITHREADING':False,'OPTIONS':'','DATA_TYPE':0,'EXTRA':'','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

    print('done')

    return cliped_raster

cliped_raster = rasterio.open(clip_raster(tiff_path,marked_area_raster_images_path))

def create_tiles(tile_size,grid_file_path,non_cliped_raster):

    print('creating tiles')

    x_min_d = non_cliped_raster.bounds[0] 
    x_max_d = non_cliped_raster.bounds[2]  
    y_min_d = non_cliped_raster.bounds[1]  
    y_max_d = non_cliped_raster.bounds[3]  

    image = non_cliped_raster.read()
    image_x_pix = image.shape[2]
    image_y_pix = image.shape[1]

    x_delta = (x_max_d - x_min_d) / image_x_pix
    y_delta = -1 * (y_max_d - y_min_d) / image_y_pix

    x_tiles_num = image_x_pix // tile_size
    y_tiles_num = image_y_pix // tile_size

    fw = open(grid_file_path, 'w')
    fw.write("x1,x2,y1,y2,filename,Geometry\n")
    fw.close()

    idx = 0
    for x_tile_idx in range(x_tiles_num):
        x1 = tile_size * x_tile_idx
        x2 = x1 + tile_size
        for y_tile_idx in range(y_tiles_num):
            y1 = tile_size * y_tile_idx
            y2 = y1 + tile_size

            x_min = float(x_min_d + x1 * x_delta)
            x_max = float(x_min_d + x2 * x_delta)
            y_max = float(y_max_d + y1 * y_delta)
            y_min = float(y_max_d + y2 * y_delta)

            polygon_str = "{} {},{} {},{} {},{} {},{} {}".format(x_min, y_min, x_max, y_min,
                                                                x_max, y_max, x_min, y_max, x_min, y_min)

            fa = open(grid_file_path, 'a')
            fa.write("%d,%d,%d,%d,%d,\"POLYGON ((%s))\"\n" % (x1,x2,y1,y2,idx, polygon_str))
            fa.close()

            idx = idx + 1
    print('done')

def overlap_analysis(expression_for_filter,grid_file_path,marked_area_raster_images_path,filterd_grid_file_path):

    print('perfoming overlap analysis')

    grid_file_path = grid_file_path.replace("\\", "/")

    overlap_analysis = processing.run("native:calculatevectoroverlaps", 
    {'INPUT':f'delimitedtext://file:///{grid_file_path}?type=csv&maxFields=10000&detectTypes=yes&wktField=Geometry&geomType=Polygon&crs=EPSG:3857&spatialIndex=no&subsetIndex=no&watchFile=no',
    'LAYERS':[f'{marked_area_raster_images_path}'],'OUTPUT':'TEMPORARY_OUTPUT','GRID_SIZE':None})['OUTPUT']

    overlap_analysis.setSubsetString(expression_for_filter)

    QgsVectorFileWriter.writeAsVectorFormat(overlap_analysis,filterd_grid_file_path,'utf-8',driverName = 'CSV',layerOptions=['GEOMETRY=AS_WKT'])

    print('done')

def create_raster_images(dataset,images_folder_path,filterd_grid_file_path):

    print('creating images')

    main_image = np.transpose(dataset.read(), (1, 2, 0))

    filterd_grid_file = pd.read_csv(filterd_grid_file_path)

    images = filterd_grid_file['filename']

    for image in images:

        img_coordinates = filterd_grid_file.query("filename == @image")

        x1 = int(img_coordinates['x1'])

        x2 = int(img_coordinates['x2'])

        y1 = int(img_coordinates['y1'])

        y2 = int(img_coordinates['y2'])

        img_crop = main_image[y1:y2, x1:x2]

        cv2.imwrite(os.path.join(images_folder_path, '{}.png'.format(image)), img_crop)

        print(f"written image : {image}")

    print('done')

def rasterize_marked_area_marking_images(non_cliped_raster,marked_area_marking_images_path):

    print('rasterizing marked_area_marking_images')

    width = non_cliped_raster.width

    height = non_cliped_raster.height

    bounds = non_cliped_raster.bounds

    rasterized = processing.run("gdal:rasterize", {'INPUT':f'{marked_area_marking_images_path}',
    'FIELD':'','BURN':1,'USE_Z':False,'UNITS':0,'WIDTH':width,'HEIGHT':height,'EXTENT':f'{bounds[0]},{bounds[2]},{bounds[1]},{bounds[3]} [EPSG:3857]',
    'NODATA':0,'OPTIONS':'','DATA_TYPE':0,'INIT':None,'INVERT':False,'EXTRA':'','OUTPUT':'TEMPORARY_OUTPUT'})['OUTPUT']

    print('done')

    return rasterized

def create_markings_npy_files(dataset,images_folder_path,filterd_grid_file_path):

    print('creating npy files')

    main_image = np.transpose(dataset.read(), (1, 2, 0))

    filterd_grid_file = pd.read_csv(filterd_grid_file_path)

    images = filterd_grid_file['filename']

    for image in images:

        img_coordinates = filterd_grid_file.query("filename == @image")

        x1 = int(img_coordinates['x1'])

        x2 = int(img_coordinates['x2'])

        y1 = int(img_coordinates['y1'])

        y2 = int(img_coordinates['y2'])

        img_crop = main_image[y1:y2, x1:x2]

        np.save(os.path.join(images_folder_path, '{}.npy'.format(image)), img_crop)

        print(f"written file : {image}")

    print('done')

create_tiles(tile_size,grid_file_path,non_cliped_raster)

overlap_analysis(expression_for_filter,grid_file_path,marked_area_raster_images_path,filterd_grid_file_path)

create_raster_images(cliped_raster,raster_images_folder_path,filterd_grid_file_path)

rasterized_marked_area_marking_images = rasterio.open(rasterize_marked_area_marking_images(non_cliped_raster,marked_area_marking_images_path))

create_markings_npy_files(rasterized_marked_area_marking_images,marking_images_folder_path,filterd_grid_file_path)

print('--------------------------')

print('end')
