import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'util'))
import pandas as pd 
from get_region import pick_region
from get_year_list import *
import shutil

def get_points(idx_matrix,uvals, ulons,ulats):
	'''
	gets the nearest neighbours of a high res point from the low res grid for interpolation to the high res point
	'''
	points = []
	for s in range(len(idx_matrix[:,0])):
		#print(s)
		lat_idx = idx_matrix[s][0]
		lon_idx = idx_matrix[s][1]
		u_point = uvals[lat_idx,lon_idx]

		lon_new = ulons[lon_idx] 
		lat_new = ulats[lat_idx] 
		points.append(tuple([lon_new,lat_new,u_point]))
			
		
	points = tuple(points)
	return points
	

def bi_inter(co_ord, points):

	'''
	given the co-ordinate and point to interpolate, this function performs the interpolation
	'''
	x,y = co_ord[1], co_ord[0]
	points = sorted(points)               # order points by x, then by y
	(x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

	if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
		raise ValueError('points do not form a rectangle')
	if not x1 <= x <= x2 or not y1 <= y <= y2:
		raise ValueError('(x, y) not within the rectangle')

	return (q11 * (x2 - x) * (y2 - y) +
	    q21 * (x - x1) * (y2 - y) +
	    q12 * (x2 - x) * (y - y1) +
	    q22 * (x - x1) * (y - y1)
	   ) / ((x2 - x1) * (y2 - y1) + 0.0)

def get_interp_grid(region_size,map_matrix,co_ord_matrix,ulons,ulats,uvals):
	'''
	this function calls the get_points and bi_inter function and performs the interpolation 
	over the region we specify. 
	
	'''
	pred_matrix = np.zeros(region_size)

	for s in range(region_size[0]):

		for t in range(region_size[1]):

			co_ord = co_ord_matrix[s,t]
			idx_matrix = map_matrix[s,t,:,:]
			points = get_points(idx_matrix,uvals,ulons,ulats)
			f = bi_inter(co_ord,points) 
			pred_matrix[s,t] = f


	return pred_matrix



def get_day_interp(component,date, ulons,ulats,region_size,map_matrix,co_ord_matrix,num_hours):
		'''
		this function interpolates a days worth of data
		'''
		
		pred_matrix = np.zeros((num_hours,region_size[0],region_size[1]))
		date = date.to_datetime64()
		daily_data = component[(component['time'].time >= date)&(component['time'].time < (date+pd.Timedelta(days=1)).to_datetime64())]

		for j in range(num_hours):
			vals = daily_data.values[j,:,:]
			pred_matrix[j,:,:] = get_interp_grid(region_size,map_matrix,co_ord_matrix,ulons,ulats,vals)


		return pred_matrix



def get_period_interp_region(counter,ERA_file,component_str,region_size,centre_point,hr_component_str,year,save_path):
	'''
	this function gets the interpolated data for the period we specify, by the list of years defined by 
	years, in code below this function.

	'''

	#check if save directory exists, if so delete it and create new for saving.
	region_dir = os.path.join(os.getcwd(),'interp_data_year'+str(counter)+'/'+save_path)
	if os.path.exists(region_dir):
    		shutil.rmtree(region_dir)

	os.makedirs(region_dir)
	
	#we just need this to get an example of the shape of hr_file output
	hr_file = (os.path.join(os.getcwd(),'jan01_corrected/upd_wind_2000-01-01.nc'))
	hr_data = xr.open_dataset(hr_file)
	component_dataset = xr.open_dataset(ERA_file)

	component = component_dataset[component_str]
	num_hours = 24

	
	list_of_days, year_list_with_dummy  = get_list_old(ERA_file) 	#old means operating on old ERA file, which runs from 2000-2011, new file runs from 1999 -2011
	
	days_in_year = list_of_days[(list_of_days >= year_list_with_dummy[year[0]]) & (list_of_days < year_list_with_dummy[year[1]])] 
	

	co_ord_matrix,map_matrix,lon_bounds,lat_bounds	=pick_region(centre_point,region_size,num_neighbours = 4)

	ulons = component.longitude.values
	ulats = component.latitude.values
	#get first num_days of interpolated data
	zero_fill = np.zeros(hr_data[hr_component_str].shape)
	for i in days_in_year:
		start_time = time.time()
		date = i 
		print(date)
		pred_matrix = get_day_interp(component = component,date = date,region_size = region_size,
						ulons = ulons,ulats = ulats,map_matrix= map_matrix,
						co_ord_matrix= co_ord_matrix,num_hours = num_hours)

		new_dataset = hr_data[hr_component_str].copy()			

		new_dataset.values = zero_fill			
		new_dataset.values[:,lon_bounds[0]:lon_bounds[1],lat_bounds[0]:lat_bounds[1]] = pred_matrix
		new_dataset.to_netcdf(region_dir+'/inter_' +str(date.date())+'.nc')
		loop_time= time.time()



#######################################################################################
co_ord_matrix = np.load('co_ord_matrix.npy')

ERA_100m_u_filepath = os.path.join(os.getcwd(),'ERA_data_source/100m_u_extended.nc')
ERA_100m_v_filepath =os.path.join(os.getcwd(),'ERA_data_source/100m_v_extended.nc')

years =[(6,7),(7,8),(8,9),(9,10)]
save_paths = ['v_newcastle','v_fort','v_coventry','v_north_sea']
locations = [(54.9783,-1.6178),(57.144364,-4.7190652),(52.4068,-1.5197),(58.38,-0.706)] 
for k in years:
	print(k)
	counter = k[0]
	for i in range(len(save_paths)):
		save_path = save_paths[i]
		centre_point = locations[i]
		get_period_interp_region(counter,ERA_file = ERA_100m_v_filepath,component_str = 'u100',region_size = (50,50),centre_point = centre_point,hr_component_str = 'U',year = k,save_path = save_path)
		print('Finished for {}'.format(save_path))

# np.ma.masked_where(new_dataset.values >0,new_dataset.values).mask
#torridon 57.5465033,-5.5147306
#fort augustus 57.144364,-4.7190652
#north east sea 58.38,-0.706
#coventry 52.4068 , -1.5197
#south west sea 51.0463, -7.242



 






