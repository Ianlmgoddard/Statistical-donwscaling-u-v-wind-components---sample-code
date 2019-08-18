import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd 
import time


def get_mapping_n(co_ord,grid_spacing,n):
		'''
		this function generates the map matrix which saves the indices of the nearest neighbours from the low resolution (ERA data)
		to any of the grid points in the high resolution data.
		
		args:
			co_ord: (tuple), the latitude,longitude of a co-ordinate in the high res grid
			grid_spacing: float, the spacing between points in the low resolution grid, currently set to 0.25
			n: int, the number of nearest neighbours you want to include, must be an odd square number!
				
		'''

		gs = grid_spacing

		lat,lon = co_ord[0],co_ord[1]
		#this is done to break ties, i.e if a lon/lat value lies on a grid line, we check if lon/lat are whole numbers.
		if lat %1 == False or lat %0.5 == False or lat %0.25 == False:
			print('break tie -- latitude is: {}'.format(lat))
			lat += 0.0000001
		if  lon % 1 == False or lon% 0.5 == False or lon%0.25 == False:
			print('break tie -- longitude is: {}'.format(lon))
			lon += 0.0000001
			
		nearest = ERA_onetime.sel(longitude = lon,latitude = lat,method = 'nearest')
		nearest_lat,nearest_lon = float(nearest.latitude.values),float(nearest.longitude.values)

		half_row = int((np.sqrt(n)-1)/2)
		row_len = int(np.sqrt(n))
		#nn_left,nn_right  = (nearest_lon-(7*gs,nearest_lat),(nearest_lon+gs,nearest_lat)
		nn_middle = list(zip(np.linspace(nearest_lon-(half_row *gs),nearest_lon+(half_row *gs),row_len ),[nearest_lat]*row_len ))
		nns = nn_middle
		for k in range(1,half_row+1):
			nn_up_row = list(zip(np.linspace(nearest_lon-(half_row *gs),nearest_lon+(half_row *gs),row_len ),[nearest_lat+(k*gs)]*row_len ))
			nn_down_row = list(zip(np.linspace(nearest_lon-(half_row *gs),nearest_lon+(half_row *gs),row_len ),[nearest_lat-(k*gs)]*row_len ))
			
			nns = nns + nn_up_row +nn_down_row
		nns = np.asarray(nns)

		xs = nns[:,0]
		ys = nns[:,1]
		lon_idxs =  np.empty(0,dtype = int)
		lat_idxs = np.empty(0,dtype = int)
		for k in range(len(xs)):			#loop gets idxs of nearest neighbours for efficient look up of nn values, in component values array
			#print(xs[k],ys[k])
			lon_idx = np.argwhere(ERA_onetime.longitude.values == xs[k])[0]
			lon_idxs = np.concatenate([lon_idxs,lon_idx])
			lat_idx = np.argwhere(ERA_onetime.latitude.values == ys[k])[0]
			lat_idxs = np.concatenate([lat_idxs,lat_idx])

		idxs_new = np.vstack((lat_idxs,lon_idxs)).T.astype(int)

		return idxs_new


ERA_100m_u_filepath = os.path.join(os.getcwd(),'ERA_data_source/100m_u_extended.nc')
ERA_100m_u_data = xr.open_dataset(ERA_100m_u_filepath)

co_ord_matrix = np.load('co_ord_matrix.npy')


timepoint = '2000-01-01T00:00:00.000000000'

ERA_onetime = ERA_100m_u_data.sel(time = timepoint)	


region_size = (390,330)
loop1_time = time.time()
n = 9	#needs to be an odd square number!
map_matrix = np.zeros((region_size[0],region_size[1],n,2),dtype = int)
for i in range(region_size[0]):
	for j in range(region_size[1]):
		co_ord = co_ord_matrix[i,j]		
		idxs = get_mapping_n(co_ord,0.25,n)
		#print(idxs)
		map_matrix[i,j,:,:] = idxs

#np.save('121_nearest_neighbour_matrix_auto.npy',map_matrix)
