import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.basemap import Basemap
import pandas as pd 
from plotting import ERA_plotter,hr_plotter
from scipy.stats import iqr
import pylab
import shutil



def get_error_over_all_vals(error_list,all_vals):
	'''
	this function takes a list of the predction errors and a list og the Ground truth values
	and outputs the RMSE/IQR, where IQR is over the the ground truth values
	'''
	error_list = np.asarray(error_list)
	mean_error = np.mean(error_list**2)
	sq_mean_error = np.sqrt(mean_error)
	
	if type(all_vals) != list:
		all_vals = all_vals.flatten()

	IQR = np.subtract(*np.percentile(all_vals, [75, 25]))

	#print('RMSE {}, IQR {}'.format(sq_mean_error, IQR))
	return sq_mean_error, IQR


def get_daily_relative_errors(predictions, ground_truth,num_years):
	'''
	args:	predictions - the predicted data
		ground_truth - the ground truth values
		num_years - the number of years over which we have made predictions
	
	returns:
		list of daily RMSE values, has shape (365,)
		spatial errors of shape (region_size)

	computes the reltaive RMSE over a daily period and all validation set years and the spatial errors
	'''
	
	daily_preds = predictions.reshape(num_years,int(predictions.shape[0]/(24*num_years)),24,50,50)
	daily_GT = ground_truth.reshape(num_years,int(predictions.shape[0]/(24*num_years)),24,50,50)

	get_error_over_all_vals((daily_preds-daily_GT).flatten(),daily_GT.flatten())

	daily_sq_error = (daily_preds- daily_GT)**2
	mean_daily_sq_error = np.mean(daily_sq_error,axis = (0,2,3,4))
	root_mean_daily_sq_error  =np.sqrt(mean_daily_sq_error)
	daily_iqr = iqr(daily_GT,axis = (0,2,3,4))

	daily_spatial_iqr = iqr(daily_GT,axis =(0,1,2))

	mean_error_spatial = np.sqrt(np.mean(daily_sq_error,axis = (0,1,2)))/daily_spatial_iqr
	
	return root_mean_daily_sq_error/daily_iqr,mean_error_spatial



def get_daily_raw_errors(predictions,ground_truth,num_years):
	'''
	args:	predictions - the predicted data
		ground_truth - the ground truth values
		num_years - the number of years over which we have made predictions
	
	returns:
		list of daily raw error values, has shape (365,)
		list of daily stds, has shape (365,)

	computes the raw errors and standard deviation over a daily period and 
				all validation set years and all points in region
	'''

	daily_preds = predictions.reshape(num_years,int(predictions.shape[0]/(24*num_years)),24,50,50)
	daily_GT = ground_truth.reshape(num_years,int(predictions.shape[0]/(24*num_years)),24,50,50)

	daily_errors = abs(daily_preds-daily_GT)
	mean_daily_errors = np.mean(daily_errors ,axis =(0,2,3,4))
	daily_std = np.std(daily_errors,axis = (0,2,3,4))

	return mean_daily_errors,daily_std


def plt_figs(spatial_errs,daily_raw_errs,full_GT,location,temp,spatial,daily_stds):
	'''
	This plots the spatial and yearly error plots and save them directories specified by temp and spatial
	args:
		spatial errors: this is the second output from get_daily_relative_errors	
		daily_raw_errs: the first output from  get_daily_raw_errors
		full_GT: the ground truth values, has shape (num_hours_predicted, region_size)
		location: string-  which region we are analysing
		temp: string - directory in which to save the temporal error plots
		spatial: string - directory in which to save the spatial error plots
		daily_stds: the second output from  get_daily_raw_errors
		
		
		

	'''

	fig1,axs = plt.subplots(1,1,figsize = (10,8))
	
	mean_raw_val = np.around(np.mean(daily_raw_errs),4)
	raw_smoothed = smooth(np.asarray(daily_raw_errs),14)	
	axs.plot(raw_smoothed,'r',label = r'MAE')
	axs.plot(smooth(np.asarray(daily_stds),14),'k',label = r'$\sigma$')
	error_plus_std = daily_raw_errs + daily_stds
	error_minus_std =daily_raw_errs - daily_stds
	axs.fill_between(np.arange(0,len(raw_smoothed),1),smooth(np.asarray(error_minus_std),14),smooth(np.asarray(error_plus_std),14),
										label = r'MAE $\pm + \sigma$',alpha = 0.2)
	#axs.plot(daily_raw_errs,'b',alpha = 0.2)
	axs.set_ylim(0,4.5)
	axs.set_xlabel('Day',fontsize = 20)
	axs.set_ylabel(r'$MAE^{day}$',fontsize = 20)
	#if location == 'fort_agustus':
	#	axs.legend(fontsize = 20,loc = 'center left',bbox_to_anchor= (1.05,0.5))


	fig1.savefig(os.path.join(temp,(location +'.pdf')),bbox_inches = 'tight') 
	#fig1.savefig(year +save_str +'.pdf') 


	fig2,axs2 = plt.subplots(1,2,figsize = (15,8))
	im_GT = axs2[1].imshow(np.flipud(np.mean(full_GT,axis = 0)))
	axs2[1].set_title('Average Ground Truth')
	fig2.colorbar(im_GT,ax = axs2[1])
	
	im = axs2[0].imshow(np.flipud(spatial_errs))
	axs2[0].set_title(r'$\frac{RMSE}{IQR}$ error (over time)')
	fig2.colorbar(im,ax = axs2[0])
	
	fig2.savefig(os.path.join(spatial,(location +'.pdf')),bbox_inches = 'tight')


	plt.clf()
	plt.close()


'''
def get_error_small_region(hr_data,interp_data,component_str):
	GT_vals = hr_data[component_str].values
	pred_vals = interp_data[component_str].values
	mask = np.ma.masked_where(abs(pred_vals) >0,abs(pred_vals)).mask.astype(int)
	mask = np.where(mask ==0,np.nan,mask)
	GT_vals_small_region = np.multiply(mask,GT_vals)

	diff_map = GT_vals_small_region - pred_vals
	error = diff_map.flatten()
	mean_error = np.nanmean(error**2)
	sq_mean_error = np.sqrt(mean_error)
	#print('RMSE: {}'.format(sq_mean_error))
	
	#get relative error
	vals_in_region = np.multiply(mask,pred_vals)
	vals_in_region =list(vals_in_region[~np.isnan(vals_in_region)].flatten())
	#mean_val = np.mean(abs(vals_in_region))
	
	return sq_mean_error, vals_in_region

'''



def smooth(y, box_pts):
	'''
	this funtion takes an array/list and smooths the values over a period defined by box_pts
	y are the values to smooth, box_pts is the region over which we smooth
	'''

	box = np.ones(box_pts)/box_pts
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth












