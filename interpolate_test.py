import sys
sys.path.append('/home/iangoddard/Desktop/MSC_proj/util')
from plotting import ERA_plotter,hr_plotter
from compare import *
from get_region import *






rootdir_suffix = ['v_fort']
locs =['fort_augustus']
year_dirs = [r'/media/iangoddard/Elements/MSC_proj/interp_data_year8/',r'/media/iangoddard/Elements/MSC_proj/interp_data_year9/']

f = open('Interpolate_test_v.txt', 'w')
scores = []
region_size = (50,50)


region_dir_spatial = '/home/iangoddard/Desktop/MSC_proj/testing_code/final_spatial'
if os.path.exists(region_dir_spatial):
	shutil.rmtree(region_dir_spatial)

os.makedirs(region_dir_spatial)

region_dir_temp = '/home/iangoddard/Desktop/MSC_proj/testing_code/final_temp'
if os.path.exists(region_dir_temp):
	shutil.rmtree(region_dir_temp)

os.makedirs(region_dir_temp)

for q in range(0,len(rootdir_suffix)):
	location = locs[q]
	suffix = rootdir_suffix[q]

	extension  = '.nc'

	many_year_raw_errors = np.empty((len(year_dirs),365))
	many_year_rel_errors = np.empty((len(year_dirs),365))
	many_spatial_errors = np.empty((len(year_dirs),region_size[0],region_size[1]))
	many_years_std = np.empty((len(year_dirs),365))

	for counter1,year in enumerate(year_dirs):
		rootdir =year +suffix 
		ground_truth_dir = r'/media/iangoddard/Elements/MSC_proj/Wind_corrected'

		#need to get the lon/lat bounds to select the correct region from the ground truth data
		centre_point =  get_coord_from_name(location)
		print(centre_point)
		co_ord_matrix,map_matrix,lon_bounds,lat_bounds = pick_region(centre_point,region_size,num_neighbours = 4) #just need the lon/lat bounds


		component_str ='V'

		period_error = []
		all_vals = []
		mat = np.empty((366,50,50))
		iqr_matrix = np.empty((366,50,50))
		for subdir,dirs,files in os.walk(rootdir):
			for counter,file in enumerate(sorted(files)):
				ext = os.path.splitext(file)[-1].lower()
				if ext == extension:
					pred_data = xr.open_dataset(os.path.join(rootdir,file))[component_str].values
					date = file[6:]
					GT_file_end = 'upd_wind_' +str(date)
					GT_file_str = os.path.join(ground_truth_dir,GT_file_end)
					GT_data = xr.open_dataset(GT_file_str)
					GT_data_region = GT_data[component_str].values[:,lon_bounds[0]:lon_bounds[1],lat_bounds[0]:lat_bounds[1]]
					pred_data = pred_data[:,lon_bounds[0]:lon_bounds[1],lat_bounds[0]:lat_bounds[1]]
					diff_map = (pred_data - GT_data_region)

			
					if counter == 0:
						full_GT = GT_data_region
						full_pred = pred_data
					else:
						full_GT = np.concatenate([full_GT,GT_data_region],axis = 0)
						full_pred = np.concatenate([full_pred,pred_data],axis = 0)
			
					all_vals.extend(GT_data_region.flatten())

		print('shape of predictions is {}'.format(full_pred.shape))

		rel_errs,spatial_errs = get_daily_relative_errors(full_pred, full_GT,1)

		daily_raw_errors,daily_std = get_daily_raw_errors(full_pred, full_GT,1)

		full_pred = full_pred[:8760,:,:]
		full_GT = full_GT[:8760,:,:]
		diff_map = full_pred - full_GT
		daily_diff_map = diff_map.reshape(int(diff_map.shape[0]/24),24,50,50)
		error = diff_map.flatten()
		RMSE,IQR = get_error_over_all_vals(error, full_GT)
		score = np.around(RMSE/IQR, 5)
		scores.append(score)
		

		print('the mean relative error over the whole spatio-temporal period is {}'.format(RMSE/IQR))

		if counter1 == 0:
			many_years_pred = np.empty((len(year_dirs),full_pred.shape[0],full_pred.shape[1],full_pred.shape[2]))
		many_years_pred[counter1,:,:,:] = full_pred
		many_year_raw_errors[counter1,:] = daily_raw_errors[:365]
		many_year_rel_errors[counter1,:] = rel_errs[:365]
		many_spatial_errors[counter1,:,:]= spatial_errs
		many_years_std[counter1,:] = daily_std[:365]

		#f.write(location +'\t' + str(year[-2]) +'\t' + str(score) +'\n')


		#plt_figs_inter(rel_errs,spatial_errs,daily_raw_errors,full_GT,location,year[-2])
		#plt.close()

	#save prediction matrix for plot
	#np.save('V!_'+str(location)+'_interp_.npy',np.mean(many_years_pred,axis =(0,1)))
	raw_errors_avg_over_years = np.mean(many_year_raw_errors,axis = 0)
	rel_errors_avg_over_years = np.mean(many_year_rel_errors,axis = 0)
	daily_stds_over_years = np.mean(many_years_std,axis =0)
	spatial_errors_avg_over_years =np.mean(many_spatial_errors,axis = 0)

	#f.write(location +'\t' + str(np.mean(scores)) +'\n')


	plt_figs(rel_errors_avg_over_years,spatial_errors_avg_over_years,raw_errors_avg_over_years,
				full_GT,location,region_dir_temp,region_dir_spatial,daily_stds_over_years)
	#plt.close()
	#scores = []
	


'''


#iqr = np.subtract(*np.percentile(all_vals, [75, 25]))
#smoothed = smooth(np.asarray(period_error),7)	
		
#plot
plt.clf()
plt.plot(smoothed,'r')
plt.plot(np.asarray(period_error),'b',alpha = 0.2)
plt.xlabel('day', fontsize = 14)
plt.ylabel('relative RMSE', fontsize = 14)
plt.axhline(np.mean(period_error),label = 'mean RMSE:' +str(np.around(np.mean(period_error)/iqr,decimals = 3)))
plt.legend(bbox_to_anchor = (1,1))
plt.savefig('year1_ed_newrel.pdf',bbox_inches = 'tight')
'''

