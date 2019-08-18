import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.getcwd(),'util'))
import time
import datetime
from get_region import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import *
from eff_get_in_out import *
from compare import *
from scipy.stats import mode
from get_lasso_group_idxs import *
from scipy import optimize


def scale_input(data_matrix):
	'''
	scales the data matrix before pushing through model
	'''
	scaler = StandardScaler()
	scaler = scaler.fit(data_matrix)
	train_scaled =scaler.transform(data_matrix)

	return train_scaled,scaler


def fit_MLR(scaled_data_matrix,ground_truth,verbose,penalty,alpha):
	'''
	Fits MLR model to scaled input data matrix
	'''
	if verbose:
		print('data matrix has shape {}'.format(scaled_data_matrix.shape))
		print('ground_truth has shape {}'.format(ground_truth.shape))


	if penalty == None:
		reg_model = LinearRegression(fit_intercept = True)
		fitted_model = reg_model.fit(scaled_data_matrix,ground_truth)

	if penalty == 'L2':
		L2_reg_model = Ridge(alpha = alpha,fit_intercept = True,random_state=51)
		fitted_model = L2_reg_model.fit(scaled_data_matrix,ground_truth)
	
	if penalty == 'L1':
		Gram_matrix = np.dot(scaled_data_matrix.T,scaled_data_matrix)
		L1_reg_model = Lasso(alpha = alpha,fit_intercept = True,random_state=51,selection = 'random',precompute = Gram_matrix,max_iter = 1000000,tol = 0.0001)
		fitted_model = L1_reg_model.fit(scaled_data_matrix,ground_truth)
		
	if verbose:
		print('The fitted weights are {}'.format(fitted_model.coef_))
	#return fitted_model.coef_
	return fitted_model


def get_best_lambda(penalty,time_shift,map_matrix,ERA_data_train,GT_data,comp_files,
		predictor_comp_strs,region_size,centre_point,num_neighbours,
		lambdas,train_year,ERA_data_val,val_GT_data, val_year):
	
	if lambdas != []:

		s_s = [1,15,24,31,47,36,41,7,12,28]
		t_s = [2,17,28,19,43,12,46,37,34,15]
		num_points = len(s_s)*len(t_s)

		opt_lambda_errs = []
		best_lambdas = []
		reg_paths = np.empty((num_points,len(lambdas)))

		num_params = (time_shift +1)*num_neighbours*len(comp_files)
		print('the number of parameters is {}'.format(num_params))
		coeff_tracker = np.empty((region_size[0]*region_size[1],num_params))
		point_and_lambda_counter = 0
		point_counter= 0

		for x in s_s:
			for y in t_s:
				t_1 = time.time()
				rel_error_lambdas = []
				for count_a,alpha in enumerate(lambdas):

					fitted_model,scaler = load_and_train_model(penalty,
										time_shift,map_matrix,ERA_data_train,x,y,
										GT_data,comp_files,predictor_comp_strs,region_size,
										centre_point,num_neighbours,year = train_year,alpha = alpha,indices = indices )


					pred_matrix,rel_error = load_and_validate(scaler,time_shift,map_matrix,ERA_data_val,x,y,
											val_GT_data,fitted_model,predictor_comp_strs,
											region_size,centre_point,num_neighbours,year = val_year,indices = indices )

					rel_error_lambdas.append(rel_error)

					if rel_error_lambdas != []:
						if rel_error_lambdas[count_a-1] < rel_error_lambdas[count_a]:

							#save best lambda and re train +val using it.
							opt_lambda_error = rel_error_lambdas[count_a-1]
							break
						if count_a == len(lambdas):
							print('best lambda not in search space')
				opt_lambda_errs.append(opt_lambda_error)
				t_2 = time.time()
				
				best_lambdas.append(np.log10(lambdas[np.argmin(rel_error_lambdas)]))



		

		mean_best_lambda = np.mean(best_lambdas)


		print('best lambda{}'.format(mean_best_lambda))
		mean_best_lambda = 10*(10**(mean_best_lambda))
		print('best lambda{}'.format(mean_best_lambda))
		
		return mean_best_lambda,opt_lambda_errs,s_s,t_s


def validate_lambda_choice(penalty,time_shift,map_matrix,ERA_data_train,GT_data,comp_files,
		predictor_comp_strs,region_size,centre_point,num_neighbours,
		lambdas,train_year,ERA_data_val,val_GT_data, val_year,mean_best_lambda,opt_lambda_errs,s_s,t_s):

		mean_lambda_errs = []
		s_s = [1,15,24,31,47,36,41,7,12,28]
		t_s = [2,17,28,19,43,12,46,37,34,15]
		alpha = mean_best_lambda

		for x in s_s:
			for y in t_s:

				fitted_model,scaler = load_and_train_model(penalty,time_shift,map_matrix,ERA_data_train,x,y,
											GT_data,comp_files,predictor_comp_strs,region_size,
											centre_point,num_neighbours,year = train_year,alpha = alpha )


				pred_matrix,rel_error = load_and_validate(scaler,time_shift,map_matrix,ERA_data_val,
											x,y,val_GT_data,fitted_model,predictor_comp_strs,
											region_size,centre_point,num_neighbours,year = val_year )
				mean_lambda_errs.append(rel_error)


		
		losses_using_mean_lambda = abs(np.asarray(mean_lambda_errs)-np.asarray(opt_lambda_errs))

		#save plot of lossed per gridpoint
		#plt.plot(losses_using_mean_lambda)
		#plt.ylabel('percentage loss using mean lambda', fontsize = 18)
		#plt.xlabel('grid point')
		#plt.savefig('L1_LPG.pdf',bbox_inches = 'tight')			
		#print('the mean percentage loss using sub-optimal lambda {}'.format(np.mean(losses_using_mean_lambda)))


				

			


def fit_glasso(scaled_points, ground_truth_points,group_idxs_list,reg_const):
	
	glasso_model = GLM(distr="gaussian", tol=1e-3,
			group=group_idxs_list, score_metric="pseudo_R2",
			alpha=1.0,verbose = True,solver ='cdfast',reg_lambda = reg_const,max_iter = 100000,learning_rate=0.1)

	glasso_fit = glasso_model.fit(scaled_points, ground_truth_points)
	score = glasso_fit.score(scaled_points,ground_truth_points)	
	print('score is {}'.format(score))

	return glasso_fit


def get_error_MLR_region(prediction,ground_truth):
	'''
	Gets the error of predictions - ground truth values for a specific region.
	Returns the RMSE of the predictions and the IQR of all the ground truth values
	'''
	diff_map = ground_truth.values - prediction
	error = diff_map.flatten()
	mean_error = np.mean(error**2)
	sq_mean_error = np.sqrt(mean_error)
	
	#IQR = np.subtract(*np.percentile(ground_truth.values.flatten(), [75, 25]))
	relative_error = sq_mean_error
	#print(IQR)
	print('RMSE: {}'.format(relative_error))
	


def load_and_validate(scaler,time_shift,map_matrix,ERA_data, s,t,val_GT, fitted_model,component_strs,region_size,centre_point,num_neighbours,year,indices):
	'''
	loads validation data matrix and validates an MLR model
	args:
		time_shift: integer, gives the number of previous time points to include as covariates in the regression model
		map_matrix: array, matrix which gives the nearest indices for a given co-ordinate in the HR data
		ERA_data: list, this list contains elements for the ERA data components, which will be passed to the function 
			which loads the data matrix for training
		s,t: gives the current cell in the region for which we are predicting
		val_GT: the validation ground truth values, which we aim to predict.
		centre_point: tuple, gives the centre of the region we are considering
		region_size: tuple, gives the region size we consider
		component_strs: list, a list of the strs which allow indexing of the predictor componenet files
		num_niehgbours: int, how many neighbours we will use as predictors in the regression model
		year: tuple, string '2001' would mean get the ground truth values for the year 2001.
		fitted_model: sklearn model, The model object which has been fitted during load_and_train

	returns
		predction matrix of shape (number of hours, region_size[0],region_size[1])
	'''

	val_points = get_nearest_data_matrix(time_shift,map_matrix,ERA_data,s,t,component_strs,region_size,centre_point,num_neighbours,year)	
	if indices == []:
		pass

	else: 
		val_points = val_points[:,indices]

	scaled_val_points = scaler.transform(val_points)
	prediction = fitted_model.predict(scaled_val_points)#(scaled_val_points,weight_matrix)
	pred_matrix = prediction


	diff_map = pred_matrix - val_GT.values[:,s,t]
	error = diff_map.flatten()

	RMSE,IQR = get_error_over_all_vals(error, val_GT.values[:,s,t])

	#RMSE where mean is taken over alll time points, for one grid cell

	rel_error = RMSE/IQR
	
	

	return pred_matrix,rel_error


def load_and_train_model(penalty,time_shift,map_matrix,ERA_data,s,t,GT_data,
			component_files,component_strs,region_size,centre_point,num_neighbours,year,alpha,indices):
	'''
	loads data matrix and trains an MLR model
	args:
		time_shift: integer, gives the number of previous time points to include as covariates in the regression model
		map_matrix: array, matrix which gives the nearest indices for a given co-ordinate in the HR data
		ERA_data: list, this list contains elements for the ERA data components, which will be passed to the function 
			which loads the data matrix for training
		s,t: gives the current cell in the region for which we are predicting
		G_T_data: array, This is the ground truth values which we train on. 
		centre_point: tuple, gives the centre of the region we are considering
		region_size: tuple, gives the region size we consider
		component_files: list, a list of the predictor component files which store the data for the predictors
		component_strs: list, a list of the strs which allow indexing of the predictor componenet files
		num_niehgbours: int, how many neighbours we will use as predictors in the regression model
		year: tuple, string '2001' would mean get the training data values for the year 2001.

	returns
		weight matrix of shape (number of hours, number of covariates)
	'''

	train_data_matrix = get_nearest_data_matrix(time_shift,map_matrix,ERA_data,s,t,component_strs,region_size,centre_point,num_neighbours,year)
	if indices == []:
		pass
	else:
		train_data_matrix = train_data_matrix[:,indices]
		

	num_input_dims =train_data_matrix.shape[-1]
	ground_truth_points = GT_data[:,s,t].values #get ground truth values i.e Y
	scaled_points,scaler = scale_input(train_data_matrix)



	fitted_model = fit_MLR(scaled_points, ground_truth_points,penalty=penalty,alpha=alpha,verbose = False )		#fit the MLR model



	return fitted_model,scaler


###############################################################################################################################################################




def train_and_validate(centre_point,penalty,lambdas,time_shift,train_year,val_GT_year,val_year,loc,region_size, 		
				prediction_comp_str,comp_files,predictor_comp_strs,num_neighbours,train_GT_year,indices,track_coeffs,
				optimise_lambda,validate_lambda):
	'''
	Trains and validates the MLR model
	args:
		time_shift: integer, gives the number of previous time points to include as covariates in the regression model
		train_year: tuple (0,3) would mean the training year is 2000-2003
		val_year: tuple (1,2) would mean the validation year is 2001-2002 
		loc: str, which location we are considering
		region_size: tuple, gives the region size we consider
		prediction_comp_str: string, this is the string for the component we are trying to predict ('U'/ 'V')
		comp_files: list, a list of the predictor component files which store the data for the predictors
		predictor_comp_strs: list, a list of the strs which allow indexing of the predictor componenet files
		num_niehgbours: int, how many neighbours we will use as predictors in the regression model
		train_GT_year: list, each element is a string, ['2001','2002'] would mean get the ground truth values for the year 2000 - 2002

	returns: 
		A prediction matrix of the same size as the predicted region dims: (number of hours,region_size[0],region_size[1])
	'''
	
	

	#get training and validation outputs. both can span more than one year
	for counter,y in enumerate(train_GT_year):
		if counter==0:
			GT_data = load_GT(centre_point,prediction_comp_str,year = y)
		else:
			next_year_GT = load_GT(centre_point,prediction_comp_str,year = y)
			GT_data = xr.concat([GT_data,next_year_GT],dim = 'Time')

	
	for counter_val,v in enumerate(val_GT_year):
		if counter_val ==0:
			val_GT_data = get_val_set(centre_point,prediction_comp_str,year = v)
		else:
			next_year_val_GT = get_val_set(centre_point,prediction_comp_str,year = v)
			val_GT_data = xr.concat([val_GT_data,next_year_val_GT],dim = 'Time')

	full_pred_matrix = np.empty((val_GT_data.shape[0],region_size[0],region_size[1]))
	
	#when validating on train set
	#full_pred_matrix = np.empty((GT_data.shape[0],region_size[0],region_size[1]))

	ERA_data_train = []
	ERA_data_val = []
	
	#load in all files here
	for k in range(len(comp_files)):
		year_of_data_train = get_year_of_comp_data(time_shift,comp_files[k],predictor_comp_strs[k],year = train_year)
		ERA_data_train.append(year_of_data_train)				#now have all data in list of arrays

		year_of_data_val = get_year_of_comp_data(time_shift, comp_files[k],predictor_comp_strs[k],year = val_year)
		ERA_data_val.append(year_of_data_val)


	co_ord_matrix,map_matrix,lon_bounds,lat_bounds = pick_region(centre_point,region_size,num_neighbours)
	co_ord_matrix = None

	all_weights = np.empty((region_size[0],region_size[1],90))

	rel_errors = np.empty((region_size[0],region_size[1]))

	if optimise_lambda == True:
		mean_best_lambda,opt_lambda_errs,s_s,t_s = get_best_lambda(penalty,time_shift,map_matrix,ERA_data_train
			,GT_data,comp_files,predictor_comp_strs,region_size,centre_point,num_neighbours,lambdas,train_year,ERA_data_val,val_GT_data, val_year)
	else:
		mean_best_lambda = None

	if validate_lambda == True:
		validate_lambda_choice(penalty,time_shift,map_matrix,ERA_data_train,GT_data,comp_files,
		predictor_comp_strs,region_size,centre_point,num_neighbours,
		lambdas,train_year,ERA_data_val,val_GT_data, val_year,mean_best_lambda,opt_lambda_errs,s_s,t_s)


	if track_coeffs == True:
		point_counter = 0
		if indices != None:
			num_params = len(indices)
		else:
			num_params = (time_shift +1)*num_neighbours*len(comp_files)

		coeff_tracker = np.empty((region_size[0]*region_size[1],num_params))
	else:
		coeff_tracker = None


	for s in range(0,region_size[0]):
		times = []
	
		print(s)
		for t in range(0,region_size[1]):

			fitted_model,scaler = load_and_train_model(penalty,
								time_shift,map_matrix,ERA_data_train,s,t,
								GT_data,comp_files,predictor_comp_strs,
								region_size,centre_point,num_neighbours,year = train_year,
								alpha = mean_best_lambda,indices = indices)
			if track_coeffs == True:
				coeff_tracker[point_counter,:]= fitted_model.coef_
				point_counter +=1

	

			pred_matrix,rel_error = load_and_validate(scaler,time_shift,map_matrix,ERA_data_val,s,t,val_GT_data,fitted_model,predictor_comp_strs,
									region_size,centre_point,num_neighbours,year = val_year,indices = indices)




				
			rel_errors[s,t] = rel_error


			full_pred_matrix[:,s,t] = pred_matrix


		print('error check: {}'.format(rel_error))


	return full_pred_matrix, val_GT_data,fitted_model,rel_errors,all_weights,coeff_tracker,ERA_data_train


def convert_year(year): 
    years = [] 
    for i in range(year[0],year[1]): 
         years.append('200'+str(i)) 
    return years 



#get training and validation inputs
  
u100_file = os.path.join(os.getcwd(),'ERA_data_source/100m_u_extended.nc')
v100_file  = os.path.join(os.getcwd(),'ERA_data_source/100m_v_extended.nc')
vort_file = os.path.join(os.getcwd(),'ERA_data_source/1000hpa_Vort.nc')
temp_file = os.path.join(os.getcwd(),'ERA_data_source/1000hpa_T.nc')
pres_file = os.path.join(os.getcwd(),'ERA_data_source/Pressure.nc')



#here we define which met variables to use and their associated component strings
component_files = [u100_file,v100_file,vort_file,temp_file,pres_file]
component_strs = ['u100','v100','vo','t','sp']
#define the target variable string
target_str = 'U'
#define the penalty - #None implies unregularised MLR, other options are 'L1' or 'L2'
model_penalty = None

#define the locations to perform analysis.
locs = ['fort_augustus'] #['north_east_sea','coventry','newcastle','fort_augustus']  
#define training year
train_year = (1,5)
train_GT_year = convert_year(train_year)

#define validation years
validation_years = [(6,7),(7,8)]
val_year = (validation_years[0][0],validation_years[-1][1])
#define number of previous time points
taus = [0]
#define number of nearest neighbours
nn = 4
#define region size 
region_size = (50,50)
#define whether we should track the weights and save them
track_coeffs = False
#define if we should optimise and validate our choice of lambda
val_lambda = False
opt_lambda = False

#define the search space for lambda optimisation
if opt_lambda == True:
	lambdas = [10e-3,10e-2,10e-1,10e-0,10e1,10e2,10e3]
else:
	lambdas = None




#if you want to use the most important subset, load in the sorted indices:
indices_to_choose = np.load(os.path.join(os.getcwd(),'important_var_indices/L1_coeffs_sorted_indices.npy'))
#partitions defines the number of vars you want in the subset.
partition = None

#only include the indices up to the partition, if defined.
if partition != None:
	indices = indices_to_choose[:partition]
else:
	indices = []


experiment_name = 'trial'

#make directories to save plots to
region_dir_spatial = os.path.join(os.getcwd(),experiment_name+'_spatial')
if os.path.exists(region_dir_spatial):
	shutil.rmtree(region_dir_spatial)

os.makedirs(region_dir_spatial)

region_dir_temp = os.path.join(os.getcwd(),experiment_name+'_Temporal')
if os.path.exists(region_dir_temp):
	shutil.rmtree(region_dir_temp)

os.makedirs(region_dir_temp)

#open file to save results
f = open(experiment_name+'.txt', 'w')



for tau in taus:
	for location in locs:
		centre_point = get_coord_from_name(location)
		full_pred_matrix,val_GT_data,fitted_model,rel_errors,all_weights,coeff_tracker,ERA_data = train_and_validate(centre_point = 												centre_point, penalty =model_penalty,lambdas = lambdas,
											time_shift =tau,train_year = train_year,
											train_GT_year = train_GT_year,val_GT_year = validation_years,
											val_year = val_year,loc = location,region_size = region_size,
											prediction_comp_str = target_str,comp_files = component_files,
											predictor_comp_strs = component_strs,num_neighbours = nn,indices = 												indices,track_coeffs = track_coeffs, optimise_lambda = 												opt_lambda,validate_lambda = val_lambda)

		diff_map = full_pred_matrix - val_GT_data.values
		daily_diff_map = diff_map.reshape(int(diff_map.shape[0]/24),24,region_size[0],region_size[1])
		error = diff_map.flatten()

		RMSE,IQR = get_error_over_all_vals(error, val_GT_data.values)
		print('the mean relative error over the whole spatio-temporal period is {}'.format(RMSE/IQR))
		score = np.around(RMSE/IQR, 5)

		print('relative error {}'.format(score))

		rel_errs,spatial_errs = get_daily_relative_errors(full_pred_matrix, val_GT_data.values)
		daily_raw_errors = get_daily_raw_errors(full_pred_matrix, val_GT_data.values)


		
		#write results to file
		f.write(location +'\t' + str(partition) +'\t' + str(score) +'\n')

		#save seasonal and spatial error plots
		plt_figs(spatial_errs,daily_raw_errors,val_GT_data.values,location,region_dir_temp,region_dir_spatial,daily_std)
		plt.close()




		#if you want to save the weights, the code below will do it
		if track_coeffs == True:
			centre_point = get_coord_from_name(location)
			co_ord_matrix,map_matrix,lon_bounds,lat_bounds = pick_region(centre_point,region_size,nn)
			co_ord_matrix = None
			train_data_matrix = get_nearest_data_matrix(tau,map_matrix,ERA_data,
									1,1,component_strs,region_size,centre_point,
									nn,(0,1))

			group_idxs,df = get_group_idxs_from_data_matrix(train_data_matrix,nn,component_strs)



			cols = list(df.columns)
			cols = [x.split('|')[0] for x in cols]
			
			np.save(experiment_name,coeff_tracker)
			np.save('Columns' +experiment_name,np.asarray(cols))
			

f.close()




