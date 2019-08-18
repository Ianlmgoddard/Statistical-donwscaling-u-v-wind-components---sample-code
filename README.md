# Combined Statistical-Dynamical downscaling of wind components over British Isles and Surrounding waters ##

This file provides the installation and run details to be able to perfrom multiple linear regression, ridge regression,lasso regression and bilinear interpolation on low resolution ERA5 renanalysis data, to generate high resolution data.

### Installing ##

to set-up the environment, you just need to run the following command with ananconda:

conda env create -f environment.yml

the env is quite large, this may take a few minutes.


## Data acquisition ##

To run the models you'll need to download the ERA5 data, this can be done using the codes inside ERA_data_source, but firstly, you'll need to make an account at the following link: 

https://cds.climate.copernicus.eu/user/register?destination=%2F%23!%2Fhome

Once you have a login in you'll need to Install the CDS API key; see the following link for more details

https://cds.climate.copernicus.eu/api-how-to

Now you're all set up to make requests to download ERA5 data. To get the data used for this study, you just need to run the .py files inside ERA_data_source. Each one of these will take several hours to download.


## Running the interpolation model. ##

All you need to do is run inter_day.py, this will generate 4 years of interpolated data, for the 4 regions considered in the study.

It's currently set to interpolate the U component, but this can be changed to V by changing the paramters 
passed to get_period_interp_region. you need to change:

component_str = 'u100' to component_str = 'v100'
hr_component_str = 'U' to hr_component_str = 'V'

You'll have to email me however if you want to test the interpolated model, because the ground truth data required is too large to include (>10GB).

## Running the MLR model. ##

This can be done with eff_lin_regress.py. There are many model parameters, which I'll outline here: you can change these in the script itself to run lasso and ridge regression. This code trains and validates the models, and spits out the final results (on the years of data which you specify) to a file, which again you can specify the name of. Also, the code outputs spatial and temporal error plots to new directories which are created when you run the file.


#here we define which met variables to use and their associated component strings
component_files = [u100_file,v100_file,vort_file,temp_file,pres_file]
component_strs = ['u100','v100','vo','t','sp']
#define the target variable string
target_str = 'U' ## can be this or 'V'
#define the penalty - #None implies unregularised MLR, other options are 'L1' or 'L2'
model_penalty = None

#define the locations to perform analysis.
locs = ['fort_augustus'] # I've only provided enough data to do this region, you'll have to contact me if you wan't the ground truth for the other regions.
#define training year
train_year = (1,5) #this means years 2001,2002,2003,2004 for training


#define validation years
validation_years = [(6,7),(7,8)] # this means 2006-2008
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

#partitions defines the number of vars you want in the subset.
partition = None  #Only define this if you want to do subset analysis with the most important
		  # variables. This must be an int.




## notes ##
There are various other .py files which are just utility functions mainly, I've omitted lots of the .py files which I used for analysing model results etc. If you would like these, please get in touch.


## Authors

* **Ian Goddard** - *Initial work* - contact email: iangoddard95@gmail.com
