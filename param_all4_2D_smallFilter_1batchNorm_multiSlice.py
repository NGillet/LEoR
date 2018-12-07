### name of the run
name = '_smallFilter_1batchNorm_multiSlice' ### _confirm

### selected parameter
paramNum = 4
paramName = [ 'ZETA', 'Tvir', 'LX', 'E0', 'all4' ]

### LC parameter
dim=2           ### dimention of the input : 3
fullres = True  ### reduce average cube (25,25,550) of full resolution (200,200,2200)
BoxSize = 200   ### raw resolution
BoxInLC = 11    ### number of concatenate box in the LC
Nsimu   = 10000 ### total number of LC

### learning parameters
RandomSeed = 4321 ### 9510 confirm ### 4321 REF ##2235 old ### should be fixed, in order to be able to reproduce the training-testing set
trainSize = 0.8 ### if LHS useless
LHS = False
Nbins_LHS = 8000

batch_size = 20 ### number of sub sample, /!\ has to be a diviseur of the training set
epochs = 200    ### number of passage over the full data set

### Network PARAMETERS
### LOSS FUNCTION
loss = 'mean_squared_error' ### classic loss function for regression, see also 'mae'
### DEFINE THE OPTIMIZER
optimizer = 'RMSprop' #'adagrad'  #'adadelta' #'adam' # 'adamax' # 'Nadam' # 'RMSprop' # sgd
### DEFINE THE LEARNING RATE
factor=0.5
patience=5

### DEFINE THE DATABASE TO USE
DATABASE = '100_2200_slice_10' ### '300Mpc_r200_2D' 
multiple_slice = True

### OPTIONS
reduce_LC      = False    ### FOR 2D slice, use half of the image
substract_mean = False   ### substract mean(f)  o the LC
apply_gauss    = False   ### apply an half gaussian of the LC
reduce_CNN     = True    ### smaller CNN
use_dropout    = 0.2    

validation     = True    ### MAKE VALIDATION DATA

batchNorm      = False ### batchnorm after all layer == LONG TIME
FirstbatchNorm = True  ### batchnorm just after the first conv
LeackyRelu_alpha = 0

Nfilter1 = 8 ### First convolution
Nfilter2 = 16 ### 2nd convolution
Nfilter3 = 64 ### First Dense

######################
### INDUCED PARAMS ###
######################
if paramNum==4: ### given as arg
    all4 = True ### LEARN THE 4 PARAMS AT THE SAME TIME
else:
    all4 = False
    
if reduce_LC:
    BoxSize = 100 
    
### save files
model_template = '%s_2D'
model_file = model_template%(paramName[paramNum]) + name
history_file = model_file + '_history'
prediction_file = model_file + '_pred'
prediction_file_val = model_file + '_pred_val'

### save folder
CNN_folder = 'CNN_save/'