

import numpy as np
from time import time

import sys, argparse, textwrap

### to use openMP
### export OMP_NUM_THREADS=2

#############
### INPUT ###
#############

#######################################################
### FIDUCIAL :                                      ###
### param_all4_2D_smallFilter_1batchNorm_multiSlice ###
#######################################################

### param_all4_2D_smallFilter_1batchNorm_multiSlice.py ### fiducial + 10 slices per training
from param_all4_2D_smallFilter_1batchNorm_multiSlice import * 

#############
### KERAS ###
#############

from keras import backend as K

### automatic detection of the back-end and choose of the the data format
if( K.backend()=='tensorflow'):
    import tensorflow as tf
    import sys
    K.set_image_data_format('channels_last')
else:
    import theano
    theano.config.exception_verbosity='high'
    theano.config.openmp = True
    K.set_image_dim_ordering('th')
    K.set_image_data_format('channels_first')
    
######################
### CODE PARAMETER ###
######################

### INITIATE RANDOM STATE
np.random.RandomState( np.random.seed(RandomSeed) )

print( 'Param          :', paramName[paramNum] )
print( 'file           :', model_file )
print( 'RandomSeed     :', RandomSeed )
print( 'trainSize      :', trainSize )
print( 'LHS            :', LHS )
if( LHS ):
    print( 'Nbins_LHS      :', Nbins_LHS )
print( 'epochs         :', epochs )
print( 'batch_size     :', batch_size )
print( 'DATABASE       :', DATABASE )
print( 'validation     :', validation )
print( 'all4           :', all4           )
print( 'reduce_LC      :', reduce_LC      )
print( 'substract_mean :', substract_mean )
print( 'apply_gauss    :', apply_gauss    )
print( 'reduce_CNN     :', reduce_CNN    )
print( 'use_dropout    :', use_dropout    )
print( 'CNN loss       :', loss )
print( 'CNN optimizer  :', optimizer )
print( 'LR factor      :', factor)
print( 'LR patience    :', patience)

### Variables not define in all parameter file!!
### TODO: clean param files - set to all the same list of params
try:
    print( 'LeackyRelu     :',LeackyRelu_alpha )
except:
    LeackyRelu_alpha = 0
    print( 'LeackyRelu     :',LeackyRelu_alpha )
    
try:
    print( 'batchNorm      :',batchNorm )
except:
    batchNorm = False
    print( 'batchNorm      :',batchNorm )
    
try:
    print( 'FirstbatchNorm :',FirstbatchNorm )
except:
    FirstbatchNorm = False
    print( 'FirstbatchNorm :',FirstbatchNorm )
     
try:
    print( 'Nfilter1       :',Nfilter1 )
    print( 'Nfilter2       :',Nfilter2 )
    print( 'Nfilter3       :',Nfilter3 )
except:
    Nfilter1 = 16 
    Nfilter2 = 32 
    Nfilter3 = 64 
    print( 'Nfilter1       :',Nfilter1 )
    print( 'Nfilter2       :',Nfilter2 )
    print( 'Nfilter3       :',Nfilter3 )

    
############################
### LOAD DADA: LIGHTCONE ###
############################

from lightcone_functions import LOAD_DATA

paramToLoad = [ DATABASE,
                RandomSeed,
                trainSize,
                LHS,
                Nbins_LHS,
                True, ### vervose
                False, ### justparam
                reduce_LC, 
                substract_mean,
                apply_gauss,
                validation,
                False ###justDataID
              ]

if validation:
    (LC_train,LC_test,LC_val),(Param_train,Param_test,Param_val),(Param_raw_train,Param_raw_test,Param_raw_val) = LOAD_DATA(*paramToLoad)
else:
    (LC_train,LC_test),(Param_train,Param_test),(Param_raw_train,Param_raw_test) = LOAD_DATA(*paramToLoad)  
    
### adjustment of data dimention 
if( K.image_data_format()=='channels_first' ):
    LC_train = np.squeeze(     LC_train, axis=3 )
    LC_train = np.expand_dims( LC_train, axis=1 )
    LC_test  = np.squeeze(     LC_test , axis=3 )
    LC_test  = np.expand_dims( LC_test , axis=1 )
    if validation:
        LC_val  = np.squeeze(     LC_val , axis=3 )
        LC_val  = np.expand_dims( LC_val , axis=1 )

####################################
### CONVOLUTIONAL NEURAL NETWORK ###
####################################
   
input_shape = LC_train.shape[1:]
print('input shape : ',input_shape)
padding = 'valid' ### 'same' or 'valid
filter_size = (10,10)
activation = 'relu' ### 'linear' 'relu'
use_bias=True
    
from keras.layers import Activation
from keras.layers.advanced_activations import LeakyReLU
    
if( batchNorm ):
    use_bias=False
    activation = 'linear' ### 'linear' 'relu'
    
if( LeackyRelu_alpha ):
    activation = 'linear' ### 'linear' 'relu'

from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.models import Sequential
model = Sequential()

### CONV 1
model.add( Convolution2D( Nfilter1, filter_size, activation=activation, 
                          input_shape=input_shape, name='Conv-1', padding=padding, use_bias=use_bias ) )
if( batchNorm ):
    model.add( BatchNormalization() )
if( LeackyRelu_alpha ):
    model.add( LeakyReLU(alpha=LeackyRelu_alpha) )
if( ( batchNorm ) and not(LeackyRelu_alpha) ):
    model.add( Activation('relu') )
        
### MAXPOOL 1
model.add( MaxPooling2D( pool_size=(2,2), name='Pool-1' ) )

### CONV 2
model.add( Convolution2D( Nfilter2, filter_size, activation=activation, 
                          name='Conv-2', padding=padding, use_bias=use_bias ) )
if( batchNorm ):
    model.add( BatchNormalization() )
if( LeackyRelu_alpha ):
    model.add( LeakyReLU(alpha=LeackyRelu_alpha) )
if( batchNorm and not(LeackyRelu_alpha) ):
    model.add( Activation('relu') )
        
### MAXPOOL 2
model.add( MaxPooling2D( pool_size=(2,2), name='Pool-2' ) )

### FLATTEN
model.add( Flatten( name='Flat' ) )
if use_dropout: 
    model.add( Dropout(use_dropout) )
        
### DENSE 1
model.add( Dense( Nfilter3, activation=activation, name='Dense-1', use_bias=use_bias ) )
if( batchNorm or FirstbatchNorm ):
    model.add( BatchNormalization() )
if( ( batchNorm or FirstbatchNorm ) and not(LeackyRelu_alpha) ):
    model.add( Activation('relu') )
    
### DENSE 2
model.add( Dense( Nfilter2, activation=activation, name='Dense-2', use_bias=use_bias ) )
if( batchNorm ):
    model.add( BatchNormalization() )
if( batchNorm and not(LeackyRelu_alpha) ):
    model.add( Activation('relu') )
    
### DENSE 3
model.add( Dense( Nfilter1, activation=activation, name='Dense-3', use_bias=use_bias ) )
if( batchNorm ):
    model.add( BatchNormalization() )
if( batchNorm and not(LeackyRelu_alpha) ):
    model.add( Activation('relu') )
        
### DENSE OUT
if all4:
    model.add( Dense( 4, activation='linear', name='Out' ) )
else:
    model.add( Dense( 1, activation='linear', name='Out' ) )
    
##############################    
model.summary(line_length=120) 

######################
### LEARNING PHASE ###
######################

### DEFINE THE LEARNING RATE

### set the learning rate callback
callbacks_list=[]
if( 1 ):
    from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
    lrate = ReduceLROnPlateau( monitor='loss', factor=factor, patience=patience )
    callbacks_list.append( lrate )

### to print the Learning Rate
from keras.callbacks import Callback, EarlyStopping
class LR_tracer(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        lr = K.eval( self.model.optimizer.lr )
        print( ' LR: %.10f '%(lr) )
callbacks_list.append( LR_tracer() )

### R2 coefficient
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

### STOP when it stop to learn
early_stopping = EarlyStopping(monitor='loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
callbacks_list.append( early_stopping )

### model compilations
model.compile( loss=loss,
               optimizer=optimizer,
               metrics=[coeff_determination] )

### THE LEARNING FUNCTION
history = model.fit( LC_train, Param_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     callbacks=callbacks_list,
                     validation_data=( LC_test, Param_test ),
                     verbose=True )

np.save( CNN_folder + history_file, history.history )

########################
### SAVING THE MODEL ###
########################

def save_model( model, fileName ):
    """
    save a model
    """
    ### save the model
    model_json = model.to_json(  )
    with open( fileName+'.json', 'w' ) as json_file:
        json_file.write( model_json )
    ### save the weights
    model.save_weights( fileName+'_weights.h5' )

save_model( model, CNN_folder + model_file )

##################
### PREDICTION ###
##################

predictions = model.predict( LC_test, verbose=True )

### PRINT SCORE
if all4:  
    print( 'R2: ', 1 - (((predictions[:,0] - Param_test[:,0])**2).sum(axis=0)) / ((predictions[:,0] - predictions[:,0].mean(axis=0) )**2).sum(axis=0) )
    print( 'R2: ', 1 - (((predictions[:,1] - Param_test[:,1])**2).sum(axis=0)) / ((predictions[:,1] - predictions[:,1].mean(axis=0) )**2).sum(axis=0) )
    print( 'R2: ', 1 - (((predictions[:,2] - Param_test[:,2])**2).sum(axis=0)) / ((predictions[:,2] - predictions[:,2].mean(axis=0) )**2).sum(axis=0) )
    print( 'R2: ', 1 - (((predictions[:,3] - Param_test[:,3])**2).sum(axis=0)) / ((predictions[:,3] - predictions[:,3].mean(axis=0) )**2).sum(axis=0) )
else: 
    print( 'R2: ', 1 - (((predictions[:,0] - Param_test[:,paramNum])**2).sum(axis=0)) / ((predictions[:,0] - predictions[:,0].mean(axis=0) )**2).sum(axis=0) )

np.save( CNN_folder + prediction_file, predictions )

### Predict the validation, to be use only at the end end end ....
predictions_val = model.predict( LC_val, verbose=True )
np.save( CNN_folder + prediction_file_val, predictions_val )



