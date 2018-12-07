import numpy as np
from time import time

import sys, argparse, textwrap

### USE THIS TO OMP !!!!!!!!
### export OMP_NUM_THREADS=2

#############
### INPUT ###
#############
### param_all4_2D
### param_all4_2D_drop == REF
### param_all4_2D_drop_30
### param_all4_2D_drop_35
### param_all4_2D_drop_20_seed_9876
### param_all4_2D_batchNorm   == param_all4_2D_drop + batchNorm
### param_all4_2D_LeackyRelu  == param_all4_2D_drop + all layer LeakyReLU + small filters
### param_all4_2D_allDrop     == param_all4_2D_drop + all layer dropOut
### param_all4_2D_1batchNorm  == param_all4_2D_drop + 1 batchNorm at beggening
### param_all4_2D_smallFilter == param_all4_2D_drop + smmal numbe filter
### param_all4_2D_smallFilter_1batchNorm == param_all4_2D_drop + smmal numbe filter + 1 batchNorm at beggening

########################################
### FIDUCIAL : 
### param_all4_2D_smallFilter_1batchNorm
########################################

### param_all4_2D_smallFilter_1batchNorm_multiSlice.py ### fiducial + 10 slices per training

#from param_all4_2D_smallFilter_1batchNorm_multiSlice import * 

from param_all4_2D_smallFilter_1batchNorm_multiSlice_dataGenerator import * 

#############
### KERAS ###
#############

from keras import backend as K

### automatic detection of the back-end and choose of the the data format
if( K.backend()=='tensorflow'):
    import tensorflow as tf
    import sys
    # njobs = np.int( sys.argv[1] )
    njobs = np.int( 20 )
    config = tf.ConfigProto( intra_op_parallelism_threads=njobs,
                             inter_op_parallelism_threads=njobs,
                             allow_soft_placement=True,
                             log_device_placement=True,
                             device_count = {'CPU':njobs})
    session = tf.Session(config=config)
    K.set_session(session)
    K.set_image_dim_ordering('tf')
    #print( 'image_dim_ordering : ', K.image_dim_ordering() )
    K.set_image_data_format('channels_last')
    #K.set_image_data_format('channels_first')
    #print( 'image_data_format  : ', K.image_data_format() )
else:
    import theano
    theano.config.exception_verbosity='high'
    theano.config.openmp = True
    #theano.config.blas.ldflags = '-lopenblas'
    #print( 'mode         :', theano.config.mode )
    #print( 'openmp       :', theano.config.openmp )
    #print( 'device       :', theano.config.device )
    #print( 'force_device :', theano.config.force_device )
    #print( 'floatX       :', theano.config.floatX )
    #print( 'ldflags      :', theano.config.blas.ldflags )
    #K.set_image_dim_ordering('th')
    K.set_image_dim_ordering('th')
    #print( 'image_dim_ordering : ', K.image_dim_ordering() )
    #K.set_image_data_format('channels_last')
    K.set_image_data_format('channels_first')
    #print( 'image_data_format  : ', K.image_data_format() )
    
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

    
from lightcone_functions import DataGenerator 
# Data Generator Parameters
params_train = {'Ndata'      : 80000,
                'batch_size' : 20, 
                'dim'        : (100,2200), 
                'n_channels' : 1,
                'n_params'   : 4, 
                'shuffle'    : True, 
                'DATA_DIR'   : '/amphora/nicolas.gillet/LC_data/LC_SLICE10_px100_2200_N10000_randICs/train/', }

params_valid = {'Ndata'      : 1000,
                'batch_size' : 20, 
                'dim'        : (100,2200), 
                'n_channels' : 1,
                'n_params'   : 4, 
                'shuffle'    : True, 
                'DATA_DIR'   : '/amphora/nicolas.gillet/LC_data/LC_SLICE10_px100_2200_N10000_randICs/validation/', }

params_test  = {'Ndata'      : 1000,
                'batch_size' : 20, 
                'dim'        : (100,2200), 
                'n_channels' : 1,
                'n_params'   : 4, 
                'shuffle'    : True, 
                'DATA_DIR'   : '/amphora/nicolas.gillet/LC_data/LC_SLICE10_px100_2200_N10000_randICs/test/', }

# Generators
training_generator   = DataGenerator( **params_train)
validation_generator = DataGenerator( **params_valid)
test_generator       = DataGenerator( **params_test)

### ADAPTE IT BY HAND! BE CAREFULL 
       
#K.image_data_format()='channels_last'
K.set_image_data_format('channels_last')
input_shape = (100,2200,1)
print('input shape : ',input_shape)

if reduce_CNN :

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
    #if use_dropout:
    #    model.add( Dropout(use_dropout) )

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
    if use_dropout: ### AFTER TEST THIS ONE AT 0.2 WORK WELL
        model.add( Dropout(use_dropout) )
        
    ### DENSE 1
    model.add( Dense( Nfilter3, activation=activation, name='Dense-1', use_bias=use_bias ) )
    if( batchNorm or FirstbatchNorm ):
        model.add( BatchNormalization() )
    #if( LeackyRelu_alpha ):
    #    model.add( LeakyReLU(alpha=LeackyRelu_alpha) )
    if( ( batchNorm or FirstbatchNorm ) and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
    #if use_dropout:
    #    model.add( Dropout(use_dropout) )
    
    ### DENSE 2
    model.add( Dense( Nfilter2, activation=activation, name='Dense-2', use_bias=use_bias ) )
    if( batchNorm ):
        model.add( BatchNormalization() )
    #if( LeackyRelu_alpha ):
    #    model.add( LeakyReLU(alpha=LeackyRelu_alpha) )
    if( batchNorm and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
    #if use_dropout:
    #    model.add( Dropout(use_dropout) )
    
    ### DENSE 3
    model.add( Dense( Nfilter1, activation=activation, name='Dense-3', use_bias=use_bias ) )
    if( batchNorm ):
        model.add( BatchNormalization() )
    #if( LeackyRelu_alpha ):
    #    model.add( LeakyReLU(alpha=LeackyRelu_alpha) )
    if( batchNorm and not(LeackyRelu_alpha) ):
        model.add( Activation('relu') )
    #if use_dropout:
    #    model.add( Dropout(use_dropout) )
        
    ### DENSE OUT
    if all4:
        model.add( Dense( 4, activation='linear', name='Out' ) )
    else:
        model.add( Dense( 1, activation='linear', name='Out' ) )
    
else:
    Nfilter1 = 32
    Nfilter2 = 64
    Nfilter3 = 128
    Nfilter4 = 256

    padding = 'valid' ### 'same' or 'valid
    filter_size = (10,10)
    filter_size_last = (4,4)

    from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
    from keras.models import Sequential
    model = Sequential()

    model.add( Convolution2D( Nfilter1, filter_size, activation='relu', input_shape=input_shape, name='Conv-1', padding=padding ) )
    #model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), name='Pool-1' ) )

    model.add( Convolution2D( Nfilter2, filter_size, activation='relu', name='Conv-2', padding=padding) )
    #model.add( BatchNormalization() )
    model.add( MaxPooling2D( pool_size=(2,2), name='Pool-2' ) )

    #model.add( Convolution2D( Nfilter3, filter_size, activation='relu', name='Conv-3', padding=padding) )
    #model.add( MaxPooling2D( pool_size=(2,2), name='Pool-3' ) ) 

    #model.add( Convolution2D( Nfilter4, filter_size_last, activation='relu', name='Conv-4', padding=padding) )
    #model.add( MaxPooling2D( pool_size=(2,2), name='Pool-4' ) ) 

    model.add( Flatten( name='Flat' ) )
    model.add( Dense( Nfilter3, activation='relu', name='Dense-1' ) )
    #model.add( Dropout(0.5) )
    model.add( Dense( Nfilter2, activation='relu', name='Dense-2' ) )
    model.add( Dense( Nfilter1, activation='relu', name='Dense-3' ) )

    if all4:
        model.add( Dense( 4, activation='linear', name='Out' ) )
    else:
        model.add( Dense( 1, activation='linear', name='Out' ) )
    
model.summary(line_length=120) 

######################
### LEARNING PHASE ###
######################

### DEFINE THE LEARNING RATE
import math
def step_decay(epoch):
    ### MAIN TUNNING HERE
    #elif(optimizer=='RMSprop'):
    dec=0.5
    epoch_drop = 2
    init_lr = 0.001 #0.0003 # 0.001

    lr =  init_lr 
    lr =  init_lr * np.power( dec, np.floor( (1+epoch) /epoch_drop ) )
    if( lr < 1.e-6 ):
        lr = 1.e-6
    return lr

### set the learning rate callback
callbacks_list=[]
if( 1 ):
    from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
    #lrate = LearningRateScheduler( step_decay )
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
if all4:
#    history = model.fit( LC_train, Param_train,
#                         epochs=epochs,
#                         batch_size=batch_size,
#                         callbacks=callbacks_list,
#                         validation_data=( LC_test, Param_test ),
#                         verbose=True )
    
    history = model.fit_generator(generator=training_generator,
                                  validation_data=validation_generator,
                                  epochs=epochs,
                                  callbacks=callbacks_list,
                                  verbose=True,
                                  use_multiprocessing=True,
                                  workers=30,
                                 )
else:
    history = model.fit( LC_train[:], Param_train[:,paramNum],
                         epochs=epochs,
                         batch_size=batch_size,
                         callbacks=callbacks_list,
                         #validation_data=( LC_test[:1000], Param_test[:1000,paramNum] ),
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

#predictions = model.predict( LC_test, verbose=True )
predictions = model.predict_generator( test_generator, verbose=True )

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



