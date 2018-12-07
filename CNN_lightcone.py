import numpy as np
from time import time

import sys, argparse, textwrap

### USE THIS TO OMP !!!!!!!!
### export OMP_NUM_THREADS=2

#############
### INPUT ###
#############

parser = argparse.ArgumentParser( description='Convolution Neural Network on Light Cone',
                                  formatter_class=argparse.RawTextHelpFormatter)

### The parameter to be learn
param_help = textwrap.dedent("""\
0 : ZETA
1 : Tvir (default)
2 : LX
3 : E0

""")
parser.add_argument( '-p', '--param' , choices=[0,1,2,3], action='store', dest='paramNum'  , default=1, type=int,
                     help=param_help )

### Type of Lightcone DEPRECATED
format_help = textwrap.dedent('''\
DEPRECATED : no choose to do here for the moment
3 : dim=3 and fullres=True  => high resolution cube  (20,20,440) on 150Mpc (default)
''')
parser.add_argument( '-f', '--format', choices=[3], action='store', dest='optionDATA', default=3, type=int,
                      help=format_help, required=False)

### Name to save the result
format_help = textwrap.dedent('''\
model file name = param + format + name
history_file = param + format + '_history' + name
''')
parser.add_argument( '-n', '--name', action='store', dest='name', default='', type=str, required=False )

### the database to use
ICs_help = textwrap.dedent('''\
DEPRECATED : no choose to do here for the moment
Initial condition of database : 
3=random + different ICs (default)
''')
parser.add_argument( '-i', '--ics', choices=[3], action='store', dest='ICs', default=3, type=int,
                      help=ICs_help, required=False)

### input management
input_args = parser.parse_args()
paramNum   = input_args.paramNum
optionDATA = input_args.optionDATA
ICs        = input_args.ICs
if( input_args.name != '' ):
    name = '_' + input_args.name
else:
    name = input_args.name

### DIR to save files
CNN_folder = 'CNN_save/'

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

paramName = [ 'ZETA', 'Tvir', 'LX', 'E0' ]

print('Param : %s'%(paramName[paramNum]))

dim=3          ### dimentiopn of the input : 3
fullres = True ### reduce average cube (25,25,550) of full resolution (200,200,2200)
BoxSize = 200  ### raw resolution
BoxInLC = 11   ### number of concatenate box in the LC

model_file = '%s_cnn_3D_highres'%(paramName[paramNum]) + name
print( model_file )
history_file = model_file + '_history'
prediction_file = model_file + '_pred'

Nsimu = 10000 ### total number of LC

### learning parameters
RandomSeed = 2235 ### should be fixed, in order to be able to reproduce the training-testing set
np.random.RandomState( np.random.seed(RandomSeed) )
trainSize = 0.9 ### if LHS useless
LHS = True

if( LHS ):
    Nbins_LHS = 9000
    print( 'Nbins_LHS:',Nbins_LHS )

batch_size = 200      ### number of sub sample, /!\ has to be a diviseur of the training set
epochs = 100         ### number of passage over the full data set

print( 'epochs:',epochs )
print( 'batch_size:',batch_size )

#steps_per_epoch = 200 ### for the generator

############################
### LOAD DADA: LIGHTCONE ###
############################

from lightcone_functions import LOAD_DATA, shrink_2D

DATABASE = '150Mpc_r25' # '75Mpc_r50'

### THIS IS TEST
reshapeLC = False 

(LC_train,LC_test),(Param_train,Param_test), (Param_raw_train,Param_raw_test) = LOAD_DATA( RandomSeed=2235, 
                                                                                            trainSize=0.9, 
                                                                                            LHS=LHS,
                                                                                            verbose=True,
                                                                                            Nbins_LHS=Nbins_LHS,
                                                                                            DATABASE=DATABASE,
                                                                                            justParam=False )
### LITTLE BUG IN THE DATABASE cf BRADLEY GREIG
Nnan = np.isnan(LC_train).sum()
if( Nnan ):
    print( '/!\ NAN IN TRAIN LIGHTCONES: %d'%(Nnan) )
    LC_train[ np.where( np.isnan(LC_train) ) ] = np.zeros( Nnan, dtype=np.float32 )
    
Nnan = np.isnan(LC_test).sum()
if( Nnan ):
    print( '/!\ NAN IN TEST LIGHTCONES: %d'%(Nnan) )
    LC_test[ np.where( np.isnan(LC_test) ) ] = np.zeros( Nnan, dtype=np.float32 )
    
####################################
### CONVOLUTIONAL NEURAL NETWORK ###
####################################

### automatic adjustment of data dimention 
axis_concatenate=4
if( K.image_data_format()=='channels_first' ):
    axis_concatenate=1
    if( dim==3 ):
        LC_train = np.squeeze(     LC_train, axis=4 )
        LC_train = np.expand_dims( LC_train, axis=1 )
        LC_test  = np.squeeze(     LC_test , axis=4 )
        LC_test  = np.expand_dims( LC_test , axis=1 )
    if( dim==2 ):
        LC_train = np.squeeze(     LC_train, axis=3 )
        LC_train = np.expand_dims( LC_train, axis=1 )
        LC_test  = np.squeeze(     LC_test , axis=3 )
        LC_test  = np.expand_dims( LC_test , axis=1 )

### ADAPTE IT BY HAND! BE CAREFULL 
if( reshapeLC ):
    beginLC = 0
    endLC = 50*4
    if( K.image_data_format()=='channels_first' ):
        LC_train = LC_train[ :, :, :, :, beginLC:endLC ]
        LC_test  = LC_test [ :, :, :, :, beginLC:endLC ]
    else:
        LC_train = LC_train[ :, :, :, beginLC:endLC, : ]
        LC_test  = LC_test [ :, :, :, beginLC:endLC, : ]
        
input_shape = LC_train.shape[1:]
print('input shape : ',input_shape)

Nfilter1 = 32
Nfilter2 = 64
Nfilter3 = 128

model_debase = True
model_inception = False
model_lowCNN = False
model_lowCNN_NOTvir = False
model_lowCNN_Tvir = False
model_lowCNN_Tvir_Zeta = False
model_dense = False
model_heavy = False

if( model_debase ):
    
    Nfilter1 = 32
    Nfilter2 = 64
    Nfilter3 = 128
    
    from keras.layers import Convolution3D, MaxPooling3D, Dropout, Flatten, Dense
    from keras.models import Sequential
    model = Sequential()

    model.add( Convolution3D( Nfilter1, (2,2,2), activation='relu', input_shape=input_shape, name='Conv-1' ) )
    model.add( Convolution3D( Nfilter1, (2,2,2), activation='relu', name='Conv-2') )
    model.add( MaxPooling3D( pool_size=(2,2,2), name='Pool-1' ) )

    #model.add( Dropout(0.25) )

    model.add( Convolution3D( Nfilter2, (2,2,2), activation='relu', name='Conv-3') )
    model.add( Convolution3D( Nfilter2, (2,2,2), activation='relu', name='Conv-4') )
    model.add( MaxPooling3D( pool_size=(2,2,2), name='Pool-2' ) )

    #model.add( Dropout(0.25) )

    model.add( Convolution3D( Nfilter3, (2,2,2), activation='relu', name='Conv-5') )
    model.add( Convolution3D( Nfilter3, (2,2,2), activation='relu', name='Conv-6') )
    model.add( MaxPooling3D( pool_size=(2,2,2), name='Pool-3' ) ) ### TODO : try with-without

    model.add( Flatten( name='Flat' ) )
    model.add( Dense( Nfilter3, activation='relu', name='Dense-1' ) )
    #model.add( Dropout(0.5) )
    model.add( Dense( Nfilter2, activation='relu', name='Dense-2' ) )
    model.add( Dense( Nfilter1, activation='relu', name='Dense-3' ) )
    model.add( Dense( 1, activation='linear', name='Out' ) )

if( model_inception ):
    
    from keras.layers import Conv3D, MaxPooling3D, Input, concatenate, Flatten, Dense, merge
    from keras.models import Model

    ### input layer
    input_img = Input( shape = input_shape )
    print(input_img._keras_shape)
    
    Nfilter1 = 16
    Nfilter2 = 32
    Nfilter3 = 64
    filter1 = (2,2,2)
    filter2 = (3,3,3)

    tower_1 = Conv3D( Nfilter1, filter1, padding='same', activation='relu' )(input_img) #(tower_1)
    tower_2 = Conv3D( Nfilter1, filter2, padding='same', activation='relu' )(input_img) #(tower_2)

    inc_out_1 = concatenate( [tower_1,tower_2], axis=axis_concatenate )
    pooling1 = MaxPooling3D( pool_size=(2,2,2) )(inc_out_1)

    tower_1 = Conv3D( Nfilter2, filter1, padding='same', activation='relu' )(pooling1) #(tower_1)
    tower_2 = Conv3D( Nfilter2, filter2, padding='same', activation='relu' )(pooling1) #(tower_2)

    inc_out_2 = concatenate( [tower_1,tower_2], axis=axis_concatenate )
    pooling2 = MaxPooling3D( pool_size=(2,2,2) )(inc_out_2)

    tower_1 = Conv3D( Nfilter3, filter1, padding='same', activation='relu' )(pooling2) #(tower_1)
    tower_2 = Conv3D( Nfilter3, filter2, padding='same', activation='relu' )(pooling2) #(tower_2)

    inc_out_3 = concatenate( [tower_1,tower_2], axis=axis_concatenate )
    pooling3 = MaxPooling3D( pool_size=(6,6,6) )(inc_out_3)

    flat = Flatten()(pooling3)
    dense1 = Dense( 128, activation='relu' )(flat)
    dense2 = Dense( 64, activation='relu' )(dense1)
    dense3 = Dense( 32, activation='relu' )(dense2)
    out = Dense( 1, activation='linear' )(dense3)

    model = Model( inputs=input_img, outputs=out )
    
if( model_lowCNN ):
    filter1_size = 2
    
    padding='same' ### 'valid' or 'same'
    
    Nfilter1 = 32
    Nfilter2 = 64
    Nfilter3 = 128
    Nfilter4 = 256
    from keras.layers import Convolution3D, MaxPooling3D, Dropout, Flatten, Dense
    from keras.models import Sequential
    model = Sequential()

    model.add( Convolution3D( Nfilter1, 
                              (filter1_size,filter1_size,filter1_size), 
                              padding='same', activation='relu', 
                              input_shape=input_shape, name='Conv-1' ) )
    #model.add( Convolution3D( Nfilter1, (2,2,2), padding='same', activation='relu', name='Conv-2') )
    model.add( MaxPooling3D( pool_size=(2,2,2), name='Pool-1' ) )

    model.add( Convolution3D( Nfilter2, (2,2,2), padding='same', activation='relu', name='Conv-2') )
    #model.add( Convolution3D( Nfilter2, (2,2,2), padding='same', activation='relu', name='Conv-4') )
    model.add( MaxPooling3D( pool_size=(2,2,2), name='Pool-2' ) )

    model.add( Convolution3D( Nfilter3, (2,2,2), padding='same', activation='relu', name='Conv-3') )
    #model.add( Convolution3D( Nfilter3, (2,2,2), padding='same', activation='relu', name='Conv-6') )
    model.add( MaxPooling3D( pool_size=(2,2,2), name='Pool-3' ) ) ### TODO : try with-without

    model.add( Convolution3D( Nfilter4, (2,2,2), padding='same', activation='relu', name='Conv-5') )
    #model.add( Convolution3D( Nfilter3, (2,2,2), padding='same', activation='relu', name='Conv-6') )
    model.add( MaxPooling3D( pool_size=(2,2,2), name='Pool-4' ) ) ### TODO : try with-without

    model.add( Flatten( name='Flat' ) )
    model.add( Dense( Nfilter4, activation='relu', name='Dense-0' ) )
    model.add( Dense( Nfilter3, activation='relu', name='Dense-1' ) )
    model.add( Dense( Nfilter2, activation='relu', name='Dense-2' ) )
    model.add( Dense( Nfilter1, activation='relu', name='Dense-3' ) )
    model.add( Dense( 1, activation='linear', name='Out' ) )
    
if( model_lowCNN_NOTvir ):

    from keras.layers import Conv3D, MaxPooling3D, Input, concatenate, Flatten, Dense, merge
    from keras.models import Model

    ### input layer
    input_img = Input( shape = input_shape )
    print(input_img._keras_shape)

    Nfilter1 = 8
    Nfilter2 = 16
    Nfilter3 = 32
    Nfilter4 = 64
    filter1 = (2,2,2)

    conv_1 = Conv3D( Nfilter1, filter1, padding='same', activation='relu' )(input_img) #
    pooling1 = MaxPooling3D( pool_size=filter1 )(conv_1)

    conv_2 = Conv3D( Nfilter2, filter1, padding='same', activation='relu' )(pooling1) #
    pooling2 = MaxPooling3D( pool_size=filter1 )(conv_2)

    conv_3 = Conv3D( Nfilter3, filter1, padding='same', activation='relu' )(pooling2) #
    pooling3 = MaxPooling3D( pool_size=filter1 )(conv_3)

    conv_4 = Conv3D( Nfilter4, filter1, padding='same', activation='relu' )(pooling3) #
    pooling4 = MaxPooling3D( pool_size=filter1 )(conv_4)

    flat = Flatten()(pooling4)
    print(flat._keras_shape)

    dense1 = Dense( Nfilter4, activation='relu' )(flat)
    dense2 = Dense( Nfilter3, activation='relu' )(dense1)
    dense3 = Dense( Nfilter2, activation='relu' )(dense2)
    dense4 = Dense( Nfilter1, activation='relu' )(dense3)
    out = Dense( 1, activation='linear' )(dense4)

    model = Model( inputs=input_img, outputs=out )
    
if( model_lowCNN_Tvir ):

    from keras.layers import Conv3D, MaxPooling3D, Input, concatenate, Flatten, Dense, merge
    from keras.models import Model

    ### input layer
    input_img = Input( shape = input_shape )
    print(input_img._keras_shape)

    Nfilter1 = 8
    Nfilter2 = 16
    Nfilter3 = 32
    Nfilter4 = 64
    filter1 = (2,2,2)

    conv_1 = Conv3D( Nfilter1, filter1, padding='same', activation='relu' )(input_img) #
    pooling1 = MaxPooling3D( pool_size=filter1 )(conv_1)

    conv_2 = Conv3D( Nfilter2, filter1, padding='same', activation='relu' )(pooling1) #
    pooling2 = MaxPooling3D( pool_size=filter1 )(conv_2)

    conv_3 = Conv3D( Nfilter3, filter1, padding='same', activation='relu' )(pooling2) #
    pooling3 = MaxPooling3D( pool_size=filter1 )(conv_3)

    conv_4 = Conv3D( Nfilter4, filter1, padding='same', activation='relu' )(pooling3) #
    pooling4 = MaxPooling3D( pool_size=filter1 )(conv_4)

    flat = Flatten()(pooling4)
    print(flat._keras_shape)

    ### add Tvir in the NN
    input_Tvir_shape = (1,)
    input_Tvir = Input( shape = input_Tvir_shape )
    print(input_Tvir._keras_shape)
    #dense_Tvir = Dense( 1, activation='relu' )(input_Tvir)

    #merge = concatenate( [flat, dense_Tvir], axis=-1 )

    dense1 = Dense( Nfilter4, activation='relu' )(flat)
    dense2 = Dense( Nfilter3, activation='relu' )(dense1)
    dense3 = Dense( Nfilter2, activation='relu' )(dense2)
    
    merge = concatenate( [dense3, input_Tvir], axis=-1 )
    
    dense4 = Dense( Nfilter1, activation='relu' )(merge)
    out = Dense( 1, activation='linear' )(dense4)

    model = Model( inputs=[input_img,input_Tvir], outputs=out )
    
if( model_lowCNN_Tvir_Zeta ):

    from keras.layers import Conv3D, MaxPooling3D, Input, concatenate, Flatten, Dense, merge
    from keras.models import Model

    ### input layer
    input_img = Input( shape = input_shape )
    print(input_img._keras_shape)

    Nfilter1 = 8
    Nfilter2 = 16
    Nfilter3 = 32
    Nfilter4 = 64
    filter1 = (2,2,2)

    conv_1 = Conv3D( Nfilter1, filter1, padding='same', activation='relu' )(input_img) #
    pooling1 = MaxPooling3D( pool_size=filter1 )(conv_1)

    conv_2 = Conv3D( Nfilter2, filter1, padding='same', activation='relu' )(pooling1) #
    pooling2 = MaxPooling3D( pool_size=filter1 )(conv_2)

    conv_3 = Conv3D( Nfilter3, filter1, padding='same', activation='relu' )(pooling2) #
    pooling3 = MaxPooling3D( pool_size=filter1 )(conv_3)

    conv_4 = Conv3D( Nfilter4, filter1, padding='same', activation='relu' )(pooling3) #
    pooling4 = MaxPooling3D( pool_size=filter1 )(conv_4)

    flat = Flatten()(pooling4)
    print(flat._keras_shape)

    ### add Tvir in the NN
    input_Tvir_shape = (1,)
    input_Tvir = Input( shape = input_Tvir_shape )
    print(input_Tvir._keras_shape)
    #dense_Tvir = Dense( 1, activation='relu' )(input_Tvir)
    
    ### add Zeta in the NN
    input_Zeta_shape = (1,)
    input_Zeta = Input( shape = input_Zeta_shape )
    print(input_Zeta._keras_shape)
    #dense_Zeta = Dense( 1, activation='relu' )(input_Zeta)

    #merge = concatenate( [flat, dense_Tvir], axis=-1 )

    dense1 = Dense( Nfilter4, activation='relu' )(flat)
    dense2 = Dense( Nfilter3, activation='relu' )(dense1)
    dense3 = Dense( Nfilter2, activation='relu' )(dense2)
    
    merge = concatenate( [dense3, input_Tvir, input_Zeta], axis=-1 )
    
    dense4 = Dense( Nfilter1, activation='relu' )(merge)
    out = Dense( 1, activation='linear' )(dense4)

    model = Model( inputs=[input_img,input_Tvir,input_Zeta], outputs=out )
    
if( model_dense ):
    Nfilter1 = 8
    Nfilter2 = 16
    Nfilter3 = 32
    Nfilter4 = 64

    from keras.layers import Convolution3D, MaxPooling3D, Dropout, Flatten, Dense
    from keras.models import Sequential
    model = Sequential()

    model.add( Flatten( name='Flat', input_shape=input_shape ) )
    model.add( Dense( Nfilter4, activation='relu', name='Dense-0' ) )
    model.add( Dense( Nfilter3, activation='relu', name='Dense-1' ) )
    model.add( Dense( Nfilter2, activation='relu', name='Dense-2' ) )
    model.add( Dense( Nfilter1, activation='relu', name='Dense-3' ) )
    model.add( Dense( 1, activation='linear', name='Out' ) )
    
if( model_heavy ):
    
    from keras.layers import Conv3D, MaxPooling3D, Input, concatenate, Flatten, Dense, merge
    from keras.models import Model

    ### input layer
    input_img = Input( shape = input_shape )
    print(input_img._keras_shape)

    Nfilter1 = 8
    Nfilter2 = 16
    Nfilter3 = 32
    Nfilter4 = 64
    filter1 = (2,2,2)

    conv_1 = Conv3D( Nfilter1, filter1, padding='same', activation='relu' )(input_img) #
    pooling1 = MaxPooling3D( pool_size=filter1 )(conv_1)

    conv_2 = Conv3D( Nfilter2, filter1, padding='same', activation='relu' )(pooling1) #
    pooling2 = MaxPooling3D( pool_size=filter1 )(conv_2)

    conv_3 = Conv3D( Nfilter3, filter1, padding='same', activation='relu' )(pooling2) #
    pooling3 = MaxPooling3D( pool_size=filter1 )(conv_3)

    conv_4 = Conv3D( Nfilter4, filter1, padding='same', activation='relu' )(pooling3) #
    pooling4 = MaxPooling3D( pool_size=(6,6,6) )(conv_4)

    flat = Flatten()(pooling4)
    dense1 = Dense( 128, activation='relu' )(flat)
    dense2 = Dense( 64, activation='relu' )(dense1)
    dense3 = Dense( 32, activation='relu' )(dense2)
    out = Dense( 1, activation='linear' )(dense3)

    model = Model( inputs=input_img, outputs=out )

    
model.summary(line_length=120) 

######################
### LEARNING PHASE ###
######################
        
loss = 'mean_squared_error' ### classic loss function for regression, see also 'mae'

### DEFINE THE OPTIMIZER
# from keras.optimizers import SGD
# lrate = 0.01
# decay = lrate/epochs 
# momentum = 0.9
# sgd = SGD( lr=lrate, momentum=momentum, decay=decay, nesterov=False )
#optimizer = sgd ### 'adam' ###  classic optimizer function for regression, see also 'sgd'
optimizer = 'adadelta' #'adagrad'  #'adadelta' #'adam' # 'adamax' # 'Nadam' # 'RMSprop' # sgd
print( 'optimizer: %s'%optimizer )

### DEFINE THE LEARNING RATE
import math
def step_decay(epoch):
    #dec=0.5
    #epoch_drop = 2
    #if(optimizer=='adagrad'):
    #    init_lr = 0.04
    #elif(optimizer=='adadelta'):
    #    dec=0.8
    #    epoch_drop = 1
    #    init_lr = 4.
    #elif(optimizer=='adam'):
    #    init_lr = 0.004
    #elif(optimizer=='adamax' or optimizer=='Nadam' ):
    #    init_lr = 0.008
        
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
if( 0 ):
    from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
    #lrate = LearningRateScheduler( step_decay )
    lrate = ReduceLROnPlateau( monitor='val_loss', factor=0.5, patience=2 )
    callbacks_list.append( lrate )
    
from keras.callbacks import Callback
class LR_tracer(Callback):
    def on_epoch_begin(self, epoch, logs={}):
        lr = K.eval( self.model.optimizer.lr )
        print( ' LR: %.6f '%(lr) )
callbacks_list.append( LR_tracer() )

### model compilations
model.compile( loss=loss,
               optimizer=optimizer,
               metrics=['mae'] )

### THE LEARNING FUNCTION
if( model_lowCNN_Tvir ):
    history = model.fit( [LC_train[:],Param_train[:,1]], Param_train[:,paramNum],
               epochs=epochs,
               batch_size=batch_size,
               callbacks=callbacks_list,
               validation_data=( [LC_test[:1000],Param_test[:1000,1]], Param_test[:1000,paramNum] ),
               verbose=True )
elif( model_lowCNN_Tvir_Zeta ):
    history = model.fit( [LC_train[:],Param_train[:,1],Param_train[:,2]], Param_train[:,paramNum],
               epochs=epochs,
               batch_size=batch_size,
               callbacks=callbacks_list,
               validation_data=( [LC_test[:1000],Param_test[:1000,1],Param_test[:1000,2]], Param_test[:1000,paramNum] ),
               verbose=True )
else:
    history = model.fit( LC_train[:], Param_train[:,paramNum],
           epochs=epochs,
           batch_size=batch_size,
           callbacks=callbacks_list,
           validation_data=( LC_test[:1000], Param_test[:1000,paramNum] ),
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

if( model_lowCNN_Tvir ):
    predictions = model.predict( [LC_test,Param_test[:,1]], verbose=True )
elif( model_lowCNN_Tvir_Zeta ):
    predictions = model.predict( [LC_test,Param_test[:,1],Param_test[:,2]], verbose=True )
else:
    predictions = model.predict( LC_test, verbose=True )

print( 'R2: ', 1 - (((predictions[:,0] - Param_test[:,paramNum])**2).sum(axis=0)) / ((predictions[:,0] - predictions[:,0].mean(axis=0) )**2).sum(axis=0) )

np.save( CNN_folder + prediction_file, predictions[:,0] )



