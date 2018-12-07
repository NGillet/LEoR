import numpy as np
import matplotlib.pyplot as plt
### LC color from Brad and Andrei
import matplotlib
EoR_colour = matplotlib.colors.LinearSegmentedColormap.from_list('EoR_colour', 
                                                                 [(0, 'white'),
                                                                  (0.21, 'yellow'),
                                                                  (0.42, 'orange'),
                                                                  (0.63, 'red'),
                                                                  (0.86, 'black'),
                                                                  (0.9, 'blue'),
                                                                  (1, 'cyan')])
plt.register_cmap(cmap=EoR_colour)
#plt.rcParams["figure.figsize"] = (20,5)
plt.rcParams['image.cmap'] = 'EoR_colour'

def shrink_1D(data, rows, func=np.mean ):
    return func( data.reshape(rows, data.shape[0]//rows), axis=1 )

def shrink_2D(data, rows, cols, func=np.mean ):
    #return data.reshape(rows, data.shape[0]//rows, cols, data.shape[1]//cols).sum(axis=1).sum(axis=2)
    #return data.reshape(rows, data.shape[0]//rows, cols, data.shape[1]//cols).mean(axis=1).mean(axis=2)

    return func(func(data.reshape(rows, data.shape[0]//rows, cols, data.shape[1]//cols), axis=1), axis=2)

def shrink_3D(data, rows, cols, depths):
    return data.reshape(rows, data.shape[0]//rows, cols, data.shape[1]//cols, depths, data.shape[2]//depths).mean(axis=1).mean(axis=2).mean(axis=3)

######################################################################################################

def read_N_LC( list_LC, BoxSize=200, NsubBox=11, verbose=False, oldMethode=False ):
    NLC = list_LC.shape[0]
    
    ### old loading function
    if(oldMethode):
        L = ( np.stack( read_1_LC( param=None, delta=s, verbose=verbose, return_data=False ), axis=2 ).\
                   reshape( (BoxSize,BoxSize,BoxSize*NsubBox ) )\
                   for s in range( NLC ) )
        return  np.stack( L ).reshape( (NLC*BoxSize, BoxSize, BoxSize*NsubBox) )
    
    ### new loading function
    else:
        DATA = np.zeros( (NLC, BoxSize, BoxSize, BoxSize*NsubBox), dtype=np.float32 )
        for j, d in enumerate(list_LC):
            for i in range(NsubBox):
                ax3 = i * BoxSize
                DATA[ j, :, :, ax3:ax3+BoxSize ] = read_1_LC_2( i, ICs=3, param=None, delta=d, verbose=False )
        return DATA
       
######################################################################################################

def read_1LCSlice( delta=10, ICs=3, param=None, verbose=True ):
    
    """
    THE ONE TO USE
    Read one SLICE
    """
    
    ### file name
    #Path = '/amphora/bradley.greig/Light-cones_for_movies/'
    #BoxLocation = 'Lightcone_boxes'
    
    Path = '/amphora/bradley.greig/21CMMC_wTs_LCGrid/ICs_%d/Programs/'%ICs
    if(ICs==3):
        Path = '/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/'

    BoxLocation = 'LightConeSlices'
    
    # delta_T_0.000000_9999.000000_200_300Mpc_LightConeSlice
    
    BoxName = 'delta_T'
    BoxRes = '200'
    N_boxes = 11
    BoxSize = '300Mpc'
        
    ### the parameter that have been changed
    ### 3=Zeta; 4=MFP; 5=Tvir; 6=LX; 7=E0; 8=aX
    WalkerID_One_list = ['3.000000','4.000000','5.000000','6.000000','7.000000','8.000000']
    if( param=='Zeta' ):
        WalkerID_One = WalkerID_One_list[0]
    elif( param=='MFP' ):
        WalkerID_One = WalkerID_One_list[1]
    elif( param=='Tvir' ):
        WalkerID_One = WalkerID_One_list[2]
    elif( param=='LX' ):
        WalkerID_One = WalkerID_One_list[3]
    elif( param=='E0' ):
        WalkerID_One = WalkerID_One_list[4]
    elif( param=='aX' ):
        WalkerID_One = WalkerID_One_list[5]
    elif( param==None ):
        WalkerID_One = '0.000000'
        
    ### for the parameter values: delta
    WalkerID_Two = '%1.6f'%(float(delta))
    
    
    fname = Path+'%s/%s_%s_%s_%s_%s_LightConeSlice'%\
             ( BoxLocation, BoxName, WalkerID_One, WalkerID_Two, BoxRes, BoxSize)
    
    if verbose:
        print( fname )
 

    return np.fromfile(open('%s'%(fname),'rb'), dtype = np.dtype('float32'), \
                                             count = int(BoxRes)*int(BoxRes)*N_boxes).\
                                reshape((int(BoxRes),int(BoxRes)*N_boxes) ) 

    ######################################################################################################
    
def read_1_LC_2( Numbox, ICs=3, param=None, delta=10, verbose=True ):
    
    """
    THE ONE TO USE
    Read one cubic box, part of lightcone
    """
    
    ### file name
    #Path = '/amphora/bradley.greig/Light-cones_for_movies/'
    #BoxLocation = 'Lightcone_boxes'
    
    Path = '/amphora/bradley.greig/21CMMC_wTs_LCGrid/ICs_%d/Programs/'%ICs
    if(ICs==3):
        Path = '/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/'

    BoxLocation = 'LightConeBoxes'
    
    BoxName = 'delta_T'
    BoxRes = '200'
    BoxSize = '300Mpc'
        
    ### the parameter that have been changed
    ### 3=Zeta; 4=MFP; 5=Tvir; 6=LX; 7=E0; 8=aX
    WalkerID_One_list = ['3.000000','4.000000','5.000000','6.000000','7.000000','8.000000']
    if( param=='Zeta' ):
        WalkerID_One = WalkerID_One_list[0]
    elif( param=='MFP' ):
        WalkerID_One = WalkerID_One_list[1]
    elif( param=='Tvir' ):
        WalkerID_One = WalkerID_One_list[2]
    elif( param=='LX' ):
        WalkerID_One = WalkerID_One_list[3]
    elif( param=='E0' ):
        WalkerID_One = WalkerID_One_list[4]
    elif( param=='aX' ):
        WalkerID_One = WalkerID_One_list[5]
    elif( param==None ):
        WalkerID_One = '0.000000'
        
    ### for the parameter values: delta
    WalkerID_Two = '%1.6f'%(float(delta)) 
    
    ### all redshift time step
    #LightConeRedshifts = np.loadtxt(Path+'Log_LightconeRedshift.txt', usecols=(0,))
    
    ### ?
    #Redshifts_LightCone_Begin = ['006.00059','006.75588','007.63959','008.68273','009.92623',
    #                         '011.42502','013.25422','015.51872','018.36853','022.02430','026.82133']
    #Redshifts_LightCone_End = ['006.75588','007.63959','008.68273','009.92623','011.42502',
    #                       '013.25422','015.51872','018.36853','022.02430','026.82133','033.28920']
    #Redshifts_LightCone_Begin = [ '006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42504',
    #                              '013.25424', '015.51874', '018.36856', '022.02435', '026.82139' ]
    Redshifts_LightCone_Begin = [ '006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42503',
                                  '013.25424', '015.51874', '018.36856', '022.02434', '026.82138' ]
    Redshifts_LightCone_End = [ '006.75589', '007.63960', '008.68274', '009.92624', '011.42503',
                                '013.25424', '015.51874', '018.36856', '022.02434', '026.82138', '033.28927' ]
    
    ### ? number of boxes
    NumBoxes = len(Redshifts_LightCone_Begin)
    ### ?
    zmin = 6.0
    zmax = float(Redshifts_LightCone_End[-1])
    
    ### get the number of reshift step = number of images => usefull to reduce the redshift range
    #for jj in range(len(LightConeRedshifts)):
    #    if np.fabs(LightConeRedshifts[jj] - zmax) <= 0.01:
    #        Nmax = jj + 1
    Nmax = len(Redshifts_LightCone_Begin)

    ### all the images
    #LightConeBox = np.zeros( (int(BoxRes), int(BoxRes), Nmax), dtype=np.float32) ### lightcone z slice 2D+20parameter
            
    ### first define all the files names
    #fname = [ Path+'%s/%s_%s_%s__zstart%s_zend%s_FLIPBOXES0_%s_%s_lighttravel'%\
    #         (BoxLocation,BoxName,WalkerID_One,WalkerID_Two,Redshifts_LightCone_Begin[k],\
    #          Redshifts_LightCone_End[k],BoxRes,BoxSize) for k in range(NumBoxes) ]
    
    
    fname = Path+'%s/%s_%s_%s__zstart%s_zend%s_FLIPBOXES0_%s_%s_lighttravel'%\
             (BoxLocation,BoxName,WalkerID_One,WalkerID_Two,Redshifts_LightCone_Begin[Numbox],\
              Redshifts_LightCone_End[Numbox],BoxRes,BoxSize)
    
    if verbose:
        print( Path+'%s/%s_%s_%s__zstart_zend_FLIPBOXES0_%s_%s_lighttravel'%\
             (BoxLocation,BoxName,WalkerID_One,WalkerID_Two,BoxRes,BoxSize) )
 

    return np.fromfile(open('%s'%(fname),'rb'), dtype = np.dtype('float32'), \
                                             count = int(BoxRes)*int(BoxRes)*int(BoxRes)).\
                                reshape((int(BoxRes),int(BoxRes),int(BoxRes)) ) 

######################################################################################################

def read_1_LC( param=None, delta=10, verbose=True, return_data=True ):
    
    """DEPRECATED Read one lightcone"""
    
    ### file name
    #Path = '/amphora/bradley.greig/Light-cones_for_movies/'
    #BoxLocation = 'Lightcone_boxes'
    
    Path = '/amphora/bradley.greig/21CMMC_wTs_LCGrid/ICs_0/Programs/'
    BoxLocation = 'LightConeBoxes'
    
    BoxName = 'delta_T'
    BoxRes = '200'
    BoxSize = '300Mpc'
        
    ### the parameter that have been changed
    ### 3=Zeta; 4=MFP; 5=Tvir; 6=LX; 7=E0; 8=aX
    WalkerID_One_list = ['3.000000','4.000000','5.000000','6.000000','7.000000','8.000000']
    if( param=='Zeta' ):
        WalkerID_One = WalkerID_One_list[0]
    elif( param=='MFP' ):
        WalkerID_One = WalkerID_One_list[1]
    elif( param=='Tvir' ):
        WalkerID_One = WalkerID_One_list[2]
    elif( param=='LX' ):
        WalkerID_One = WalkerID_One_list[3]
    elif( param=='E0' ):
        WalkerID_One = WalkerID_One_list[4]
    elif( param=='aX' ):
        WalkerID_One = WalkerID_One_list[5]
    elif( param==None ):
        WalkerID_One = '0.000000'
        
    ### for the parameter values: delta
    WalkerID_Two = '%1.6f'%(float(delta)) 
    
    ### all redshift time step
    #LightConeRedshifts = np.loadtxt(Path+'Log_LightconeRedshift.txt', usecols=(0,))
    
    ### ?
    #Redshifts_LightCone_Begin = ['006.00059','006.75588','007.63959','008.68273','009.92623',
    #                         '011.42502','013.25422','015.51872','018.36853','022.02430','026.82133']
    #Redshifts_LightCone_End = ['006.75588','007.63959','008.68273','009.92623','011.42502',
    #                       '013.25422','015.51872','018.36853','022.02430','026.82133','033.28920']
    Redshifts_LightCone_Begin = [ '006.00060', '006.75589', '007.63960', '008.68274', '009.92624', '011.42504',
                                  '013.25424', '015.51874', '018.36856', '022.02435', '026.82139' ]
    Redshifts_LightCone_End = [ '006.75589', '007.63960', '008.68274', '009.92624', '011.42504',
                                '013.25424', '015.51874', '018.36856', '022.02435', '026.82139', '033.28927' ]
    
    ### ? number of boxes
    NumBoxes = len(Redshifts_LightCone_Begin)
    ### ?
    zmin = 6.0
    zmax = float(Redshifts_LightCone_End[-1])
    
    ### get the number of reshift step = number of images => usefull to reduce the redshift range
    #for jj in range(len(LightConeRedshifts)):
    #    if np.fabs(LightConeRedshifts[jj] - zmax) <= 0.01:
    #        Nmax = jj + 1
    Nmax = len(Redshifts_LightCone_Begin)

    ### all the images
    #LightConeBox = np.zeros( (int(BoxRes), int(BoxRes), Nmax), dtype=np.float32) ### lightcone z slice 2D+20parameter
            
    ### first define all the files names
    #fname = [ Path+'%s/%s_%s_%s__zstart%s_zend%s_FLIPBOXES0_%s_%s_lighttravel'%\
    #         (BoxLocation,BoxName,WalkerID_One,WalkerID_Two,Redshifts_LightCone_Begin[k],\
    #          Redshifts_LightCone_End[k],BoxRes,BoxSize) for k in range(NumBoxes) ]
    
    
    fname = ( Path+'%s/%s_%s_%s__zstart%s_zend%s_FLIPBOXES0_%s_%s_lighttravel'%\
             (BoxLocation,BoxName,WalkerID_One,WalkerID_Two,Redshifts_LightCone_Begin[k],\
              Redshifts_LightCone_End[k],BoxRes,BoxSize) for k in range(NumBoxes) )
    
    if verbose:
        print( Path+'%s/%s_%s_%s__zstart_zend_FLIPBOXES0_%s_%s_lighttravel'%\
             (BoxLocation,BoxName,WalkerID_One,WalkerID_Two,BoxRes,BoxSize) )
    #if verbose:
    #    for f in fname:
    #        print(f)
    
    ### Generator of all the sub-boxes
    #IndividualLightConeBox = [ np.fromfile(open('%s'%(fn),'rb'), dtype = np.dtype('float32'), count = int(BoxRes)*int(BoxRes)*int(BoxRes)).reshape( (int(BoxRes),int(BoxRes),int(BoxRes)) ) for fn in fname ]
    
    #return np.concatenate( IndividualLightConeBox, axis=2 )
    
    ### 1 line return
    if(return_data):
        #return np.concatenate( [ np.fromfile(open('%s'%(fn),'rb'), dtype = np.dtype('float32'), \
        #                                    count = int(BoxRes)*int(BoxRes)*int(BoxRes)).\
        #                        reshape((int(BoxRes),int(BoxRes),int(BoxRes)) ) for fn in fname ], axis=2 )
        return np.stack( ( np.fromfile(open('%s'%(fn),'rb'), dtype = np.dtype('float32'), \
                                            count = int(BoxRes)*int(BoxRes)*int(BoxRes)).\
                                reshape((int(BoxRes),int(BoxRes),int(BoxRes)) ) for fn in fname ), axis=2).\
               reshape( (int(BoxRes),int(BoxRes),int(BoxRes)*Nmax ) )
    else:
        return ( np.fromfile(open('%s'%(fn),'rb'), dtype = np.dtype('float32'), \
                                             count = int(BoxRes)*int(BoxRes)*int(BoxRes)).\
                                reshape((int(BoxRes),int(BoxRes),int(BoxRes)) ) for fn in fname )

######################################################################################################

def read_1_param( param=None, ICs=3, delta=10, verbose=True ):
    
    ### file name
   # Path = '/amphora/bradley.greig/Light-cones_for_movies/'
    #Path = '/amphora/bradley.greig/21CMMC_wTs_LCGrid/ICs_%d/Programs/'%(ICs)
    Path = '/amphora/bradley.greig/21CMMC_wTs_LC_RSDs/21CMMC_wTs_LCGrid/ICs_%d/Programs/'%(ICs)
    if( ICs==3 ):
        Path = '/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/'
    ParamLocation = 'GridPositions'
        
    ### the parameter that have been changed
    ### 3=Zeta; 4=MFP; 5=Tvir; 6=LX; 7=E0; 8=aX
    WalkerID_One_list = ['3.000000','4.000000','5.000000','6.000000','7.000000','8.000000']
    if( param=='Zeta' ):
        WalkerID_One = WalkerID_One_list[0]
    elif( param=='MFP' ):
        WalkerID_One = WalkerID_One_list[1]
    elif( param=='Tvir' ):
        WalkerID_One = WalkerID_One_list[2]
    elif( param=='LX' ):
        WalkerID_One = WalkerID_One_list[3]
    elif( param=='E0' ):
        WalkerID_One = WalkerID_One_list[4]
    elif( param=='aX' ):
        WalkerID_One = WalkerID_One_list[5]
    elif( param==None ):
        WalkerID_One = '0.000000'
        
    ### for the parameter values: delta
    WalkerID_Two = '%1.6f'%(float(delta)) 
            
    ### first get the params
    ParamNames = Path+'%s/Walker_%s_%s.txt'%(ParamLocation,WalkerID_One,WalkerID_Two)
    with open('%s'%(ParamNames),'r') as f:
        lines = f.readlines()
        Zeta = np.float( lines[2].split()[1] )
        MFP = np.float( lines[3].split()[1] )
        Tvir = np.float( lines[4].split()[1] )
        LX = np.float( lines[5].split()[1] )
        Eo = np.float( lines[6].split()[1] )
        aX = np.float( lines[9].split()[1] )

    return np.array([ Zeta, MFP, Tvir, LX, Eo, aX ])

def load_LC( Yslice=150, verbose=False ):
    
    """read several LC"""
    
    ### file name
    Path = '/amphora/bradley.greig/Light-cones_for_movies/'
    BoxLocation = 'Lightcone_boxes'
    BoxName = 'delta_T'
    BoxRes = '200'
    BoxSize = '300Mpc'
    
    ParamLocation = 'ParameterValues'
    
    ### the parameter that have been changed
    ### 3=Zeta; 4=MFP; 5=Tvir; 6=LX; 7=E0; 8=aX
    WalkerID_One = ['3.000000','4.000000','5.000000','6.000000','7.000000','8.000000']
    
    ### position Y of the slice
    FixedYPos_coord = Yslice ### 
    
    ### all redshift time step
    LightConeRedshifts = np.loadtxt(Path+'Log_LightconeRedshift.txt', usecols=(0,))
    ### ?
    Redshifts_LightCone_Begin = ['006.00059','006.75588','007.63959','008.68273','009.92623',
                             '011.42502','013.25422','015.51872','018.36853','022.02430','026.82133']
    Redshifts_LightCone_End = ['006.75588','007.63959','008.68273','009.92623','011.42502',
                           '013.25422','015.51872','018.36853','022.02430','026.82133','033.28920']
    ### ? number of boxes
    NumBoxes = len(Redshifts_LightCone_Begin)
    ### ?
    zmin = 6.0
    zmax = float(Redshifts_LightCone_End[-1])
    
    ### get the number of reshift step = number of images => usefull to reduce the redshift range
    for jj in range(len(LightConeRedshifts)):
        if np.fabs(LightConeRedshifts[jj] - zmax) <= 0.01:
            Nmax = jj + 1

    ### all the images
    LightConeBox = np.zeros( (21, int(BoxRes), Nmax), dtype=np.float32) ### lightcone z slice 2D+20parameter
    ### all the images parameters
    params = np.zeros( (21, 6) )
    
    ### for Parameter
    for i in range(1):
        i = 2 ### = Tvir
        
        ### for the parameter values
        for j in np.arange(21):
            WalkerID_Two = '%1.6f'%(float(j)) 
            
            ### first get the params
            ParamNames = Path+'%s/Walker_%s_%s.txt'%(ParamLocation,WalkerID_One[i],WalkerID_Two)
            f = open('%s'%(ParamNames),'r')
            lines = f.readlines()
            Zeta = np.float( lines[2].split()[1] )
            MFP = np.float( lines[3].split()[1] )
            Tvir = np.float( lines[4].split()[1] )
            LX = np.float( lines[5].split()[1] )
            Eo = np.float( lines[6].split()[1] )
            aX = np.float( lines[9].split()[1] )
            f.close()
            params[j,:] = [ Zeta, MFP, Tvir, LX, Eo, aX ]

            ### for each sub-Boxes
            for k in range(NumBoxes): 

                ### get the LC data
                BoxNames = Path+'%s/%s_%s_%s__zstart%s_zend%s_FLIPBOXES0_%s_%s_lighttravel'%(BoxLocation,BoxName,WalkerID_One[i],WalkerID_Two,Redshifts_LightCone_Begin[k],Redshifts_LightCone_End[k],BoxRes,BoxSize)
                if verbose:
                    print( 'READING : '+BoxNames )

                f = open('%s'%(BoxNames),'rb')
                IndividualLightConeBox = np.fromfile(f, dtype = np.dtype('float32'), count = int(BoxRes)*int(BoxRes)*int(BoxRes))
                f.close()
                
                ### HERE I play with the indices to avoid the double loop
                ind1 = int(BoxRes)*( FixedYPos_coord + int(BoxRes)*0 )
                ind2 = int(BoxRes)*( FixedYPos_coord + int(BoxRes)*(int(BoxRes)-1) )
                indt = np.arange( ind1, ind2+1, int(BoxRes)**2 )
                ind = np.array( list( (np.arange( indt[ii], indt[ii]+200 ) for ii in range(int(BoxRes)) ) ) )
                
                LightConeBox[ j, :, 0+int(BoxRes)*k:int(BoxRes)+int(BoxRes)*k ] = IndividualLightConeBox[ ind ]
    
    return LightConeBox, params
#                 ### ORIGINAL double loop to alocate the slice
#                 ### for each coordinate of the box
#                 for ii in range(int(BoxRes)):
#                     if ii+int(BoxRes)*k < Nmax:
#                         for kk in range(int(BoxRes)):
#                             LightConeBox[kk][j][ii+int(BoxRes)*k] = IndividualLightConeBox[ii + int(BoxRes)*( FixedYPos_coord + int(BoxRes)*kk )]
#     return LightConeBox

def LOAD_DATA_old( dim=3, fullres=True, ICs=3, RandomSeed=2235, trainSize=0.9, verbose=True, LHS=True ): 
    """
    POSSIBILITIES
    1) low resolution 2D slice  : dim=2 and fullres=False (default)
    2) low resolution 3D slice  : dim=3 and fullres=False
    3) full resolution 2D slice : dim=2 and fullres=True
    4) high resolution 3D slice  : dim=3 and fullres=True
    5) medium resolution 3D slice : dim=3 Lside=150Mpc, resolution=20, ICs=3
    6) high resolution 3D slice : dim=3 Lside=150Mpc, resolution=25, ICs=4
    
    ICs seed : 0=>grid, 1=>random
    
    RandomSeed the seed of numpy.random. IMPORTANT if you want to test again a sve network
    
    trainSize: fraction size of the training set (<1)
    """
    import pyDOE
    np.random.RandomState( np.random.seed(RandomSeed) ) ### fixe the random seed for testing

    Nsimu = 12959 ### total number of simu/dataset
    if( (ICs==3) ):
        Nsimu = 10000
    BoxInLC = 11  ### number of concatenate boxes/LC
   
    ############################################################################################
    ### load the DATA
    if(verbose):
        print('Data loading')
    from time import time
    tin = time()
    if(fullres):
        if(dim==2):
            ### Full slice (200,2200) 12959 LC, with random axe, random axe
            BoxSize = 200 ### full resolution 
            file_name = '/amphora/nicolas.gillet/LC_data/LC_fullSlice_12959_grid_ICs_%d.npy'%(ICs)
        if(dim==3):
            ### High resolution cube (20,20,440) 12959 LC,
            BoxSize = 20 ### full resolution 
            if(ICs==3):
                file_name = '/amphora/nicolas.gillet/LC_data/LC_3D_40_halfsize_10000_rand_ICs_%d.npy'%(ICs)
            else:
                file_name = '/amphora/nicolas.gillet/LC_data/LC_3D_40_halfsize_12959_grid_ICs_%d.npy'%(ICs)
    else:
        ### reduce box (20,20,220) 12959 LC
        BoxSize = 20  ### low resolution 
        file_name = '/amphora/nicolas.gillet/LC_data/LC_12959_grid_ICs_%d.npy'%(ICs)
    LightConeBox = np.load( file_name )
    print('Loading time: %f'%(time()-tin))

    ############################################################################################
    ### data  normalization
    if(verbose):
        print('Data normalization')
    val_min = -250
    val_max = 50
    LightConeBox = np.clip( LightConeBox, val_min, val_max )
    def normalize_array_inplace( array, vmin=-250, vmax=50 ):
        array -= vmin
        array /= (vmax-vmin)
    normalize_array_inplace( LightConeBox )

    ############################################################################################
    ### reshape data => need an additionnal dimension at the end
    if(verbose):
        print('Data reshape')
    ### 3D
    if( (dim==3 and not(fullres)) or (dim==2 and not(fullres)) ):
        LightConeBox = LightConeBox.reshape( (Nsimu, BoxSize, BoxSize, BoxSize*BoxInLC, 1) )
    if(dim==3 and fullres):
        LightConeBox = LightConeBox.reshape( (Nsimu, BoxSize, BoxSize, BoxSize*BoxInLC*2, 1) )
    ### 2D Full resolution
    if(dim==2 and fullres):
        LightConeBox = LightConeBox.reshape( (Nsimu, BoxSize, BoxSize*BoxInLC, 1) )

    print( 'Data shape :', LightConeBox.shape )

    ############################################################################################
    ### randomly choose axes and reverse data to 'blur' IC
    ### IN THE CASE OF FULLRES, THE SLICE ARE ALREDY RANDOM
    if(verbose):
        print('Data augmentation')
        
    ### random direction of LC (3D)
    if( dim==3 and ICs<3 ):
        ### if indAxe change the 2 small axes
        indAxe = np.random.np.random.randint(0, 2, Nsimu).astype('bool')
        ### if reverseAxe reverse the image on the axis
        reverseAxe1 = np.random.np.random.randint(0, 2, Nsimu).astype('bool')
        reverseAxe2 = np.random.np.random.randint(0, 2, Nsimu).astype('bool')
        for lc in range(Nsimu):
            if indAxe[lc]:
                LightConeBox[lc,:,:,:,0] = np.swapaxes( LightConeBox,1,2 )[lc,:,:,:,0]
            if reverseAxe1[lc]:
                LightConeBox[lc,:,:,:,0] = LightConeBox[lc,::-1,:,:,0]
            if reverseAxe2[lc]:
                LightConeBox[lc,:,:,:,0] = LightConeBox[lc,:,::-1,:,0]
                
    if( dim==2 and not(fullres) ):
        LightConeBox_2 = np.zeros( (Nsimu, BoxSize, BoxInLC*BoxSize, 1), dtype=np.float32 )
        ### Select random slice (2D)
        ### if indAxe take the slice in the othe axe
        indAxe = np.random.np.random.randint(0, 2, Nsimu).astype('bool')
        numSlice = np.random.np.random.randint(0, BoxSize, Nsimu)
        for lc in range(Nsimu):
            if indAxe[lc]:
                LightConeBox_2[lc,:,:,0] = LightConeBox[lc,:,numSlice[lc],:,0]
            else:
                LightConeBox_2[lc,:,:,0] = LightConeBox[lc,numSlice[lc],:,:,0]
        LightConeBox = LightConeBox_2
 
        
    ############################################################################################
    ### load the parameters of each LC
    if(verbose):
        print('Load the parameters')
    from lightcone_functions import read_1_param
    param_reduce = np.array( [ read_1_param( param=None, ICs=ICs, delta=s, verbose=False ) for s in range( Nsimu ) ] )
    if(ICs==3):
        param_reduce = param_reduce[:,[0,2,3,4]]
    param = (param_reduce - param_reduce.min(axis=0)) / (param_reduce.max(axis=0) - param_reduce.min(axis=0) )
        
    ############################################################################################
    ### random separate TRAIN to TEST
    ### here I define the indices of the 2 set
    if(verbose):
        print('Defining Train-Test sets')

    Ndata = LightConeBox.shape[0] ### total number of data
    Ntrain = np.int(Ndata * trainSize) ###controle the size of the training sample

    shuffle_index = np.arange( Ndata )
    np.random.shuffle( shuffle_index )

    train_select = shuffle_index[ :Ntrain ] ### indices of the traning set
    test_select  = shuffle_index[ Ntrain: ] ### indices of the testing set
    
    if(verbose):
        print('Latin-Hypercube Sampling')
    if( LHS ): ### Latin Hypercube sample
        if(ICs==3):
            number_of_param = 4
            number_of_bin_per_param = 12
            k = pyDOE.lhs( number_of_param, samples=number_of_bin_per_param )
        else:
            number_of_param = 6
            number_of_bin_per_param = 6
            k = pyDOE.lhs( number_of_param, samples=number_of_bin_per_param )
        ### build approximative LHS
        import scipy.spatial.kdtree as KD
        tree = KD.KDTree( param )
        nn = tree.query( k, k=2 ) ### k=2 : return 2 neighbors
        train_select = np.unique( nn[1] )
        
        np.random.shuffle( train_select )
        Ntrain = train_select.shape[0]

        tempo = np.arange( Ndata )
        tempo[ train_select ]=-1
        test_select = np.where(tempo>=0)[0]
        np.random.shuffle( test_select )
        Ntest = test_select.shape[0]



    print( 'Number of LC:', Ndata, ', train set: ', Ntrain, ', test set: ', Ndata-Ntrain )
    if(0):
        return (LightConeBox[train_select], LightConeBox[test_select]) , (param[train_select], param[test_select]), (param_reduce[train_select], param_reduce[test_select])
    else:
        return (LightConeBox[train_select], LightConeBox[test_select]) , (param[train_select], param[test_select])
    
    
    
def lightcone_frequency():
    
    BoxRes = 200
    
    Redshifts = [ '6.000594', '6.140606', '6.283418', '6.429087', '6.577668', '6.729221', '6.883806', '7.041489', '7.202312', '7.366358', '7.533685', '7.704358', '7.878446', '8.056021', '8.237134', '8.421877', '8.610314', '8.802521', '8.998578', '9.198543', '9.402514', '9.610563', '9.822775', '10.039230', '10.260015', '10.485220', '10.714919', '10.949220', '11.188202', '11.431970', '11.680605', '11.934220', '12.192902', '12.456760', '12.725895', '13.000420', '13.280421', '13.566030', '13.857350', '14.154500', '14.457587', '14.766740', '15.082080', '15.403720', '15.731789', '16.066420', '16.407751', '16.755911', '17.111023', '17.473240', '17.842709', '18.219561', '18.603954', '18.996031', '19.395954', '19.803869', '20.219950', '20.644350', '21.077230', '21.518780', '21.969151', '22.428539', '22.897110', '23.375050', '23.862551', '24.359800', '24.867001', '25.384340', '25.912029', '26.450270', '26.999269', '27.559259', '28.130440', '28.713051', '29.307310', '29.913460', '30.531731', '31.162359', '31.805611', '32.461720', '33.130951', '33.813568', '34.509838' ]

    Redshifts_LightCone_Begin = [ '006.00059', '006.75588', '007.63959', '008.68273', '009.92623', '011.42502', '013.25422', '015.51872', '018.36853', '022.02430', '026.82133']
    Redshifts_LightCone_End = [ '006.75588', '007.63959', '008.68273', '009.92623', '011.42502', '013.25422', '015.51872', '018.36853', '022.02430', '026.82133', '033.28920' ]

    NumBoxes = len(Redshifts_LightCone_Begin)
    zmin = 6.0
    zmax = float( Redshifts_LightCone_End[-1] )
    
    AVG_file = '/amphora/bradley.greig/21CMMC_wTs_LC_RSDs_Nicolas/Programs/AveData/AveData_0.000000_414.000000.txt'
    
    LightConeRedshifts = np.loadtxt( AVG_file, usecols=(0,) )
    
    Nmax = 1
    for jj, LC_z in LightConeRedshifts:
        if np.fabs( LC_z - zmax ) <= 0.01:
            Nmax = jj + 1 ### number of box
      
    NUM_Z_PS = len(Redshifts)
    
    LightConeRedshifts_ForFigure = np.zeros(Nmax)
    Y_Vals = np.zeros(int(BoxRes))
    X_Vals = np.zeros(Nmax)

    for ii in range(int(BoxRes)):
        Y_Vals[ii] = (0.0 + 300.*(float(ii) + 0.5)/((int(BoxRes)-1)))
    for ii in range(Nmax):
        LightConeRedshifts_ForFigure[ii] = LightConeRedshifts[ii]
        X_Vals[ii] = ii + 0.5
    
    
    
def LOAD_DATA( DATABASE='150Mpc_r25', 
               RandomSeed=2235, 
               trainSize=0.9, 
               LHS=True, 
               Nbins_LHS=1000, 
               verbose=True, 
               justParam=False, 
               reduce_LC=False, 
               substract_mean=False, 
               apply_gauss=False, 
               validation=False, 
               justDataID=False, 
               multiple_slice=False): 
    """
    return Light-cones ready for learning
        
    inputs:
        - DATABASE       : '150Mpc_r25', the database to use
        - RandomSeed     : the seed of numpy.random. IMPORTANT if you want to test again a saved network
        - trainSize      : fraction size of the training set (float in [0;1])
        - LHS            : use Latin-Hypercube Sampling for the learning, it is approximate, work for small number of LS (<1000)
        - Nbins_LHS      : number of bins is LHS activated, trainSize is useless if LHS
        - reduce_LC      : divide the high of LC slice per 2, /!\ use with DATABASE=='300Mpc_r200_2D'
        - substract_mean : (TEST) if the mean has to be remove from the LC
        - apply_gauss    : (TEST) aply a gaussian multipication to change the contrast
        - validation     : Do a validation sample
        - justDataID     : return just the ID of the training, testing and validation sample for a given RandomSeed
        - justParam      : return only the param for the given RandomSeed
        - verbose        : Do some control print
        
        /!\ MOST OF THE THINGS ARE HARD-CODED AND DATABASE SPECIFIC !
        
    
    """
    from time import time
    import pyDOE
    np.random.RandomState( np.random.seed(RandomSeed) ) ### fixe the random seed for testing

    if( DATABASE=='150Mpc_r25' ):
        file_name = '/amphora/nicolas.gillet/LC_data/LCDB_150Mpc_r25.npy'
        Nsimu = 10000 ### number of LC
        BoxInLC = 11  ### number of concatenate boxes/LC
        BoxSize = 25 ###
        BoxLong = BoxSize*BoxInLC*2 ### 
    elif( DATABASE=='75Mpc_r50' ):
        file_name = '/amphora/nicolas.gillet/LC_data/LC_3D_px50-2200_N10000_randICs.npy'
        Nsimu = 10000 ### number of LC
        BoxInLC = 11  ### number of concatenate boxes/LC
        BoxSize = 50 ###
        BoxLong = BoxSize*BoxInLC*4
        
    ### MAIN ONE    
    elif( DATABASE=='300Mpc_r200_2D' ):
        file_name = '/amphora/nicolas.gillet/LC_data/LC_SLICE_px2200_N10000_randICs.npy'
        Nsimu = 10000 ### number of LC
        BoxInLC = 11  ### number of concatenate boxes/LC
        BoxSize = 200 ###
        BoxLong = BoxSize*BoxInLC
        
    elif( DATABASE=='avg_1D' ):
        file_name = '/amphora/nicolas.gillet/LC_data/LC_1Davg_px2200_N10000_randICs.npy'
        Nsimu = 10000 ### number of LC
        BoxInLC = 11  ### number of concatenate boxes/LC
        BoxSize = 200 ###
        BoxLong = BoxSize*BoxInLC
        
    ### LC SLICE, 100x2200, 10 slices per LC
    elif( DATABASE=='100_2200_slice_10' ):
        file_name = '/eos_data/nicolas.gillet/LC_data/LC_SLICE10_px100-2200_N10000_randICs.npy'
        Nsimu = 10000 ### number of LC
        BoxInLC = 11  ### number of concatenate boxes/LC
        BoxSize = 200 ### transverse box size
        BoxLong = BoxSize*BoxInLC ### LoS lenght
        Nslice = 10   ### number of slice per LC
        ImgHeight = BoxSize//2
        
        ### LC slice are already 100px high
        reduce_LC = False
        ### 10 slices per parameters
        multiple_slice = True
        
    if( (Nbins_LHS>1000) and LHS ):
        LHS=False
        trainSize = Nbins_LHS / Nsimu
    
    ############################################################################################
    ### IF HAVE TO LOAD THE DATA
    ############################################################################################
    if( not(justParam) and not(justDataID) ):

        ############################################################################################
        ### load the DATA
        ############################################################################################
        if(verbose):
            print('Data loading')
        tin = time()
        LightConeBox = np.load( file_name )
        print('Loading time: %f'%(time()-tin))
        
        ############################################################################################
        ### data  normalization
        ############################################################################################
        if(verbose):
            print('Data normalization')
            
        ### reduce LC size
        if reduce_LC:
            if(verbose):
                print('Reshaping LC')
            LightConeBox = LightConeBox[:,:BoxSize//2,:]
            if(verbose):
                print('LC shape',LightConeBox.shape)
            
        ### set value limit
        val_min = -250
        val_max = 50
        if(verbose):
            print('Clip LC to vmin=%.0f, vmax=%.0f'%(val_min,val_max))
        np.clip( LightConeBox, val_min, val_max, LightConeBox ); ### in-place clipping

        ### (TEST) substract mean in the LC
        if substract_mean:
            if(verbose):
                print('Substracting mean')
            
            if len( LightConeBox.shape )==2: ### 1D
                print( "!!! NO SUBSTRACT MEAN IN 1D" )
                pass
            
            if len( LightConeBox.shape )==3: ### 2D
                LightConeBox = np.swapaxes( np.swapaxes(LightConeBox,0,1) -np.average( LightConeBox, axis=1 ), 0, 1)
             
            if len( LightConeBox.shape )==4: ### 3D
                LightConeBox = np.swapaxes( np.swapaxes(LightConeBox,0,2) -np.average( LightConeBox, axis=(1,2) ), 0, 2)
            
        ### normalize between [0-1]
        def normalize_array_inplace( array, vmin=-250, vmax=50 ):
            if np.isscalar(vmin):
                array -= vmin
                array /= (vmax-vmin)
            else:
                array -= vmin[:,np.newaxis,np.newaxis]
                array /= (vmax-vmin)[:,np.newaxis,np.newaxis]
        if(verbose):
            print('Normalization')
        if substract_mean:
            normalize_array_inplace( LightConeBox, vmin=LightConeBox.min(axis=(1,2)), vmax=LightConeBox.max(axis=(1,2)) )
        else:
            normalize_array_inplace( LightConeBox )
        
        ### (TEST) APPLY A GAUSSIAN CONTRAST
        if apply_gauss:
            if(verbose):
                print('Apply Gauss filter')
                
            def gauss_func( x, mu=0, std=1 ):
                return np.exp( -0.5 * ( (x-mu)/std )**2) / (std*np.sqrt(2.*np.pi))

            x = np.arange( 2200 )
            contrast =  gauss_func( x, mu=0, std=800 )
            contrast /= contrast.max()
            #LightConeBox = np.swapaxes( LightConeBox, 1, 2 )
            LightConeBox *= contrast
            #LightConeBox = np.swapaxes( LightConeBox, 1, 2 )
            
        ############################################################################################
        ### reshape data => need an additionnal dimension at the end for CNN
        ############################################################################################
        if(verbose):
            print('Data reshape')
        LightConeBox = LightConeBox.reshape( *LightConeBox.shape, 1 )
        print( 'Data shape :', LightConeBox.shape )
        
    ############################################################################################
    ### load the parameters of each LC
    ### and normalize parameters
    ############################################################################################
    if( not(justDataID) ):
        if(verbose):
            print('Load the parameters')
        from lightcone_functions import read_1_param
        param_raw = np.array( [ read_1_param( param=None, ICs=3, delta=s, verbose=False ) for s in range( Nsimu ) ] )
        param_raw = param_raw[:,[0,2,3,4]]
        param = (param_raw - param_raw.min(axis=0)) / (param_raw.max(axis=0) - param_raw.min(axis=0) )
        
    ############################################################################################
    ### random separate TRAIN to TEST and validation
    ### here I define the indices of the 3 set
    ### the random seed is import here for reproductibility of the loading
    ############################################################################################
    if(verbose):
        print('Defining Train-Test-Validation sets')

    Ndata = param_raw.shape[0] ### total number of data
    Ntrain = np.int(Ndata * trainSize) ### controle the size of the training sample

    shuffle_index = np.arange( Ndata ) ### ordered indices
    np.random.shuffle( shuffle_index ) ### shuffled indices

    train_select = shuffle_index[ :Ntrain ] ### indices of the traning set (the first part of the shuffled indices)
    
    Ntest_val = Ndata-Ntrain
    test_select  = shuffle_index[ Ntrain: ] ### indices of the testing set
    
    if validation:
        test_select       = shuffle_index[ Ntrain:Ntrain+Ntest_val//2 ] ### indices of the testing set
        validation_select = shuffle_index[ Ntrain+Ntest_val//2: ] ### indices of the testing set
        
    if justDataID:
        if validation:
            return train_select, test_select, validation_select
        else:
            return train_select, test_select
        
        
    if( LHS ): ### Latin Hypercube sample
        if(verbose):
            print('Latin-Hypercube Sampling')
        ### here the points that are the closer to the LHS matrix
        ### it build an approximate LHS
        ### in pratique, the approximation will reduce it
        number_of_param = 4
        number_of_bin_per_param = Nbins_LHS
        
        #k = pyDOE.lhs( number_of_param, samples=number_of_bin_per_param**(number_of_param-1)+1 ) ### LHS matrix
        k = pyDOE.lhs( number_of_param, samples=number_of_bin_per_param ) ### LHS matrix

        ### find the closest point to the LHS matrix
        import scipy.spatial.kdtree as KD
        tree = KD.KDTree( param ) ### kd of the normalize LC param
        nn = tree.query( k ) ### 
        
        train_select = np.unique( nn[1] ) ### remove double point detection (it may occur!)
        
        NEW_number_of_bin_per_param = number_of_bin_per_param
        count_while = 0
        while train_select.shape[0]<number_of_bin_per_param :
            count_while += 1
            #print( 'GET IN WHILE :', count_while )
              
            #print( number_of_bin_per_param, train_select.shape[0], NEW_number_of_bin_per_param )
              
            NEW_number_of_bin_per_param += (number_of_bin_per_param - train_select.shape[0])

            k = pyDOE.lhs( number_of_param, samples=NEW_number_of_bin_per_param )
            nn = tree.query( k )
            train_select = np.unique( nn[1] )
            
        train_select = train_select[:number_of_bin_per_param]
        
        #### TRY TO SOLVE THE 2POINTS ISSUES
        #print('query shape', nn[1][:,0].shape) 
        
        #train_select = np.zeros( number_of_bin_per_param ).astype( np.int, copy=False )
        ### get the unique ID of nearest points AND their ID in nn[1][:,0]
        ### there is N_unique unique value <= N
        #unique_train_select, unique_ID = np.unique( nn[1][:,0], return_index=True )
        
        #print('unique ID shape', unique_ID.shape)
        ### save the N_unique first nearest points
        #train_select[ unique_ID ] = nn[1][:,0][unique_ID]
        
        ### search which of the N nearest points are not used
        ### look for IDs not in unique_ID
        #not_used_kID = np.in1d( np.arange( number_of_bin_per_param ), unique_ID, invert=True )
        #train_select[ not_used_kID ] = nn[1][:,1][not_used_kID]
        
        #print( 'unique ID shape :', unique_ID.shape )
        #print( '+ non used ID', not_used_kID.sum() )
        #print( 'should equal :', number_of_bin_per_param )
        
        #print( 'second unique ID shape', np.unique( nn[1][:,1][not_used_kID] ).shape )
        #print( 'should equal :', not_used_kID.sum() )
        ###
        
        np.random.shuffle( train_select ) ### randomize the order of the LC 
        Ntrain = train_select.shape[0]

        tempo = np.arange( Ndata )
        tempo[ train_select ]=-1
        test_select = np.where(tempo>=0)[0] ### test sample are the other point
        np.random.shuffle( test_select )
        
    Ntest = test_select.shape[0]
    
    if validation:
        Nval  = validation_select.shape[0]
        print( 'Number of LC:', Ndata, ', train set: ', Ntrain, ', test set: ', Ntest, ' Validation set: ', Nval )
    else:
        print( 'Number of LC:', Ndata, ', train set: ', Ntrain, ', test set: ', Ntest )
    
    ############################################################################################
    ### return the train-test sample of: 
    ### 1-LC 
    ### 2-norm param 
    ### 3-raw param 
    ############################################################################################
    ### multi slices per parameter
    ############################################################################################
    if multiple_slice:
        if verbose:
            print( 'multiple_slice' )
        shuffle_index_multiSlice = np.arange( Ntrain*Nslice ) ### ordered indices
        np.random.shuffle( shuffle_index_multiSlice )         ### shuffled indices
        
        ### HAVE TO FLATTEN OVER THE MULTI-SLICE, AND SHUFFLE
        ### too much copy here, to long, to memory...
        #tmp_param = np.array( list( param[train_select] ) * Nslice )[shuffle_index_multiSlice]   
        #tmp_param_raw = np.array( list( param_raw[train_select] ) * Nslice )[shuffle_index_multiSlice]  
        #tmp_LC = LightConeBox[train_select].reshape( (Ndata*Nslice,ImgHeight,BoxLong,1) )[shuffle_index_multiSlice]
        ###      
            
        if( not(justParam) ):
            ### just one slice per test and validation
            return (LightConeBox[train_select,:Nslice].reshape( (Ntrain*Nslice,ImgHeight,BoxLong,1) )[shuffle_index_multiSlice], \
                    LightConeBox[test_select,0], LightConeBox[validation_select,0]), \
                    (np.repeat( param[train_select], Nslice, axis=0 )[shuffle_index_multiSlice], \
                    param[test_select], param[validation_select]), \
                    (np.repeat( param_raw[train_select], Nslice, axis=0 )[shuffle_index_multiSlice], \
                    param_raw[test_select], param_raw[validation_select])
        else:
            return (np.array( list( param[train_select] ) * Nslice )[shuffle_index_multiSlice], \
                    param[test_select], param[validation_select]), \
                   (np.array( list( param_raw[train_select] ) * Nslice )[shuffle_index_multiSlice], \
                    param_raw[test_select], param_raw[validation_select])
            
    ############################################################################################
    ### 
    ############################################################################################
    if validation:
        if( not(justParam) ):
            return (LightConeBox[train_select], LightConeBox[test_select], LightConeBox[validation_select]), \
                   (param[train_select], param[test_select], param[validation_select]), \
                   (param_raw[train_select], param_raw[test_select], param_raw[validation_select])
        else:
            return (param[train_select], param[test_select], param[validation_select]), \
                   (param_raw[train_select], param_raw[test_select], param_raw[validation_select])
    else:   
        if( not(justParam) ):
            return (LightConeBox[train_select], LightConeBox[test_select]), \
                   (param[train_select], param[test_select]), \
                   (param_raw[train_select], param_raw[test_select])
        else:
            return (param[train_select], param[test_select]), \
                   (param_raw[train_select], param_raw[test_select])

            
            
###################################
###################################
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, 
                 #list_IDs, 
                 #labels,
                 Ndata,
                 batch_size=20, 
                 dim=(100,2200), 
                 n_channels=1,
                 n_params=4, 
                 shuffle=True, 
                 DATA_DIR='/amphora/nicolas.gillet/LC_data/LC_SLICE10_px100_2200_N10000_randICs/train', ):
        
        """Initialization"""
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.Ndata = Ndata
        #self.list_IDs = list_IDs
        self.list_IDs = np.arange( self.Ndata )
        self.n_channels = n_channels
        self.n_params = n_params
        self.shuffle = shuffle
        self.DATA_DIR = DATA_DIR
        
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        #return int(np.floor(len(self.list_IDs) / self.batch_size))
        return int(np.floor(self.Ndata / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        #X = np.empty((self.n_channels, self.batch_size, *self.dim))
        y = np.empty((self.batch_size, self.n_params), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X_temp, y[i] = np.load(self.DATA_DIR + '%d.npy'%ID)
            #y[i] = np.copy( y_temp )
            X[i,] = X_temp.reshape( *self.dim, self.n_channels )
            #X[:,i,] = X_temp.reshape( self.n_channels, *self.dim )

            # Store class
            #y[i] = self.labels[ID]

        return X, y #keras.utils.to_categorical(y, num_classes=self.n_classes)
    