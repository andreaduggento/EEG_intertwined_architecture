import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, Flatten, Dense, Dropout, LSTM, Input, TimeDistributed, Bidirectional, GlobalMaxPooling1D, AveragePooling1D, MaxPooling1D, Permute, Reshape, Conv1D
from keras import initializers, Model, optimizers, callbacks
from keras.models import load_model

## INTERTWINED MODEL
modelFamily = "intertwined_v4"

def makeIntertwinedModel(input_shape,tdNnum,tdNact,sdCnum,sdCker,sdCact,poolSize,pooltype,LSnum,LSdrop,FNnum,FNact,FNdrop,NUMCATEGORIES,printModelSummary=0,initseed=42,printLayerShapes=0):
        
    """
    input_shape:
    


    """
    
    lecun = initializers.lecun_normal(seed=initseed)
    inputs = Input(shape=input_shape)
    x = inputs
    # Convolutional over SPACE layers block
    tdNnum = np.trim_zeros(tdNnum,'b')
    for idx, val in enumerate(tdNnum):
        ########  TIME-DISTRIBUTED (fully connected over space)
        x = TimeDistributed(Dense(val, kernel_initializer=lecun), name='tdNdense'+str(idx))(x)
        x = BatchNormalization(name='tdNbanorm'+str(idx))(x)
        x = Activation(tdNact, name='tdNact'+str(idx))(x)
        x = Reshape((val,-1,1),name='reshtd'+str(idx))(Permute((2,1),name='permtd'+str(idx))(x))
        ########  SPACE-DISTRIBUTED (convolution over time and pooling)
        x = TimeDistributed(Conv1D(sdCnum[idx], sdCker[idx]
                        #, padding='same'
                        , data_format='channels_last', kernel_initializer=lecun),name='sdC'+str(idx))(x)
        x = BatchNormalization(name='sdCbanorm'+str(idx))(x)
        x = Activation(sdCact, name='sdCact'+str(idx))(x)
        if (pooltype=='max'):
            x = TimeDistributed(MaxPooling1D(pool_size=poolSize[idx], padding="valid", data_format="channels_last"),name='pool'+str(idx))(x)
        if (pooltype=='ave'):
            x = TimeDistributed(AveragePooling1D(pool_size=poolSize[idx], padding="valid", data_format="channels_last"),name='pool'+str(idx))(x)
        x = Permute((2,1,3),name='permst'+str(idx))(x)
        x = Reshape((x.shape[1],x.shape[2]*x.shape[3]),name='reshsd'+str(idx))(x)  ###   BACK TO INITIAL STRUCTURE
        if printLayerShapes: print(x.shape)

    if LSnum[0]>0:
        LSnum = np.trim_zeros(LSnum,'b')
        for idx, val in enumerate(LSnum[0:-1]):
                x = LSTM(val, kernel_initializer=lecun, return_sequences=True, name='LSTM'+str(idx))(x)
                x = Dropout(LSdrop[idx], name='LSdrop'+str(idx))(x)
        x = LSTM(LSnum[-1], kernel_initializer=lecun, return_sequences=False,name='LSTM'+str(len(LSnum)-1))(x)
        x = Dropout(LSdrop[-1], name='LSdrop'+str(len(LSnum)-1))(x)
    else:
        print("ciao")
        print(x.shape)
        x = Permute((2,1),name='lastperm')(x)
        print(x.shape)
        x = Reshape((x.shape[1],x.shape[2],1),name='LastResh')(x)  ###   READY FOR GloMaxPooling
        print("ciao2")
        print(x.shape)
        x = TimeDistributed(GlobalMaxPooling1D(),name='LastGloMaxpool')(x)
        print(x.shape)
    
    x = Flatten(name='Flatten')(x)    
    if FNnum[0]>0:
            FNnum = np.trim_zeros(FNnum,'b')
            for idx, val in enumerate(FNnum):
                x = Dense(val, kernel_initializer=lecun, activation=FNact, name='FC'+str(idx))(x)
                x = Dropout(FNdrop[idx], name='FNdrop'+str(idx))(x)
    # Output layer
    outputs = Dense(NUMCATEGORIES, activation='softmax')(x)
    # Model compile
    model = Model(inputs=inputs, outputs=outputs)
    if printModelSummary: 
        print(model.summary())
    return model

def nameIntertwinedModel(TASK,SUB,nomeschema
                         ,lr,patience_e,patience_p,factor,batch_size
                         ,tdNnum,tdNact,sdCnum,sdCker,sdCact,poolSize,pooltype,LSnum,LSdrop,FNnum,FNact,FNdrop
                         ,NUMCATEGORIES,printModelSummary=0,printModelName=0
                         ,optimizer='sgd'):
    modelname=TASK+'Sub_'+SUB+'_'+modelFamily+'_'+str(nomeschema)+'__tdN'+tdNact+'_'
    for idx, val in enumerate(tdNnum):
        modelname=modelname+str(val)+'_'
    modelname=modelname+'_sdC'+sdCact+'_'
    for idx, val in enumerate(sdCnum):
        modelname=modelname+str(val)+'k'+str(sdCker[idx])+'p'+str(poolSize[idx])+pooltype+'_'
    ####  LSTM
    if LSnum[0]>0:
        LSnum = np.trim_zeros(LSnum,'b')
        modelname=modelname+'_LS_'
        for idx,val in enumerate(LSnum):
            modelname=modelname+str(val)+'_d'+str(LSdrop[idx])+'_'
    else:
        modelname=modelname+'_GlobMaxpool_'
    #####
    if FNnum[0]>0:
        FNnum = np.trim_zeros(FNnum,'b')
        modelname=modelname+'_FC'+FNact+'_'
        for idx,val in enumerate(FNnum):
            modelname=modelname+str(val)+'_d'+str(FNdrop[idx])+'_'
    modelname=modelname+'='+str(NUMCATEGORIES)+'class__'+optimizer+'_lr_'+str(lr)+'_patience_'+str(patience_e)+'_'+str(patience_p)
    modelname=modelname+'_factor_'+str(factor)+'_batch_'+str(batch_size)
    if printModelName: print(modelname)
    return modelname
    
    