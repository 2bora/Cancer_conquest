import pandas as pd
import numpy as np
import os 

import tensorflow as tf

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import LSTM, Input, Dense, concatenate, Dropout, Activation
from keras import optimizers,applications, callbacks
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l1, l2
from keras.optimizers import SGD, RMSprop


from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from math import sqrt

from lifelines.utils import concordance_index


# c_index definition
def cindex(surv, cen, y_pred):
    N = surv.shape[0]
    Comparable = np.zeros([N,N])

    for i in range(N):
        for j in range(N):
            if cen[i] == 0 and cen[j] == 0:
                if surv[i] != surv[j]:
                    Comparable[i, j] = 1

            elif cen[i] == 1 and cen[j] == 1:
                Comparable[i, j] = 0

            else: # one sample is censored and the other is not
                if cen[i] == 1:
                    if surv[i] >= surv[j]:
                        Comparable[i, j] = 1
                    else:
                        Comparable[i, j] = 0
                else: # cen[j] == 1
                    if surv[j] >= surv[i]:
                        Comparable[i, j] = 1
                    else:
                        Comparable[i, j] = 0

    p2, p1 = np.where(Comparable==1)

    Y = y_pred

    c=0
    N_valid_sample = p1.shape[0]
    for i in range(N_valid_sample):
        if cen[p1[i]] == 0 and cen[p2[i]] == 0:
            if Y[p1[i]] == Y[p2[i]]:
                c = c + 0.5
            elif Y[p1[i]] > Y[p2[i]] and surv[p1[i]] > surv[p2[i]]:
                c = c + 1
            elif Y[p2[i]] > Y[p1[i]] and surv[p2[i]] > surv[p1[i]]:
                c = c + 1

        elif cen[p1[i]] == 1 and cen[p2[i]] == 1:
            continue # do nothing - samples cannot be ordered

        else: # one sample is censored and the other is not
            if cen[p1[i]] == 1:
                if Y[p1[i]] > Y[p2[i]] and surv[p1[i]] > surv[p2[i]]:
                    c = c + 1
                elif Y[p1[i]] == Y[p2[i]]:
                    c = c + 0.5

            else: # cen[p2[i]] == 1
                if Y[p2[i]] > Y[p1[i]] and surv[p2[i]] > surv[p1[i]]:
                    c = c + 1
                elif Y[p1[i]] == Y[p2[i]]:
                    c = c + 0.5

    c = c*1.0 / N_valid_sample
    return c

def makeRNNinput_output(x,c,y):
    time_interval = np.array(range(0,18))
    feature = []
    for i in range(len(x)):
        feature_input = []
        feature_each_sample = []
        for f in x[i]:
            feature_each_sample.append(f)
            
        for j in range(len(c[0])):
            feature_each_sample.append(c[i][j])
            feature_input.append(feature_each_sample)
            feature_each_sample = feature_each_sample[:-1]
        feature.append(feature_input)

    output = []
    for i in range(len(y)):
        o = []
        for j in range(len(y[0])):
            o.append(y[i][j])
        output.append(o)

    return np.array(time_interval), np.array(feature), np.array(output)

def makeNNinput(x,censored_df):
    x_trn = []
    for i in range(len(x)):
        x_t = []
        for j in (x[i]):
            x_t.append(j)
        for k in censored_df[i]:
            x_t.append(k)
        x_trn.append(x_t)
    
    x_trn = np.array(x_trn)

    return x_trn

def makeCindexinput(data):
    cindex_input = []
    for i in range(len(data[0])):
        t = []
        for j in range(len(data)):
            t.append(data[j][i])
        cindex_input.append(t)
    
    return np.array(cindex_input)  

## set parameter
class WxHyperParameter(object):
    def __init__(self, epochs=100, batch_size=16, learning_ratio=0.001, weight_decay=1e-6, momentum=0.9):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_ratio = learning_ratio
        self.weight_decay = weight_decay
        self.momentum = momentum


## make NN model preprocessor
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def negative_log_likelihood(E, NUM_E):
    print(E.shape,NUM_E.shape)
    print(E[0],len(E[0]))

    def loss(y_true,y_pred):
        hazard_ratio = y_pred
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = (y_pred) - log_risk
        print(uncensored_likelihood[0])
        loss_log = []
        for i in range(len(E)):
            if NUM_E[i] != 0:
                censored_likelihood = (K.transpose(uncensored_likelihood))[i] * E[i] 
                neg_likelihood = -K.sum(censored_likelihood)/NUM_E[i]
        # num_observed_events = K.sum(E)
        # num_observed_events = K.cast(num_observed_events, dtype='float64')
        # neg_likelihood = -K.sum(censored_likelihood)/num_observed_events
                loss_log.append(neg_likelihood)
        neg_likelihood=K.sum(loss_log)/len(E)
        return neg_likelihood
    return loss


def ModelMlpReg(input_size, hyper_param):
    input_dim = input_size
    inputs = Input((input_dim,))
    hidden1 = Dense(256)(inputs)
    fc_out = Dense(18, kernel_initializer='zeros', bias_initializer='zeros', activation = 'sigmoid')(hidden1)
    model = Model(inputs=inputs, outputs=fc_out)
    return model   

def ModelSlpReg(input_size, hyper_param):
    input_dim = input_size
    inputs = Input((input_dim,))
    fc_out = Dense(18, kernel_initializer='zeros', bias_initializer='zeros',activation = 'sigmoid')(inputs)
    model = Model(inputs=inputs, outputs=fc_out)
    return model




## make NN model
# model = ['SLP','MLP']
from keras import losses
from sklearn import preprocessing

def TrainRNN(split_data):
    hyper = WxHyperParameter()
    hyper.learning_ratio = 1e-2
    hyper.batch_size = 32
    hyper.epochs = 100
    hyper.weight_decay = 1e-6
    hyper.momentum = 0.9

    feature_list, x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst, censored_df_trn, censored_df_dev, censored_df_tst = split_data

    time_interval, x_trn, y_trn = makeRNNinput_output(x_trn, censored_df_trn, y_trn)
    time_interval, x_dev, y_dev = makeRNNinput_output(x_dev, censored_df_dev, y_dev)
    time_interval, x_tst, y_tst = makeRNNinput_output(x_tst, censored_df_tst, y_tst)
    
    rmsprop=RMSprop(lr=hyper.learning_ratio, rho=0.9, epsilon=1e-8)

    model = Sequential()
    model.add(LSTM(len(x_trn), input_shape = (len(time_interval),len(x_trn[0][0]))))
    model.add(Dense(len(time_interval), activation = 'sigmoid'))
    model.compile(optimizer = rmsprop, loss= losses.mean_squared_error,
              metrics=['mse'])

    base_path = './save_model/RNN/'
    best_model_path= base_path+str(model)+'rnn_weights_best.hdf5'
    if os.path.exists(base_path) == False:
        os.makedirs(base_path)
    save_best_model = ModelCheckpoint(best_model_path, monitor="val_mean_squared_error", verbose=0, save_best_only=True, mode='auto')

    #learning schedule
    def step_decay(epoch):
        exp_num = int(epoch/30)+1       
        return float(hyper.learning_ratio/(10 ** exp_num))    
    change_lr = LearningRateScheduler(step_decay)


    history = model.fit(x_trn, y_trn,validation_data=[x_dev, y_dev],epochs=hyper.epochs, batch_size=hyper.batch_size, verbose=0, shuffle=True, callbacks=[save_best_model, change_lr])
    model.load_weights(best_model_path, by_name=True)

    #x_trn, y_trn = get_train()

    y_pred = model.predict(x_tst, verbose = 0)

    y_pred_cindex_input = makeCindexinput(y_pred)
    y_tst_cindex_input = makeCindexinput(y_tst)

    c_index = []
    rmse =[]
    for i in range(0,18):
        c_index_p = cindex(s_tst,c_tst,y_pred_cindex_input[i])
        rmse_p = sqrt(mean_squared_error(y_tst_cindex_input[i], (y_pred_cindex_input[i])))
        c_index.append(c_index_p)
        rmse.append(rmse_p)

    return rmse, c_index

def TrainNNnaive(split_data, method):
    feature_list, x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst, censored_df_trn, censored_df_dev, censored_df_tst = split_data
    x_trn = makeNNinput(x_trn,censored_df_trn)
    x_dev = makeNNinput(x_dev,censored_df_dev)
    x_tst = makeNNinput(x_tst,censored_df_tst)
    
    hyper = WxHyperParameter()
    hyper.learning_ratio = 1e-2
    hyper.batch_size = 32
    hyper.epochs = 100
    hyper.weight_decay = 1e-6
    hyper.momentum = 0.9

    if method == 'SLP' :
        model = ModelSlpReg(len(x_trn[0]),hyper)
    elif method == 'MLP':
        model = ModelMlpReg(len(x_trn[0]),hyper)
    
    #build a optimizer
    sgd = optimizers.SGD(lr=hyper.learning_ratio, decay=hyper.weight_decay, momentum=hyper.momentum, nesterov=True)
    #adam = optimizers.Adam(lr=hyper.learning_ratio, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #rmsprop=RMSprop(lr=hyper.learning_ratio, rho=0.9, epsilon=1e-8)
    model.compile(loss=losses.mean_squared_error, optimizer=sgd ,metrics=["mse","acc"])    
    
    base_path = './save_model/'+str(method)+'/'
    best_model_path= base_path+str(model)+'_reg_weights_best.hdf5'
    if os.path.exists(base_path) == False:
        os.makedirs(base_path)
    save_best_model = ModelCheckpoint(best_model_path, monitor="val_mean_squared_error", verbose=0, save_best_only=True, mode='auto')

    #learning schedule
    def step_decay(epoch):
        exp_num = int(epoch/30)+1       
        return float(hyper.learning_ratio/(10 ** exp_num))    
    change_lr = LearningRateScheduler(step_decay)

    history = model.fit(x_trn, y_trn,validation_data=[x_dev, y_dev],epochs=hyper.epochs, batch_size=hyper.batch_size, verbose=0, shuffle=True, callbacks=[change_lr,save_best_model])
    
    model.load_weights(best_model_path, by_name=True)
    Y_pred = model.predict(x_tst, batch_size=1, verbose=0)
    
    y_pred_cindex_input = makeCindexinput(Y_pred)
    y_tst_cindex_input = makeCindexinput(y_tst)

    c_index = []
    rmse =[]
 
    for i in range(0,18):
        c_index_p = cindex(s_tst,c_tst,y_pred_cindex_input[i])
        rmse_p = sqrt(mean_squared_error(y_tst_cindex_input[i], (y_pred_cindex_input[i])))
        c_index.append(c_index_p)
        rmse.append(rmse_p)

    return rmse, c_index


# model = only 'SLP'
from sklearn import preprocessing
def TrainNNLogLikeLoss(split_data):
    feature_list, x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst, censored_df_trn, censored_df_dev, censored_df_tst = split_data
    
    x_trn = makeNNinput(x_trn,censored_df_trn)
    x_dev = makeNNinput(x_dev,censored_df_dev)
    x_tst = makeNNinput(x_tst,censored_df_tst)

    hyper = WxHyperParameter()
    hyper.learning_ratio = 1e-8
    hyper.epochs = 500
    hyper.batch_size = len(x_trn)
    rmse = []
    c_index = []

    #Sorting for negative log likelyhood loss
    sort_idx = np.argsort(s_trn)[::-1]
    x_trn=x_trn[sort_idx]
    y_trn=(1.000-y_trn)[sort_idx]
    #c_trn=c_trn[sort_idx]
    #print(c_trn)
    #c_trn_reverse= np.array([not elem for elem in c_trn]).astype(int)
    censored_df_trn=censored_df_trn[sort_idx]
    censored_df_trn = makeCindexinput(censored_df_trn)
    c_trn_reverse = []
    c_trn_reverse_sum = []
    for i in range(len(censored_df_trn)):
        c_trn_reverse.append(np.asarray([not elem for elem in censored_df_trn[i]]).astype(int))
        c_trn_reverse_sum.append([np.sum(c_trn_reverse[i])])

    model = ModelSlpReg(len(x_trn[0]),hyper)
    #adam = optimizers.Adam(lr=hyper.learning_ratio, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #sgd = SGD(lr=hyper.learning_ratio, decay=0.01, momentum=0.9, nesterov=True)
    rmsprop=RMSprop(lr=hyper.learning_ratio, rho=0.9, epsilon=1e-8) 
    model.compile(loss=[negative_log_likelihood(np.array(c_trn_reverse),np.array(c_trn_reverse_sum))], optimizer=rmsprop)    
    model.summary()

    base_path = './save_model/LL/'
    best_model_path= base_path+str(model)+'_reg_weights_best_ll.hdf5'
    if os.path.exists(base_path) == False:
        os.makedirs(base_path)
    save_best_model = ModelCheckpoint(best_model_path, monitor="loss", verbose=1, save_best_only=True, mode='auto')

    #learning schedule
    def step_decay(epoch):
        exp_num = int(epoch/100)+1       
        return float(hyper.learning_ratio/(10 ** exp_num))    
    change_lr = LearningRateScheduler(step_decay)

    history = model.fit(x_trn, y_trn, epochs=hyper.epochs, batch_size=hyper.batch_size, 
                        verbose=1, shuffle=False, callbacks=[save_best_model, change_lr])                        

    model.load_weights(best_model_path, by_name=True)

    sp_pred_val = model.predict(x_dev, batch_size=1, verbose=0)   
    print(sp_pred_val)
    sp_pred_val=np.exp(sp_pred_val) 
    rmse_val = sqrt(mean_squared_error((1-y_dev), sp_pred_val)) 

    sp_pred_test = model.predict(x_tst, batch_size=1, verbose=0)
    sp_pred_test=np.exp(sp_pred_test)
    rmse_test = sqrt(mean_squared_error((1-y_tst), sp_pred_test))

    c_index_val = cindex(s_dev, c_dev, 1-sp_pred_val)
    c_index_test = cindex(s_tst, c_tst, 1-sp_pred_test)

    rmse.append(rmse_test)
    c_index.append(c_index_test)

    return rmse, c_index

# make CPH
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

def make_cph_event_at_df(survival_function_df,i):
    row = []
    now = survival_function_df.loc[survival_function_df.index.values >= i]
    now = now.loc[now.index.values < i+1]
    sf = now.mean()
    for j in range(len(sf)):
        row.append(sf[j])
        
    sf_ary = np.array(row)

    return sf_ary

def TrainCPH(split_data):
    feature_list, x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst, censored_df_trn, censored_df_dev, censored_df_tst = split_data

    cph_head = ['S','E']
    for f in feature_list:
        cph_head.append(f)
    #for i in range(0,18):
     #   cph_head.append('time_interval_'+str(i))


    # make_training_df
    cph_data = []
    for i in range(len(x_trn)):
        row = []
        row.append(s_trn[i])
        row.append(c_trn[i])
        for j in range(0, len(feature_list)):
            row.append(x_trn[i][j])
        #for k in (censored_df_trn[i]):
         #   row.append(k)

        cph_data.append(row)
    
    cph_df = pd.DataFrame(cph_data,columns=cph_head)
    cph = CoxPHFitter()
    cph.fit(cph_df, duration_col = 'S', event_col = 'E', step_size = 0.001)

    
    #make_test_df
    cph_data_test = []
    for i in range(len(x_tst)):
        row = []
        row.append(s_tst[i])
        row.append(c_tst[i])
        for j in range(0, len(feature_list)):
            row.append(x_tst[i][j])
        #for k in (censored_df_tst[i]):
         #   row.append(k)

        cph_data_test.append(row)
    
    cph_df_test = pd.DataFrame(cph_data_test,columns=cph_head)

    train_c = []
    test_c = []
    for i in range(0,18):
        x = make_cph_event_at_df(cph.predict_survival_function(cph_df),i)
        x_ = make_cph_event_at_df(cph.predict_survival_function(cph_df_test),i)


        ci_trn = concordance_index(cph_df.S.values,x,cph_df.E.values)
        #ci_trn = cindex(cph_df.S.values, cph_df.E.values, x)

        train_c.append(ci_trn)

        ci_tst = concordance_index(cph_df_test.S.values, x_, cph_df_test.E.values)
        #ci_tst = cindex(cph_df_test.S.values, cph_df_test.E.values, x_)                   

        test_c.append(ci_tst)


    return test_c
