import pandas as pd
import numpy as np
import os 

import tensorflow as tf

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, concatenate, Dropout, Activation
from keras import optimizers,applications, callbacks
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l1, l2
from keras.optimizers import SGD, RMSprop

from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_score, recall_score, f1_score
from sklearn import metrics
from sklearn.multioutput import MultiOutputRegressor
from math import sqrt

# c_index
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

## set parameter
class WxHyperParameter(object):
    def __init__(self, epochs=100, batch_size=16, learning_ratio=0.001, weight_decay=1e-6, momentum=0.9):

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_ratio = learning_ratio
        self.weight_decay = weight_decay
        self.momentum = momentum

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

## make NN model preprocessor
def negative_log_likelihood_binary(E, NUM_E):
    def loss(y_true,y_pred) : 
        hazard_ratio = K.exp(y_pred)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = (y_pred) - log_risk
        censored_likelihood = uncensored_likelihood * E
        # num_observed_events = K.sum(E)
        # num_observed_events = K.cast(num_observed_events, dtype='float64')
        # neg_likelihood = -K.sum(censored_likelihood)/num_observed_events
        neg_likelihood = -K.sum(censored_likelihood)/NUM_E
        return neg_likelihood
    return loss


def ModelMlpReg(input_size, output_size, hyper_param, method):
    input_dim = input_size
    inputs = Input((input_dim,))
    hidden1 = Dense(256)(inputs)
    if method == 'binary':
        fc_out = Dense(int(output_size), kernel_initializer='zeros', bias_initializer='zeros', activation = 'sigmoid')(hidden1)
    if method == 'time':
        fc_out = Dense(int(output_size), kernel_initializer='zeros', bias_initializer='zeros', activation = 'tanh')(hidden1)

    model = Model(inputs=inputs, outputs=fc_out)
    return model   

def ModelSlpReg(input_size, output_size, hyper_param,method):
    input_dim = input_size
    inputs = Input((input_dim,))
    if method == 'binary':
        fc_out = Dense(int(output_size), kernel_initializer='zeros', bias_initializer='zeros', activation = 'sigmoid')(inputs)
    if method == 'time':
        fc_out = Dense(int(output_size), kernel_initializer='zeros', bias_initializer='zeros', activation = 'tanh')(inputs)
    model = Model(inputs=inputs, outputs=fc_out)
    return model


## make NN model
# model = ['SLP','MLP']
from keras import losses
from sklearn import preprocessing
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical

def TrainNNnaive(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst, c_tst, s_tst, auc_trn, auc_dev, auc_tst, out_method, NN_method, followup_years):
    hyper = WxHyperParameter()
    hyper.learning_ratio = 1e-2
    hyper.batch_size = 10
    hyper.epochs = 100
    hyper.weight_decay = 1e-6
    hyper.momentum = 0.9
    output_size = y_trn.shape[1]

    if NN_method == 'SLP' :
        model = ModelSlpReg(len(x_trn[0]), output_size, hyper, out_method)
    elif NN_method == 'MLP':
        model = ModelMlpReg(len(x_trn[0]), output_size, hyper, out_method)

    #build a optimizer
    #sgd = optimizers.SGD(lr=hyper.learning_ratio, decay=hyper.weight_decay, momentum=hyper.momentum, nesterov=True)
    #adam = optimizers.Adam(lr=hyper.learning_ratio, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    rmsprop=RMSprop(lr=hyper.learning_ratio, rho=0.9, epsilon=1e-8)

    if out_method == 'binary' :
        model.compile(loss=losses.binary_crossentropy, optimizer=rmsprop ,metrics=['accuracy'])
    if out_method == 'time' :
        model.compile(loss=losses.mean_squared_error, optimizer=rmsprop ,metrics=["mse","acc"])     
   
    base_path = './save_model/'+str(NN_method)+'/'
    best_model_path= base_path+str(model)+'_reg_weights_best.hdf5'
    if os.path.exists(base_path) == False:
        os.makedirs(base_path)
    save_best_model = ModelCheckpoint(best_model_path, monitor='loss', verbose=0, save_best_only=True, mode='min')

    #learning schedule
    def step_decay(epoch):
        exp_num = int(epoch/30)+1       
        return float(hyper.learning_ratio/(10 ** exp_num))    
    change_lr = LearningRateScheduler(step_decay)

    history = model.fit(x_trn, y_trn, validation_data=[x_dev,y_dev],epochs=hyper.epochs, batch_size=hyper.batch_size, verbose=0, shuffle=True, callbacks=[change_lr,save_best_model])
    model.load_weights(best_model_path, by_name=True)
    Y_pred = model.predict(x_tst, batch_size=1, verbose=0)  

    if out_method == 'binary':
        divide_group = []
        for pred in Y_pred:
            if pred > 0.5 :
                divide_group.append(1)
            else :
                divide_group.append(0)
        fpr, tpr, thresholds = roc_curve(y_tst, Y_pred, pos_label = 1)
        auc_score = auc(fpr,tpr)

    elif out_method == 'time':
        Y_pred_survival_duration = []
        Y_pred_survival_proba = []
        for predict in Y_pred:
            duration = 0
            weight = 0
            for i in range(len(predict)):
                duration += (i+1) * predict[i]
                weight += predict[i]
            duration =duration / weight
            Y_pred_survival_duration.append(duration)
            Y_pred_survival_proba.append(sigmoid(duration-int(followup_years)))
        
        c_score = cindex(s_tst, c_tst, Y_pred_survival_duration)

        divide_group = []
        for pred in Y_pred_survival_proba:
            if pred > 0.5 :
                divide_group.append(1)
            else :
                divide_group.append(0)

        fpr, tpr, thresholds = roc_curve(auc_tst, np.array(Y_pred_survival_proba), pos_label = 1)
        auc_score = auc_score = auc(fpr,tpr)

    if out_method == 'binary':
        return auc_score, divide_group
    elif out_method == 'time' : 
        return c_score, auc_score, divide_group

"""
# model = only 'SLP'
from sklearn import preprocessing
def TrainNNLogLikeLoss(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst, auc_trn, auc_dev, auc_tst, c_trn, c_dev, c_tst, s_trn, s_dev, s_tst, out_method, followup_years):
    hyper = WxHyperParameter()
    hyper.learning_ratio = 1e-8
    hyper.epochs = 500
    hyper.batch_size = len(x_trn)
    output_size = y_trn.shape[1]

    #Sorting for negative log likelyhood loss
    sort_idx = np.argsort(s_trn)[::-1]
    x_trn=x_trn[sort_idx]
    y_trn=y_trn[sort_idx]
    c_trn=c_trn[sort_idx]
    c_trn_reverse= np.array([not elem for elem in c_trn]).astype(int)

    model = ModelSlpReg(len(x_trn[0]),output_size, hyper)

    #adam = optimizers.Adam(lr=hyper.learning_ratio, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #sgd = SGD(lr=hyper.learning_ratio, decay=0.01, momentum=0.9, nesterov=True)
    rmsprop=RMSprop(lr=hyper.learning_ratio, rho=0.9, epsilon=1e-8) 
    if out_method == 'binary':
        model.compile(loss=[negative_log_likelihood_binary(c_trn_reverse,np.sum(c_trn_reverse))], optimizer=rmsprop)    
    #elif out_method == 'time':
     #   model.compile(loss=[negative_log_likelihood_time(c_trn_reverse,np.sum(c_trn_reverse),followup_years)], optimizer=rmsprop)  


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

    history = model.fit(x_trn, y_trn, validation_data=[x_dev,y_dev] ,epochs=hyper.epochs, batch_size=hyper.batch_size, 
                        verbose=1, shuffle=False, callbacks=[save_best_model, change_lr])                        

    model.load_weights(best_model_path, by_name=True)

    sp_pred_val = model.predict(x_dev, batch_size=1, verbose=0)   
    print(sp_pred_val)


    sp_pred_test = model.predict(x_tst, batch_size=1, verbose=0)
    sp_pred_test=np.exp(sp_pred_test)


    return rmse, c_index
"""