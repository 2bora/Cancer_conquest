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

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error
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

# data processing
def preprcossing_data(filename):
    base_df = pd.read_csv("./"+str(filename)+".csv")
    print "raw input data shape = ", str(base_df.shape)

    base_df["years_to_followup"] =(base_df["days_to_followup"]/(30*12))
    for i in range(len(base_df)):
        base_df.iloc[i,-1] = round(base_df.iloc[i,-1],3)

    print " Followup years max, min = " , (base_df["years_to_followup"].max(), base_df["years_to_followup"].min())
    # year max = 18, time interval = 1 year
    #0th = 0 -1, ... , 17th = 17 - 18

    return base_df

## make output&censored df
def make_output_data(base_df):
    header = []
    for i in range(0,18):
        header.append("t_interval_"+str(i))

    out_df_pre = []
    censored_df_pre = []
    for i,row in base_df.iterrows():
        out_row =[]
        censored_row = []
        sy = int(row["years_to_followup"])
        d = 0
        if row['Event_flag'] == 1 :#dead, censored = 0
            for j in range(0,sy):
                out_row.append(0)
            for k in range(sy,18):
                out_row.append(1)
            for l in range(0,18):
                censored_row.append(0)
        elif row['Event_flag'] == 0 :#Censored, censored = 1
            for j in range(0,sy):
                out_row.append(0)
                censored_row.append(0)
            for k in range(sy,18):
                n = len(base_df.loc[base_df.years_to_followup >= k])
                d_p = base_df.loc[base_df.Censored_flag == 1]
                d_p.reindex(range(0,len(d_p)))
                for l in range(len(d_p)):
                    if k<d_p.iloc[l, -1] and d_p.iloc[l,-1]<k+1:
                        d += 1
                if n != 0:
                    if d != 0 :
                        out_row.append(round(float(d)/n,4))
                    else :
                        out_row.append(1e-08)
                else :
                    out_row.append(1e-08)
                censored_row.append(1)
                d = 0

        out_df_pre.append(out_row)
        censored_df_pre.append(censored_row)

    Output_df=pd.DataFrame(out_df_pre,columns = header)
    Output_df=1-Output_df
    Censored_df = pd.DataFrame(censored_df_pre,columns = header)

    print "Shape of Output_df = ", Output_df.shape
    print "Shape of Censored_by_t_df = ", Censored_df.shape
    return Output_df, Censored_df

## make input df
def make_input_data(base_df):
    output_related_f = ['Censored_flag','Event_flag','years_to_followup',"days_to_followup","PID"]
    Input_df = base_df.drop(output_related_f,axis =1)
    print "Shape of Iutput_df = ", Input_df.shape

    return Input_df

## kfold prepare
def split_trn_dev_tst(filename):
    base_df=preprcossing_data(filename)
    censored_flag = np.array(base_df["Censored_flag"].tolist())
    followup_time = np.array(base_df["years_to_followup"].tolist())
    output_df, censored_df = make_output_data(base_df)
    output_df = (output_df.T).values.astype(float)
    censored_df = (censored_df.T).values.astype(float)
    input_df = make_input_data(base_df)
    input_df = input_df.values.astype(float)
    x_trn_l = []
    y_trn_l = []
    s_trn_l = []
    c_trn_l = []
    x_dev_l = []
    y_dev_l = []
    s_dev_l = []
    c_dev_l = []
    x_tst_l = []
    y_tst_l = []
    s_tst_l = []
    c_tst_l = []
    for i in range(0,18):
        cv_trn = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=7)
        for i_trn, i_devtst in cv_trn.split(input_df, censored_df[i]):
            x_trn = input_df[i_trn]
            x_devtst = input_df[i_devtst]
            y_trn = output_df[i][i_trn]
            y_devtst = output_df[i][i_devtst]
            c_trn = censored_df[i][i_trn]
            c_devtst = censored_df[i][i_devtst]
            s_trn = followup_time[i_trn]
            s_devtst = followup_time[i_devtst]
        cv_dev = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=7)
        for i_dev, i_tst in cv_dev.split(x_devtst, c_devtst):
            x_dev = x_devtst[i_dev]
            x_tst = x_devtst[i_tst]
            y_dev = y_devtst[i_dev]
            y_tst = y_devtst[i_tst]
            s_dev = s_devtst[i_dev]
            s_tst = s_devtst[i_tst]
            c_dev = c_devtst[i_dev]
            c_tst = c_devtst[i_tst]
        x_trn_l.append(x_trn)
        y_trn_l.append(y_trn)
        s_trn_l.append(s_trn)
        c_trn_l.append(c_trn)
        x_dev_l.append(x_dev)
        y_dev_l.append(y_dev)
        s_dev_l.append(s_dev)
        c_dev_l.append(c_dev)
        x_tst_l.append(x_tst)
        y_tst_l.append(y_tst)
        s_tst_l.append(s_tst)
        c_tst_l.append(c_tst)
    
    return x_trn_l, y_trn_l, s_trn_l, c_trn_l, x_dev_l, y_dev_l, s_dev_l, c_dev_l, x_tst_l, y_tst_l, s_tst_l, c_tst_l

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
    def loss(y_true,y_pred):
        hazard_ratio = K.exp(y_pred)
        log_risk = K.log(K.cumsum(hazard_ratio))
        uncensored_likelihood = K.transpose(y_pred) - log_risk
        censored_likelihood = uncensored_likelihood * E
        # num_observed_events = K.sum(E)
        # num_observed_events = K.cast(num_observed_events, dtype='float64')
        # neg_likelihood = -K.sum(censored_likelihood)/num_observed_events
        neg_likelihood = -K.sum(censored_likelihood)/NUM_E
        return neg_likelihood
    return loss

def ModelMlpReg(input_size, hyper_param):
    input_dim = input_size
    inputs = Input((input_dim,))
    hidden1 = Dense(256)(inputs)
    fc_out = Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(hidden1)
    model = Model(inputs=inputs, outputs=fc_out)
    return model   

def ModelSlpReg(input_size, hyper_param):
    input_dim = input_size
    inputs = Input((input_dim,))
    fc_out = Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(inputs)
    model = Model(inputs=inputs, outputs=fc_out)
    return model

## make NN model
# model = ['SLP','MLP']
from keras import losses
from sklearn import preprocessing
def TrainNNnaive(split_data, model):
    x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst = split_data

    hyper = WxHyperParameter()
    hyper.learning_ratio = 1e-2
    hyper.batch_size = 32
    hyper.epochs = 100
    hyper.weight_decay = 1e-6
    hyper.momentum = 0.9

    rmse = []
    c_index = []

    for i in range(0,18):
        if model == 'SLP' :
            model = ModelSlpReg(len(x_trn[i][0]),hyper)
        elif model == 'MLP':
            model = ModelMlpReg(len(x_trn[i][0]),hyper)
        #build a optimizer
        #sgd = optimizers.SGD(lr=hyper.learning_ratio, decay=hyper.weight_decay, momentum=hyper.momentum, nesterov=True)
        adam = optimizers.Adam(lr=hyper.learning_ratio, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #rmsprop=RMSprop(lr=hyper.learning_ratio, rho=0.9, epsilon=1e-8)
        model.compile(loss=losses.mean_squared_error, optimizer=adam ,metrics=['mse'])    

        base_path = './save_model/'
        best_model_path= base_path+str(model)+str(i)+'_reg_weights_best.hdf5'
        if os.path.exists(base_path) == False:
            os.makedirs(base_path)
        save_best_model = ModelCheckpoint(best_model_path, monitor="val_mean_squared_error", verbose=0, save_best_only=True, mode='auto')

        #learning schedule
        def step_decay(epoch):
            exp_num = int(epoch/30)+1       
            return float(hyper.learning_ratio/(10 ** exp_num))    
        change_lr = LearningRateScheduler(step_decay)

        history = model.fit(x_trn[i], y_trn[i] , validation_data=[x_dev[i], y_dev[i]],epochs=hyper.epochs, batch_size=hyper.batch_size, verbose=0, shuffle=True, callbacks=[change_lr,save_best_model])
    
        model.load_weights(best_model_path, by_name=True)
        Y_pred = model.predict(x_tst[i], batch_size=1, verbose=0)

        rmse_p = sqrt(mean_squared_error(y_tst[i], Y_pred))
        c_index_p = cindex(s_tst[i],c_tst[i],Y_pred)
        rmse.append(rmse_p)
        c_index.append(c_index_p)

    #print(rmse, c_index)
    return rmse, c_index

# model = only 'SLP'
# ????????????????????
from sklearn import preprocessing
def TrainNNLogLikeLoss(split_data):
    x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst = split_data
    
    hyper = WxHyperParameter()
    hyper.learning_ratio = 1e-5
    hyper.epochs = 500
   
    rmse = []
    c_index = []

    for i in range(0,18):
        #Sorting for negative log likelyhood loss
        sort_idx = np.argsort(y_trn[i])[::-1]
        
        x_trn_i=x_trn[i][sort_idx]
        y_trn_i=y_trn[i][sort_idx]
        c_trn_i=c_trn[i][sort_idx]

        hyper.batch_size = len(x_trn_i)
        model = ModelSlpReg(len(x_trn_i[0]),hyper)

        #adam = optimizers.Adam(lr=hyper.learning_ratio, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #sgd = SGD(lr=hyper.learning_ratio, decay=0.01, momentum=0.9, nesterov=True)
        rmsprop=RMSprop(lr=hyper.learning_ratio, rho=0.9, epsilon=1e-8)    
        c_trn_reverse = np.asarray([not elem for elem in c_trn_i]).astype(int)
        model.compile(loss=[negative_log_likelihood(c_trn_reverse,np.sum(c_trn_reverse))], optimizer=rmsprop)    
        #model.summary()

        base_path = './save_model/'
        best_model_path= base_path+str(i)+'_reg_weights_best_ll.hdf5'
        if os.path.exists(base_path) == False:
            os.makedirs(base_path)
        save_best_model = ModelCheckpoint(best_model_path, monitor="loss", verbose=0, save_best_only=True, mode='auto')

        #learning schedule
        def step_decay(epoch):
            exp_num = int(epoch/100)+1       
            return float(hyper.learning_ratio/(10 ** exp_num))    

        change_lr = LearningRateScheduler(step_decay)

        history = model.fit(x_trn_i, y_trn_i, epochs=hyper.epochs, batch_size=hyper.batch_size, 
                        verbose=1, shuffle=False, callbacks=[save_best_model, change_lr])                        

        model.load_weights(best_model_path, by_name=True)

        sp_pred_val = model.predict(x_dev[i], batch_size=1, verbose=0)   
        sp_pred_val=np.exp(sp_pred_val) 
        rmse_val = sqrt(mean_squared_error(y_dev[i], sp_pred_val)) 

        sp_pred_test = model.predict(x_tst[i], batch_size=1, verbose=0)
        sp_pred_test=np.exp(sp_pred_test)
        rmse_test = sqrt(mean_squared_error(y_tst[i], sp_pred_test))

        c_index_val = cindex(s_dev[i], c_dev[i], sp_pred_val)
        c_index_test = cindex(s_tst[i], c_tst[i], sp_pred_test)

        rmse.append(rmse_test)
        c_index.append(c_index_test)

    #print(rmse, c_index)
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
def TrainCPH(split_data, method):
    x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst = split_data

    cph_head = ['S','E']
    
    clinical_features = ['Sex','Age','BMI','Comorbidity','PCEA','GT','pT','pN','Stage','CRM','Harvested_LNs', 'Lnmeta_num', 'Differentiation', 'Neural_Invasion', 'Vascular_Invasion','Lymphatic_Invasion', 'K-ras', 'Adjuvant_Tx', 'recurrence']
    for f in clinical_features:
        cph_head.append(f)

    train = []
    test = []

    # make_training_df
    for k in range(0,18):
        cph_data = []
        for i in range(0,len(x_trn[k])):
            row = []
            row.append(s_trn[k][i])
            row.append(c_trn[k][i])
            for j in range(0, len(clinical_features)):
                row.append(x_trn[k][i][j])
            cph_data.append(row)
    
        cph_df = pd.DataFrame(cph_data,columns=cph_head)
        cph = CoxPHFitter()
        cph.fit(cph_df, duration_col = 'S', event_col = 'E', step_size = 0.001)

        #make_test_df
        cph_data_test = []
        for i in range(0,len(x_tst[k])):
            row = []
            row.append(s_tst[k][i])
            row.append(c_tst[k][i])
            for j in range(0,len(clinical_features)):
                row.append(x_tst[k][i][j])
            cph_data_test.append(row)
    
        cph_df_test = pd.DataFrame(cph_data_test,columns=cph_head)

        x = make_cph_event_at_df(cph.predict_survival_function(cph_df),k)
        x_ = make_cph_event_at_df(cph.predict_survival_function(cph_df_test),k)


 
        ci_trn = concordance_index(cph_df.S.values,
                                   -x,
                                   cph_df.E.values)
        #ci_trn = cindex(cph_df.S.values, cph_df.E.values, x)
        print(k,ci_trn)
        train.append(ci_trn)

        ci_tst = concordance_index(cph_df_test.S.values,
                                   -x_,
                                 cph_df_test.E.values)
        #ci_tst = cindex(cph_df_test.S.values, cph_df_test.E.values, x_)                   
        print(k,ci_tst)
        test.append(ci_tst)

    #print(train,test)
    #print('Train ',train)
    #print('Test ',test)

    return test

def RunModel(filename,iter_num):
    iter_num = int(iter_num)
    results_mlp = []
    results_cox = []
    results_ll = []
    for i in range(0, iter_num) :
        split_data = split_trn_dev_tst(filename)
        rmse, cindex = TrainNNnaive(split_data,'MLP')
        #c_tst = TrainCPH(split_data, "CPH")
        #rmse_l, c_index_l = TrainNNLogLikeLoss(split_data)
        results_mlp.append([rmse, cindex])
        #results_cox.append(c_tst)
        #results_ll.append([rmse_l, c_index_l ])
        print str(i),"th process done"
    avg_mlp = np.mean(results_mlp,axis = 0)
    #avg_ll = np.mean(results_ll,axis = 0)
    #avg_cox = np.mean(results_cox)
    print(' MLP Result mean RMSE, Cindex ', avg_mlp)
    print(' LL Result mean RMSE, Cindex ', avg_ll)
    print(' COX Result mean Cindex ', avg_cox)

RunModel('clinic',2)


