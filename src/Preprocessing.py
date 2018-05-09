import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit

# data processing
def preprcossing_data(filename):
    base_df = pd.read_csv("./data/"+str(filename)+".csv")
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
                        out_row.append(round(float(0.0001),5))
                else :
                    out_row.append(round(float(0.0001),5))
                censored_row.append(1)
                d = 0

        out_df_pre.append(out_row)
        censored_df_pre.append(censored_row)

    Output_df=pd.DataFrame(out_df_pre,columns = header)
    Output_df=1-Output_df
    Censored_df = pd.DataFrame(censored_df_pre,columns = header)

    print "Shape of Output_df = ", Output_df.shape
    print "Shape of Censored_by_t_df = ", Censored_df.shape
    Output_df.to_csv("./Output_ex.csv")
    Censored_df.to_csv("./Censored_ex.csv")
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
    followup_time = np.array(base_df["years_to_followup"].tolist())
    censored_flag = np.array(base_df["Censored_flag"].tolist())
    output_df, censored_df = make_output_data(base_df)
    output_df = output_df.values.astype(float)
    input_df = make_input_data(base_df)
    feature_list = input_df.columns.tolist()
    #input_df = pd.concat([input_df,censored_df],axis=1)
    input_df = input_df.values.astype(float)
    censored_df = censored_df.values.astype(float)

    cv_trn = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=None)
    for i_trn, i_devtst in cv_trn.split(input_df, censored_flag):
        x_trn = input_df[i_trn]
        x_devtst = input_df[i_devtst]
        y_trn = output_df[i_trn]
        y_devtst = output_df[i_devtst]
        censored_df_trn = censored_df[i_trn]
        censored_df_devtest = censored_df[i_devtst]
        c_trn = censored_flag[i_trn]
        c_devtst = censored_flag[i_devtst]
        s_trn = followup_time[i_trn]
        s_devtst = followup_time[i_devtst]
    cv_dev = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=None)
    for i_dev, i_tst in cv_dev.split(x_devtst, c_devtst):
        x_dev = x_devtst[i_dev]
        x_tst = x_devtst[i_tst]
        y_dev = y_devtst[i_dev]
        y_tst = y_devtst[i_tst]
        s_dev = s_devtst[i_dev]
        s_tst = s_devtst[i_tst]
        c_dev = c_devtst[i_dev]
        c_tst = c_devtst[i_tst]
        censored_df_dev = censored_df_devtest[i_dev]
        censored_df_tst = censored_df_devtest[i_tst]
    print("data split done")
    return feature_list, x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst, censored_df_trn, censored_df_dev, censored_df_tst
