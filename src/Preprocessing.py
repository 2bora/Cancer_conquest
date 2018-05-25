import pandas as pd
import numpy as np
import os
from keras.utils import to_categorical

# data processing
def preprcossing_data(filename, followup_years):
    base_df = pd.read_csv("../data/"+str(filename)+".csv")
    print "raw input data shape = ", str(base_df.shape)

    base_df["years_to_followup"] =(base_df["days_to_followup"]/(30*12))
    for i in range(len(base_df)):
        base_df.iloc[i,-1] = round(base_df.iloc[i,-1],3)

    base_df_row =[]
    for i, row in base_df.iterrows():
        if int(row["years_to_followup"]) <= followup_years and row["Censored_flag"] == 1:
            pass
        else : 
            base_df_row.append(row.tolist())

    base_df = pd.DataFrame(base_df_row, columns = base_df.columns.tolist())

    print " Followup years max, min = " , (base_df["years_to_followup"].max(), base_df["years_to_followup"].min())
    # year max = 18, time interval = 1 year
    #0th = 0 -1, ... , 17th = 17 - 18
    return base_df

## make censored df
def makeTimeintervalCensored(base_df):
    header = []
    max_interval = int(base_df["years_to_followup"].max()) + 1

    for i in range(0,max_interval):
        header.append("t_interval_"+str(i))
    
    censored_df_pre = []
    for i,row in base_df.iterrows():
        censored_row = []
        sy = int(row["years_to_followup"])
        d = 0
        if row['Event_flag'] == 1 :#dead, censored = 0
            for l in range(0,max_interval):
                censored_row.append(0)
        elif row['Event_flag'] == 0 :#Censored, censored = 1
            for j in range(0,sy):
                censored_row.append(0)
            for k in range(sy,max_interval):
                censored_row.append(1)

        censored_df_pre.append(censored_row)

    Censored_df = pd.DataFrame(censored_df_pre,columns = header) # whether censored or not

    print "Shape of Censored_by_t_df = ", Censored_df.shape
    return Censored_df

## make Outputdf (timeinterval, binary, multitask)
def makeTimeintervalOutput(base_df):
    header = []
    max_interval = int(base_df["years_to_followup"].max()) + 1

    for i in range(0,max_interval):
        header.append("t_interval_"+str(i))

    out_df_pre = []
    for i,row in base_df.iterrows():
        out_row =[]
        sy = int(row["years_to_followup"])
        d = 0
        if row['Event_flag'] == 1 :#dead, censored = 0
            for j in range(0,sy):
                out_row.append(0)
            for k in range(sy,max_interval):
                out_row.append(1)
        elif row['Event_flag'] == 0 :#Censored, censored = 1
            for j in range(0,sy):
                out_row.append(0)
            for k in range(sy,max_interval):
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
                d = 0

        out_df_pre.append(out_row)

    Output_df=pd.DataFrame(out_df_pre,columns = header) # hazard function
    Output_df=1-Output_df # survival probability

    print "Shape of Output_df = ", Output_df.shape
    return Output_df

def makeBinaryOutput(base_df, followup_years): # survival = 0, event(death) = 1
    out_df_row = []
    for i, row in base_df.iterrows():
        if int(row["years_to_followup"]) <= followup_years and row["Event_flag"] == 1:
            out_df_row.append(0)
        else :
            out_df_row.append(1)

    Output_df = pd.DataFrame(out_df_row, columns =[str(followup_years)+'yr survival bianry'])

    return Output_df

## make input df
def makeInput(base_df):
    output_related_f = ['Censored_flag','Event_flag','years_to_followup',"days_to_followup","PID"]
    Input_df = base_df.drop(output_related_f,axis =1)
    print "Shape of Iutput_df = ", Input_df.shape

    return Input_df

def makeAllData(filename, method, followup_years):
    # Prepare base dataframes
    base_df=preprcossing_data(filename, followup_years)
    auc_df = makeBinaryOutput(base_df, followup_years)
    test_binary_out = pd.concat([base_df,auc_df], axis =1 )
    test_binary_out.to_csv("../data/binary_output_"+str(followup_years)+".csv")
    auc_df = auc_df.values.astype(float)
 
    if method =='time':
        output_df = makeTimeintervalOutput(base_df)
        output_df = output_df.values.astype(float)
        censored_df = makeTimeintervalCensored(base_df)
        censored_df = censored_df.values.astype(float)

    elif method == 'binary':
        output_df = auc_df 
        censored_df = makeTimeintervalCensored(base_df)
        censored_df = censored_df.values.astype(float)      
        

    followup_time = np.array(base_df["years_to_followup"].tolist())
    censored_flag = np.array(base_df["Censored_flag"].tolist())    
    input_df = makeInput(base_df)
    feature_list = input_df.columns.tolist()
    input_df = input_df.values.astype(float)
    return auc_df, output_df, censored_df, followup_time, censored_flag, input_df, feature_list
