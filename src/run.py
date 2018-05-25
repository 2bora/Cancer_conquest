import ML as ml
import numpy as np
import pandas as pd
import Preprocessing as pp
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import StratifiedKFold
import pickle 
import os 

def RunNNModel(filename,out_method,followup_years,iter_num):
    # results_cox = []
    header = []
    for method in out_method:
        header.append(str(method)+'_SLP')
        header.append(str(method)+'_MLP')
        if method == 'time':
            header.append('timeSLP_Cindex')
            header.append('timeMLP_Cindex')
    
    row= []
    cph_data = []
    cph_score_group = []


    for y in followup_years :
        tot_score= []
        for method in out_method:
            auc_df, output_df, censored_df, followup_time, censored_flag, input_df, feature_list = pp.makeAllData(filename, method, y)
            scores = []
            c_index_scores = []

            for i in range(0,iter_num):
                x_trn, x_tst, y_trn, y_tst, auc_trn, auc_tst, c_trn, c_tst, s_trn, s_tst = train_test_split(input_df, output_df, auc_df, censored_flag, followup_time, test_size=0.2, random_state=None,stratify = auc_df)
                x_trn, x_dev, y_trn, y_dev, auc_trn, auc_dev, c_trn, c_dev, s_trn, s_dev = train_test_split(x_trn, y_trn, auc_trn, c_trn, s_trn, test_size=0.2, random_state=None,stratify = auc_trn)
                
                print "Running Fold", i+1,"/",iter_num
                
                if method == 'binary':
                    score_slp, divide_group_slp = ml.TrainNNnaive(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst,c_tst, s_tst, auc_trn, auc_dev, auc_tst, method, 'SLP', y)
                    score_mlp, divide_group_mlp = ml.TrainNNnaive(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst, c_tst, s_tst, auc_trn, auc_dev, auc_tst, method, 'MLP', y)
                    test_binary_out = (pd.Series([[len(y_trn),np.sum(y_trn),len(y_trn)-np.sum(y_trn)], [len(y_dev),np.sum(y_dev),len(y_dev)-np.sum(y_dev)], [len(y_tst),np.sum(y_tst),len(y_tst)-np.sum(y_tst)]], index = ['y_trn', 'y_dev', 'y_tst'])).to_frame()
                    test_binary_out.to_csv("../data/binary_output_"+str(i)+"_"+str(y)+".csv")
                    #score_ll = ml.TrainNNLogLikeLoss(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst, auc_trn, auc_dev, auc_tst, c_trn, c_dev, c_tst, s_trn, s_dev, s_trn, method, y)
                
                elif method == 'time' :
                    c_index_slp, score_slp, divide_group_slp = ml.TrainNNnaive(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst, c_tst, s_tst,auc_trn, auc_dev, auc_tst, method, 'SLP', y)
                    c_index_mlp, score_mlp, divide_group_mlp = ml.TrainNNnaive(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst, c_tst, s_tst,auc_trn, auc_dev, auc_tst, method, 'MLP', y)
                    #score_ll = ml.TrainNNLogLikeLoss(x_trn, x_dev, x_tst, y_trn, y_dev, y_tst, auc_trn, auc_dev, auc_tst, c_trn, c_dev, c_tst, s_trn, s_dev, s_trn, method, y)
                    c_index_scores.append([c_index_slp, c_index_mlp])
                
                cph_data.append([i, y, method, feature_list, x_trn, y_trn, s_trn, c_trn, x_tst, y_tst, s_tst, c_tst])
                cph_score_group.append([i,y,method,score_slp, divide_group_slp,score_mlp, divide_group_mlp])
                
                scores.append([score_slp,score_mlp])                 
            print(scores)
                #scores.append(score_ll)


            avg_score_auc = np.mean(scores,axis = 0)
            
            for s in avg_score_auc:
                tot_score.append(s)

            if method == 'time' :
                avg_score_cindex = np.mean(c_index_scores,axis = 0)
                for s in avg_score_cindex:
                    tot_score.append(s)

        row.append(tot_score)
    result = pd.DataFrame(row, index = followup_years, columns= header)
    print(result)

    pickle.dump(cph_data, open("./cph_input.pickle",'wb'))
    pickle.dump(cph_score_group, open("./cph_group.pickle",'wb'))

RunNNModel('clinic',['binary','time'],[3,5,7],5)

