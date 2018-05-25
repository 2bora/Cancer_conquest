import numpy as np
from lifelines import CoxPHFitter, KaplanMeierFitter
import pandas as pd
import pickle

import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 100})

def loadKMinput():
    with open('./cph_input.pickle','rb') as f:
        inputs = pickle.load(f)
    with open('./cph_group.pickle','rb') as f:
        groups = pickle.load(f)

    return inputs, groups

# make CPH
def make_cph_event_at_df(survival_function_df,i):
    row = []
    now = survival_function_df.loc[survival_function_df.index.values >= i]
    now = now.loc[now.index.values < i+1]

    sf = now.mean()
    for j in range(len(sf)):
        row.append(sf[j])
        
    sf_ary = np.array(row)
    return sf_ary

def makeKMplot(cph_input, cph_group,NN_method):
    k= 0
    for inputs in cph_input:
        print(k)
        iter_num, followup, method, feature_list, x_trn, y_trn, s_trn, c_trn, x_tst, y_tst, s_tst, c_tst = inputs
        cph_head = ['S','E']
        for f in feature_list:
            cph_head.append(f)
        cph_head.append('group')
        iter_num, followup ,method, score_slp, divide_group_slp,score_mlp, divide_group_mlp = cph_group[k]

        for method in NN_method:
            #make_test_df
            cph_data_test = []
            for i in range(len(x_tst)):
                row = []
                row.append(s_tst[i])
                row.append(c_tst[i])
                for j in range(0, len(feature_list)):
                    row.append(x_tst[i][j])
                if method == 'SLP' :
                    if divide_group_slp[i] == 0:      
                        row.append('d')
                    elif divide_group_slp[i] == 1 :
                        row.append('s')
                elif method == 'MLP':
                    if divide_group_mlp[i] == 0:      
                        row.append('d')
                    elif divide_group_mlp[i] == 1 :
                        row.append('s')
                cph_data_test.append(row)

            cph_df_test = pd.DataFrame(cph_data_test,columns=cph_head)

            kmf = KaplanMeierFitter()
            if len(cph_df_test.loc[cph_df_test.group == 'd']) > 1 :
                print('a')
                groups = cph_df_test["group"]
                T = cph_df_test["S"]
                E = cph_df_test["E"]
                ix = (groups == 'd')
                kmf.fit(T[~ix], E[~ix], label='survival')
                ax = kmf.plot()
                kmf.fit(T[ix], E[ix], label='death')
                kmf.plot(ax=ax)
                plt.title(str(iter_num)+'th trial of '+str(followup)+'year survival with '+method)
                plt.savefig('../data/KMplot/'+str(iter_num)+'th trial of '+str(followup)+'year survival with '+method+'.png')
            else :
                pass

        k+=1

inputs, groups = loadKMinput()
makeKMplot(inputs, groups, ["SLP","MLP"])