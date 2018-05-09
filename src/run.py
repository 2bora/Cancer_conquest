import ML as ml
import numpy as np
import Preprocessing as pp


def RunModel(filename,iter_num):
    iter_num = int(iter_num)
    results_rnn = []
    results_NN_mlp = []
    results_NN_slp = []
    results_cox = []
    for i in range(0, iter_num) :
        split_data=pp.split_trn_dev_tst(filename)
        #rmse, cindex = ml.TrainNNLogLikeLoss(split_data)
        rmse_slp, cindex_slp = ml.TrainNNnaive(split_data,'SLP')
        rmse_mlp, cindex_mlp = ml.TrainNNnaive(split_data,'MLP')
        rmse_rnn, cindex_rnn = ml.TrainRNN(split_data)
        c_tst = ml.TrainCPH(split_data)
        results_rnn.append([rmse_rnn, cindex_rnn])
        results_NN_mlp.append([rmse_mlp, cindex_mlp])
        results_NN_slp.append([rmse_slp, cindex_slp])        
        results_cox.append(c_tst)
        
        print str(i),"th process done"
    avg_rnn = np.mean(results_rnn,axis = 0)
    avg_slp = np.mean(results_NN_slp,axis = 0)
    avg_mlp = np.mean(results_NN_mlp,axis = 0)
    avg_cox = np.mean(results_cox)

    print(' RNN Result mean RMSE, Cindex ', avg_rnn)
    print(' SLP Result mean RMSE, Cindex ', avg_slp)
    print(' MLP Result mean RMSE, Cindex ', avg_mlp)
    print(' COX Result mean Cindex ', avg_cox)



RunModel('clinic',5)