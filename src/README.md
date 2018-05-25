## Data form description
```
split_data = feature_list, x_trn, y_trn, s_trn, c_trn, x_dev, y_dev, s_dev, c_dev, x_tst, y_tst, s_tst, c_tst, 
             censored_df_trn, censored_df_dev, censored_df_tst = split_data


x_trn = row : 환자, column : features
y_trn = row : 환자, column : time interval에서의 survival probability(총 17개)
c_trn = row : 환자, column : censored flag
censored_df_trn = row : 환자, column : time interval에서의 censored flag(총 17개)
s_trn = row : 환자, column : survival duration => c_index에서 사용하고 model build에서는 제외
```

## Input description
1) RNN input (sample, time_step, features)
- feature_form => [array(sample_1), array(sample_2), ...]
- array(sample_1) => array([feature1, ..., feature_n]*17)
```
array(sample_1)의 17개의 리스트에서 feature 1 ~ feature n-1 까지는 동일(clinical data)
feature_n은 time interval에서의 censored 유무로 censored되기 전까지는 0, 된 후부터는 1
```

2) MLP & SLP input
- x_trn과 censored_df_trn concat

## Result
- rmse, cindex 
- prediction한 결과 값이 survival possibility이기 때문에, survival duration과 비교될 수 있다고 생각하여 prediction 값을 c_index input으로 넣어줌
- 아래 표는 17개의 time interval에서 나온 결과의 average 

score|RNN|MLP|SLP
 --|--|--|--
 RMSE|0.197791543|0.228285343|0.233011736
 Cindex|0.799676628|0.607161243|0.560969756
 
