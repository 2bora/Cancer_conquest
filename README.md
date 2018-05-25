# Cancer_conquest

## 1. Objective
- 대장암 환자들의 Clinical 정보(후에 유전정보, 이미지 정보도 추가될 예정)를 가지고 특정 time interval마다 각 환자의 생존 확률 예측.
- 3,5,7년 후 생존여부 예측을 위해서 1년 time interval에서의 생존 확률 예측하는 모델과 3,5,7년 후 생존여부(0/1)를 바이너리로 예측하는 모델 각각 build

## 2. Input Data
- Clinical information list(ALL)

        총 환자 : 496명
        Sex : 1(294), 2(202)
        Age : 0(20-30, 1), 1(30-40, 18), 2(40-50, 46), 3(50-60, 126), 4(60-70, 122), 5(70-80, 143),6(80-90, 38),7(90-100, 2)
        BMI : 0(0-18.5, 31), 1(18.5-25, 276), 2(25-30, 166), 3(30-40, 22), 4(40-50, 1)
        Comorbidity : 0(244), 1(132), 2(118), 3(2)
        GT : 1(35), 2(231), 3(203), 4(1), 5(1), 6(25)   
        pT : 1(4), 2(24), 3(404), 4(64)
        pN : 0(244), 1(147), 2(105) 
        Stage : 1(1), 2(243), 3(252) 
        CRM : 0(147), 1(27), -1(NA, 322) 
        Differentiation : 1(well, 16), 2(moderate, 457), 3(poorly, 22), -1(NA, 1)
        Neural_Invasion : 0(No, 317), 1(Yes, 179)
        Vascular_Invasion : 0(No, 420), 1(Yes, 76)
        Lymphatic_Invasion : 0(No, 308), 1(Yes, 188)  
        K-ras : 0(154), 1(90), 2(252)
        Adjuvant_Tx : 0(Yes, 115), 1(No, 381)
        Censored_flag : 0(Not censored(dead), 56) , 1(Censored(followup이 끊긴 환자 + followup이 끝났을 때 생존한 환자), 440)
        recurrence : 0(367), 1(123), 2(6) 
        Event_flag : 0(Not daed, 440), 1(daed, 56)
        PCEA, Harvested_LNs, Lnmeta_num, days_to_followup, years_to_followup : continuous variables

- 이 중에서 Sex, Age, BMI, Comorbidity, GT, pT, pN, stage, CRM, Differentiation, Neural_Invasion,
Vascular_Invation, K-ras, Adjuvant_Tx,recurrence, PCEA, Harvested_LNs, Lnmeta_num만 Input data로 사용

## 3. Output Data
### 3-1. Survival Probability Prediction for each Time interval 
<img width="1279" alt="2018-04-30 11 46 36" src="https://user-images.githubusercontent.com/30252311/39414122-2417da26-4c6f-11e8-86df-1fdd2ceb58f4.png">

- 환자별로 각 time interval에서 생존확률을 Output으로 지정
        
**1. uncensored data**

        1(살아있는 기간동안), 0(죽은 시점부터~)

**2. censored data**

        1(censored되지 않은 기간 동안), 
        1-d/n(censored 된 시점 부터, d = the number of deceased subjects, n = total number of subjects alive at the beginning of time, Kaplan-Meier hazard probabilities)
        
### 3-2. Survival binary prediction for 3,5,7 year
- 3년 생존율을 구할 때는 3년 이전에 censored 된 데이터는 제외 (5, 7년 분석시에도 동일하게)
- 1 = 생존, 0 = 사망, 아래표에서 숫자는 [전체 sample 수(생존 sample 수/사망 sample 수)]를 나타냄

x_yr survival|overall|y_trn|y_dev|y_tst
--|--|--|--|--
3|1=372,0=35|260(237/23)|66(60/6)|82(75/7)
5|1=208,0=44|160(132/28)|41(34/7)|51(42/9)
7|1=124,0=49|110(79/31)|28(20/8)|35(25/10)

## 4. Censored_dataframe
<img width="1277" alt="2018-04-30 12 17 36" src="https://user-images.githubusercontent.com/30252311/39414271-82c03310-4c70-11e8-8242-23651e33dea3.png">
- 환자별로 각 time interval에서 censored여부를 나타냄(censored = 1, un-censored = 0)

## 5. Model description
- MLP, SLP 모델로 test 

## 6. Result
### 6-1. Score

year|binary_SLP_auc|binary_MLP_auc|time_SLP_auc|time_MLP_auc|timeSLP_Cindex|timeMLP_Cindex
--|--|--|--|--|--|--
3|0.641524|0.706286|0.786667|0.716952|0.773900|0.725073
5|0.662434|0.759788|0.806349|0.789418|0.780226|0.745122
7|0.676000|0.744800|0.743200|0.714400|0.697204|0.678235
