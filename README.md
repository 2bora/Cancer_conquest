# Cancer_conquest

## 1. Objective
- 대장암 환자들의 Clinical 정보(후에 유전정보, 이미지 정보도 추가될 예정)를 가지고 특정 time interval마다 각 환자의 생존 확률 예측.
- 현재 코드에서는 time interval을 1년으로 지정

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
<img width="1279" alt="2018-04-30 11 46 36" src="https://user-images.githubusercontent.com/30252311/39414122-2417da26-4c6f-11e8-86df-1fdd2ceb58f4.png">

- 환자별로 각 time interval에서 생존확률을 Output으로 지정
        
**1. uncensored data**

        1(살아있는 기간동안), 0(죽은 시점부터~)

**2. censored data**

        1(censored되지 않은 기간 동안), 
        1-d/n(censored 된 시점 부터, d = the number of deceased subjects, n = total number of subjects alive at the beginning of time, Kaplan-Meier hazard probabilities)
        
## 4. Censored_dataframe
<img width="1277" alt="2018-04-30 12 17 36" src="https://user-images.githubusercontent.com/30252311/39414271-82c03310-4c70-11e8-8242-23651e33dea3.png">
- 환자별로 각 time interval에서 censored여부를 나타냄(censored = 1, un-censored = 0)

## 5. Model description
- RNN, MLP, SLP 세가지 모델로 test 중

