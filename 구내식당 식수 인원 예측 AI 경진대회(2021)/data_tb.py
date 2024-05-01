# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 19:08:09 2021

@author: 1105t
"""

# =============================================================================
# <구내식당 식수 인원 예측 AI 경진대회>
# =============================================================================

# =============================================================================
#사용용어 변환
# 일자 : date
# 요일 : day
# 본사정원수 : total 
# 본사휴가자수 : vacation
# 본사출장자수 : bt
# 본사시간외근무명령서승인건수 : ow
# 현본사소속재택근무자수 : home
# 중식메뉴 : lun_menu
# 석식메뉴 : din_menu
# 중식계: lun_count
# 석식계 : din_count
# 출근자수 : commuter
# =============================================================================

import pandas as pd
import scipy.stats as ss  ##zscore이용 표준화
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc, style, font_manager

font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/GULIM.ttc").get_name()
rc('font', family=font_name)

#엑셀 데이터 불러오기
test = pd.read_csv('C:/Users/1105t/OneDrive/Desktop/dongari/koonaesickdang/test.csv')
train = pd.read_csv('C:/Users/1105t/OneDrive/Desktop/dongari/koonaesickdang/train.csv')
sample_submission = pd.read_csv('C:/Users/1105t/OneDrive/Desktop/dongari/koonaesickdang/sample_submission.csv')

#head 한국어에서 영어로 바꾸기
train.columns = ['date', 'day','total', 'vacation', 'bt', 'ow', 'home', 'bre_menu','lun_menu', 'din_menu', 'lun_count', 'din_count']
test.columns = ['date', 'day','total', 'vacation', 'bt', 'ow', 'home', 'bre_menu','lun_menu', 'din_menu']

#요일을 숫자로 매핑
train['day'] = train['day'].map({'월':0, '화':1, '수':2, '목':3, '금':4})
test['day'] = test['day'].map({'월':0, '화':1, '수':2, '목':3, '금':4})

#날짜를 데이터타임으로 변환
train['date']=pd.to_datetime(train['date'])
test['date']=pd.to_datetime(test['date'])

#휴가자수 재택근무자수 둘다 빼서 출근자수 컬럼 만들기 
train['commuter'] = train.total - train.vacation - train.home 
test['commuter'] = test.total - test.vacation - test.home 

#결측치 확인
test.isnull().sum()
train.isnull().sum()
sample_submission.isnull().sum() ##결측치 없음

test.c = test.iloc[:,1:7] ##숫자값만 열추출
train.c = test.iloc[:,1:7]


#상관관계 확인
df_t = pd.DataFrame(test).T
corr_t = df_t.corr(method = 'pearson') ##안나옴

df_r = pd.DataFrame(train.c).T
corr_r = df_r.corr(method = 'pearson')


#표준화 후 다시 확인
test_ss=ss.zscore(test.c)
df_ss = pd.DataFrame(test.c).T
corr_ss = df_ss.corr(method = 'pearson')

train_ss=ss.zscore(train.c)
df_ss2 = pd.DataFrame(train.c).T
corr_ss2 = df_ss2.corr(method = 'pearson')
##표준화전도 후도 상관관계가 높게 나옴 -> 뭔가 잘못된듯 ㅎ;

    
#일별로 따로 확인
dday=train.date.apply(lambda x : x.day)
l_d_count=train.iloc[:,10:12]
result = pd.concat([dday,l_d_count], axis=1) 

sns.lineplot(x=result.iloc[:,0], y=result.iloc[:,1]) ##알아서 평균처리해서 보여줌
sns.lineplot(x=result.iloc[:,0], y=result.iloc[:,2])

df_corr = result.corr()
sns.heatmap(df_corr, cmap='Reds')

sns.stripplot(x='date', y='lun_count', data=result)
sns.stripplot(x='date', y='din_count', data=result)
sns.violinplot(x='date', y='lun_count', data=result)
sns.boxplot(x='date', y='lun_count', data=result)
sns.boxplot(x='date', y='din_count', data=result)


#새로운 train파일을 엑셀에 저장
train.to_excel('C:/Users/1105t/OneDrive/Desktop/dongari/koonaesickdang/train.xlsx'
                   , sheet_name='new_name')


#토큰화
def token(data, col):
    token = []
    for i in range(len(train)):
        tmp = train.loc[i, col].split(' ') # 공백으로 문자열 구분 
        tmp = ' '.join(tmp).split()    # 빈 원소 삭제
        
        for menu in tmp:
            if i >1066:
                continue
            if ('(' in menu) | (')' in menu) : # 원산지 정보는 삭제
                tmp.remove(menu)
                
        token.append(tmp)
    
    return token

lunch = token(train, 'lun_menu')
dinner = token(train, 'din_menu')
    
# 점심=============================================================================
train['lun_rice']=0
for i in range(0,len(lunch)):
    z=lunch[i]
    train['lun_rice'][i]=z[0]

train['lun_guk']=0
for i in range(0,1067):
    z=lunch[i]
    train['lun_guk'][i]=z[1]
    
for i in range(1067,len(train)):
    z=lunch[i]
    train['lun_guk'][i]=z[1]

#저녁==============================================================================
train['din_rice']=0
for i in range(0,len(dinner)):
    if len(dinner[i])<4:          #*표시나 자기개발의 날 0처리
        train['din_rice'][i]=0
    else:
        z=dinner[i]
        train['din_rice'][i]=z[0]

train['din_guk']=0
for i in range(0,1066):
    if len(dinner[i])<4:
        train['din_guk'][i]=0
    else:    
        z=dinner[i]
        train['din_guk'][i]=z[1]
    
for i in range(1067,len(train)):
    if len(dinner[i])<4:
        train['din_guk'][i]=0
    else:
        z=dinner[i]
        train['din_guk'][i]=z[1]
        
##############반찬나누기##################
train['lun_banchan1']=0
for i in range(0,len(lunch)):
    z=lunch[i]
    train['lun_banchan1'][i]=z[2]

train['lun_banchan2']=0
for i in range(0,len(lunch)):
    z=lunch[i]
    train['lun_banchan2'][i]=z[3]
    
################################################## 
    
train['din_banchan1']=0
for i in range(0,len(dinner)):
    if len(dinner[i])<4:          #*표시나 자기개발의 날 0처리
        train['din_banchan1'][i]=0
    else:
        z=dinner[i]
        train['din_banchan1'][i]=z[2]

train['din_banchan2']=0
for i in range(0,1066):
    if len(dinner[i])<4:
        train['din_banchan2'][i]=0
    else:    
        z=dinner[i]
        train['din_banchan2'][i]=z[3]
    
for i in range(1067,len(train)):
    if len(dinner[i])<4:
        train['din_banchan2'][i]=0
    else:
        z=dinner[i]
        train['din_banchan2'][i]=z[3]

#1142,1171# 예외처리 (쌀밥은 보편적임으로 제거하였음)
train['lun_rice'][1142]='곤드레밥/찰현미밥'
train['lun_guk'][1142]='된장찌개'
train['lun_rice'][1171]='(New)바지락비빔밥'
train['lun_guk'][1171]='팽이장국'


#점심 특식 라벨링
train['lunch_teuksik']=1
for i in range(0,len(lunch)):
    z=lunch[i]
    for j in range (0,len(lunch[i])):
        if z[j].find('/')!=-1:
            train['lunch_teuksik'][i]=0
           
            
#저녁 특식 라벨링
train['dinner_teuksik']=1
for i in range(0,1080):
   z=dinner[i]
   for j in range(0,len(dinner[i])):
       if z[j].find('/')!=-1:
           train['dinner_teuksik'][i]=0
           
for i in range(1080,len(dinner)):
    z=dinner[i]
    for j in range(0,len(dinner[i])):
        if z[j].find('흑미')!=-1:
            train['dinner_teuksik'][i]=0

#################################################################

#밥, 국 (점심/저녁) 빈도수
l_r = train['lun_rice'].value_counts().reset_index()
l_g = train['lun_guk'].value_counts().reset_index()
d_r = train['din_rice'].value_counts().reset_index()
d_g = train['din_guk'].value_counts().reset_index()

#빈도수 EDA
sns.barplot(data=l_r.sort_values('lun_rice',ascending= False)[2:13]
            , x='index', y='lun_rice') #barplot for 3등부터 12등까지
plt.pie('lun_rice', labels='index', autopct='%.1f%%'
        ,data=l_r.sort_values('lun_rice',ascending= False)[2:13]) #원그래프
plt.show()
sns.barplot(data=l_g.sort_values('lun_guk',ascending= False)[0:10]
            , x='index', y='lun_guk') #barplot for 1등부터 10등까지
plt.pie('lun_guk', labels='index', autopct='%.1f%%'
        ,data=l_g.sort_values('lun_guk',ascending= False)[0:10]) #원그래프
plt.show()

sns.barplot(data=d_r.sort_values('din_rice',ascending= False)[3:14]
            , x='index', y='din_rice') #barplot for 4등부터 13등까지
plt.pie('din_rice', labels='index', autopct='%.1f%%'
        ,data=d_r.sort_values('din_rice',ascending= False)[3:14]) #원그래프
plt.show()
sns.barplot(data=d_g.sort_values('din_guk',ascending= False)[1:11]
            , x='index', y='din_guk') #barplot for 2등부터 12등까지
plt.pie('din_guk', labels='index', autopct='%.1f%%'
        ,data=d_g.sort_values('din_guk',ascending= False)[1:12]) #원그래프
plt.show()

##도수 1 -> 기타로 합치기
count=0
for i in range(len(l_r)):
    if l_r.lun_rice[i] == 1:
        count=count+1
        
#인덱스, 런라이스으로 바꿔서 시리즈어펜드, 콘켓
Series = pd.Series(['기타', count], index=['index', 'lun_rice'])
l_r = l_r.append(Series, ignore_index=True)

##도수 1 -> 행삭제
for i in range(len(l_r)):
    if l_r.lun_rice[i] == 1:
        l_r = l_r.drop(index=i, axis=0) 
        
#오름차순 정렬
l_r = l_r.sort_values(by='lun_rice' ,ascending=False).reset_index()

flg ,axes=plt.subplots(1,1,sharey=True)
plt.pie('lun_rice', labels='index', autopct='%.1f%%'
        ,data=l_r.sort_values('lun_rice',ascending= False)[2:]) #원그래프
#plt.show()

plt.bar('index','lun_rice',data=l_r[2:])

for i, v in enumerate(l_r.loc[2:]):
    plt.text(v, l_r.lun_rice[i+2], l_r.lun_rice[i+2],                 # 좌표 (x축 = v, y축 = y[0]..y[1], 표시 = y[0]..y[1])
             fontsize = 9, 
             color='blue',
             horizontalalignment='center',  # horizontalalignment (left, center, right)
             verticalalignment='bottom')    # verticalalignment (top, center, bottom)

plt.show()


##도수 1 -> 기타로 합치기
count=0
for i in range(len(l_g)):
    if l_g.lun_guk[i] == 1:
        count=count+1
        
#인덱스, 런라이스으로 바꿔서 시리즈어펜드, 콘켓
Series = pd.Series(['기타', count], index=['index', 'lun_guk'])
l_g = l_g.append(Series, ignore_index=True)

##도수 1 -> 행삭제
for i in range(len(l_g)):
    if l_g.lun_guk[i] == 1:
        l_g = l_g.drop(index=i, axis=0) 
        
#오름차순 정렬
l_g = l_g.sort_values(by='lun_guk' ,ascending=False).reset_index()

plt.pie('lun_guk', labels='index', autopct='%.1f%%'
        ,data=l_g.sort_values('lun_guk',ascending= False)) #원그래프









##도수 1 -> 기타로 합치기
count=0
for i in range(len(d_r)):
    if d_r.din_rice[i] == 1:
        count=count+1
        
#인덱스, 런라이스으로 바꿔서 시리즈어펜드, 콘켓
Series = pd.Series(['기타', count], index=['index', 'din_rice'])
d_r = d_r.append(Series, ignore_index=True)

##도수 1 -> 행삭제
for i in range(len(d_r)):
    if d_r.din_rice[i] == 1:
        d_r = d_r.drop(index=i, axis=0) 
##index 0 -> 행삭제   
d_r = d_r.drop(index=2, axis=0) 
        
#오름차순 정렬
d_r = d_r.sort_values(by='din_rice' ,ascending=False).reset_index()

flg ,axes=plt.subplots(1,1,sharey=True)
plt.pie('din_rice', labels='index', autopct='%.1f%%'
        ,data=d_r.sort_values('din_rice',ascending= False)[2:]) #원그래프



##도수 1 -> 기타로 합치기
count=0
for i in range(len(d_g)):
    if d_g.din_guk[i] == 1:
        count=count+1
        
#인덱스, 런라이스으로 바꿔서 시리즈어펜드, 콘켓
Series = pd.Series(['기타', count], index=['index', 'din_guk'])
d_g = d_g.append(Series, ignore_index=True)

##도수 1 -> 행삭제
for i in range(len(d_g)):
    if d_g.din_guk[i] == 1:
        d_g = d_g.drop(index=i, axis=0)
 ##index 0 -> 행삭제   
d_g = d_g.drop(index=1, axis=0) 
        
#오름차순 정렬
d_g = d_g.sort_values(by='din_guk' ,ascending=False).reset_index()

plt.pie('din_guk', labels='index', autopct='%.1f%%'
        ,data=d_g.sort_values('din_guk',ascending= False)) #원그래프


#gensim의 Word2Vec으로 토픽모델링 -> 안씀
from gensim.models import Word2Vec
 ############참고 https://ebbnflow.tistory.com/153



#DBSCAN -> 안씀
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

db = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
y_db = db.fit_predict(model_b.wv.vectors)
y_db

############################umm
def scale(menu,count):
    a=train.groupby(y).mean()[count]
    aa=a.index
    aaa=a.values
    x={}
    
    for i in  range(0,len(a)):
        x[aa[i]]=aaa[i]
        
    return(x)

def scale_r(menu,count):
    a=train.groupby(menu).mean()[count].rank()
    aa=a.index
    aaa=a.values
    x={}
    
    for i in  range(0,len(a)):
        x[aa[i]]=aaa[i]
        
    return(x)

scale('din_rice','din_count')
scale_r('din_rice','din_count')


