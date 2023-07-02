"""## **Тестовое задание на позицию аналитика в MyFin**"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
danform=pd.read_excel('/content/ТЗ Данные формы — копия.xlsx')
status=pd.read_excel('/content/ТЗ статусы по заявкам — копия.xlsx')

"""Уточнение по поводу файла "Данные формы": загружать в ноутбук следует мой отредактированный файл. В нем в одной строчке двойные кавычки были заменены на одинарные, ввиду того, что у модуля json возникал с ними конфликт"""

print(danform.shape)
print(status.shape)

"""В таблице "Статусы по заявкам" заявок больше, чем в таблице "Данные формы"
Проверим, являются ли эти "лишиние заявки" дубликатами, либо же мы к ним не сможем подобрать соответствий из другой таблицы и их допустимо удалить:

"""

list_1=list(danform['Id заявки банка'])
list_2=list(status['Id заявки банка'])
list_3 = set(list_2) - set(list_1)
list_3

"""Да, гипотеза подтвердилась, есть 3 "лишних" заявки, поэтому соединение таблиц будет производиться по принципу INNER JOIN"""

full=status.merge(danform, how='inner')
full.shape

full.head()

full['Статус'].unique()

"""Соответсвенно последовательность операций будет следующая:
Заявка принята - Обработка завершена - ***Предварительно  одобрено/***Отказ - Выдано

Столбец получившейся таблицы "данные формы" содержит данные, которые можно преобразовать и разнести по новым столбцам в таблице и удобно их извлекать и использовать

Применим функцию json_normalize
"""

from pandas.io.json import json_normalize
list1=list(full['данные формы'])
list2=[]
import ast
for i in range(len(list1)):
  list2.append(ast.literal_eval(list1[i]))
forma=json_normalize(list2)
forma.head()

"""Сделаем финальную таблицу из двух имеющихся"""

full1=full[['Id заявки банка','Статус','id']]
finalfull=full1.merge(forma, left_index=True, right_index=True)
finalfull.head()

"""## Общий заработок с партнера будет рассчитываться как стоимость целевого действия умноженную на количество целевых действий"""

total_earning = 350 * len(finalfull.loc[(finalfull['Статус'] == 'Предварительно одобрено') | (finalfull['Статус']=='Выдано'), 'Статус'])
  total_earning

"""## **Стоимость одной заявки**=(CPA*Количество целевых действий)/Количество заявок




"""

(350*finalfull.loc[(finalfull['Статус'] == 'Предварительно одобрено') | (finalfull['Статус']=='Выдано'), 'Статус'].count()/finalfull.shape[0]).round(2)

"""## **AR(approve rate)** рассчитывается как отношение количества целевых действий к общему количеству заявок"""

(finalfull.loc[(finalfull['Статус'] == 'Предварительно одобрено') | (finalfull['Статус']=='Выдано'), 'Статус'].count()/finalfull.shape[0]*100).round(2)

"""## **Рекомендации касательно отправляемых партнеру заявок, для увеличения процента AR**

Чтобы предложить идею для повышения количества отправляемых заявок и количества действий, нужно провести анализ

В финальной таблице мы получили следующие столбцы: Id заявки банка,	Статус,	id,	amount,	credit_term_id,	credit_city_id,	education_id,	date_of_birth,type_of_employment,	gender,	registration_address_city,	living_address_postal_code

Очевидно, что не все эти показатели влияют на вероятность одобрения кредита. Так, можно отбросить Id заявки банка; id клиента(хотя с другой стороны id клиента может быть присвоен с того момента, как клиент впервые начал работу с банком, поэтому меньшие(более старые) id могут свидетельстовать о том, что клиент проверенный и "знаком" банку); credit_city_id(город, в котором находится отделение банка, который будет выдавать кредит), город регистрации и индекс тоже большой роли не играют, хотя можно классифицировать города по обеспеченности и по уровню заработных плат регионов, в которых они находятся, но эта гипотеза в случае, если она подтвердится не будет иметь большого веса; пол кредитуемого лица имеет право на участие в модели, так как в интернете достаточно споров о том, кто является более надежным заемщиком.

Так, создадим таблицу для анализа:
"""

finalfull.registration_address_city.nunique()

import requests

json1 = requests.get('https://raw.githubusercontent.com/Photon74/russian-cities-python/master/russian-cities.json')

import numpy as np
import json

extracted = pd.io.json.json_normalize(json1.json())

extracted = extracted[['name', 'subject']]
extracted.head()

len(set(extracted.name) & set(finalfull.registration_address_city))

sublist = list(extracted.name)

analysis_df = finalfull[['Статус', 'amount', 'credit_term_id', 'education_id', 'date_of_birth', 'type_of_employment', 'gender', 'registration_address_city']]
analysis_df.head()

subjects = []
non_subjects =[]
for i in analysis_df.registration_address_city:
  if i in sublist:
    subjects.append(extracted.loc[extracted.name==i].iloc[0,1])
  else:
    subjects.append('village')
    non_subjects.append(i)

len(subjects)

analysis_df['city'] = subjects

analysis_df.info()

analysis_df.city = analysis_df.city.astype(str)

analysis_df[['city']]=analysis_df[['city']].replace(' ', '_', regex=True)

analysis_df.city.unique()

from sklearn.preprocessing import OneHotEncoder
ohe= OneHotEncoder()
enc = ohe.fit_transform(analysis_df[['city']])

a22= pd.DataFrame(enc.toarray(), columns= ohe.categories_)
a22.columns = a22.columns.map(''.join)
a22.columns

analysis_df = analysis_df.join(a22)

analysis_df.education_id.unique()

analysis_df.head()

"""Условимся, что 30 - среднее специальное, 40 - общее среднее образование, 50 - высшее, 60 - магистратура, аспирантура и т.д.
Люди
"""

#educ_dict = {'30' : 0, '40' : 1, '50' : 2, '60' : 3}
#analysis_df['education_id'] = analysis_df['education_id'].apply(lambda x: educ_dict[x])

"""Дату рождения можно заменить на число полных лет, тип занятости тоже ранжировать, а пол сделать бинарным признаком"""

import datetime
analysis_df['date_of_birth'] = analysis_df['date_of_birth'].replace('-', '', regex=True)

import numpy as np
analysis_df['years_old'] = (pd.to_datetime(analysis_df['date_of_birth'], format='%d%m%Y') - datetime.datetime.now())/ np.timedelta64 ( -1 , 'Y')
analysis_df['years_old'] = analysis_df['years_old'].round()

plt.figure(figsize=(10,5))
plt.hist(analysis_df['Статус'], orientation = 'horizontal')

bins = 5
plt.hist(analysis_df.loc[analysis_df['Статус'] == 'Отказ', 'years_old'], bins, label = 'Неодобренные')
plt.hist(analysis_df.loc[(analysis_df['Статус'] == 'Выдано') |(analysis_df['Статус'] == "Предварительно одобрено"), 'years_old'], bins = 4,  label = 'Одобренные')

plt.legend(loc = 'upper right')
plt.xlabel('Количество лет', fontsize = 16)
plt.ylabel('Количество наблюдений', fontsize = 16)
plt.title('Распределение возраста для двух типов статуса', fontsize = 16)

plt.hist(analysis_df.loc[analysis_df['Статус'] == 'Отказ', 'type_of_employment'], label = 'Неодобренные')
plt.hist(analysis_df.loc[(analysis_df['Статус'] == 'Выдано') |(analysis_df['Статус'] == "Предварительно одобрено"), 'type_of_employment'],  label = 'Одобренные')

plt.ylim([0, 60])
plt.legend(loc = 'upper left')
plt.xlabel('Вид занятости', fontsize = 16)
plt.ylabel('Количество наблюдений', fontsize = 16)
plt.title('Зависимость выдачи кредита от вида занятости', fontsize = 16)

plt.hist(analysis_df.loc[analysis_df['Статус'] == 'Отказ', 'education_id'], label = 'Неодобренные')
plt.hist(analysis_df.loc[(analysis_df['Статус'] == 'Выдано') |(analysis_df['Статус'] == "Предварительно одобрено"), 'education_id'], label = 'Одобренные')

plt.ylim([0, 60])
plt.legend(loc = 'upper left')
plt.xlabel('Уровень образования', fontsize = 16)
plt.ylabel('Количество наблюдений', fontsize = 16)
plt.title('Зависимость выдачи кредита от уровня образования', fontsize = 16)

plt.hist(analysis_df['amount'], bins = 5, color = 'purple')

plt.xscale('linear')
plt.xlabel('Сумма кредита', fontsize = 16)
plt.ylabel('Количество заявок', fontsize = 16)
plt.title('Распределение величины суммы кредита', fontsize = 16)

"""## Подводя итог, можно сказать, что кредиты пользуются наибольшим спросом среди людей 30-50 лет с постоянным местом работы и среднеспециальным образованием на сумму до 400 000 рублей. Исходя из этого, можно представить образ целевого клиента, поэтому было бы целесообразно проводить рекламную кампанию, настроенную на такую аудиторию. **С большей вероятностью типичный кредитполучатель будет проживать в небольших городах и берет кредит на потребительские расходы - это может быть покупка машины, участка, бытовой техники, кредит на учебу.**"""

analysis_df.type_of_employment.unique()

#employ_dict = {'unemployed' : 0, 'pensioner': 1, 'own-business': 2, 'employ' : 3}
#analysis_df['type_of_employment'] = analysis_df['type_of_employment'].apply(lambda x: employ_dict[x])

analysis_df = pd.get_dummies(analysis_df, columns = ['type_of_employment'])

analysis_df

gend_dict = {'Ж' : 0, 'М': 1}
analysis_df['gender'] = analysis_df['gender'].apply(lambda x: gend_dict[x])

analysis_df.columns

analysis_df = analysis_df.drop(columns = ['date_of_birth', 'registration_address_city', 'city'])

analysis_df

analysis_df.loc[analysis_df['Статус']=='Выдано']

"""Переменную статуса выберем бинарную 0 - в случае отказа, 1 - в случае одобрения и выдачи кредита
Так, одобрение - это статусы "Предварительно одобрено" и "Выдано, Отказ - "Отказ, а статус "Заявка принята" и "Обработка завершена" будут тестовыми выборками
"""

analysis_df['Статус'].unique()

X_test = analysis_df.loc[(analysis_df['Статус']=='Заявка принята') | (analysis_df['Статус']=='Обработка завершена')].copy()
X_test = X_test.drop(columns = 'Статус', axis=1)
X_test.head()

X_train = analysis_df.loc[(analysis_df['Статус'] == 'Отказ') | (analysis_df['Статус'] == 'Предварительно одобрено') | (analysis_df['Статус'] == 'Выдано')]
status_dict = {'Отказ' : 0, 'Предварительно одобрено' : 1,  'Выдано' : 1}
X_train['status'] = X_train['Статус'].apply(lambda x: status_dict[x])
y_train = X_train['status'].copy()
X_train = X_train.drop(columns = ['status', 'Статус'], axis=1)

"""Далее построим модель классификации заявок, используя возможности логистической регрессии"""

X_train = X_train.astype('int64')
y_train = y_train.astype('int64')
X_test = X_test.astype('int64')

X_train

"""Исходя из графика корреляций, предпочтительная будет модель с 9 признаками

"""

X_train_5_feat= X_train.copy()[['amount', 'credit_term_id', 'education_id', 'gender', 'years_old']]
X_train_9_feat = X_train.copy()[['amount', 'credit_term_id', 'education_id', 'gender', 'years_old', 'type_of_employment_employ',
       'type_of_employment_own-business', 'type_of_employment_pensioner',
       'type_of_employment_unemployed']]
X_train_full = X_train.copy()

data_train_5_feat = X_train_5_feat.copy().join(y_train)
data_train_9_feat = X_train_9_feat.copy().join(y_train)
data_train_full = X_train_full.copy().join(y_train)

plt.figure(figsize=(10,10))
sns.heatmap(data_train_9_feat.corr())

"""# Проверим баланс классов и при необходимости их сбалансируем"""

rat = len(data_train_9_feat.status)/sum(data_train_9_feat.status)

balanced_data = pd.concat([data_train_9_feat.loc[data_train_9_feat.status==0], data_train_9_feat.loc[data_train_9_feat.status==1].loc[data_train_9_feat.loc[data_train_9_feat.status==1].index.repeat(rat)]] )

balanced_data.status.value_counts()

balanced_data = balanced_data.sample(frac=1).reset_index(drop=True)

X_train_9_feat = balanced_data.copy().drop(columns = 'status', axis=1)
y_train = balanced_data.copy()[['status']]

"""Разделим тренировочную выборку еще на тренировочную и тестовую, чтобы максимизировать качество модели, а после предсказать на реаьлных данных X_test

"""

from sklearn.model_selection import train_test_split
X_tr, X_ts, y_tr, y_ts = train_test_split(X_train_9_feat, y_train, test_size = 0.4)

#from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()
#X_tr_sc = X_tr.copy()
#X_ts_sc = X_ts.copy()
#X_tr_sc[['amount','credit_term_id',	'education_id', 'years_old']] = ss.fit_transform(X_tr_sc[['amount','credit_term_id',	'education_id', 'years_old']])
#X_ts_sc[['amount','credit_term_id',	'education_id', 'years_old']] = ss.transform(X_ts_sc[['amount','credit_term_id',	'education_id', 'years_old']])
#y_tr = np.array(y_tr)

"""# Построим модель классификации заявок, используя возможности "Случайного леса"
"""

from sklearn.ensemble import RandomForestClassifier
model1 = RandomForestClassifier()
model1.fit(X_tr, y_tr)
y_pred1 = model1.predict(X_ts)

y_pred1

from sklearn.metrics import precision_score, accuracy_score, recall_score
accuracy_score(y_ts, y_pred1), precision_score(y_ts, y_pred1), recall_score(y_ts, y_pred1)

# from sklearn.decomposition import PCA
# pca_test = PCA(n_components=9)
# pca_test.fit(X_tr)
# sns.set(style='whitegrid')
# plt.plot(np.cumsum(pca_test.explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance')
# plt.axvline(linewidth=4, color='r', linestyle = '--', x=10, ymin=0, ymax=1)
# display(plt.show())
# evr = pca_test.explained_variance_ratio_
# cvr = np.cumsum(pca_test.explained_variance_ratio_)
# pca_df = pd.DataFrame()
# pca_df['Cumulative Variance Ratio'] = cvr
# pca_df['Explained Variance Ratio'] = evr
# display(pca_df.head(10))

from sklearn.model_selection import GridSearchCV
n_estimators = [5,10, 20, 30]
max_depth = [5, 10]
min_samples_split = [4,5, 10, 15]
min_samples_leaf = [4,5, 10, 15]
param_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion' : ['gini', 'entropy', 'log_loss']}
gs = GridSearchCV(model1, param_grid, cv = 5, verbose = 1, n_jobs=-1)
gs.fit(X_tr, y_tr)

gs.best_params_

ypr = gs.predict(X_ts)

precision_score(y_ts, ypr)

"""**Модель model1 случайного леса дала хорошие показатели точности и полноты, которые можно применять в предсказании решения выдачи кредита получателю**"""
