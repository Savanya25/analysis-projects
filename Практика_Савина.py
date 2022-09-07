import sqlite3
import sys,time,datetime,os
import fileinput, pathlib, math
import numpy as np
import seaborn as sns # библиотека для создания статистических графиков на Python.
import pandas as pd
import matplotlib.pyplot as plt
import statistics 
from sklearn import datasets
from sklearn import metrics #для вычисления точности предсказания 
from sklearn.tree import DecisionTreeClassifier #испортируем клссификатор, который будет использоваться
from sklearn.model_selection import train_test_split #для разделения данных
from IPython.display import Image
from sklearn.tree import export_graphviz
from subprocess import call
import math
import pandas
import random 
import seaborn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


os.chdir('C:\\soxranyt vce suda\\Any faif sem\\KR')
flag_BD = 2
pth = 'C:\\soxranyt vce suda\\Any faif sem\\KR\\номер 5 скважина 133 ЭЦН'
pattern = '*.txt'
files_path = pathlib.Path(pth)
list_files = files_path.glob(pattern)
new_file = 'Full_archive_1.txt'
flag_extremes = False
#Создаем базу данных
conn= sqlite3.connect('Arhiv1.db')  # подключение к базе данных
cur = conn.cursor()     # курсор
cur.execute( """CREATE TABLE IF NOT EXISTS Arhiv1 (Data DATE,
    Time TEXT,
    V_Zv_Centr_Trub REAL,
    V_Zv_Perefer_Trub REAL,
    Gaz_Centr_Trub REAL,
    Gaz_Perefer_Trub REAL,
    Temp_Potoka REAL,
    Obvodn_Centr_Trub REAL,
    Obvodn_Perefer_Trub REAL,
    Kod_Skvag INTEGER,
    Integral_gaz_Potok REAL,
    Integral_Obvodn_Potok REAL);
    """)
conn.commit()
   
#Считываем скважину в базу данных 
if (flag_BD == 1): 
    if list_files:
     with fileinput.FileInput(files=list_files) as file1, open(new_file, 'w') as fw:
        for line in file1:
            if file1.isfirstline():
                file_name = file1.filename()
            fw.write(line)
            Data_s=line[:10]
            line=line[len(Data_s)+1:]
            Data_s=datetime.date(int(Data_s[:4]), int(Data_s[5:7]), int(Data_s[8:10]))
            Time_s=line[:8]
            line=line[len(Time_s)+1:]
            V_Zv_Centr_Trub_s = line[:line.find('\t')]
            line=line[len(V_Zv_Centr_Trub_s)+1:]
            V_Zv_Perefer_Trub_s = line[:line.find('\t')]
            line=line[len(V_Zv_Perefer_Trub_s)+1:]
            Gaz_Centr_Trub_s = line[:line.find('\t')]
            line=line[len(Gaz_Centr_Trub_s)+1:]
            Gaz_Perefer_Trub_s = line[:line.find('\t')]
            line=line[len(Gaz_Perefer_Trub_s)+1:]
            Temp_Potoka_s = line[:line.find('\t')]
            line=line[len(Temp_Potoka_s)+1:]
            Obvodn_Centr_Trub_s = line[:line.find('\t')]
            line=line[len(Obvodn_Centr_Trub_s)+1:]
            Obvodn_Perefer_Trub_s = line[:line.find('\t')]
            line=line[len(Obvodn_Perefer_Trub_s)+1:]
            Kod_Skvag_s=line[:line.find('\t')]
            line=line[len(Kod_Skvag_s)+1:]
            Integral_gaz_Potok_s = line[:line.find('\t')]
            line=line[len(Integral_gaz_Potok_s)+1:]
            Integral_Obvodn_Potok_s =line[:line.find('\t')]
            new_data=(Data_s,Time_s,V_Zv_Centr_Trub_s,V_Zv_Perefer_Trub_s,Gaz_Centr_Trub_s,Gaz_Perefer_Trub_s,Temp_Potoka_s, Obvodn_Centr_Trub_s,Obvodn_Perefer_Trub_s,Kod_Skvag_s,Integral_gaz_Potok_s,Integral_Obvodn_Potok_s)
            cur.execute ( "INSERT INTO Arhiv1 VALUES (?,?,?,?,?,?,?,?,?,?,?,?);", new_data)
            conn.commit()
            
#Выводим базу данных      
a=cur.execute ("select Data from Arhiv1;")    
# print(a)

#Узнать кол-во строк в базе данных о
cur.execute("select Data from Arhiv1;")    
rows = cur.fetchall()
print(len(rows))
conn.commit()

sql = 'SELECT * FROM {}'.format('Arhiv1')    


#Переделать наши данные  в DataFrame
Data_base = cur.execute(sql).fetchall() #заносим в переменную всю нашу базу данных
Data_base= pd.DataFrame(Data_base, columns=['Дата', 'Время', 'Vзв. в цн.т.', 'Vзв. на п.т.','Газ.сод. в цн.т.','Газ.сод. на п.т.', 'Т потока', 'Обв. в цн.т.', 'Обв.на п.т.','Код','Инт. газосодер. потока','Инт. Обв. потока'])
print(Data_base)
 

#определить класс для каждой строки  
def define_class (column, _class, n_class):
     min_value = min(column)
     max_value = max (column)
     step = round((max_value-min_value)/n_class, 3)  
     index = 1
     for i in range (len(column)):
         index = 1
         for j in range (1, n_class+1):
              if (min_value  + step*(index-1) <= column[i] < min_value + step*index):
                    _class[i] = j
              index +=1
         if (min_value  + step*(n_class-1) <= column[i] <=  max_value):
            _class[i] = n_class
             
                
   
    
          
#Обучение
def classification(x, y, max_depth_, min_samples):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    model = DecisionTreeClassifier(
                                # функция для impurity ('gini' или 'entropy')
                                criterion='gini',
                                # максимальная глубина дерева
                                max_depth=max_depth_,
                                # минимальное число элементов в узле для разбиения (может быть долей)
                                min_samples_split=4,
                                # минимальное число элементов в листе (может быть долей)
                                # min_samples_leaf=5,
                               )

    # # # Обучаем модель
    model.fit(x_train, y_train)
    print('Accuracy on training set: {:.3f}'.format(model.score(x_train,y_train)))
    print('Accuracy on test set: {:.3f}'.format(model.score(x_test,y_test)))
    export_graphviz(model,
                out_file='tree.dot',
                # задать названия классов
                class_names=['1','2','3', '4', '5'],
    #             # раскрашивать узлы в цвет преобладающего класса
                filled=True,
    #             # показывать значение impurity для каждого узла
                impurity=False,
    #             # показывать номера узлов
                node_ids=True,
    #             # Число точек после запятой для отображаемых дробей
                precision=2
                )
    # Преобразуем файл tree.dot в tree.png
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png']);
    # Вставляем картинку в блокнот
    Image("tree.png")
    


# Разделить набор данных на основе значения
def data_division (index, value, dataset):
    true = list()
    false  = list()
    for row in dataset:
        if row[index] < value:
            false.append(row)
        else:
            true.append(row)
    return false, true
 
#передаем наше группы после разбиения и уникальные классы 
def gini_index(groups, classes):
	# Сколько изначально было значений
    n_selection = 0
    #Проходим по 2 группам
    for group in groups:
       n_selection +=len(group)
    n_selection = float(n_selection)    
    gini_index = 0.0
    for group in groups: #проходим по данным
        size_group = float(len(group))
        if size_group == 0:
            continue
        score = 0.0
        class_number = []
        for class_val in classes: #проходимся по классам (по всем которые бывают)
            n_class_val = 0
        # считываем все классы в группе 1 переменную
            class_number = []
            for row in group:  
                 class_number.append(row[-1])
            # После этого считаем сколько данного класса в значениях 
            for i in class_number:
                if i == class_val:
                    n_class_val += 1
            p = n_class_val/size_group
            score += p**2
        gini_index  += (1.0 - score) * (size_group / n_selection)
    return gini_index 


#Выбираем лучшую точку для разбиения
def get_split(dataset):
    class_values =[]
    for row in dataset:
        if row[-1] not in class_values:
            class_values.append(row[-1])   
    new_index, new_value, new_score = 1000, 1000, 1000
    new_groups = None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            #для этого делим по каждой точки данные на 2 разбиения
             groups_after_division = data_division(index, row[index], dataset) 
            #и считаем для них индекс Джини
             gini = gini_index(groups_after_division, class_values)
             if gini < new_score:
                    new_index, new_value, new_score, new_groups = index, row[index], gini, groups_after_division  
    return {'index':new_index, 'value':new_value, 'groups':new_groups, 'gini': new_score}
 
# возращает наиболее частый класс в группе
def often_value_class (group):
    value = []
    for row in group:
        value.append(row[-1])
    return max(set(value), key=value.count)
 
#Реализуем условия расщепления 
def if_split(best_split, max_depth, min_size, depth):
    false, true = best_split['groups']
    gini = best_split['gini']
    del(best_split['groups'])
    #проверяем пустые группы строк, если да, то создаем конечный узел 
    if not false or not true:
        best_split['false'] = best_split['true'] = often_value_class(false + true)
        return 
    # проверяем достигли ли мы максимальной глубины
    if depth >= max_depth:
        best_split['false'], best_split['true'] = often_value_class(false), often_value_class(true)
        return
    if gini == 0.0:
         best_split['false'] = best_split['true'] = often_value_class(true)
    if len(false) <= min_size:
        best_split['false'] = often_value_class(false)
    else:
        best_split['false'] = get_split(false)
        if_split(best_split['false'], max_depth, min_size, depth+1)
    if len(true) <= min_size:
        best_split['true'] = often_value_class(true)
    else:
        best_split['true'] = get_split(true)
        if_split(best_split['true'], max_depth, min_size, depth+1)
   
 
# Построение дерева
def tree_build(dataset, max_depth, min_size):
    #выбираем наилучшее разбиение
    best_split = get_split(dataset)
    #Проверка критерий остановы 
    if_split(best_split, max_depth, min_size, 1)
    return best_split
 
#Выводим дерево 
def tree_output(best_split, depth=0):
    if isinstance(best_split, dict):
        print('{0}x[{1}] < {2}  Gini:{3}'.format(depth*'  ',(best_split['index']+1), (best_split['value']), round(best_split['gini'], 2)))
        tree_output(best_split['false'], depth+1)
        tree_output(best_split['true'], depth+1)
    else:
        print('{0}Class:{1}'.format(depth*'  ', best_split))





n_folds = 1
max_depth = 4 
min_size = 50

Data_base = Data_base.assign(Class = 0) #Добавляем столбец, который будет определять класс и заполним пока его 0
define_class (Data_base['Инт. Обв. потока'], Data_base['Class'],  5)

# 
#берем выборку данный, чтобы наглядно посмотреть результат
Data_base_1 = Data_base[0:65000:50]
print('Data_base_1')
print(Data_base_1.loc[:,'Vзв. в цн.т.':'Инт. газосодер. потока'],Data_base_1.loc[:,'Class'])

classification (Data_base_1.loc[:,'Vзв. в цн.т.':'Инт. газосодер. потока'],Data_base_1.loc[:,'Class'],max_depth, min_size )




dataset=[]
i=0
print('dataset')
# for j in range(len(Data_base_1['Инт. Обв. потока'])):
#         dataset.append([Data_base_1['Vзв. в цн.т.'][i], Data_base_1['Vзв. на п.т.'][i], 
#                        Data_base_1['Газ.сод. в цн.т.'][i], Data_base_1['Газ.сод. на п.т.'][i],
#                        Data_base_1['Т потока'][i], Data_base_1['Обв. в цн.т.'][i],
#                        Data_base_1['Обв.на п.т.'][i], Data_base_1['Код'][i], 
#                        Data_base_1['Инт. газосодер. потока'][i], Data_base_1['Class'][i]])
#         i +=50
       
# print()


# tree = tree_build(dataset, max_depth, min_size)
# tree_output(tree)


