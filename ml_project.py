from math import log
from random import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import LinearSVC
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import preprocessing
import time
from sklearn.ensemble import AdaBoostClassifier

"""# **Difficult dataset generation algorithm**"""

def Generate(n,N,k,sf):
  clas = 1
  relation = np.zeros((N,n+2))
  row_similar = N + 1
  while row_similar >= N + 1:
    for i in range(0,N):
      for j in range(0,n):
        relation[i][j] = randint(0,round(log(N),0))
      relation[i][n+1] = clas
      if clas == k: clas = 0
      clas = clas + 1

    row_similar = 0
    for row1 in relation:
      for row2 in relation:
        similar_attribute = True
        for i in range (0,len(row1)):
          if row1[i] == row2[i]: similar_attribute = False  
        if similar_attribute == True: row_similar = row_similar + 1 
  
  dataset = relation
  copy = relation
  for i in range(2,sf):
    copy = Duplicate(copy, k)
    dataset = np.concatenate([dataset,copy]) 
  return dataset            

def Duplicate(r,k):
  duplicate = []
  for row in r:
    new = row
    new[len(row)-1] = (row[len(row)-1] + 1)%k 
    duplicate.append(new)
  return duplicate

"""# **Generazione dei file CSV di test e train dei dataset per i dataset generati casualmente**"""

def csv_generate(n,N,k,sf,index,path):
  setDati = Generate(n,N,k,sf)
  attribute=[]
  listDati = range(len(setDati))
  X_train, X_test, Y_train, Y_test=train_test_split(setDati,listDati)

  with open(path + '/train'+str(index)+'.csv', mode='w',newline='') as file:
    writer = csv.writer(file)
    for i in range(0,(len(setDati[0])-1)):
      attribute.append("A"+str(i+1))
    attribute.append("Classe")
    writer.writerow(attribute)
    
    for row in X_train:
      attribute=[]
      for i in range(0,len(row)):
         attribute.append(row[i])
      writer.writerow(attribute)
    
  attribute=[]
  with open(path + '/test'+str(index)+'.csv', mode='w',newline='') as file:
    writer = csv.writer(file)
    for i in range(0,(len(setDati[0])-1)):
      attribute.append("A"+str(i+1))
    attribute.append("Classe")
    writer.writerow(attribute)
    for row in X_test:
      attribute=[]
      for i in range(0,len(row)):
         attribute.append(row[i])
      writer.writerow(attribute)

"""# **Data cleaning dei dataset reali**"""
def data_cleaning():
  my_file=open("Real_dataset/mammographic_masses.data",mode="r")
  with open('Real_dataset/mammographic_masses.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["BI-RADS_assessment","Age","Shape","Margin","Density","Severity"])
    
    for line in my_file:
      array = []
      for i in range(0,6):
        array.append(line.split(",")[i])
      writer.writerow(array)
      
  my_file=open("Real_dataset/parkinsons.data",mode="r")
  with open('Real_dataset/parkinsons.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    
    for line in my_file:
      array = []
      for i in range(0,23):
        array.append(line.split(",")[i])
      temp = array[17]
      array[17] = array[22]
      array[22] = temp 
      writer.writerow(array)

  my_file=open("Real_dataset/allUsers.lcl.csv",mode="r")
  with open('Real_dataset/allUsers.lcl2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    reader = csv.reader(my_file)
    for line in reader:
      array = []
      for i in range(1,17): #tolte 21 colonne perchè il valore è "?"
        array.append(line[i])
      array.append(line[0])
      writer.writerow(array)

  my_file=open("Real_dataset/log2.csv",mode="r")
  with open('Real_dataset/log2-2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    reader = csv.reader(my_file)
    for line in reader:
      array = []
      for i in range(0,12): #tolte 21 colonne perchè il valore è "?"
        if i != 4: array.append(line[i])
      array.append(line[4])
      writer.writerow(array)

  my_file=open("Real_dataset/PS_20174392719_1491204439457_log.csv",mode="r")
  with open('Real_dataset/PS_20174392719_1491204439457_log2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    reader = csv.reader(my_file)
    for line in reader:
      array = []
      for i in range(0,11): #tolte 21 colonne perchè il valore è "?"
        if i != 9: array.append(line[i])
      array.append(line[9])
      writer.writerow(array)

  my_file=open("Real_dataset/clean1.data",mode="r")
  with open('Real_dataset/clean2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    reader = csv.reader(my_file)
    
    array = []
    for i in range(0,168):
      array.append("A"+str(i))
    writer.writerow(array)

    for line in my_file:
      array = []
      for i in range(0,168):
        array.append(line.split(",")[i])
      writer.writerow(array)

  my_file=open("Real_dataset/letter-recognition.data",mode="r")
  with open('Real_dataset/letter-recognition2.data', 'w', newline='') as file:
    writer = csv.writer(file)
    reader = csv.reader(my_file)
     
    array = []
    for i in range(0,16):
      array.append("A"+str(i))
    writer.writerow(array)

    for line in reader:
      array = []
      for i in range(1,16): 
        array.append(line[i])
      array.append(line[0])
      writer.writerow(array)

"""# **Generazione dei file CSV di test e train per i dataset con anche la conversione dei dataset con stringhe in valori per la lavorazione con i modelli**"""
def generate_test_and_train():
  path = ["Real_dataset/mammographic_masses.csv",
        "Real_dataset/parkinsons.csv",
        "Real_dataset/log2-2.csv",
        "Real_dataset/allUsers.lcl2.csv",
        "Real_dataset/PS_20174392719_1491204439457_log2.csv",
        "Real_dataset/clean2.csv",
        "Real_dataset/letter-recognition2.data"]
  # per ogni dataset andiamo a dividire il train e il test con le proporzioni di default della libreria model_selection e successivamente vengono creati i file di train e di test in formato csv
  for p in range(0,len(path)):
    with open(path[p], 'r') as file:
      reader = csv.reader(file)
      x = []
      y = []
      j = 0
      for row in reader:
        j = j + 1
        if j == 1:
          for i in range(0,len(row)):
              if i != len(row)-1:
                x.append(row[i])
              else:
                y.append(row[i])    
          fd = [x,y]

    setDati = []
    j = 0

    with open(path[p], 'r') as file:
      reader = csv.reader(file)
      for row in reader:
        dati = []
        j = j + 1
        if j != 1 :
          for w in range(0,len(row)):
            if w != len(row) - 1: dati.append(row[w])
            else: dati.append(row[w].split("\n")[0])
          setDati.append(dati)

    #conversione degli attributi in interi
    for r in range(0,(len(fd[0])+len(fd[1]))):
      arr = []
      try:
        prova = float(setDati[0][r])
      except:
        for m in range(0,len(setDati)):
          arr.append(setDati[m][r])
        le = preprocessing.LabelEncoder()
        le.fit(arr)
        arr_tr = le.transform(arr)
        for m in range(0,len(setDati)):
          setDati[m][r] = arr_tr[m]
          print("colonna " + str(r) + " riga " + str(m))
 
    #split del train e del test
    listDati = range(len(setDati))
    X_train, X_test, Y_train, Y_test=train_test_split(setDati,listDati)

    #creazione file csv di train
    attribute=[]
    with open('file_train_real/train'+str(p)+'.csv', mode='w',newline='') as file:
      writer = csv.writer(file)
      for i in range(0,(len(fd[0]))):
        attribute.append(fd[0][i])
      attribute.append(fd[1][0])
      writer.writerow(attribute)
    
      for row in X_train:
        attribute=[]
        conf = True
        for i in range(0,len(row)):
          if row[i] == "?":
            conf = False
          else:
            attribute.append(row[i])
        if conf == True: writer.writerow(attribute)

    #creazione file csv di test
    attribute=[]
    with open('file_test_real/test'+str(p)+'.csv', mode='w',newline='') as file:
      writer = csv.writer(file)
      for i in range(0,(len(fd[0]))):
        attribute.append(fd[0][i])
      attribute.append(fd[1][0])
      writer.writerow(attribute)
      for row in X_test:
        attribute=[]
        conf = True
        for i in range(0,len(row)):
          if row[i] == "?":
            conf = False
          else:
            attribute.append(row[i])
        if conf == True: writer.writerow(attribute)

"""# **Linear Support Vector Machine**"""

def SVM(p,fd,tipo_test):
  X_train = []
  Y_train = []

  #Se tipo test è 0 stiamo facendo i test dell'algoritmo che genera casualmente gli algoritmi altrimenti con 1 i dataset reali
  if tipo_test == 0:
    path_train = "file_train_casual/esperimento1"
    path_test = "file_test_casual/esperimento1"
    path_result = "result_casual/esperimento1"
  elif tipo_test == 1:
    path_train = "file_train_casual/esperimento2"
    path_test = "file_test_casual/esperimento2"
    path_result = "result_casual/esperimento2"
  elif tipo_test == 2:
    path_train = "file_train_casual/esperimento3"
    path_test = "file_test_casual/esperimento3"
    path_result = "result_casual/esperimento3"
  elif tipo_test == 3:
    path_train = "file_train_real"
    path_test = "file_test_real"
    path_result = "result_real"

  with open(path_train + "/train"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
        
    j = 0
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_train.append(x)
        Y_train.append(y)
        
  X_test = []
  Y_test = []
  with open(path_test + "/test"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
    j = 0
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_test.append(x)
        Y_test.append(y)
  
  model=LinearSVC()
  model.fit(X_train,np.ravel(Y_train))
  pred=model.predict(X_test)

  test_df = pd.read_csv(path_test + '/test'+str(p)+'.csv')
  test_df['Classe_pred'] = pred
  test_df.to_csv(path_result + "/test_svm_pred"+str(p)+".csv", index=False)
  return (metrics.accuracy_score(Y_test,y_pred=pred))

"""# **AdaBoost**"""

def Ada(p,fd,tipo_test):
  X_train = []
  Y_train = []

  #Se tipo test è 0 stiamo facendo i test dell'algoritmo che genera casualmente gli algoritmi altrimenti con 1 i dataset reali
  if tipo_test == 0:
    path_train = "file_train_casual/esperimento1"
    path_test = "file_test_casual/esperimento1"
    path_result = "result_casual/esperimento1"
  elif tipo_test == 1:
    path_train = "file_train_casual/esperimento2"
    path_test = "file_test_casual/esperimento2"
    path_result = "result_casual/esperimento2"
  elif tipo_test == 2:
    path_train = "file_train_casual/esperimento3"
    path_test = "file_test_casual/esperimento3"
    path_result = "result_casual/esperimento3"
  elif tipo_test == 3:
    path_train = "file_train_real"
    path_test = "file_test_real"
    path_result = "result_real"

  with open(path_train + "/train"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
        
    j = 0
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_train.append(x)
        Y_train.append(y)
        
  X_test = []
  Y_test = []
  with open(path_test+ "/test"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
    j = 0
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_test.append(x)
        Y_test.append(y)
  
  abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
  model = abc.fit(X_train, Y_train)
  pred = model.predict(X_test)

  test_df = pd.read_csv(path_test + '/test'+str(p)+'.csv')
  test_df['Classe_pred'] = pred
  test_df.to_csv(path_result + "/test_ada_pred"+str(p)+".csv", index=False)
  return (metrics.accuracy_score(Y_test, pred))

"""## **Random Forest**"""

def SF(p,fd, tipo_test):
  X_train = []
  Y_train = []

  #Se tipo test è 0 stiamo facendo i test dell'algoritmo che genera casualmente gli algoritmi altrimenti con 1 i dataset reali
  if tipo_test == 0:
    path_train = "file_train_casual/esperimento1"
    path_test = "file_test_casual/esperimento1"
    path_result = "result_casual/esperimento1"
  elif tipo_test == 1:
    path_train = "file_train_casual/esperimento2"
    path_test = "file_test_casual/esperimento2"
    path_result = "result_casual/esperimento2"
  elif tipo_test == 2:
    path_train = "file_train_casual/esperimento3"
    path_test = "file_test_casual/esperimento3"
    path_result = "result_casual/esperimento3"
  elif tipo_test == 3:
    path_train = "file_train_real"
    path_test = "file_test_real"
    path_result = "result_real"

  with open(path_train + "/train"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
    j = 0 
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_train.append(x)
        Y_train.append(y)
        
  X_test = []
  Y_test = []
  with open(path_test + "/test"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
    j = 0
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_test.append(x)
        Y_test.append(y)

  model=RandomForestClassifier()
  model.fit(X_train,np.ravel(Y_train))
  pred=model.predict(X_test)

  test_df = pd.read_csv(path_test + '/test'+str(p)+'.csv')
  test_df['Classe_pred'] = pred
  test_df.to_csv(path_result + "/test_rf_pred"+str(p)+".csv", index=False)
  return (metrics.accuracy_score(Y_test,y_pred=pred))

"""# **Decision** **Tree** ##"""

def DT(p,fd,tipo_test):
  X_train = []
  Y_train = []

  # Se tipo test è 0 stiamo facendo i test dell'algoritmo che genera casualmente gli algoritmi altrimenti con 1 i dataset reali
  if tipo_test == 0:
    path_train = "file_train_casual/esperimento1"
    path_test = "file_test_casual/esperimento1"
    path_result = "result_casual/esperimento1"
  elif tipo_test == 1:
    path_train = "file_train_casual/esperimento2"
    path_test = "file_test_casual/esperimento2"
    path_result = "result_casual/esperimento2"
  elif tipo_test == 2:
    path_train = "file_train_casual/esperimento3"
    path_test = "file_test_casual/esperimento3"
    path_result = "result_casual/esperimento3"
  elif tipo_test == 3:
    path_train = "file_train_real"
    path_test = "file_test_real"
    path_result = "result_real"

  with open(path_train + "/train"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
    j = 0 
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_train.append(x)
        Y_train.append(y)
        
  X_test = []
  Y_test = []
  with open(path_test + "/test"+str(p)+".csv", 'r') as file:
    reader = csv.reader(file)
    j = 0
    for row in reader:
      j = j + 1
      x = []
      y = []
      conf = True
      if j != 1:
        for i in range(0,len(row)):
          if i == len(row)-1:
            y.append(float(row[i]))
          else:
            x.append(float(row[i]))    

        X_test.append(x)
        Y_test.append(y)

  model=DecisionTreeClassifier()
  model.fit(X_train,Y_train)
  pred=model.predict(X_test)

  test_df = pd.read_csv(path_test + '/test'+str(p)+'.csv')
  test_df['Classe_pred'] = pred
  test_df.to_csv(path_result + "/test_dt_pred"+str(p)+".csv", index=False)
  return (metrics.accuracy_score(Y_test,y_pred=pred))

"""## **G3 Error algoritmo**"""

#procedure ComputeG3;
def G3_function(r,fd):
  map = {}
  for i in range(0,len(r)):
    x = ""
    for j in range (0,len(fd[0])):
      x = x + str(r[fd[0][j]][i]) + "-" 
    y = str(r[fd[1][0]][i])
    try: 
      map[x]
      try: 
        map[x][y]
        map[x][y] = map[x][y] + 1 
      except: 
        map[x][y] = 1
    except:
      map[x] = {}
      map[x][y] = 1 

  maxsum = 0
  for x in map:
    max = 0
    for y in map[x]:
      if max < map[x][y]: 
        max = map[x][y]
    maxsum = maxsum + max
  return maxsum/len(r)

"""## **Impurità di Gini algoritmo**"""

def Gini_function(r,fd):
  map = {}
  for i in range(0,len(r)):
    x = ""
    for j in range (0,len(fd[0])):
      x = x + str(r[fd[0][j]][i]) + "-" 
    y = str(r[fd[1][0]][i])
    try: 
      map[x]
      try: 
        map[x][y]
        map[x][y] = map[x][y] + 1 
      except: 
        map[x][y] = 1
    except:
      map[x] = {}
      map[x][y] = 1 

  gini_result = 0
  for x in map:
    tot = 0
    numero_tuple = 0
    for y in map[x]:
      tot = tot + (map[x][y]/len(r))
      numero_tuple = numero_tuple + map[x][y]
    gini_result = gini_result + (tot*(numero_tuple/len(r)))
  return 1 - gini_result

"""## **Esecuzione dei classificatori e dell'algoritmo del G3 sul Difficult dataset generation algorithm**"""
def test_with_difficult_dataset_generation(N,k,sf,tipo_esperimento):
  #array di supporto contenenti le accuracy dei classificatori
  SVM_accuracy = []
  Random_forest_accuracy = []
  Decision_tree_accuracy = []
  Ada_accuracy = []
  G3 = []
  Number_row = []
  Number_class = []

  fd = [["A1","A2","A3","A4"],["Classe"]]
  #nei commenti troviamo i 3 esperimenti riguardanti i dataset creati dal Difficult dataset generation algorithm nei quali usiamo i classificatori e il G3
  for i in range (1,15):
    #csv_generate(n,N,k,sf,index)
    #esperimento 1: N = (i*15),k = 5, sf = 10
    #esperimento 2: N = 100 ,k = 5, sf = (10*i)
    #esperimento 3: N = 100,k = (i*10), sf = 10
    if tipo_esperimento == 0:
      csv_generate(N*15,k,sf,i,"file_test_casual/esperimento1")
      path = "result_casual/esperimento1"
    elif tipo_esperimento == 1:
      csv_generate(N,k,sf*i,i,"file_test_casual/esperimento2")
      path = "result_casual/esperimento2"
    else:
      csv_generate(N,i*sf, k, sf, i,"file_test_casual/esperimento3")
      path = "result_casual/esperimento3"

    #esecuzione dei classificatori
    Number_row.append((15*i))
    SVM_accuracy.append(SVM(i,fd,tipo_esperimento))
    Random_forest_accuracy.append(SF(i,fd,tipo_esperimento))
    Decision_tree_accuracy.append(DT(i,fd,tipo_esperimento))
    Ada_accuracy.append(Ada(i,fd,tipo_esperimento))

    r = pd.read_csv(path + "/test_svm_pred"+str(i)+".csv")
    Number_row.append(len(r))

    #esecuzione G3 e Gini (da eliminare gini)
    G3.append(G3_function(r,fd))

    arr = []
    le = preprocessing.LabelEncoder()
    for m in range(1,len(r)):
      arr.append(r[fd[1][0]][m])
    le.fit(arr)
    Number_class.append(len(le.classes_))
  
    #output dei risultati sul file csv
    with open(path + '/Result.csv', mode='w',newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["Accuratezza_Ada","Accuratezza_SVM","Accuratezza_Random_forest","Accuratezza_Decision_tree","G3","Number_row","Number_class"])
    
      for i in range(0, len(SVM_accuracy)):
        writer.writerow([Ada_accuracy[i],SVM_accuracy[i],Random_forest_accuracy[i],Decision_tree_accuracy[i],
                       G3[i],Number_row[i],Number_class[i]])

"""## **Esecuzione dei classificatori e dell'algoritmo del G3 e Gini sui dataset del mondo reale**"""
def test_with_real_dataset():
  SVM_accuracy = []
  Random_forest_accuracy = []
  Decision_tree_accuracy = []
  Ada_accuracy = []
  Gini = []
  G3 = []
  Number_row = []
  Number_class = []
  timer = []

  path = ["Real_dataset/mammographic_masses.csv",
        "Real_dataset/parkinsons.csv",
        "Real_dataset/log2-2.csv",
        "Real_dataset/allUsers.lcl2.csv",
        "Real_dataset/PS_20174392719_1491204439457_log2.csv",
        "Real_dataset/clean2.csv",
        "Real_dataset/letter-recognition2.data"]

  for p in range (0,len(path)):
    with open(path[p], 'r') as file:
      reader = csv.reader(file)
      x = []
      y = []
      j = 0
      for row in reader:
        j = j + 1
        if j == 1:
          for i in range(0,len(row)):
              if i != len(row)-1:
                x.append(row[i])
              else:
                y.append(row[i])    
          fd = [x,y]
  
    SVM_accuracy.append(SVM(p,fd,3))
    Random_forest_accuracy.append(SF(p,fd,3))
    Decision_tree_accuracy.append(DT(p,fd,3))
    Ada_accuracy.append(Ada(p,fd,3))
  
    r = pd.read_csv("result_real/test_svm_pred"+str(p)+".csv")
    Number_row.append(len(r))
  
    start = time.time()
    G3.append(G3_function(r,fd))
    Gini.append(Gini_function(r,fd))
    end = time.time()
    timer.append(end - start)

    arr = []
    le = preprocessing.LabelEncoder()
    for m in range(1,len(r)):
      arr.append(r[fd[1][0]][m])
    le.fit(arr)
    Number_class.append(len(le.classes_))
  
    with open('result_real/Result.csv', mode='w',newline='') as file:
      writer = csv.writer(file)
      writer.writerow(["Path_file","Accuratezza_Ada","Accuratezza_SVM","Accuratezza_Random_forest","Accuratezza_Decision_tree","G3","Gini","Number_row","Number_class","Time_work"])
    
      for i in range(0, len(SVM_accuracy)):
        writer.writerow([path[i],round(Ada_accuracy[i],2),
                      round(SVM_accuracy[i],2),round(Random_forest_accuracy[i],2),
                      round(Decision_tree_accuracy[i],2), round(G3[i],2),round(Gini[i],2),round(Number_row[i],2),round(Number_class[i],2),timer[i]])

"""## **Funzioni per costruire i plot dei risultati sui classificatori, il G3 e Gini**"""

def plt_G3_to_accuracy(G3,SVM_accuracy,Ada_accuracy,Random_forest_accuracy, Decision_tree_accuracy,esperimenti):
  plt.plot(esperimenti, SVM_accuracy, linestyle='--', marker='o',label="Linear SVM")
  plt.plot(esperimenti, Random_forest_accuracy, linestyle='--', marker='o', label="Random Forest") 
  plt.plot(esperimenti, Decision_tree_accuracy, linestyle='--', marker='o', label="Decision Tree" )
  plt.plot(esperimenti, Ada_accuracy, linestyle='--', marker='o',label="Ada Boost") 
  plt.plot(esperimenti, G3, marker='o', label="G3" ) 
  x0 = [0.85] 
  y0 = [0.3]
  plt.xlabel("Dataset") 
  plt.ylabel("Accuracy") 
  plt.title("Confronto tra accuratezza e valore G3 dei modelli") 
  plt.plot(x0, y0)
  plt.legend(bbox_to_anchor=(1.4, 1))
  plt.show()

def plt_Gini_to_accuracy(Gini,SVM_accuracy,Ada_accuracy,Random_forest_accuracy, Decision_tree_accuracy,esperimenti):
  plt.plot(esperimenti, SVM_accuracy, linestyle='--', marker='o',label="Linear SVM")
  plt.plot(esperimenti, Random_forest_accuracy, linestyle='--', marker='o', label="Random Forest") 
  plt.plot(esperimenti, Decision_tree_accuracy, linestyle='--', marker='o', label="Decision Tree" )
  plt.plot(esperimenti, Ada_accuracy, linestyle='--', marker='o',label="Ada Boost") 
  plt.plot(esperimenti, Gini, marker='o', label="Gini" ) 
  x0 = [0.85] 
  y0 = [0.3]
  plt.xlabel("Dataset") 
  plt.ylabel("Accuracy") 
  plt.title("Confronto tra accuratezza e valore Gini dei modelli") 
  plt.plot(x0, y0)
  plt.legend(bbox_to_anchor=(1.4, 1))
  plt.show()

def plt_accuracy_to_number_class(G3,Number_Class, SVM_accuracy,Ada_accuracy,Random_forest_accuracy,Decision_tree_accuracy):
  plt.plot(Number_Class, SVM_accuracy, linestyle='--', marker='o',label="Linear SVM")
  plt.plot(Number_Class, Random_forest_accuracy, linestyle='--', marker='o', label="Random Forest") 
  plt.plot(Number_Class, Decision_tree_accuracy, linestyle='--', marker='o', label="Decision Tree") 
  plt.plot(Number_Class, Ada_accuracy, linestyle='--', marker='o',label="Ada Boost") 
  plt.plot(Number_Class, G3,marker='o', label="G3") 
  x0 = [20] 
  y0 = [0.35]
  plt.xlabel("Numero classi") 
  plt.ylabel("Accuracy dei modelli") 
  plt.title("Confronto accuratezza tra i modelli con numero di classi diverse") 
  plt.plot(x0, y0)
  plt.show()

def plt_accuracy_to_number_row(G3,Number_row, SVM_accuracy,Ada_accuracy,Random_forest_accuracy,Decision_tree_accuracy):
  plt.plot(Number_row, SVM_accuracy, linestyle='--', marker='o',label="Linear SVM")
  plt.plot(Number_row, Random_forest_accuracy, linestyle='--', marker='o', label="Random Forest") 
  plt.plot(Number_row, Decision_tree_accuracy, linestyle='--', marker='o', label="Decision Tree") 
  plt.plot(Number_row, Ada_accuracy, linestyle='--', marker='o',label="Ada Boost") 
  plt.plot(Number_row, G3,marker='o', label="G3") 
  x0 = [max(Number_row)] 
  y0 = [0.35]
  plt.xlabel("Numero tuple") 
  plt.ylabel("Accuracy dei modelli") 
  plt.title("Confronto accuratezza tra i modelli con numero di tuple") 
  plt.plot(x0, y0)
  plt.show()

