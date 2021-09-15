import ml_project
import pandas as pd

#esperimento 1: N = (i*15),k = 5, sf = 10

#la seguente funzione serve per la generazione generazione dei risultati ma bloccata perchè già presenti nelle cartelle
#ml_project.test_with_difficult_dataset_generation(15, 5, 10, 0)

dataSet_risultati = pd.read_csv("result_casual/esperimento1/Result.csv")
Ada_accuracy = list(zip(dataSet_risultati.Accuratezza_Ada.values))
SVM_accuracy = list(zip(dataSet_risultati.Accuratezza_SVM.values))
Random_forest_accuracy = list(zip(dataSet_risultati.Accuratezza_Random_forest.values))
Decision_tree_accuracy = list(zip(dataSet_risultati.Accuratezza_Decision_tree.values))
G3 = list(zip(dataSet_risultati.G3.values))
Number_row = list(zip(dataSet_risultati.Number_row.values))
Number_Class = list(zip(dataSet_risultati.Number_class.values))

esperimenti = []
for i in range (0,len(G3)):
  esperimenti.append(i+1)

#plot delle accuracy sul numero di tuple in maniera crescente
ml_project.plt_G3_to_accuracy(G3, SVM_accuracy, Ada_accuracy, Random_forest_accuracy, Decision_tree_accuracy, esperimenti)




# esperimento 2: N = 100 ,k = 5, sf = (10*i)

#la seguente funzione serve per la generazione generazione dei risultati ma bloccata perchè già presenti nelle cartelle
#ml_project.test_with_difficult_dataset_generation(100, 5, 10, 1)

dataSet_risultati = pd.read_csv("result_casual/esperimento2/Result.csv")
Ada_accuracy = list(zip(dataSet_risultati.Accuratezza_Ada.values))
SVM_accuracy = list(zip(dataSet_risultati.Accuratezza_SVM.values))
Random_forest_accuracy = list(zip(dataSet_risultati.Accuratezza_Random_forest.values))
Decision_tree_accuracy = list(zip(dataSet_risultati.Accuratezza_Decision_tree.values))
G3 = list(zip(dataSet_risultati.G3.values))
Number_row = list(zip(dataSet_risultati.Number_row.values))
Number_Class = list(zip(dataSet_risultati.Number_class.values))

esperimenti = []
for i in range (0,len(G3)):
  esperimenti.append(i+1)

#plot delle accuracy sul numero di tuple in maniera crescente
ml_project.plt_G3_to_accuracy(G3, SVM_accuracy, Ada_accuracy, Random_forest_accuracy, Decision_tree_accuracy, esperimenti)




# esperimento 3: N = 100,k = (i*10), sf = 10

#la seguente funzione serve per la generazione generazione dei risultati ma bloccata perchè già presenti nelle cartelle
#ml_project.test_with_difficult_dataset_generation(100, 10, 10, 2)

dataSet_risultati = pd.read_csv("result_casual/esperimento3/Result.csv")
Ada_accuracy = list(zip(dataSet_risultati.Accuratezza_Ada.values))
SVM_accuracy = list(zip(dataSet_risultati.Accuratezza_SVM.values))
Random_forest_accuracy = list(zip(dataSet_risultati.Accuratezza_Random_forest.values))
Decision_tree_accuracy = list(zip(dataSet_risultati.Accuratezza_Decision_tree.values))
G3 = list(zip(dataSet_risultati.G3.values))
Number_row = list(zip(dataSet_risultati.Number_row.values))
Number_Class = list(zip(dataSet_risultati.Number_class.values))

esperimenti = []
for i in range (0,len(G3)):
  esperimenti.append(i+1)

#plot delle accuracy sul numero di tuple in maniera crescente
ml_project.plt_G3_to_accuracy(G3, SVM_accuracy, Ada_accuracy, Random_forest_accuracy, Decision_tree_accuracy, esperimenti)


# esperimento dataset reali

#le seguenti funzione servono per la generazione generazione dei risultati ma bloccata perchè già presenti nelle cartelle
#ml_project.data_cleaning()
#ml_project.generate_test_and_train()
#ml_project.test_with_real_dataset()

#estrazione dei risultati delle accuracy in formato lista
dataSet_risultati = pd.read_csv("result_real/Result.csv")
Ada_accuracy = list(zip(dataSet_risultati.Accuratezza_Ada.values))
SVM_accuracy = list(zip(dataSet_risultati.Accuratezza_SVM.values))
Random_forest_accuracy = list(zip(dataSet_risultati.Accuratezza_Random_forest.values))
Decision_tree_accuracy = list(zip(dataSet_risultati.Accuratezza_Decision_tree.values))
G3 = list(zip(dataSet_risultati.G3.values))
Gini = list(zip(dataSet_risultati.Gini.values))
Number_row = list(zip(dataSet_risultati.Number_row.values))
Number_Class = list(zip(dataSet_risultati.Number_class.values))

esperimenti = []
for i in range (0,len(G3)):
  esperimenti.append(i+1)

#funzioni per richiamare i plot
ml_project.plt_G3_to_accuracy(G3, SVM_accuracy, Ada_accuracy, Random_forest_accuracy, Decision_tree_accuracy, esperimenti)
ml_project.plt_Gini_to_accuracy(Gini, SVM_accuracy, Ada_accuracy, Random_forest_accuracy, Decision_tree_accuracy, esperimenti)



