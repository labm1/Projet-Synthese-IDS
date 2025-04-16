#----------------------------------------------------------------------------------------------------------
#   
#    L�UTILISATION DE MOD�LES D�INTELLIGENCE ARTIFICIELLE POUR LA D�TECTION DES INTRUSIONS DANS LES R�SEAUX
#   
#   Auteure: M�gane Labelle
#   Superviseure: Hajar Moudoud
#   Coordonateur: Karim El Guemhioui
#   
#   Cours: INF4173 Projet Synth�se
#
#   Le 25 avril 2025
#----------------------------------------------------------------------------------------------------------


#importations de librairies
import pandas as pd
pd.options.mode.chained_assignment = None
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve 




#----------------------------------------------------------------------------------------------------------
#
#                                TRAITEMENT DE LA BASE DE DONN�E NSL-KDD
#
#----------------------------------------------------------------------------------------------------------

#noms des colonnes
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

df = pd.read_csv("NSL_KDD.csv", names=col_names)    #importer la base de donn�es
df_2 = df.copy()
df['label'].value_counts() #compter le nombre de valeurs de label et les afficher

#normal = pas d'attaque = 0, attaque = 1
label_class_dict={'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                          'ipsweep' : 1,'nmap' : 1,'portsweep' : 1,'satan' : 1,'mscan' : 1,'saint' : 1
                           ,'ftp_write': 1,'guess_passwd': 1,'imap': 1,'multihop': 1,'phf': 1,'spy': 1,'warezclient': 1,'warezmaster': 1,'sendmail': 1,'named': 1,'snmpgetattack': 1,'snmpguess': 1,'xlock': 1,'xsnoop': 1,'httptunnel': 1,
                           'buffer_overflow': 1,'loadmodule': 1,'perl': 1,'rootkit': 1,'ps': 1,'sqlattack': 1,'xterm': 1}

label_class_dict_2={ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4}

df['label']=df['label'].replace(label_class_dict) #remplacer les cat�gories d'attaques par 0 ou 1
df_2['label']=df_2['label'].replace(label_class_dict_2) #remplacer les cat�gories d'attaques par 0, 1, 2, 3 ou 4

#afficher le nombre d'attaques de chaque type
print("Nombre normal: %d" % (df['label'] == 0).sum())
print("Nombre attaques: %d" % (df['label'] == 1).sum())
print("Nombre DoS: %d" % (df_2['label'] == 1).sum())
print("Nombre Probe: %d" % (df_2['label'] == 2).sum())
print("Nombre R2L: %d" % (df_2['label'] == 3).sum())
print("Nombre U2R: %d" % (df_2['label'] == 4).sum())

#enlever les duplicats
df = df.drop_duplicates()
df_2 = df_2.drop_duplicates()

#mettre les donn�es en vrai/faux
df=pd.get_dummies(df)
df_2=pd.get_dummies(df_2)

#Pour pr�dire un certain type d'attaque, garder seulement ce type et les normales
to_drop_DoS = [0,1]
to_drop_Probe = [0,2]
to_drop_R2L = [0,3]
to_drop_U2R = [0,4]

DoS_df=df_2[df_2['label'].isin(to_drop_DoS)];

#Remplacer les �tiquettes d'attaques de 2 � 1
Probe_df=df_2[df_2['label'].isin(to_drop_Probe)];
Probe_df['label']=Probe_df['label'].replace({2:1})

#Remplacer les �tiquettes d'attaques de 3 � 1
R2L_df=df_2[df_2['label'].isin(to_drop_R2L)];
R2L_df['label']=R2L_df['label'].replace({3:1})

#Remplacer les �tiquettes d'attaques de 4 � 1
U2R_df=df_2[df_2['label'].isin(to_drop_U2R)];
U2R_df['label']=U2R_df['label'].replace({4:1})

#D�finir les colonnes de donn�es et les colonnes � pr�dire
X = df.drop(columns="label")    #X = le tableau moins la colonne label
Y = df["label"]                 #Y = la colonne label

X_DoS = DoS_df.drop(columns="label")
Y_DoS = DoS_df["label"]

X_Probe = Probe_df.drop(columns="label")
Y_Probe = Probe_df["label"]

X_R2L = R2L_df.drop(columns="label")
Y_R2L = R2L_df["label"]

X_U2R = U2R_df.drop(columns="label")
Y_U2R = U2R_df["label"]

#normaliser les donn�es
sc = StandardScaler()
X=sc.fit_transform(X)           
X_DoS=sc.fit_transform(X_DoS)           
X_Probe=sc.fit_transform(X_Probe)           
X_R2L=sc.fit_transform(X_R2L)           
X_U2R=sc.fit_transform(X_U2R)
                 

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30)                                       #s�parer le set en train et test
X_DoS_train, X_DoS_test, Y_DoS_train, Y_DoS_test = train_test_split(X_DoS, Y_DoS, test_size=0.30)               #s�parer le set DoS en train et test
X_Probe_train, X_Probe_test, Y_Probe_train, Y_Probe_test = train_test_split(X_Probe, Y_Probe, test_size=0.30)   #s�parer le set Probe en train et test
X_R2L_train, X_R2L_test, Y_R2L_train, Y_R2L_test = train_test_split(X_R2L, Y_R2L, test_size=0.30)               #s�parer le set R2L en train et test
X_U2R_train, X_U2R_test, Y_U2R_train, Y_U2R_test = train_test_split(X_U2R, Y_U2R, test_size=0.30)               #s�parer le set U2R en train et test

#initialiser les donn�es de la courbe ROC et du tableau de comparaison
X_ROC = []
tableau = [['','Exactitude','Precision','Rappel','F-mesure','Faux-positifs','Faux-negatifs', 'Temps d\'entrainement (s)']]

#----------------------------------------------------------------------------------------------------------
#
#                                                FONCTIONS
#
#----------------------------------------------------------------------------------------------------------
#Cr�er le graphe ROC curve et le tableau de comparaison des m�triques
def create_ROC_curve_and_table(attack_name, Y_test):
    global X_ROC, tableau
    plt.clf()
    
    #ajouter la courbe ROC pour chaque mod�le
    for i in range(len(X_ROC)):
        fpr, tpr, thresholds = roc_curve(Y_test, X_ROC[i])
        plt.plot(fpr, tpr,label='data model ' + str(i+1))

    #ajouter les titres au graphe et le sauvegarder
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=0)
    plt.savefig('ROC_curve/ROC_curve_' + attack_name + '.png')

    #cr�er le tableau et le sauvegarder
    tableau = np.array(tableau)
    np.savetxt('tableaux/' + attack_name + '_tableau.csv', tableau.transpose(), fmt="%s", delimiter=";")

    #r�initialiser les donn�es de la courbe ROC et du tableau de comparaison
    X_ROC=[]
    tableau = [['','Exactitude','Precision','Rappel','F-mesure','Faux-positifs','Faux-negatifs', 'Temps d\'entrainement (s)']]



#Cr�er la matrice de confusion
def create_confusion_matrix(Y_test, Y_pred, name):
    plt.clf()
    cm=confusion_matrix(Y_test, Y_pred) #Cr�er une matrice de confusion selon les estimations du mod�le et les vraies donn�es
    accuracy_score(Y_test,Y_pred)

    #Cr�er la matrice de confusion
    df=pd.DataFrame(cm,columns=['normal','anormal'],index=['normal','anormal'])
    plt.rcParams['figure.figsize']=[10,6]
    sns_plot =sns.heatmap(df,annot=True, annot_kws={"size": 16},fmt='g',cmap="YlGnBu",linewidths=.5)

    #construction d'un graphique
    figure = sns_plot.get_figure()
    plt.xlabel("Classe Predite")
    plt.ylabel('Vraie Classe')
    plt.savefig('confusion_matrix/confusion_matrix_' + name + '.png')



#Entrainer le mod�le
def train_model(model_name, model, X_test, X_train, Y_test, Y_train):
    global X_ROC, tableau
    start=time.time()               #temps que le mod�le prends (d�but)
    model.fit(X_train,Y_train)    #Entrainer le mod�le avec les donn�es d'entrainement
    finish= time.time()             #temps que le mod�le prends (fin)
    t=finish-start                  #temps que le mod�le prends

    print(f'training time={t}')
    Y_pred = model.predict(X_test)  #Pr�dire les attaques ou non selon les donn�es

    #imprimer les m�triques d'�valuation
    accuracy = cross_val_score(model, X_test, Y_test, cv=10, scoring='accuracy').mean()
    print("Accuracy: %0.5f" % accuracy)

    precision = cross_val_score(model, X_test, Y_test, cv=10, scoring='precision').mean()
    print("Precision: %0.5f" % precision)

    recall = cross_val_score(model, X_test, Y_test, cv=10, scoring='recall').mean()
    print("Recall: %0.5f" % recall)

    f = cross_val_score(model, X_test, Y_test, cv=10, scoring='f1').mean()
    print("F-measure: %0.5f" % f)

    #afficher les faux positifs et faux n�gatifs
    FP = ((Y_pred == 1) & (Y_test == 0)).sum()
    FN = ((Y_pred == 0) & (Y_test == 1)).sum()
    print("Nombre faux positifs: %d" % FP)
    print("Nombre faux negatifs: %d" % FN)

    #cr�er la matrice de confusion
    create_confusion_matrix(Y_test, Y_pred, model_name)
    
    #ajouter les donn�es pour la cr�ation de la courbe ROC
    Y_score = model.predict_proba(X_test)[:, 1]
    X_ROC.extend([Y_score])

    #ajouter les donn�es pour la cr�ation du tableau
    data = [model_name, accuracy, precision, recall, f, FP, FN, t]
    tableau.extend([data])

#----------------------------------------------------------------------------------------------------------
#
#                                             MOD�LES UTILIS�S
#
#----------------------------------------------------------------------------------------------------------
model_1=DecisionTreeClassifier()
model_2=RandomForestClassifier()
model_3=KNeighborsClassifier()
model_4=LogisticRegression(max_iter=200)    #nombre d'it�rations max augment� car sinon erreur
model_5=GaussianNB()
model_6=SVC(probability=True)    #probability = true pour avoir predict_proba, mais prends beaucoup plus de temps



#----------------------------------------------------------------------------------------------------------
#
#                                   ENTRAINEMENT DES MOD�LES - G�N�RAL
#
#----------------------------------------------------------------------------------------------------------

#D�TECTION: NORMALE OU ATTAQUE
print("-------- DETECTION: NORMALE OU ATTAQUE --------")

#MOD�LE 1 - DECISION TREE
print("MODELE 1 - DECISION_TREE")
train_model('DECISION_TREE_attaque', model_1, X_test, X_train, Y_test, Y_train)


#MOD�LE 2 - RANDOM FOREST
print("\nMODELE 2 - RANDOM FOREST")
train_model('RANDOM_FOREST_attaque', model_2, X_test, X_train, Y_test, Y_train)


#MOD�LE 3 - K-NEAREST NEIGHBORS (KNN)
print("\nMODELE 3 - K-NEAREST NEIGHBORS (KNN)")
train_model('KNN_attaque', model_3, X_test, X_train, Y_test, Y_train)


#MOD�LE 4 - LOGISTIC REGRESSION
print("\nMODELE 4 - LOGISTIC REGRESSION")
train_model('LOGISTIC_REGRESSION_attaque', model_4, X_test, X_train, Y_test, Y_train)


#MOD�LE 5 - NAIVE BAYES
print("\nMODELE 5 - NAIVE BAYES")
train_model('NAIVE_BAYES_attaque', model_5, X_test, X_train, Y_test, Y_train)


#MOD�LE 6 - SUPPORT VECTOR MACHINES (SVMS)
print("\nMODELE 6 - SUPPORT VECTOR MACHINES (SVMS)")
train_model('SVMS_attaque', model_6, X_test, X_train, Y_test, Y_train)

#cr�er la courbe ROC et le tableau de comparaison
create_ROC_curve_and_table('attaque', Y_test)



#----------------------------------------------------------------------------------------------------------
#
#                                   ENTRAINEMENT DES MOD�LES - DOS
#
#----------------------------------------------------------------------------------------------------------

print("\n\n-------- DETECTION: NORMALE OU DOS --------")

#MOD�LE 1 - DECISION TREE
print("MODELE 1 - DECISION_TREE")
train_model('DECISION_TREE_DoS', model_1, X_DoS_test, X_DoS_train, Y_DoS_test, Y_DoS_train)


#MOD�LE 2 - RANDOM FOREST
print("\nMODELE 2 - RANDOM FOREST")
train_model('RANDOM_FOREST_DoS', model_2, X_DoS_test, X_DoS_train, Y_DoS_test, Y_DoS_train)


#MOD�LE 3 - K-NEAREST NEIGHBORS (KNN)
print("\nMODELE 3 - K-NEAREST NEIGHBORS (KNN)")

train_model('KNN_DoS', model_3, X_DoS_test, X_DoS_train, Y_DoS_test, Y_DoS_train)


#MOD�LE 4 - LOGISTIC REGRESSION
print("\nMODELE 4 - LOGISTIC REGRESSION")
train_model('LOGISTIC_REGRESSION_DoS', model_4, X_DoS_test, X_DoS_train, Y_DoS_test, Y_DoS_train)


#MOD�LE 5 - NAIVE BAYES
print("\nMODELE 5 - NAIVE BAYES")
train_model('NAIVE_BAYES_DoS', model_5, X_DoS_test, X_DoS_train, Y_DoS_test, Y_DoS_train)


#MOD�LE 6 - SUPPORT VECTOR MACHINES (SVMS)
print("\nMODELE 6 - SUPPORT VECTOR MACHINES (SVMS)")
train_model('SVMS_DoS', model_6, X_DoS_test, X_DoS_train, Y_DoS_test, Y_DoS_train)

#cr�er la courbe ROC et le tableau de comparaison
create_ROC_curve_and_table('DoS', Y_DoS_test)



#----------------------------------------------------------------------------------------------------------
#
#                                   ENTRAINEMENT DES MOD�LES - PROBE
#
#----------------------------------------------------------------------------------------------------------

#D�TECTION: PROBE
print("\n\n-------- DETECTION: NORMALE OU PROBE --------")
#MOD�LE 1 - DECISION TREE
print("MODELE 1 - DECISION TREE")
train_model('DECISION_TREE_Probe', model_1, X_Probe_test, X_Probe_train, Y_Probe_test, Y_Probe_train)


#MOD�LE 2 - RANDOM FOREST
print("\nMODELE 2 - RANDOM FOREST")
train_model('RANDOM_FOREST_Probe', model_2, X_Probe_test, X_Probe_train, Y_Probe_test, Y_Probe_train)


#MOD�LE 3 - K-NEAREST NEIGHBORS (KNN)
print("\nMODELE 3 - K-NEAREST NEIGHBORS (KNN)")
train_model('KNN_Probe', model_3, X_Probe_test, X_Probe_train, Y_Probe_test, Y_Probe_train)


#MOD�LE 4 - LOGISTIC REGRESSION
print("\nMODELE 4 - LOGISTIC REGRESSION")
train_model('LOGISTIC_REGRESSION_Probe', model_4, X_Probe_test, X_Probe_train, Y_Probe_test, Y_Probe_train)


#MOD�LE 5 - NAIVE BAYES
print("\nMODELE 5 - NAIVE BAYES")
train_model('NAIVE_BAYES_Probe', model_5, X_Probe_test, X_Probe_train, Y_Probe_test, Y_Probe_train)


#MOD�LE 6 - SUPPORT VECTOR MACHINES (SVMS)
print("\nMODELE 6 - SUPPORT VECTOR MACHINES (SVMS)")
train_model('SVMS_Probe', model_6, X_Probe_test, X_Probe_train, Y_Probe_test, Y_Probe_train)

#cr�er la courbe ROC et le tableau de comparaison
create_ROC_curve_and_table('Probe', Y_Probe_test)


#----------------------------------------------------------------------------------------------------------
#
#                                   ENTRAINEMENT DES MOD�LES - R2L
#
#----------------------------------------------------------------------------------------------------------

#D�TECTION: R2L
print("\n\n-------- DETECTION: NORMALE OU R2L --------")

#MOD�LE 1 - DECISION TREE
print("MODELE 1 - DECISION TREE")
train_model('DECISION_TREE_R2L', model_1, X_R2L_test, X_R2L_train, Y_R2L_test, Y_R2L_train)


#MOD�LE 2 - RANDOM FOREST
print("\nMODELE 2 - RANDOM FOREST")
train_model('RANDOM_FOREST_R2L', model_2, X_R2L_test, X_R2L_train, Y_R2L_test, Y_R2L_train)


#MOD�LE 3 - K-NEAREST NEIGHBORS (KNN)
print("\nMODELE 3 - K-NEAREST NEIGHBORS (KNN)")
train_model('KNN_R2L', model_3, X_R2L_test, X_R2L_train, Y_R2L_test, Y_R2L_train)


#MOD�LE 4 - LOGISTIC REGRESSION
print("\nMODELE 4 - LOGISTIC REGRESSION")
train_model('LOGISTIC_REGRESSION_R2L', model_4, X_R2L_test, X_R2L_train, Y_R2L_test, Y_R2L_train)


#MOD�LE 5 - NAIVE BAYES
print("\nMODELE 5 - NAIVE BAYES")
train_model('NAIVE_BAYES_R2L', model_5, X_R2L_test, X_R2L_train, Y_R2L_test, Y_R2L_train)


#MOD�LE 6 - SUPPORT VECTOR MACHINES (SVMS)
print("\nMODELE 6 - SUPPORT VECTOR MACHINES (SVMS)")
train_model('SVMS_R2L', model_6, X_R2L_test, X_R2L_train, Y_R2L_test, Y_R2L_train)

#cr�er la courbe ROC et le tableau de comparaison
create_ROC_curve_and_table('R2L', Y_R2L_test)


#----------------------------------------------------------------------------------------------------------
#
#                                   ENTRAINEMENT DES MOD�LES - U2R
#
#----------------------------------------------------------------------------------------------------------

#D�TECTION: U2R
print("\n\n-------- DETECTION: NORMALE OU U2R --------")

#MOD�LE 1 - DECISION TREE
print("MODELE 1 - DECISION TREE")
train_model('DECISION_TREE_U2R', model_1, X_U2R_test, X_U2R_train, Y_U2R_test, Y_U2R_train)


#MOD�LE 2 - RANDOM FOREST
print("\nMODELE 2 - RANDOM FOREST")
train_model('RANDOM_FOREST_U2R', model_2, X_U2R_test, X_U2R_train, Y_U2R_test, Y_U2R_train)


#MOD�LE 3 - K-NEAREST NEIGHBORS (KNN)
print("\nMODELE 3 - K-NEAREST NEIGHBORS (KNN)")
train_model('KNN_U2R', model_3, X_U2R_test, X_U2R_train, Y_U2R_test, Y_U2R_train)


#MOD�LE 4 - LOGISTIC REGRESSION
print("\nMODELE 4 - LOGISTIC REGRESSION")
train_model('LOGISTIC_REGRESSION_U2R', model_4, X_U2R_test, X_U2R_train, Y_U2R_test, Y_U2R_train)


#MOD�LE 5 - NAIVE BAYES
print("\nMODELE 5 - NAIVE BAYES")
train_model('NAIVE_BAYES_U2R', model_5, X_U2R_test, X_U2R_train, Y_U2R_test, Y_U2R_train)


#MOD�LE 6 - SUPPORT VECTOR MACHINES (SVMS)
print("\nMODELE 6 - SUPPORT VECTOR MACHINES (SVMS)")
train_model('SVMS_U2R', model_6, X_U2R_test, X_U2R_train, Y_U2R_test, Y_U2R_train)


#cr�er la courbe ROC et le tableau de comparaison
create_ROC_curve_and_table('U2R', Y_U2R_test)