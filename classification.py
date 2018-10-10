import pandas as pd
import numpy as np
import csv
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt

df=pd.read_csv('train.tsv', sep='\t')
df2=pd.read_csv('test.tsv', sep='\t')

#convert categorical values to numerical
lb=LabelBinarizer()
att1=lb.fit_transform(df['Attribute1'])
att3=lb.fit_transform(df['Attribute3'])
att4=lb.fit_transform(df['Attribute4'])
att6=lb.fit_transform(df['Attribute6'])
att7=lb.fit_transform(df['Attribute7'])
att9=lb.fit_transform(df['Attribute9'])
att10=lb.fit_transform(df['Attribute10'])
att12=lb.fit_transform(df['Attribute12'])
att14=lb.fit_transform(df['Attribute14'])
att15=lb.fit_transform(df['Attribute15'])
att17=lb.fit_transform(df['Attribute17'])
att19=lb.fit_transform(df['Attribute19'])
att20=lb.fit_transform(df['Attribute20'])


tatt1=lb.fit_transform(df2['Attribute1'])
tatt3=lb.fit_transform(df2['Attribute3'])
tatt4=lb.fit_transform(df2['Attribute4'])
tatt6=lb.fit_transform(df2['Attribute6'])
tatt7=lb.fit_transform(df2['Attribute7'])
tatt9=lb.fit_transform(df2['Attribute9'])
tatt10=lb.fit_transform(df2['Attribute10'])
tatt12=lb.fit_transform(df2['Attribute12'])
tatt14=lb.fit_transform(df2['Attribute14'])
tatt15=lb.fit_transform(df2['Attribute15'])
tatt17=lb.fit_transform(df2['Attribute17'])
tatt19=lb.fit_transform(df2['Attribute19'])
tatt20=lb.fit_transform(df2['Attribute20'])

#change from numerical to categorical
#some numerical attributes don't need to be changed since they take less that 5 different values
#usefull for information gain
dfCategorical=df[:]
dfCategorical["Attribute2"]=pd.qcut(dfCategorical["Attribute2"], 5) #, labels=[1, 2, 3, 4, 5])
dfCategorical["Attribute5"]=pd.qcut(dfCategorical["Attribute5"], 5) 
dfCategorical["Attribute13"]=pd.qcut(dfCategorical["Attribute13"], 5) 

dfOriginal=df[:]

attcount=20
firstIterationFlag=True
svcFeaturesRemovedAccScore=[]
rfFeaturesRemovedAccScore=[]
gnbFeaturesRemovedAccScore=[]
featureRemoved=[]
igRemoved=[]
while(attcount>0):
    #have all the numerical values in one array
    d=[[None]]*len(df)
    for i in range(0, len(df)):
        d[i]=[]
        for attribute in df.columns:
            #if numerical
            if attribute=='Attribute2' or attribute=='Attribute5' or attribute=='Attribute8' or attribute=='Attribute11' or attribute=='Attribute13' or attribute=='Attribute16' or attribute=='Attribute18':
                d[i]+=[df[attribute][i]]
            #categorical
            else: 
                if attribute=='Attribute1':
                    d[i]+=list(att1[i])
                elif attribute=='Attribute3':
                    d[i]+=list(att3[i])
                elif attribute=='Attribute4':
                    d[i]+=list(att4[i])
                elif attribute=='Attribute6':
                    d[i]+=list(att6[i])
                elif attribute=='Attribute7':
                    d[i]+=list(att7[i])
                elif attribute=='Attribute9':
                    d[i]+=list(att9[i])
                elif attribute=='Attribute10':
                    d[i]+=list(att10[i])
                elif attribute=='Attribute12':
                    d[i]+=list(att12[i])
                elif attribute=='Attribute14':
                    d[i]+=list(att14[i])
                elif attribute=='Attribute15':
                    d[i]+=list(att15[i])
                elif attribute=='Attribute17':
                    d[i]+=list(att17[i])
                elif attribute=='Attribute19':
                    d[i]+=list(att19[i])
                elif attribute=='Attribute20':
                    d[i]+=list(att20[i])

    if firstIterationFlag: 
        #keep a copy of the original data for the classification method, usefull at the test prediction
        dOriginal=d[:]

    #10-fold
    kf = KFold(n_splits=10)
    svcAccScore=0
    rfAccScore=0
    gnbAccScore=0
    fold = 0
    for train_index, test_index in kf.split(d):
        clf = svm.SVC()
        clf.fit(np.array(d)[train_index], np.array(df['Label'])[train_index])
        svcPredicted=clf.predict(np.array(d)[test_index])
        svcAccScore+=metrics.accuracy_score(np.array(df['Label'])[test_index], svcPredicted)

        clf=RandomForestClassifier(n_estimators=10)
        clf.fit(np.array(d)[train_index], np.array(df['Label'])[train_index])
        rfPredicted=clf.predict(np.array(d)[test_index])
        rfAccScore+=metrics.accuracy_score(np.array(df['Label'])[test_index], rfPredicted)

        clf=GaussianNB()
        clf.fit(np.array(d)[train_index], np.array(df['Label'])[train_index])
        gnbPredicted=clf.predict(np.array(d)[test_index])
        gnbAccScore+=metrics.accuracy_score(np.array(df['Label'])[test_index], gnbPredicted)

        fold+=1
    svcAccScore/=float(fold)
    rfAccScore/=float(fold)
    gnbAccScore/=float(fold)
    svcFeaturesRemovedAccScore.append(svcAccScore)
    rfFeaturesRemovedAccScore.append(rfAccScore)
    gnbFeaturesRemovedAccScore.append(gnbAccScore)

    #hold the initial scores to run the prediction on the test file with the best method
    if firstIterationFlag:
        initialSvcAccScore=svcAccScore
        initialRfAccScore=rfAccScore
        initialGnbAccScore=gnbAccScore

        #write the accuracy results to csv file
        with open('EvaluationMetric_10fold.csv', 'wb') as csvfile:                             
            spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)     
            spamwriter.writerow(['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM'])
            spamwriter.writerow(['Accuracy', initialGnbAccScore,  initialRfAccScore, initialSvcAccScore])



    def entropy(b, p): #p an array with the p(x_i)
        sumEntropy=0
        for x in p:
            sumEntropy+=x*math.log(x, b)
        return -sumEntropy
        
    def informationGain(b1, xp, b2, tp): 
        #xp an array with p(x_i)(good/bad) in label, tp an array with p(value_i) in attribute
        #b1, b2 the number of different values respectively
        summ=0
        for x in tp:
            summ+=x*entropy(b2, [x])
        return entropy(b1, xp)-summ

    if firstIterationFlag:
        iGScores=[]
    #calculate information gain for each feature
    for att in dfCategorical.columns:
        if att=='Label' or att=='Id':
            continue
        if firstIterationFlag:
            iGScores.append(informationGain(len(df['Label'].value_counts()), [len(df[df['Label']==1])/float(len(df)), len(df[df['Label']==2])/float(len(df))], len(dfCategorical[att].value_counts()), [float(x)/len(df) for x in dfCategorical[att].value_counts()]))


    if iGScores:
        ind=np.argmin(iGScores)
        featureRemoved.append(df.columns[ind])
        igRemoved.append(iGScores[ind])
        df.drop(df.columns[ind], axis=1, inplace=True)
        dfCategorical.drop(dfCategorical.columns[ind], axis=1, inplace=True)
        del iGScores[ind]
    attcount-=1
    firstIterationFlag=False
        


t=[[None]]*len(df2)
for i in range(0, len(df2)):
    t[i]=list(tatt1[i])
    t[i]+=[df2['Attribute2'][i]]
    t[i]+=list(tatt3[i])
    t[i]+=list(tatt4[i])
    t[i]+=[df2['Attribute5'][i]]
    t[i]+=list(tatt6[i])
    t[i]+=list(tatt7[i])
    t[i]+=[df2['Attribute8'][i]]
    t[i]+=list(tatt9[i])
    t[i]+=list(tatt10[i])
    t[i]+=[df2['Attribute11'][i]]
    t[i]+=list(tatt12[i])
    t[i]+=[df2['Attribute13'][i]]
    t[i]+=list(tatt14[i])
    t[i]+=list(tatt15[i])
    t[i]+=[df2['Attribute16'][i]]
    t[i]+=list(tatt17[i])
    t[i]+=[df2['Attribute18'][i]]
    t[i]+=list(tatt19[i])
    t[i]+=list(tatt20[i])
    
#find the classification method with the best accuracy score
if initialSvcAccScore>=initialRfAccScore and initialSvcAccScore>=initialGnbAccScore:
    clf=svm.SVC()
    plt.bar([x for x in range(19)], svcFeaturesRemovedAccScore[1:])
elif initialRfAccScore>=initialGnbAccScore:
    clf=RandomForestClassifier(n_estimators=10)
    plt.bar([x for x in range(19)], rfFeaturesRemovedAccScore[1:])
else:
    clf=GaussianNB()
    plt.bar([x for x in range(19)], gnbFeaturesRemovedAccScore[1:])
plt.xlabel('# of features removed')
plt.ylabel('Accuracy')
plt.savefig('./plots/AccuracyPlot')

with open('FeaturesRemoved.csv', 'wb') as csvfile:                             
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)     
    spamwriter.writerow(['Feature','Information Gain'])
    for x in range(len(featureRemoved)):
        spamwriter.writerow([featureRemoved[x],igRemoved[x]])


#predict the test file results
clf.fit(dOriginal, dfOriginal["Label"])
predicted=clf.predict(t)

#write the prediction results to csv file
with open('testSet_Predictions.csv', 'wb') as csvfile:                             
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)     
    spamwriter.writerow(['Id', 'Predicted_Label'])
    for i in range(len(df2)):
        spamwriter.writerow([df2['Id'][i], predicted[i]])

