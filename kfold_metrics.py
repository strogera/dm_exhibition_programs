import pandas as pd                                                              
import numpy as np
import csv                                                                       
from sklearn.feature_extraction.text import TfidfVectorizer                      
from sklearn.feature_extraction.text import TfidfTransformer                     
from sklearn.decomposition import TruncatedSVD  
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

train_set=pd.read_csv('train_set.csv', sep='\t')                                        
test_set=pd.read_csv('test_set.csv', sep='\t')                                        
transformer=TfidfTransformer()                                                   
vectorizer=TfidfVectorizer(stop_words='english')                                 
svd=TruncatedSVD(n_components=200, random_state=42)                              

X_train=vectorizer.fit_transform(train_set['Content'])                                        
X_train=transformer.fit_transform(X_train)                                                   
X_train=svd.fit_transform(X_train)                                                           
Y_train=train_set['Category']

X_test=vectorizer.fit_transform(test_set['Content'])                                        
X_test=transformer.fit_transform(X_test)                                                   
X_test=svd.fit_transform(X_test)                                                           
X_test.tolist()   




kf = KFold(n_splits=10)
fold=0
svmAccuracy=0
rfAccuracy=0
gnbAccuracy=0
svmPrecision=0
rfPrecision=0
gnbPrecision=0
svmRecall=0
rfRecall=0
gnbRecall=0
svmFMeasure=0
rfFMeasure=0
gnbFMeasure=0

for train_index, test_index in kf.split(X_train):

    clf=svm.SVC(C=1.0)
    clf.fit(X_train[train_index], np.array(Y_train)[train_index])
    svmPred=clf.predict(X_train[test_index])
    svmAccuracy+=metrics.accuracy_score(np.array(Y_train)[test_index], svmPred)
    svmPrecision+=metrics.precision_score(np.array(Y_train)[test_index], svmPred, average='micro')
    svmRecall+=metrics.recall_score(np.array(Y_train)[test_index], svmPred, average='micro')
    svmFMeasure+=metrics.f1_score(np.array(Y_train)[test_index], svmPred, average='micro')

    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train[train_index], np.array(Y_train)[train_index])
    rfcPred=clf.predict(X_train[test_index])
    rfAccuracy+=metrics.accuracy_score(np.array(Y_train)[test_index], rfcPred)
    rfPrecision+=metrics.precision_score(np.array(Y_train)[test_index], rfcPred, average='micro')
    rfRecall+=metrics.recall_score(np.array(Y_train)[test_index], rfcPred, average='micro')
    rfFMeasure+=metrics.f1_score(np.array(Y_train)[test_index], rfcPred, average='micro')

    clf=GaussianNB()
    clf.fit(X_train[train_index], np.array(Y_train)[train_index])
    gnbPred=clf.predict(X_train[test_index])
    gnbAccuracy+=metrics.accuracy_score(np.array(Y_train)[test_index], gnbPred)
    gnbPrecision+=metrics.precision_score(np.array(Y_train)[test_index], gnbPred, average='micro')
    gnbRecall+=metrics.recall_score(np.array(Y_train)[test_index], gnbPred, average='micro')
    gnbFMeasure+=metrics.f1_score(np.array(Y_train)[test_index], gnbPred, average='micro')

    fold+=1
    


with open('EvaluationMetric_10fold.csv', 'wb') as csvfile:                             
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)     
    spamwriter.writerow(['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM'])
    spamwriter.writerow(['Accuracy', gnbAccuracy/fold,  rfAccuracy/fold, svmAccuracy/fold])
    spamwriter.writerow(['Precision', gnbPrecision/fold,  rfPrecision/fold, svmPrecision/fold])
    spamwriter.writerow(['Recall', gnbRecall/fold,  rfRecall/fold, svmRecall/fold])
    spamwriter.writerow(['F-Measure', gnbFMeasure/fold,  rfFMeasure/fold, svmFMeasure/fold])

if svmAccuracy>=rfAccuracy and svmAccuracy>=gnbAccuracy:
    clf=svm.SVC(C=1.0)
elif rfAccuracy>=gnbAccuracy:
    clf=RandomForestClassifier(n_estimators=100)
else:
    clf=GaussianNB()

clf.fit(X_train, train_set['Category'])
predicted=clf.predict(X_test)

with open('testSet_categories.csv', 'wb') as csvfile:                             
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)     
    spamwriter.writerow(['Id', 'Predicted_Category'])
    for i in range(len(test_set)):
        spamwriter.writerow([test_set['Id'][i], predicted[i]])

