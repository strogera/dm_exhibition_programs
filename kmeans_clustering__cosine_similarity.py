import pandas as pd
import csv
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.decomposition import TruncatedSVD 
from nltk.cluster import KMeansClusterer, cosine_distance


df=pd.read_csv('train_set.csv', sep='\t')        
transformer=TfidfTransformer()
vectorizer=TfidfVectorizer(stop_words='english')                                 
X=vectorizer.fit_transform(df['Content'])
X=transformer.fit_transform(X)
svd=TruncatedSVD(n_components=200, random_state=42)
X=svd.fit_transform(X)
X.tolist()

clusterer = KMeansClusterer(5, cosine_distance, initial_means=None)
clusters = clusterer.cluster(X, True, trace=False)

#every element of these arrays is the number of articles of this category that
#exists in the corresponding cluster ex. numOfBusinessAtCluster[0]=4 means that
#at cluster 0 there are 4 articles with the category 'Business'
numOfBusinessAtCluster=[0 for i in range(0, 5)] 
numOfPoliticsAtCluster=[0 for i in range(0, 5)]
numOfFootballAtCluster=[0 for i in range(0, 5)]
numOfFilmAtCluster=[0 for i in range(0, 5)]
numOfTechnologyAtCluster=[0 for i in range(0, 5)]

numOfElementsAtCluster=[0.0 for i in range(0, 5)]

for i in range(0, len(clusters)):
    if df['Category'][i]=='Business':
        numOfBusinessAtCluster[clusters[i]]+=1
    elif df['Category'][i]=='Politics':
        numOfPoliticsAtCluster[clusters[i]]+=1
    elif df['Category'][i]=='Football':
        numOfFootballAtCluster[clusters[i]]+=1
    elif df['Category'][i]=='Film':
        numOfFilmAtCluster[clusters[i]]+=1
    elif df['Category'][i]=='Technology':
        numOfTechnologyAtCluster[clusters[i]]+=1
    numOfElementsAtCluster[clusters[i]]+=1.0

#write the results at csv file
with open('clustering_KMeans.csv', 'wb') as csvfile:                             
    spamwriter = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)     
    spamwriter.writerow([' ', 'Business', 'Politics', 'Football', 'Film', 'Technology'])
    spamwriter.writerow(['Cluster1', numOfBusinessAtCluster[0]/numOfElementsAtCluster[0], numOfPoliticsAtCluster[0]/numOfElementsAtCluster[0], numOfFootballAtCluster[0]/numOfElementsAtCluster[0], numOfFilmAtCluster[0]/numOfElementsAtCluster[0], numOfTechnologyAtCluster[0]/numOfElementsAtCluster[0]])
    spamwriter.writerow(['Cluster2', numOfBusinessAtCluster[1]/numOfElementsAtCluster[1], numOfPoliticsAtCluster[1]/numOfElementsAtCluster[1], numOfFootballAtCluster[1]/numOfElementsAtCluster[1], numOfFilmAtCluster[1]/numOfElementsAtCluster[1], numOfTechnologyAtCluster[1]/numOfElementsAtCluster[1]])
    spamwriter.writerow(['Cluster3', numOfBusinessAtCluster[2]/numOfElementsAtCluster[2], numOfPoliticsAtCluster[2]/numOfElementsAtCluster[2], numOfFootballAtCluster[2]/numOfElementsAtCluster[2], numOfFilmAtCluster[2]/numOfElementsAtCluster[2], numOfTechnologyAtCluster[2]/numOfElementsAtCluster[2]])
    spamwriter.writerow(['Cluster4', numOfBusinessAtCluster[3]/numOfElementsAtCluster[3], numOfPoliticsAtCluster[3]/numOfElementsAtCluster[3], numOfFootballAtCluster[3]/numOfElementsAtCluster[3], numOfFilmAtCluster[3]/numOfElementsAtCluster[3], numOfTechnologyAtCluster[3]/numOfElementsAtCluster[3]])
    spamwriter.writerow(['Cluster5', numOfBusinessAtCluster[4]/numOfElementsAtCluster[4], numOfPoliticsAtCluster[4]/numOfElementsAtCluster[4], numOfFootballAtCluster[4]/numOfElementsAtCluster[4], numOfFilmAtCluster[4]/numOfElementsAtCluster[4], numOfTechnologyAtCluster[4]/numOfElementsAtCluster[4]])
