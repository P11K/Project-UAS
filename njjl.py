import pandas as pd
from sklearn.cluster import KMeans
# from sklearn.feature_extraction.text import CountVectorizer

data_buku = pd.read_excel('Mall_Customers.xlsx')
df = data_buku

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
# print(df.head())
    
if 'id' in df.columns:
    x = df.drop(['id'], axis=1)
    # Continue with your code
else:
    print("The 'id' column is not present in the DataFrame.")


# vectorizer = CountVectorizer()  
# X = vectorizer.fit_transform(x['Gender']+x['Profession'])

kmeans = KMeans(n_clusters= 3, init='random')
kmeans.fit(df)

print(kmeans.cluster_centers_)

df["Klaster"] = kmeans.labels_
kluster = kmeans.labels_

# print(kluster)
print(df)