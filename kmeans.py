import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Use 'unicode_escape' encoding to avoid UnicodeDecodeError
data = pd.read_csv('OnlineRetail.csv', encoding='unicode_escape')

# Data Preparation
print(data.head(3))
print(data.info())

# Feature selection
# Three Columns used to cluster defined using usecols parameter from the dataset
data = pd.read_csv('OnlineRetail.csv', usecols=['UnitPrice', 'Quantity', 'StockCode'], encoding='unicode_escape')
print('Only Three Columns used to cluster:\n**********************')
print(data.head())

# Visualize Data
scatter_plot = sns.scatterplot(data=data, x='UnitPrice', y='Quantity', hue='StockCode')
plt.title('Clustering Columns')
scatter_plot.legend(loc='upper right')
plt.show()

# Normalize
x_train, x_test, y_train, y_test = train_test_split(data[['UnitPrice', 'Quantity']], data['StockCode'], test_size=0.3, random_state=0)
x_train_norm = preprocessing.normalize(x_train)
x_test_norm = preprocessing.normalize(x_test)

# Function to perform KMeans clustering and capture performance results
def kmeans_performance(x_data, range_clusters):
    silhouette_scores = []
    for n_clusters in range_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        labels = kmeans.fit_predict(x_data)
        score = silhouette_score(x_data, labels, metric='euclidean')
        silhouette_scores.append((n_clusters, score))
        print(f'Number of clusters: {n_clusters}')
        print(f'Silhouette Score: {score:.4f}\n')
    return silhouette_scores

# Run the performance function
range_clusters = range(2, 10)  # You can change the range as needed
performance_results = kmeans_performance(x_train_norm, range_clusters)

# Visualize the performance results
performance_df = pd.DataFrame(performance_results, columns=['Number of Clusters', 'Silhouette Score'])
sns.lineplot(data=performance_df, x='Number of Clusters', y='Silhouette Score')
plt.title('KMeans Clustering Performance')
plt.show()
