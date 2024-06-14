import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
'''Read and Understand DATA'''
#Load data
data = pd.read_csv('OnlineRetail.csv', encoding="ISO-8859-1")
print('Data Head')
print(data.head())

#Get data shape
print('Data shape ')
print(data.shape)

#Get data information
print('Data Info')
print(data.info())


#Describe data
print('Data Describe')
print(data.describe())

#Check null values
print('Check null values')
print(data.isnull().sum())

#Calculate missing values
df_null = (100 * (data.isnull().sum() / len(data))).round(2)
print('New is null values')
print(df_null)


#Drop rows with missing values
data= data.dropna()
print('New shape after dropping rows with missing values')
print(data.shape)

#Convert CustomerID column to String
data['CustomerID'] = data['CustomerID'].astype(str)
print('New CustomerID data type')
print(data.info())


#DATA PREPARATION
#Recency Frequency Monetary
#Monetary
data['Amount'] = data['Quantity']*data['UnitPrice']
monetary = data.groupby('CustomerID')['Amount'].sum()
monetary= monetary.reset_index()
print('Monetary value',monetary.head())


# Frequency

frequency = data.groupby('CustomerID')['InvoiceNo'].count()
frequency = frequency.reset_index()
frequency.columns = ['CustomerID', 'Frequency']
print('Frequency values',frequency.head())

# Merging Monetary,Frequency and Customer ID

merged_columns = pd.merge(monetary,frequency, on='CustomerID', how='inner')
print('Merged MF and CustomerID',merged_columns.head())

# New Attribute : Recency

# Convert to datetime to proper datatype

data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')



# Compute the maximum date to know the last transaction date

max_date = max(data['InvoiceDate'])
print('Max Date',max_date)


# Compute the difference between max date and transaction date

data['Diff'] = max_date - data['InvoiceDate']
print('Date differences',data.head())


# Compute last transaction date to get the recency of customers

last_date = data.groupby('CustomerID')['Diff'].min()
last_date = last_date.reset_index()
print('Last date',last_date.head())

# Extract number of days only

last_date['Diff'] = last_date['Diff'].dt.days
print('Last date',last_date.head())



# Merge tha dataframes to get the final RFM dataframe

m = pd.merge(merged_columns, last_date, on='CustomerID', how='inner')
m.columns = ['CustomerID', 'Amount', 'Frequency', 'Diff']
print('Merged Columns',m.head())

# Rescaling the attributes

rfm_df = m[['Amount', 'Frequency', 'Diff']]

# Instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['Amount', 'Frequency', 'Diff']
print('Standardized values',rfm_df_scaled.head())


# k-means with some arbitrary k

kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)

print('KMEANS labels',kmeans.labels_)

# Elbow-curve/SSD

ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
plt.plot(ssd)
plt.show()

# Silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


    

