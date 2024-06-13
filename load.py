import pandas as pd
'''Read and Understand DATA'''
#Load data
data = pd.read_csv('OnlineRetail.csv',encoding='latin1')
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

'''DATA PREPARATION'''
'''Recency Frequency Monetary'''

#New attribute:Monetary(revenue contributed)
data['Amount']= data['Quantity']*data['UnitPrice']
print('Check new Amount Column')
print(data.head())

'''
#Customer spending the most
data_monetary = data.groupby('CustomerID')['Amount'].sum()
print('Monetary Value ')
print(data_monetary)
#Customer 12347 has spent the most amount


#Product sold the most
product_monetary = data.groupby('Description')['Amount'].sum()
print('Monetary Value  of Products')
print(product_monetary)

#Country selling product the most
country_monetary = data.groupby('Country')['Amount'].sum()
print('Monetary Value  of Country')
print(country_monetary)'''


# checking which customer makes the most sales
customer_monitoring = data.groupby('CustomerID')['Amount'].sum()
customer_most_purchase = customer_monitoring.idxmax()
print("This is the customer who made the most purchases",customer_most_purchase)
#Find max amount
amount_bought = customer_monitoring.max()
print("This is the amount bought",amount_bought)



#  which product is sold the most
product_monitoring = data.groupby('Description')['Quantity'].sum()
product_most_sold = product_monitoring.idxmax()
print("This is the most sold product quantity",product_most_sold)
# Find the amount sold for the most popular product
amount_sold = product_monitoring.max()
print("This is the amount sold for most popular product",amount_sold)


# what region is most product sold
country_monitoring = data.groupby('Country')['Amount'].sum()
country_most_sales = country_monitoring.idxmax()
print("This is the country with most sales",country_most_sales)

#Find Amount
max_country_amount = country_monitoring.max()
print('This is the amount of highest country sales',max_country_amount)

''' FREQUENCY OF ITEMS SOLD'''
#Frequently bought product
frequent_sold= data.groupby('Description')['InvoiceNo'].count()
print('This is the frequently sold product',frequent_sold)

#Recency

#Convert date and time
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')

#Compute max date to know last transaction date
max_date= max(data['InvoiceDate'])
print('Max date is',max_date)

#Compute the min date
min_date= min(data['InvoiceDate'])
print('Min date is',min_date)

#Find sales days
days= max_date-min_date
print(days)

#Find sales for last 30 days
from datetime import timedelta
new_min_date = max_date - timedelta(days=30)
last_30_days_sales =  data[(data['InvoiceDate'] >= new_min_date) & (data['InvoiceDate'] <= max_date)]['Amount'].count()
print('Print last day sales',last_30_days_sales) 
#Find optimal number of K in Kmeans Clustering
