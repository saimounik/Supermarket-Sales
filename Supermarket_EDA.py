#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()


# In[8]:


import plotly.express as px


# In[9]:


# conda install seaborn


# In[10]:


sns.set_style("darkgrid")


# In[11]:


# Importing all required libraries
import warnings
warnings.filterwarnings("ignore")

# The normal imports
import numpy as np
from numpy.random import randn
import pandas as pd

# Import the stats librayr from numpy
from scipy import stats

# These are the plotting modules adn libraries we'll use:
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Command so that plots appear in the iPython Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


# Importing Dataset to the enviorment
sm = pd.read_csv("/Users/christopherdevlin/Downloads/supermarket_sales - Sheet1.csv")


# In[13]:


# Exploring Dataset before proceeding with analysis
sm


# In[14]:


# Exploring dataset by describing it to see statisctical values of quantitative variables in dataset
sm.describe()


# In[15]:


# Extracting Year, Month and Day from dataset and creating new columns to perform time series analysis
# Here first i converted the column Date is in object formate. I converted it into Date Time formate

sm.info()
sm["Date"] = pd.to_datetime(sm["Date"])
sm.Date


# In[16]:


# Year
sm["Date"].dt.year

# Month
sm["Date"].dt.month

# Day
sm["Date"].dt.day


# In[17]:


# Creating Year, Month and Day Columns

sm["Year"] = sm["Date"].dt.year
sm["Month"] = sm["Date"].dt.month
sm["Day"] = sm["Date"].dt.day

# Checking dataset by prinitng it to see the changes
sm


# In[18]:


# Now assigning names to our newly created month column

month_names = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}
sm['Month'] = sm['Month'].map(month_names)

# Checking dataset by prinitng it to see the changes
sm


# In[19]:


# Creating Year, Month and Day Columns in the dataset

# Assuming 'sm' is your DataFrame
# Adding year column
sm["Year"] = sm["Date"].dt.year
# Assuming 'sm' is your DataFrame
# Adding Month column 
sm["Month"] = sm["Date"].dt.month
# Assuming 'sm' is your DataFrame
# Adding Day column 
sm["Day"] = sm["Date"].dt.day

#Printing the dataset to check the new columns
sm


# # Here are the Conclusions of our Analysis
# 
# #1) The highest number of branches are in branch 'A'.
# 
# #2) Though 'A' has the highest number of branches, Branch 'C' contributing more revenue through sales.
# 
# #3) The highest earning productline in business is 'Food and Beverages', It is contributing 17.4% of sales.
# 
# #4) Through time series analysis we found that the highest amount of sales hit in January and lowest is February.
# 
# #5) Through comparison of sales vs tax, We found that branch 'C' is paying highest taxes because of it's high volume of sales.
# 
# #6) The average rating of the all sales is 6.5 / 10.
# 
# #7) Through heat map, We found that the highest sale recorded in 11nth of February.
# 
# #8) Productline Health and Beauty earned highest gross income among all.
# 
# #9) Most used payment method is E-Wallet.
# 
# #10) Product line electronic accessories have purchased the highest quantity.
# 
# #11) The purchases of Males are higher then females.
# 
# #12) Health and Beauty is the only product line where female purchases are higher then males.
# 
# #13) Branch 'A' has highest male purchases and banch 'B' has highest female purchases
# 
# #14) Food and beverages has spending high amount for selling goods, It has highest cogs amount.
# 
# #15) Females are earning more then males. Females has higher income.
# 
# #16) Members are the highest customer segment in branch 'C' and normal customers are highest segment in branch 'A'.

# # Total Number of Branches

# In[46]:


# Creating barplot of count of branches
sns.catplot(data=sm, x="Branch", kind="count")


# # Bar Chart: Sales distribution by branch or city

# In[24]:


# Bar chart - Sales distribution by branch
# Creating barplot of distrinbution of branches 
plt.figure
sns.barplot(x='Branch', y='Total', data=sm, estimator=sum, ci=None)
plt.title('Sales Distribution by Branch')
plt.xlabel('Branch')
plt.ylabel('Total Sales')

# Showing the plot
plt.show()


# # Pie chart - Percentage of sales by product line

# In[25]:


# Creating pie-chart of percentage of sales by product line
plt.figure
sales_productline = sm.groupby('Product line')['Total'].sum()
plt.pie(sales_productline, labels=sales_productline.index, autopct='%1.1f%%')
plt.title('Percentage of Sales by Product Line')

# Showing the plot
plt.show()


# # Line chart - Sales trend over time

# In[26]:


# Create line chart of sales trend
sm['Date'] = pd.to_datetime(sm['Date'])
monthly_sales = sm.resample('M', on='Date')['Total'].sum()
plt.figure
monthly_sales.plot(marker='o')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')

# Showing the plot
plt.show()


# # Scatter plot - Relationship between quantity sold and unit price

# In[27]:


# Creating scatter plot to check relationship between quantity sold unit price
plt.figure
sns.scatterplot(x='Quantity', y='Unit price', data=sm)
plt.title('Relationship between Quantity Sold and Unit Price')
plt.xlabel('Quantity Sold')
plt.ylabel('Unit Price')

# Showing plot
plt.show()


# # Comparison of sales and taxes by branch

# In[28]:


# Creating bar plot to compare sales and tax
salestax_bybranch = sm.groupby('Branch')[['Total', 'Tax 5%']].sum()
salestax_bybranch.plot(kind='bar', stacked=True)
plt.title('Comparison of Sales and Taxes by Branch')
plt.xlabel('Branch')
plt.ylabel('Amount')

# Showing plot
plt.show()


# # Distribution of sales ratings

# In[29]:


# Creating histogram by using histplot to see distributio of sales rating
sns.histplot(sm.Rating, kde=True)
plt.xlabel('Sales Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Sales Ratings')

# SHowing plot
plt.show()


# # Distribution of sales ratings by branch - using Matplotlib

# In[30]:


# Grouping data by branch
branches = sm.groupby('Branch')['Rating']

# Creating separate histograms for each branch
plt.figure
colors = ['blue', 'orange', 'green'] 
for i, (branch, ratings) in enumerate(branches):
    plt.hist(ratings, bins=10, alpha=0.5, label=f'Branch {branch}', color=colors[i])

plt.xlabel('Sales Ratings')
plt.ylabel('Frequency')
plt.title('Distribution of Sales Ratings by Branch')
plt.legend()

# Showing plot
plt.show()


# # Heatmap: Sales performance by date and time

# In[31]:


# Extracting Year, Month and Day from dataset and creating new columns to perform time series analysis
# Here first i converted the column Date is in object formate. I converted it into Date Time formate

sm.info()
sm["Date"] = pd.to_datetime(sm["Date"])
sm.Date


# In[32]:


# Year
# Defining date
sm["Date"].dt.year

# Month
# Defining month
sm["Date"].dt.month

# Day
# Defining day
sm["Date"].dt.day


# In[33]:


# Creating Year, Month and Day Columns in the dataset

# Assuming 'sm' is your DataFrame
# Adding year column
sm["Year"] = sm["Date"].dt.year
# Assuming 'sm' is your DataFrame
# Adding Month column 
sm["Month"] = sm["Date"].dt.month
# Assuming 'sm' is your DataFrame
# Adding Day column 
sm["Day"] = sm["Date"].dt.day

#Printing the dataset to check the new columns
sm


# In[34]:


# Creating pivot table for heatmap
s_pvt = sm.pivot_table(index='Month', columns='Day', values='Total')

# Creating heatmap
plt.figure
sns.heatmap(s_pvt, cmap='YlGnBu',fmt='g')
plt.title('Sales Performance by Month and Day')
plt.xlabel('Date')
plt.ylabel('Month')

# Showing plot
plt.show()


# # Box Plot: Distribution Gross income across product lines

# In[35]:


# Creating boxplot of gross income distribution
plt.figure
sns.boxplot(x='gross income', y='Product line', data=sm)
plt.xlabel('Gross Income')
plt.ylabel('Product line')
plt.title('Distribution of Gross Income Across Product Lines')

# Showing plot
plt.show()


# # Stacked Area Chart: Revenue contribution by branch over time

# In[36]:


# Creating pivot table for bar chart
pivot_data = sm.pivot_table(index='Date', columns='Branch', values='Total', fill_value=0)

plt.figure
plt.stackplot(pivot_data.index, pivot_data.values.T, labels=pivot_data.columns)
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.title('Revenue Contribution by Branch Over Time')
plt.legend(loc='upper left')

# Showing plot
plt.show()


# # Donut Chart: Payment method distribution

# In[37]:


# Calculating payment method frequencies
payment_counts = sm['Payment'].value_counts()

# Extracing labels (payment methods) and counts for the chart
payment_methods = payment_counts.index.tolist()
counts = payment_counts.values.tolist()

# Creating a donut chart using Matplotlib
plt.figure(figsize=(8, 8))
plt.pie(counts, labels=payment_methods, autopct='%1.1f%%', startangle=90)

# Drawing a circle to create a donut chart
circle = plt.Circle((0, 0), 0.7, color='white')
plt.gca().add_artist(circle)
plt.title('Payment Method Distribution')
plt.axis('equal')
plt.show()


# # Daywise Sales Trend 

# In[38]:


# Creating and showing plot of sales trend
sm['Total'].plot(legend=True,figsize=(10,4))


# # Heatmap

# In[39]:


# Creating and showing heatmap
sns.heatmap(np.round(sm.corr(), 2), annot= True)


# # Which product line has purchased the highest quanity ?

# In[40]:


# Creating barplot of productlines quantity wise
plt.figure
sns.barplot(x='Product line', y='Quantity', data=sm)
plt.xlabel('Product line')
plt.ylabel('Quantity')
plt.xticks(rotation = 60)

# Showing plot
plt.show()


# # Relationship between rating and sales using scatterplot(plotly lib)

# In[42]:


import plotly.express as px
fig = px.scatter(sm, x='Total',y='Rating',color = "Total",hover_name = 'Branch')
fig.show()


# # Pairwise relationships between variables within a dataset -Pairplot

# In[47]:


# Creating pairplot
sns.pairplot(sm)


# # Spending pattern of females and males as per product category

# In[48]:


# Creating countplot of monthly transactions grouping by product line
plt.figure
plt.title('Total Monthly transaction by Gender')
plt.xticks(rotation = 60)
sns.countplot(x=sm['Product line'], hue = sm.Gender)


# # Histogram of Attributes

# In[49]:


# Creating Histogram
sm.hist(figsize=(10,10))


# # Genderwise sales per branch - Stripplot

# In[50]:


# Creating strip plot of sales genderwise
sns.stripplot(y = 'Branch', x = 'Total',data = sm, hue = 'Gender')


# # Branchwise rating as per Customer Type - Swarmplot

# In[51]:


# Creating swarmplot of customer segment
sns.swarmplot(x = 'Customer type', y = 'Rating', hue = 'Branch', data = sm).set_title('Customer Type')


# # Cogs as per Product Line - Boxenplot

# In[52]:


# Creating boxplot of productlines
sns.boxenplot(y = 'Product line', x = 'cogs',data = sm)


# # Distribution of Sales Across Branches - Voilinplot

# In[53]:


# Creating the violin plot of sales grouping by branches
plt.figure(figsize=(8, 6))
sns.violinplot(x='Branch', y='Total', data=sm)
plt.title('Distribution of Sales Across Branches')
plt.xlabel('Branch')
plt.ylabel('Total')

# Showing plot
plt.show()


# # Male vs Female income

# In[54]:


# Creating boxplot of grossincome grouping by gender
sns.boxplot(x=sm['Gender'],y=sm['gross income'])
plt.title('Male vs Female Gross Income')
plt.figure(figsize=(12, 7))


# # Customertype per Branch

# In[55]:


# Creating countplot of customer segment grouping bt branches 
sns.countplot(x = 'Customer type', data = sm, hue = 'Branch').set_title('Customer Type By Branch')


# # Total sales month-wise

# In[56]:


# Assuming 'sm' is your DataFrame
# Group by 'Month' and 'Total' to get the total sales


monthlysales = sm.groupby('Month')['Total'].sum().reset_index()

# Printing the result
monthlysales


# # Highest selling product line

# In[57]:


# Assuming 'sm' is your DataFrame
# Group by 'Product Line' and 'Quantity' to get the highest selling product line
quantcount = sm.groupby('Product line')['Quantity'].sum().reset_index()

#Printing the result
quantcount.max()


# # Which payment method is using by most of the people

# In[58]:


# Assuming 'sm' is your DataFrame
# Group by 'Payment' and 'total' to get the most used payment method
most_pmethod = sm.groupby('Payment')['Total'].sum().reset_index()

# Printing the result
most_pmethod.max()


# # Most valuble customer segment as per gross income

# In[59]:


# Assuming 'sm' is your DataFrame
# Group by 'Customer type' and 'gross income' to get the most valuble customer segment
most_vsegment = sm.groupby('Customer type')['gross income'].sum().reset_index()

# Printing the result
most_vsegment


# # Who purchases more ? Men or Women ?

# In[60]:


# Assuming 'sm' is your DataFrame
# Group by 'Gender' and 'Total' find out which gender has more purchases
g_sell = sm.groupby('Gender')['Total'].sum().reset_index()

# Printing the result
g_sell.max()


# # Interests of Men & Women - Product Wise

# In[61]:


# Assuming 'sm' is your DataFrame
# Finding the total count value of males in each product line to get total product count gender wise

sm[['Product line','Gender']][(sm['Gender']=='Male')].value_counts().reset_index()

#Printing the result
print(sm[['Product line','Gender']][(sm['Gender']=='Male')].value_counts().reset_index())


# In[62]:


# Assuming 'sm' is your DataFrame
# Finding the total count value of females in each product line to get total product count gender wise

sm[['Product line','Gender']][(sm['Gender']=='Female')].value_counts().reset_index()

#Printing the result
print(sm[['Product line','Gender']][(sm['Gender']=='Female')].value_counts().reset_index())


# # Most expensive product among all

# In[63]:


# Assuming 'sm' is your DataFrame
# Finding the expensive product in total products

expensiveproduct = sm.loc[sm["Total"].idxmax()]

# Printing the result
print(expensiveproduct)


# # Relationship between rating and sales

# In[64]:


# Performing Correlation analysis to get the relationship between two columns rating and total 
sm[['Rating','Total']].corr()

#This says the rating and quantity is inversly propotional to each other. When rating increases the total sales decreased and vice versa


# # Best vs Worst performing branches

# In[65]:


# Assuming 'sm' is your DataFrame
# # Group by 'Branch' and 'Total' to get the best performing branch
best_branch = sm.groupby('Branch')['Total'].max().reset_index()

#printing the result
best_branch.max()


# In[79]:


#Worst performing branch# What is the most popular product line in each city?

city_product_counts = sm.groupby(['City', 'Product line'])['Invoice ID'].count().unstack()
most_popular_products = city_product_counts.idxmax(axis=1)
print(most_popular_products)


# In[80]:


# Calculate the average rating by customer type and gender

avg_rating = sm.groupby(['Customer type', 'Gender'])['Rating'].mean()
print(avg_rating)


# In[81]:


# Calculate the most common payment method

common_payment_method = sm['Payment'].mode().values[0]
print(common_payment_method)


# In[82]:


# Calculate the average gross income by product line

avg_gross_income_by_product = sm.groupby('Product line')['gross income'].mean()
print(avg_gross_income_by_product)


# In[83]:


# Calculate the distribution of customer types in each city

city_customer_distribution = sm.groupby(['City', 'Customer type'])['Invoice ID'].count().unstack()
print(city_customer_distribution)


# In[84]:


# Median Total Purchase Price and percentage above 

#calculating the median total purchase price by product
avg_total_by_product = sm.groupby('Product line')['Total'].mean()
print(avg_total_by_product)


# In[86]:


# Find purchases above median for each product line
purchases_above_median = sm['Total'] > sm['Product line'].map(avg_total_by_product)

# Calculate the percentage of purchases above the median for each product line
percentage_above_median = purchases_above_median.groupby(sm['Product line']).mean() * 100

# Print the result
print("Percentage of purchases above median total by product line:")
print(percentage_above_median)


# In[87]:


# What product coincides with the highest total price

# Find the product line with the highest total price
max_total_product_line = sm.loc[sm['Total'].idxmax(), 'Product line']

# Filter the DataFrame for purchases with the highest total price
max_total_purchases = sm[sm['Product line'] == max_total_product_line]

# Print the result
print(max_total_product_line)
print(max_total_purchases)


# In[88]:


# Percentage of gender for each product line

# Assuming 'sm' is your DataFrame
# Group by 'Product line' and 'Gender' to get the count of each combination
gender_product_counts = sm.groupby(['Product line', 'Gender']).size().unstack(fill_value=0)

# Calculate the percentage for each gender related to each product line
percentage_by_gender = gender_product_counts.div(gender_product_counts.sum(axis=1), axis=0) * 100

# Print the result
print(percentage_by_gender)


# In[89]:


# Gender percentages for each city and each branch

# Assuming 'sm' is your DataFrame
# Group by 'City', 'Branch', and 'Gender' to get the count of each combination
gender_city_branch_counts = sm.groupby(['City', 'Branch', 'Gender']).size().unstack(fill_value=0)

# Calculate the percentage for each gender for each city and each branch
percentage_by_gender_city_branch = gender_city_branch_counts.div(gender_city_branch_counts.sum(axis=1), axis=0) * 100

# Print the result
print(percentage_by_gender_city_branch)


# In[90]:


# Average quantity of products associated with each product line

worst_branch = sm.groupby('Branch')['Total'].min().reset_index()
worst_branch.min()


# # Calculate total sales for each branch
# 

# In[67]:


# Calculate total sales for each branch
branch_sales = sm.groupby('Branch')['Total'].sum()
print(branch_sales)


# # What is the most popular product line in each city?

# In[68]:


city_product_counts = sm.groupby(['City', 'Product line'])['Invoice ID'].count().unstack()
most_popular_products = city_product_counts.idxmax(axis=1)
print(most_popular_products)


# # Calculate the average rating by customer type and gender

# In[69]:


avg_rating = sm.groupby(['Customer type', 'Gender'])['Rating'].mean()
print(avg_rating)


# # Calculate the most common payment method

# In[70]:


common_payment_method = sm['Payment'].mode().values[0]
print(common_payment_method)


# # Calculate the average gross income by product line

# In[71]:


avg_gross_income_by_product = sm.groupby('Product line')['gross income'].mean()
print(avg_gross_income_by_product)


# # Calculate the distribution of customer types in each city

# In[72]:


city_customer_distribution = sm.groupby(['City', 'Customer type'])['Invoice ID'].count().unstack()
print(city_customer_distribution)


# # Median Total Purchase Price and percentage above 

# In[73]:


#calculating the median total purchase price by product
avg_total_by_product = sm.groupby('Product line')['Total'].mean()
print(avg_total_by_product)


# In[74]:


# Find purchases above median for each product line
purchases_above_median = sm['Total'] > sm['Product line'].map(avg_total_by_product)

# Calculate the percentage of purchases above the median for each product line
percentage_above_median = purchases_above_median.groupby(sm['Product line']).mean() * 100

# Print the result
print("Percentage of purchases above median total by product line:")
print(percentage_above_median)


# # What product coincides with the highest total price

# In[75]:


# Find the product line with the highest total price
max_total_product_line = sm.loc[sm['Total'].idxmax(), 'Product line']

# Filter the DataFrame for purchases with the highest total price
max_total_purchases = sm[sm['Product line'] == max_total_product_line]

# Print the result
print(max_total_product_line)
print(max_total_purchases)


# # Percentage of gender for each product line

# In[76]:


# Assuming 'sm' is your DataFrame
# Group by 'Product line' and 'Gender' to get the count of each combination
gender_product_counts = sm.groupby(['Product line', 'Gender']).size().unstack(fill_value=0)

# Calculate the percentage for each gender related to each product line
percentage_by_gender = gender_product_counts.div(gender_product_counts.sum(axis=1), axis=0) * 100

# Print the result
print(percentage_by_gender)


# # Gender percentages for each city and each branch

# In[77]:


# Assuming 'sm' is your DataFrame
# Group by 'City', 'Branch', and 'Gender' to get the count of each combination
gender_city_branch_counts = sm.groupby(['City', 'Branch', 'Gender']).size().unstack(fill_value=0)

# Calculate the percentage for each gender for each city and each branch
percentage_by_gender_city_branch = gender_city_branch_counts.div(gender_city_branch_counts.sum(axis=1), axis=0) * 100

# Print the result
print(percentage_by_gender_city_branch)


# # Average quantity of products associated with each product line

# In[78]:


# Assuming 'sm' is your DataFrame
# Group by 'Product line' and 'Product' to get the average quantity for each combination
average_quantity_by_product_line_product = sm.groupby('Product line')['Quantity'].mean()

# Print the result
print(average_quantity_by_product_line_product)

