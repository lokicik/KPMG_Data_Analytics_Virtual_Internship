import pandas as pd
import numpy as np
import openpyxl
import matplotlib.pyplot as plt
import seaborn as sns

#functions
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def print_repeated_values_once(dataframe, column_name):
    unique_values = set()
    repeated_values = set()

    for value in dataframe[column_name]:
        if value in unique_values:
            repeated_values.add(value)
        else:
            unique_values.add(value)

    for value in repeated_values:
        print(value)

def has_repeating_values(dataframe, column_name):
    unique_values = set()

    for value in dataframe[column_name]:
        if value in unique_values:
            return True
        else:
            unique_values.add(value)

    return False



# NewCustomerList
df = pd.read_excel("KPMG/Task-2/KPMG_VI_New_raw_data_update_final.xlsx", sheet_name="NewCustomerList")
df.head()
df.columns

#############################################
# AGE DISTRIBUTION
sns.histplot(data=df, x='Age', bins=10, kde=True, color='blue', discrete=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.savefig('KPMG/Task-2/age_distribution_new.png')
plt.show()
#############################################
######## GENDER BY BIKE PURCHASES TOTAL COUNT
plt.figure(figsize=(8, 6))  # Set the figure size
sns.barplot(data=df, x='gender', y='past_3_years_bike_related_purchases', estimator=sum, ci=None)
plt.xlabel('Gender')
plt.ylabel('Total Bike Purchases')
plt.title('Total Bike Purchases by Gender')
plt.savefig('KPMG/Task-2/gender_by_bike_count_new.png')
plt.show()
#############################################
######## GENDER BY BIKE PURCHASES THE REAL ONE
total_purchase_sum = df['past_3_years_bike_related_purchases'].sum()

# Create a bar plot using seaborn
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=df, x='gender', y='past_3_years_bike_related_purchases', estimator=sum, ci=None)

# Annotate each bar with its percentage value
for p in ax.patches:
    percentage = (p.get_height() / total_purchase_sum) * 100
    ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Total Bike Purchases')
plt.title('Total Bike Purchases by Gender')
plt.savefig('KPMG/Task-2/gender_by_bike_new.png')
# Show the plot
plt.show()
#############################################
# GENDER BY BIKE PURCHASES PERCENTAGE
gender_purchase_sum = df.groupby('gender')['past_3_years_bike_related_purchases'].sum().reset_index()

# Calculate the total sum of 'past_3_years_bike_related_purchases'
total_purchase_sum = gender_purchase_sum['past_3_years_bike_related_purchases'].sum()

# Calculate the percentage for each gender
gender_purchase_sum['percentage'] = (gender_purchase_sum['past_3_years_bike_related_purchases'] / total_purchase_sum) * 100

# Create a bar plot using seaborn
plt.figure(figsize=(8, 6))  # Set the figure size
sns.barplot(data=gender_purchase_sum, x='gender', y='percentage', errorbar=None)
ax = sns.barplot(data=gender_purchase_sum, x='gender', y='percentage', errorbar=None)
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Percentage of Total Bike Purchases')
plt.title('Percentage of Total Bike Purchases by Gender')
plt.savefig('KPMG/Task-2/gender_by_bike_percentage_new.png')
# Show the plot
plt.show()
##############################################
######## JOB INDUSTRIES DISTRIBUTION
gender_counts = df['job_industry_category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=260, textprops={'fontsize': 12})
plt.title('Job Industry Distribution', y=1.05)
plt.axis('equal')
plt.savefig('KPMG/Task-2/job_industry_distribution_new.png')
plt.show()
####################
######## WEALTH SEGMENT BY AGE
plt.figure(figsize=(10, 6))  # Set the figure size
sns.swarmplot(data=df, x='Age', y='wealth_segment')
plt.xlabel('Age')
plt.ylabel('Age')
plt.title('Age Distribution by Wealth Segment')
plt.savefig('KPMG/Task-2/wealth_segment_by_age_distribution_new.png')
plt.show()
##############################################
##############################################
######### NUMBER OF CARS OWNED BY STATE
car_by_state = df.groupby(['state', 'owns_car']).size().unstack()
ax = car_by_state.plot(kind='bar', stacked=True)
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Car Ownership Distribution by State')
plt.savefig('KPMG/Task-2/number_of_cars_owned_by_state_old.png')
plt.show()
##############################################
######### RFM ANALYSIS
from datetime import datetime

# Assuming 'DOB' is the date of the most recent purchase
# 'reference_date' can be today's date or the end of your analysis period
reference_date = datetime.today()

# Calculate recency in days
df['Recency'] = (reference_date - df['DOB']).dt.days

# Assign scores based on recency (lower recency is better)
df['Recency_Score'] = pd.qcut(df['Recency'], q=4, labels=False) + 1
# Assuming 'past_3_years_bike_related_purchases' is the frequency
# Assign scores based on frequency (higher frequency is better)
df['Frequency_Score'] = pd.qcut(df['past_3_years_bike_related_purchases'], q=4, labels=False) + 1
# Assuming 'Value' is the monetary value
# Assign scores based on monetary value (higher monetary value is better)
df['Monetary_Score'] = pd.qcut(df['Value'], q=4, labels=False) + 1
# Combine the individual scores into an RFM score
# Assuming you have calculated Recency_Score, Frequency_Score, and Monetary_Score as integers
df['RFM_Score'] = (df['Recency_Score'] * 100 + df['Frequency_Score'] * 10 + df['Monetary_Score'])

df['RFM_Score']

rfm_df = pd.DataFrame({"RFM_Score": df["RFM_Score"],'Recency': df['Recency_Score'], 'Frequency': df['Frequency_Score'], 'Monetary': df['Monetary_Score']})
df["RFM_Score"].max()
df["RFM_Score"].min()
df["RFM_Score"].quantile(0.25)
df["RFM_Score"].quantile(0.75)
# Define the segments based on the RFM scores
def assign_segment(row):
    if row['RFM_Score'] >= 421:
        return 'Very Loyal'
    elif row['RFM_Score'] >= 372:
        return 'Recent Customer'
    elif row['RFM_Score'] >= 323:
        return 'Potential Customer'
    elif row['RFM_Score'] >= 274:
        return 'High Risk Customer'
    elif row['RFM_Score'] >= 225:
        return 'Late Bloomer'
    elif row['RFM_Score'] >= 176:
        return 'Becoming Loyal'
    elif row['RFM_Score'] >= 127:
        return 'Almost Lost Customer'
    elif row['RFM_Score'] >= 111:
        return 'Lost Customer'
    elif row['RFM_Score'] >= 104:
        return 'Evasive Customer'
    else:
        return 'Bronze Customer'

# Apply the segment assignment function to each row
df['Segment'] = rfm_df.apply(assign_segment, axis=1)
rfm_df.head(50)
# Create a bar plot to visualize customer segments
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Segment', hue='wealth_segment', palette='Set2')
plt.xlabel('Segments')
plt.ylabel('Count')
plt.title('Wealth Segments by Customer Segments')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('KPMG/Task-2/rfm_analysis.png')
# Show the plot
plt.legend(title='Wealth Segment')
plt.show()
############################
############ CUSTOMER SEGMENT DISTRIBUTION

segment_counts = df['Segment'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=120, textprops={'fontsize': 8})
plt.title('Customer Segment Distribution', y=1.05)
plt.axis('equal')
plt.savefig('KPMG/Task-2/customer_segment_distribution_new.png')
plt.show()
#####################
plt.figure(figsize=(10, 6))
bars = plt.bar(segment_counts.index, segment_counts, color='skyblue')

# Annotate each bar with its count
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(int(bar.get_height())),
             ha='center', va='bottom', fontsize=10)

# Add labels and title
plt.xlabel('Segments')
plt.ylabel('Count')
plt.title('Segment Distribution with Count')

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout()
plt.savefig('KPMG/Task-2/customer_segment_distribution_bar_new.png')
plt.show()
#######################
####### Recency by Frequency
plt.figure(figsize=(8, 6))
plt.scatter(df['Recency_Score'], df['Frequency_Score'], color='purple', alpha=0.7)

# Add labels and title
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.title('Scatterplot of Frequency by Recency')

# Show the plot
plt.show()
#######################
#######  Frequency by Monetary
plt.figure(figsize=(8, 6))
plt.scatter(df['Frequency_Score'], df['Monetary_Score'], color='green', alpha=0.7)

# Add labels and title
plt.xlabel('Frequency')
plt.ylabel('Monetary')
plt.title('Scatterplot of Monetary by Frequency')

# Show the plot
plt.show()
#######################
####### Monetary by Recency
plt.figure(figsize=(8, 6))
plt.scatter(df['Recency_Score'], df['Monetary_Score'], color='blue', alpha=0.7)

# Add labels and title
plt.xlabel('Recency')
plt.ylabel('Monetary')
plt.title('Scatterplot of Monetary by Recency')

# Show the plot
plt.show()
############
df["Segment"].value_counts()
































##########################################################################################
# CustomerDemographic
df = pd.read_excel("KPMG/Task-2/KPMG_VI_New_raw_data_update_final.xlsx", sheet_name="CustomerDemographic")

df.columns

max_age = df['Age'].max()
df.drop(df[df['Age'] == max_age].index, inplace=True)

##################################################
# AGE DISTRIBUTION

sns.histplot(data=df, x='Age', bins=10, kde=True, color='blue', discrete=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.savefig('KPMG/Task-2/age_distribution_old.png')
plt.show()


######## GENDER BY BIKE PURCHASES TOTAL COUNT
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='gender', y='past_3_years_bike_related_purchases', estimator=sum, ci=None)


plt.xlabel('Gender')
plt.ylabel('Total Bike Purchases')
plt.title('Total Bike Purchases by Gender')
plt.savefig('KPMG/Task-2/gender_by_bike_count_old.png')
plt.show()
#############################################
######## GENDER BY BIKE PURCHASES THE REAL ONE
total_purchase_sum = df['past_3_years_bike_related_purchases'].sum()

# Create a bar plot using seaborn
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=df, x='gender', y='past_3_years_bike_related_purchases', estimator=sum, ci=None)

# Annotate each bar with its percentage value
for p in ax.patches:
    percentage = (p.get_height() / total_purchase_sum) * 100
    ax.annotate(f'{percentage:.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

# Add labels and title
plt.xlabel('Gender')
plt.ylabel('Total Bike Purchases (Old)')
plt.title('Total Bike Purchases by Gender')
plt.savefig('KPMG/Task-2/gender_by_bike_old.png')
# Show the plot
plt.show()
#############################################
#############################################
# GENDER BY BIKE PURCHASES PERCENTAGE
gender_purchase_sum = df.groupby('gender')['past_3_years_bike_related_purchases'].sum().reset_index()

# Calculate the total sum of 'past_3_years_bike_related_purchases'
total_purchase_sum = gender_purchase_sum['past_3_years_bike_related_purchases'].sum()

# Calculate the percentage for each gender
gender_purchase_sum['percentage'] = (gender_purchase_sum['past_3_years_bike_related_purchases'] / total_purchase_sum) * 100


plt.figure(figsize=(8, 6))
sns.barplot(data=gender_purchase_sum, x='gender', y='percentage', errorbar=None)
ax = sns.barplot(data=gender_purchase_sum, x='gender', y='percentage', errorbar=None)

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}%', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.xlabel('Gender')
plt.ylabel('Percentage of Total Bike Purchases')
plt.title('Percentage of Total Bike Purchases by Gender')
plt.savefig('KPMG/Task-2/gender_by_bike_percentage_old.png')
plt.show()
##############################################
##############################################
######## JOB INDUSTRIES DISTRIBUTION
gender_counts = df['job_industry_category'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=260, textprops={'fontsize': 12})
plt.title('Job Industry Distribution', y=1.05)
plt.axis('equal')
plt.savefig('KPMG/Task-2/job_industry_distribution_old.png')
plt.show()
##############################################
######## WEALTH SEGMENT BY AGE
plt.figure(figsize=(10, 6))  # Set the figure size
sns.swarmplot(data=df, x='Age', y='wealth_segment')
plt.xlabel('Age')
plt.ylabel('Age')
plt.title('Age Distribution by Wealth Segment')
plt.savefig('KPMG/Task-2/wealth_segment_by_age_distribution_old.png')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='wealth_segment', y='Age')
plt.xlabel('Wealth Segment')
plt.ylabel('Age')
plt.title('Age Distribution by Wealth Segment (Box Plot)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


##############################################
######### NUMBER OF CARS OWNED BY STATE
car_by_state = df.groupby(['state', 'owns_car']).size().unstack()
ax = car_by_state.plot(kind='bar', stacked=True)
plt.xlabel('State')
plt.ylabel('Count')
plt.title('Car Ownership Distribution by State')
plt.savefig('KPMG/Task-2/number_of_cars_owned_by_state_old.png')
plt.show()
##############################################
####################################################################################
age_by_wealth = df.groupby(['Age', 'wealth_segment']).size()
ax = age_by_wealth.plot(kind='bar', stacked=True)
plt.xlabel('Wealth Segment')
plt.ylabel('Age')
plt.title('Age Distribution by Wealth Segment (Box Plot)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Show the plot
plt.show()