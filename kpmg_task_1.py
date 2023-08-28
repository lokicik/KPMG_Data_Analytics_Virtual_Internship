import pandas as pd
import numpy as np
import openpyxl
import matplotlib as plt
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

# Transactions
df = pd.read_excel("KPMG/Task-1/KPMG_VI_New_raw_data_update_final.xlsx", sheet_name="Transactions")
df.head()
df.columns
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# 6 cat_cols ['order_status', 'brand', 'product_line', 'product_class', 'product_size', 'online_order']
# 7 num_cols ['transaction_id', 'product_id', 'customer_id', 'transaction_date', 'list_price', 'standard_cost', 'product_first_sold_date']
# 0 cat_but_car 0
check_df(df)
for col in num_cols:
    print(col, check_outlier(df, col))


for col in df.columns:
    print(col, df[col].nunique())

for col in df.columns:
    print(col, print_repeated_values_once(df, col))

for col in df.columns:
    print(col, has_repeating_values(df, col))
df["address"].value_counts(ascending=False)


# CustomerAddress
df = pd.read_excel("KPMG/Task-1/KPMG_VI_New_raw_data_update_final.xlsx", sheet_name="CustomerAddress")
cat_cols, num_cols, cat_but_car = grab_col_names(df)

check_df(df)
for col in num_cols:
    print(col, check_outlier(df, col))

df["state"].value_counts()

for col in df.columns:
    print(col, df[col].nunique())

for col in df.columns:
    print(col, print_repeated_values_once(df, col))

for col in df.columns:
    print(col, has_repeating_values(df, col))


# CustomerDemographic
df = pd.read_excel("KPMG/Task-1/KPMG_VI_New_raw_data_update_final.xlsx", sheet_name="CustomerDemographic")
cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.columns
check_df(df)
for col in num_cols:
    print(col, check_outlier(df, col))

for col in df.columns:
    print(col, df[col].nunique())


df.gender.value_counts()

for col in df.columns:
    print(col, print_repeated_values_once(df, col))

for col in df.columns:
    print(col, has_repeating_values(df, col))
