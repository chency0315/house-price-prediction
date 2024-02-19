# house-price-prediction
資料集來自kaggle，專案的目的在於練習分析和了解預測房價的機器學習方式。

The dataset is from Kaggle, the goal of this project is to analyze and predict house prices and practice data analysis skills.

https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation

![house_price_project](https://github.com/chency0315/house-price-prediction/assets/100465252/f7a4f895-dbf9-478d-ba7c-6df6eab8be76)

# Data visualization

為了瞭解房子特徵如何影響房價，我們必須先了解資料數值和房價分布。
我添加一行salepricerange將房價做區隔，以便好預測。區分為，SalePrice < 100k, SalePrice >= 100k, SalePrice >= 200k, SalePrice >= 300k。

To understand how the house features impact the house price, we first need to understand the data values and the distribution of the house prices. I added a column called salepricerange to do a segmentation on the house price for the convenience of predicting. It is separated like this, SalePrice < 100k, SalePrice >= 100k, SalePrice >= 200k, SalePrice >= 300k.

總共有1460筆房價資料，但是有些在alley這行的幾筆含有非數值。

There are 1460 house data, but there are a few nan values in the feature, like alley. 

### 了解資料型態和資料後，我將房價分布以圓餅圖畫出來以便了解多數房價為何?

After understanding the datatypes and data, I drew out a pie chart to find out the ratio of the house prices.

![ratio of the house prices](https://github.com/chency0315/house-price-prediction/assets/100465252/792079ad-d865-4c83-bb10-f0da649ea1c7)

### 我想比較一下30萬以上和10萬以下的價格在數量上有什麼不同。我發現2007年之前30萬多的價格賣了10萬多。

I want to compare how the prices of more than 300k and less than 100k differ in quantity.
I found out that before 2007 the price of over 300k sold more than 100k. 

```
df_300k = df1[df1['SalePriceRange'] == 'SalePrice >= 300k']
df_100k = df1[df1['SalePriceRange'] == 'SalePrice < 100k']
df_300k['quantity']=1
df_100k['quantity']=1
df_300k = df_300k.groupby(df_300k['YrSold']).sum()
df_100k = df_100k.groupby(df_100k['YrSold']).sum()
x = df_300k['quantity'].index
x_axis = np.arange(len(x))
y = df_300k['quantity']
z = df_100k['quantity']
plt.bar(x_axis-0.1,y,0.2,label = '>=300k')
plt.bar(x_axis+0.1,z,0.2,label = '<100k')
plt.xticks(x_axis,x)
plt.xlabel("year")
plt.ylabel("quantity")
plt.legend()
plt.title("Quantity of sold houses every year")
```

![quanity of house sold everywhere](https://github.com/chency0315/house-price-prediction/assets/100465252/3ed11c42-5788-4ca3-8370-4d8edaedd67a)

### 以下是實際居住面積和銷售價格的散佈圖。地面居住面積越大，價格越高，但也有少數房屋表現相反。

Below is a scatter plot of the ground living area and sale price.
The bigger the ground living area is, the higher the price is, but there are a few houses that show the opposite.

![groundlivingarea_saleprice](https://github.com/chency0315/house-price-prediction/assets/100465252/693de6f4-9e9e-4dfc-aaf4-c49bbd72d249)

### 以下是房價的常態分佈

The distribution of the prices are 

count      1460.000000

mean     180921.195890

std       79442.502883

min       34900.000000

25%      129975.000000

50%      163000.000000

75%      214000.000000

max      755000.000000

![distribution](https://github.com/chency0315/house-price-prediction/assets/100465252/881a668e-ddbd-453a-90ea-38c0c4d6921a)

# Feature engineering 

將 Id, yrsold, poolqc, fence, misfeatures 這幾行沒有對預測有幫助的特徵去除，然後標記其他data。

Drop out Id, yrsold, poolqc, fence, misfeatures, and label the other columns.

```
# Data preprocessing
# label encoding
for i,name in enumerate(df1.columns):
    c = f'c{i}'
    c = df1[name].astype('category')
    df1[name] = c.cat.codes
    if name == 'SalePriceRange':
        d_SalePrices = dict(enumerate(c.cat.categories))
        print(d_SalePrices)
display(df1)
```

<img width="1211" alt="label encode" src="https://github.com/chency0315/house-price-prediction/assets/100465252/b7d48bd5-cc96-4e61-9a3b-ce480cceaf8f">

# Model selection 

對於模型我選擇隨機森林樹，最大深度 = 10, 準確度為0.859，結果可用於房價評估參考。

For the model, I choose random forest classifier, max depth = 10 
the precision is 0.859 with test data, the result can be used for the evaluation of the house price.

<img width="487" alt="acc" src="https://github.com/chency0315/house-price-prediction/assets/100465252/cca94c39-e126-4b53-9af9-152974be5e22">


# Feature importances

利用feature importance，我發現除了售價外實際居住坪數是最重要的。
Through feature importance, I found out that the Ground living area is the most important factor.

```
print(clf.feature_importances_)
print("")

feature_name = list(xtrain.columns)
feature_dict = {}
for i, feature in enumerate(feature_name):
    feature_dict[feature] = clf.feature_importances_[i]

feature_import = [(v, k) for k, v in feature_dict.items()]
feature_import = sorted(feature_import, reverse=True)  # Sort in descending order
for i in range(len(feature_name)):
    print('Rank: {:2d}  Score: {:.5f}  Feature: {:s}'.format(i+1, feature_import[i][0], feature_import[i][1]))
```

feature importance排名目的在於，特徵值有多頻繁被用於隨機樹森林機器學習模型裡。

<img width="409" alt="feature importance" src="https://github.com/chency0315/house-price-prediction/assets/100465252/30ccaaad-6327-4e3c-80b6-479c0b951de3">
