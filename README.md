# house-price-prediction
Dataset is from Kaggle, the goal of this project is to analyze and predict house prices and practice data analysis skills.
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation
To understand the impact of the house price, we first need to understand the data values and the distribution of the house prices

# data visualization
After cleaning the data, I drew out a pie chart to find out the ratio of the house prices.
![ratio of the house prices](https://github.com/chency0315/house-price-prediction/assets/100465252/792079ad-d865-4c83-bb10-f0da649ea1c7)

I want to compare each year how the prices of more than 300k and less than 100k differ in quantity.
I found out that before 2007 the price of over 300k sold more than 100k. 
![quanity of house sold everywhere](https://github.com/chency0315/house-price-prediction/assets/100465252/3ed11c42-5788-4ca3-8370-4d8edaedd67a)

Below is a scatter plot of the ground living area and sale price.
The bigger the ground living area is, the higher the price is, but there are a few houses that shows the opposite.
![groundlivingarea_saleprice](https://github.com/chency0315/house-price-prediction/assets/100465252/693de6f4-9e9e-4dfc-aaf4-c49bbd72d249)

the distribution of the prices are 
count      1460.000000
mean     180921.195890
std       79442.502883
min       34900.000000
25%      129975.000000
50%      163000.000000
75%      214000.000000
max      755000.000000
![distribution](https://github.com/chency0315/house-price-prediction/assets/100465252/881a668e-ddbd-453a-90ea-38c0c4d6921a)

# feature engineering 
drop out Id, yrsold, poolqc, fence, miscfeature, and label the other columns
<img width="1211" alt="label encode" src="https://github.com/chency0315/house-price-prediction/assets/100465252/b7d48bd5-cc96-4e61-9a3b-ce480cceaf8f">

# model selection 
 For the model I choose random forest classifier, 
 # train test split
from sklearn.model_selection import train_test_split
#train, test = train_test_split(df1, test_size=0.2, random_state=1)
xtrain, xtest, ytrain, ytest = train_test_split( x, y,test_size=0.2,
                                                random_state=1)
# Try RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(max_depth = 15) #set model parameters max_depth = The maximum depth of the tree, the bigger the better classification
clf.fit(xtrain,ytrain)
y_pred=clf.predict(xtest)

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
conf_mx1 = confusion_matrix(ytest, y_pred)
print(confusion_matrix(ytest, y_pred))
print("acc:",accuracy_score(ytest, y_pred))

# RandomForestClassifier classification report 
from sklearn.metrics import classification_report
y1_true = ytest
target_names = ['SalePrice < 100K','SalePrice >= 100k', 'SalePrice >= 200k', 'SalePrice >= 300k']
print(classification_report(y1_true, y_pred, target_names=target_names))

# plot confusion matrix 
import seaborn as sns
import matplotlib.pyplot as plt
plt.title("RandomForestClassifier Confusion Matrix")
labels = ['SalePrice < 100K','SalePrice >= 100k', 'SalePrice >= 200k', 'SalePrice >= 300k'] 
sns.set(font_scale=1.4)
ax1=sns.heatmap(conf_mx1,annot=True,cmap="Purples",fmt="d",cbar=True, xticklabels=labels, yticklabels=labels)
ax1.set_xlabel('predicted label')
ax1.set_ylabel('true label')
![confusion matrix](https://github.com/chency0315/house-price-prediction/assets/100465252/981dda2a-b7e0-4cde-befd-1afaa44c2214)
