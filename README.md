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

# feature engineering 
Drop out Id, yrsold, poolqc, fence, miscfeature, and label the other columns

<img width="1211" alt="label encode" src="https://github.com/chency0315/house-price-prediction/assets/100465252/b7d48bd5-cc96-4e61-9a3b-ce480cceaf8f">

# model selection 

For the model I choose random forest classifier, max depth = 10 
the precision is 0.859, the result can use for the evaluation of the house price.

<img width="267" alt="acc" src="https://github.com/chency0315/house-price-prediction/assets/100465252/4ce4314d-f96b-4b94-8339-43ad0f2b7417">

#feature importances
Through feature importance, I found out that Ground living area is the most important factor.

<img width="369" alt="feature importance" src="https://github.com/chency0315/house-price-prediction/assets/100465252/5dea2db6-b70e-4bf4-a215-f5218af6a24c">

