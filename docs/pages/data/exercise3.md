## Get the Data

Data downloaded from : Spaceship Titanic (https://www.kaggle.com/competitions/spaceship-titanic/data)

## Describe the Data

The objective of this training dataset is to predict whether a passenger was transported ("Transported") or not during a space voyage, based on various personal and consumption characteristics. 

We have some numerical features, categorical features and special cases features:

``` py
numerical_features = ['Age', 'RoomService','FoodCourt','ShoppingMall', 'Spa', 'VRDeck']
categorical_features = ['HomePlanet','CryoSleep','Destination','VIP']
special_columns = ['Name', 'Cabin', 'PassengerId']
target_column = 'Transported'
```

And we need to be aware of the missing values and specificities from the dataset, we can see this aspects using describe on dataset

``` py
df.info()
```
```
RangeIndex: 8693 entries, 0 to 8692
Data columns (total 14 columns):
 #   Column        Non-Null Count  Dtype
---  ------        --------------  -----
 0   PassengerId   8693 non-null   object
 1   HomePlanet    8492 non-null   object
 2   CryoSleep     8476 non-null   object
 3   Cabin         8494 non-null   object
 4   Destination   8511 non-null   object
 5   Age           8514 non-null   float64
 6   VIP           8490 non-null   object
 7   RoomService   8512 non-null   float64
 8   FoodCourt     8510 non-null   float64
 9   ShoppingMall  8485 non-null   float64
 10  Spa           8510 non-null   float64
 11  VRDeck        8505 non-null   float64
 12  Name          8493 non-null   object
 13  Transported   8693 non-null   bool
dtypes: bool(1), float64(6), object(7)
memory usage: 891.5+ KB
```
``` py
print(df.isnull().sum())
```
```
PassengerId       0
HomePlanet      201
CryoSleep       217
Cabin           199
Destination     182
Age             179
VIP             203
RoomService     181
FoodCourt       183
ShoppingMall    208
Spa             183
VRDeck          188
Name            200
Transported       0
dtype: int64
```

## Preprocess the Data

First of all, we need to handle the missing data:

``` py
```

Transform the special feature Cabin in categorical columns:

``` py
```

Encode categorical features

``` py
```

## Visualize the Results

![Image1](../../assets/images/data/exercise3_1.png)
![Image2](../../assets/images/data/exercise3_2.png)
![Image3](../../assets/images/data/exercise3_4.png)