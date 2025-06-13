# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")

df.head()
df.dropna()
```
# Output
![image](https://github.com/user-attachments/assets/69ddec3d-8d91-4d21-b62e-cfffeb9ec3d1)

```
max_vals=np.max(np.abs(df[['Height']]))
max_vals
max_vals1=np.max(np.abs(df[['Weight']]))
max_vals1
print("Height =",max_vals)
print("Weight =",max_vals1)
```
# Output
![image](https://github.com/user-attachments/assets/ac7ef8ee-adb2-4773-ac5f-5685fd95b370)

```
df1=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
# Output
![image](https://github.com/user-attachments/assets/29752392-0a1d-4348-b5fa-16fd91f32358)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
# Output
![image](https://github.com/user-attachments/assets/d9f7d1df-c943-4d8b-b67c-29f803a779c2)

```
from sklearn.preprocessing import Normalizer
df2=pd.read_csv("/content/bmi.csv")
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
# Output
![image](https://github.com/user-attachments/assets/d828d521-9cc7-4a7e-9b13-101f1baae631)

```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
# Output
![Screenshot 2025-04-27 114528](https://github.com/user-attachments/assets/04eb2c51-353f-4553-9ab9-187cc3399d4d)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
# Output
![Screenshot 2025-04-27 114706](https://github.com/user-attachments/assets/d49d6edf-ae70-4589-87f8-a318bb1d0201)

# FEATURE SELECTION SUING KNN CLASSIFICATION
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
# Output
![image](https://github.com/user-attachments/assets/a6d6a274-3d52-4d63-ad9e-8148192c305a)

```
data.isnull().sum()
```
# Output
![image](https://github.com/user-attachments/assets/4ea3ecc3-a065-4c42-b787-559c3487cdd8)

```
missing=data[data.isnull().any(axis=1)]
missing
```
# Output
![image](https://github.com/user-attachments/assets/787cd777-2531-4bcf-b21a-957f77d7fd20)

```
data2=data.dropna(axis=0)
data2
```
# Output
![image](https://github.com/user-attachments/assets/f779c134-66c1-45ae-bef8-d702bf6eb90f)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
# Output
![image](https://github.com/user-attachments/assets/0cd7f515-4027-4808-97c8-c202b4bc49a1)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
# Output
![image](https://github.com/user-attachments/assets/7118f566-4aad-4877-bfe1-3d41ad0b9ecd)

```
data2
```
# Output
![image](https://github.com/user-attachments/assets/d9524700-5cbc-4d53-b26d-688182f79d8b)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
# Output
![Screenshot 2025-04-27 115250](https://github.com/user-attachments/assets/eac556a3-2fae-4d9e-bb07-360a9de3b174)

```
columns_list=list(new_data.columns)
print(columns_list)
```
# Output

![image](https://github.com/user-attachments/assets/42c0b4eb-ce91-45da-9ef3-fc554290d76b)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
# Output
![image](https://github.com/user-attachments/assets/05c4ade9-9fe4-4ba9-a350-72d671e06705)

```
y=new_data['SalStat']
print(y)
```
# Output
![image](https://github.com/user-attachments/assets/c229c2d3-8d09-4e24-a4d8-08a56419d1b6)

```
x=new_data[features].values
print(x)
```
# Output
![Screenshot 2025-04-27 115545](https://github.com/user-attachments/assets/2f0e83e5-0d0f-40b8-a3a6-3a880559301b)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```
# Output
![image](https://github.com/user-attachments/assets/c46dad02-624d-45d6-b8c8-ca586d5b4428)

```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```
# Output
![image](https://github.com/user-attachments/assets/927b4a99-1272-4dc6-a83e-2abb9642be9b)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)

X=df[['Feature1','Feature3']]
y=df['Target']

selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)

selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]

print("Selected Features:", selected_features)
```
# Output
![image](https://github.com/user-attachments/assets/37042179-fa08-49b9-833b-11a3900e815a)


# RESULT:
      Thus, Feature selection and Feature scaling has been used on the given dataset.
