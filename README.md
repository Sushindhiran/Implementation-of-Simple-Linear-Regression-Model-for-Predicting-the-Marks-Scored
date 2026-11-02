# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries: Import essential libraries for data manipulation, numerical operations, plotting, and regression analysis.
2. Load and Explore Data: Load a CSV dataset using pandas, then display initial and final rows to quickly explore the data's structure.
3. Prepare and Split Data: Divide the data into predictors (x) and target (y). Use train_test_split to create training and testing subsets for model building and evaluation.
4. Train Linear Regression Model: Initialize and train a Linear Regression model using the training data.
5. Visualize and Evaluate: Create scatter plots to visualize data and regression lines for training and testing. Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to quantify model performance.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARINI S
RegisterNumber: 212224240049
```
``` python
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())

# Independent and dependent variables
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values
print("X:", X)
print("Y:", Y)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict
Y_pred = regressor.predict(X_test)
print("Predicted:", Y_pred)
print("Actual:", Y_test)

# Training set visualization
plt.scatter(X_train, Y_train, color="orange")
plt.plot(X_train, regressor.predict(X_train), color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Testing set visualization
plt.scatter(X_test, Y_test, color="blue")
plt.plot(X_test, regressor.predict(X_test), color="green")
plt.title("Hours vs Scores (Testing Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

# Error metrics
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

```

## Output:
### Head:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/18f3af86-9ae2-4494-bb61-636b83a7bcd5)
### Tail:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/0a120341-9b3f-4ee2-8740-1ea5cdc71610)
### Array value of X:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/e296242a-26a5-4ba3-9736-86bc8fe4e85c)
### Array value of Y:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/c25d4900-5e51-4bf7-8db7-1ec80106ab55)
### Values of Y prediction:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/611f79c2-9b85-47f4-9a9f-6fbe4dc4e7ee)
### Array values of Y test:
![image](https://github.com/MukeshVelmurugan/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118707363/df0a81e5-6894-4e45-9037-6dce6cb3a7d7)
### Training Set Graph:
<img width="718" height="471" alt="image" src="https://github.com/user-attachments/assets/a8ef4450-69f6-4ee6-9f76-17dd9d1f718a" />

### Test Set Graph:
<img width="617" height="467" alt="image" src="https://github.com/user-attachments/assets/b068dbd3-fee3-4d14-9242-970d175de02a" />

### Values of MSE, MAE and RMSE:
<img width="516" height="77" alt="image" src="https://github.com/user-attachments/assets/d1f3db30-965b-49fe-bef3-6c95aea5a74a" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
