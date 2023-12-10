#!/usr/bin/env python
# coding: utf-8

# ## DE Project 
# 

# In[1]:


import pandas as pd


# In[2]:


solar = pd.read_csv("ProTwo.csv")


# In[3]:


solar.head()


# In[4]:


solar.info()


# In[5]:


solar.describe()


# ## Train-Test Splitting

# In[6]:


import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:] 
    return data.iloc[train_indices], data.iloc[test_indices]


# In[7]:


train_set, test_set = split_train_test(solar, 0.2)


# In[8]:


print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[9]:


solar.shape


# In[10]:


print(solar.columns)


# In[11]:


features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)", "PPA Tenure(Y)", "IRR"]
x= solar[features]
print(x.head())


# In[13]:


y = solar['Solar_price']  
print(y.head())


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your data from the CSV file
solar = pd.read_csv("ProTwo.csv")

# Define your features and target variable
features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)", "PPA Tenure(Y)", "IRR"]
# Create input features (X) and target variable (y)
X = solar[features]
y = solar["Solar_price"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model with the training data
model.fit(X_train, y_train)

# Make predictions using the test data
predictions = model.predict(X_test)

# Calculate and print the Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

# Now you can use the trained model to make predictions on new data
# For example, if you have new input features in a variable called 'new_data':
# new_predictions = model.predict(new_data)


# In[16]:


accuracy = model.score(X_test, y_test)
print("Model Accuracy on Test Data:", accuracy)


# In[18]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle


solar = pd.read_csv('ProTwo.csv')

# Define your features and target variable
features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)", "PPA Tenure(Y)", "IRR"]

X = solar[features]  # Features
y = solar['Solar_price']  # Target variable
    
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the test data
accuracy = model.score(X_test, y_test)
print(f'Model Accuracy on Test Data: {accuracy}')

# Save the trained model to a .pkl file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
# Assume X_new represents your new data for prediction (features for which you want to predict solar prices)
X_new = {
    "System Size(KWp)": [1000],
    "EB Tariff(RS/kwh)": [7.31],
    "Generation/efficiency": [1513],
    "System Cost": [53714],
    "Asset Management Cost(Inr/kwp)": [1248.22],
    "Commission In Time (Y)": [0.5],
    "PPA Tenure(Y)": [20],
    "IRR": [15.07],
}

# Convert the dictionary to a pandas DataFrame
X_new = pd.DataFrame(X_new)

# Make predictions using the loaded model
predicted_solar_prices = model.predict(X_new.values.reshape(1, -1))

# The variable 'predicted_solar_prices' now contains the predicted solar prices for the new data
print(predicted_solar_prices)


# In[22]:


# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset into a Pandas DataFrame
# Replace 'your_dataset.csv' with the actual file path or data source
solar = pd.read_csv('ProTwo.csv')

# Compute the correlation matrix
correlation_matrix = solar.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[23]:


corr_matrix = solar.corr()
corr_matrix['System Size(KWp)'].sort_values(ascending=False)


# In[25]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import pickle

# Load your dataset
solar = pd.read_csv('ProTwo.csv')

# Define your features and target variables
features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)", "IRR"]


X = solar[features]  # Features
y = solar[['PPA Tenure(Y)', 'Solar_price']]  # Target variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
base_model = LinearRegression()

# Use MultiOutputRegressor with the Linear Regression model
model = MultiOutputRegressor(base_model)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('multioutput_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the trained model
with open('multioutput_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Assume X_new represents your new data for prediction
X_new = {
    "System Size(KWp)": [1000],
    "EB Tariff(RS/kwh)": [7.31],
    "Generation/efficiency": [1513],
    "System Cost": [53714],
    "Asset Management Cost(Inr/kwp)": [1248.22],
    "Commission In Time (Y)": [0.5],
    "IRR": [15.07],
}

# Convert the dictionary to a pandas DataFrame
X_new = pd.DataFrame(X_new)

# Make predictions using the loaded model
predicted_values = model.predict(X_new)

# Extract predicted tenure and solar price
predicted_tenure, predicted_solar_price = predicted_values[:, 0], predicted_values[:, 1]

print('Predicted Tenure:', predicted_tenure[0])
print('Predicted Solar Price:', predicted_solar_price[0])


# In[26]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pickle

# Load your dataset
solar = pd.read_csv('ProTwo.csv')

# Define your features and target variables
features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)", "IRR"]

X = solar[features]  # Features
y = solar[['Solar_price', 'PPA Tenure(Y)']]  # Target variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
base_model = LinearRegression()

# Use MultiOutputRegressor with the Linear Regression model
model = MultiOutputRegressor(base_model)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('multioutput_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the trained model
with open('multioutput_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Assume X_new represents your new data for prediction (features for which you want to predict solar prices and PPA Tenure)
X_new = {
    "System Size(KWp)": [1000],
    "EB Tariff(RS/kwh)": [7.31],
    "Generation/efficiency": [1513],
    "System Cost": [53714],
    "Asset Management Cost(Inr/kwp)": [1248.22],
    "Commission In Time (Y)": [0.5],
    "IRR": [15.07],
}

# Convert the dictionary to a pandas DataFrame
X_new = pd.DataFrame(X_new)

# Make predictions using the loaded model
predicted_values = model.predict(X_new)

# The variable 'predicted_values' now contains the predicted values for both Solar_price and PPA Tenure
print('Predicted Values:', predicted_values)


# In[27]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pickle

# Load your dataset
solar = pd.read_csv('ProTwo.csv')

# Define your features and target variables
features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)", "PPA Tenure(Y)"]

X = solar[features]  # Features
y = solar[['Solar_price', 'IRR']]  # Target variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
base_model = LinearRegression()

# Use MultiOutputRegressor with the Linear Regression model
model = MultiOutputRegressor(base_model)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('multioutput_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the trained model
with open('multioutput_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Assume X_new represents your new data for prediction (features for which you want to predict solar prices and PPA Tenure)
X_new = {
    "System Size(KWp)": [1000],
    "EB Tariff(RS/kwh)": [7.31],
    "Generation/efficiency": [1513],
    "System Cost": [53714],
    "Asset Management Cost(Inr/kwp)": [1248.22],
    "Commission In Time (Y)": [0.5],
    "PPA Tenure(Y)": [20],
}

# Convert the dictionary to a pandas DataFrame
X_new = pd.DataFrame(X_new)

# Make predictions using the loaded model
predicted_values = model.predict(X_new)

# The variable 'predicted_values' now contains the predicted values for both Solar_price and PPA Tenure
print('Predicted Values:', predicted_values)


# In[28]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
import pickle

# Load your dataset
solar = pd.read_csv('ProTwo.csv')

# Define your features and target variables
features = ["System Size(KWp)", "EB Tariff(RS/kwh)", "Generation/efficiency", "System Cost", "Asset Management Cost(Inr/kwp)",
            "Commission In Time (Y)"]

X = solar[features]  # Features
y = solar[['Solar_price', 'IRR','PPA Tenure(Y)']]  # Target variables

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
base_model = LinearRegression()

# Use MultiOutputRegressor with the Linear Regression model
model = MultiOutputRegressor(base_model)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a .pkl file
with open('multioutput_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the trained model
with open('multioutput_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Assume X_new represents your new data for prediction (features for which you want to predict solar prices and PPA Tenure)
X_new = {
    "System Size(KWp)": [1000],
    "EB Tariff(RS/kwh)": [7.31],
    "Generation/efficiency": [1513],
    "System Cost": [53714],
    "Asset Management Cost(Inr/kwp)": [1248.22],
    "Commission In Time (Y)": [0.5],
}

# Convert the dictionary to a pandas DataFrame
X_new = pd.DataFrame(X_new)

# Make predictions using the loaded model
predicted_values = model.predict(X_new)

# The variable 'predicted_values' now contains the predicted values for both Solar_price and PPA Tenure
print('Predicted Values:', predicted_values)


# In[ ]:




