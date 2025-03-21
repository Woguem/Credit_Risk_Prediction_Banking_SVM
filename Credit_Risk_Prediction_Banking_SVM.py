import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import numpy as np

# Fix_the_seed_for_reproducibility
np.random.seed(42)

# Number of rows in data set
n_samples = 5000

# Creating a DataFrame
data = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n_samples),      
    'age': np.random.randint(18, 70, n_samples),               
    'loan_amount': np.random.normal(20000, 5000, n_samples),   
    'credit_score': np.random.randint(300, 850, n_samples),    
    'balance': np.random.normal(10000, 2500, n_samples),       
    'default': np.random.choice([0, 1], size=n_samples)        
})

# Display the first lines of the DataFrame
print(data.head())

# Separate features (X) and target (y)
X = data.drop('default', axis=1)  # Features
y = data['default']  # Target

# Split data into trains/tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVC model
model = SVC(C=100,
            kernel='sigmoid', 
            gamma='scale',
            random_state=42) #linear, poly, rbf, sigmoid

model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)

# Display classification report
print(classification_report(y_test, y_pred))





