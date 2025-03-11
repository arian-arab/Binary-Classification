from sklearn.datasets import fetch_covtype
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.compose import ColumnTransformer

data = fetch_covtype()
x = data.data
y = data.target
del data

class_counts = np.unique(y, return_counts = True)[1]

# classes 1 and 2 for binary classification
subset_indices = np.isin(y, [1, 2])
x_subset = x[subset_indices]
y_subset = y[subset_indices]
y_binary = (y_subset == 1).astype(int)

# select 10,000 samples randomly
np.random.seed(42) 
idx = np.random.choice(x_subset.shape[0], size=10000, replace=False) 
x_subset = x_subset[idx] 
y_binary = y_binary[idx] 
del idx

# standardize continuous features only
continuous_cols = [0,1,2,3,4,5,6,7,8,9]
preprocessor = ColumnTransformer([("scaler", StandardScaler(), continuous_cols)], remainder="passthrough")
x_subset = preprocessor.fit_transform(x_subset)

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(x_subset, y_binary, test_size=0.2, random_state=42)

# models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Neural Network": MLPClassifier(max_iter=100,activation = 'sigmoid')
}

# train models
for name, model in models.items():
    print(f"\nProcessing model: {name}")

    try:
        model.fit(x_train, y_train)
    except Exception as e:
        print(f"Error training model {name}: {e}")
        continue

    try:        
        y_probability = model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class
        data_to_save = np.array([y_test,y_probability]).T
        np.savetxt(name+".txt", data_to_save, fmt="%.4f", delimiter="\t", comments="")
    except Exception as e:
        print(f"Error predicting with model {name}: {e}")
        continue