# Sonar Data Classification

## Project Description
This project aims to classify sonar signals as either reflections from a metal cylinder (mine) or a roughly cylindrical rock. The dataset consists of sonar returns collected from various angles and under different conditions. A Logistic Regression model is used for this binary classification task.

## Dataset
The dataset used is `sonar dataset.csv`, which contains 60 features representing the energy content at different frequency bands, and a target variable indicating whether the object is a 'Rock' (R) or a 'Mine' (M).

## Dependencies
The following Python libraries are required to run this project:
- `numpy`
- `pandas`
- `sklearn` (scikit-learn)

## Installation
To set up the project locally, clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd <your-repository-name>
pip install numpy pandas scikit-learn
```

## Usage
To run the notebook and reproduce the results:

1. Open the `.ipynb` notebook in a Jupyter environment (e.g., Jupyter Lab, Google Colab).
2. Ensure the `sonar dataset.csv` file is in the same directory as the notebook or update the path in the code.
3. Run all cells in the notebook.

### Code Snippets

#### Data Loading and Preprocessing
```python
import pandas as pd
sonar_data = pd.read_csv('/sonar dataset.csv', header=None)
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
```

#### Model Training
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
model = LogisticRegression()
model.fit(X_train, Y_train)
```

#### Model Evaluation
```python
from sklearn.metrics import accuracy_score

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f'Accuracy on training data: {training_data_accuracy:.2f}')

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f'Accuracy on test data: {test_data_accuracy:.2f}')
```

## Results
The Logistic Regression model achieved the following accuracies:
- **Training Data Accuracy**: 83.42%
- **Test Data Accuracy**: 76.19%

## Predictive System Example
An example of how to use the trained model for prediction:

```python
import numpy as np

input_data = (...data to be input....)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
if(prediction[0] == 'R'):
   print('Object is rock')
else:
   print('Object is mine')
```

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
