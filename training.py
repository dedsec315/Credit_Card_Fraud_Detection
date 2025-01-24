import numpy as np  # Provides numerical computing capabilities for arrays and matrices.
import pandas as pd  # Used for data manipulation, analysis, and CSV file I/O.

from sklearn.model_selection import train_test_split  # For splitting data into training and testing sets.
from sklearn.preprocessing import StandardScaler  # For standardizing features.
from imblearn.over_sampling import SMOTE  # For oversampling the minority class.
import tensorflow  # TensorFlow library.
from tensorflow import keras  # Keras API for building neural networks.
from tensorflow.keras import Sequential  # For creating sequential models.
from tensorflow.keras.layers import Dense  # For creating dense layers.
import matplotlib.pyplot as plt  # For plotting graphs.

# Load Data
df = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')  # Reads the credit card fraud dataset.

# Data Exploration (Optional)
df.sample(5)  # Displays a random sample of 5 rows.
#df.drop([0]) # Drops the row at index 0 (generally not recommended without a specific reason).
df['Class'].value_counts()  # Counts the occurrences of each class.
df.describe()  # Provides summary statistics of numerical columns.
df.isnull().sum()  # Counts missing values in each column.
df['Amount'].value_counts() #Counts the occurences of each value in the amount column

# Data Preprocessing
x = df.drop(columns=['Class'])  # Creates feature matrix X by dropping the 'Class' column.
y = df.iloc[:, -1]  # Selects the last column as the target variable y.

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # Splits data into training (80%) and testing (20%).

# Standardization
scaler = StandardScaler()  # Creates a StandardScaler object.
new_x_train = scaler.fit_transform(x_train)  # Fits the scaler on the training data and transforms it.
new_x_test = scaler.transform(x_test)  # Transforms the testing data using the fitted scaler.

#Oversampling
smote = SMOTE(random_state=42) #Create a SMOTE object
new_x_train_resampled, y_train_resampled = smote.fit_resample(new_x_train, y_train) #Apply SMOTE to balance the training data

# Model Building
model = Sequential()  # Creates a sequential neural network model.

# Hidden Layers
model.add(Dense(32, activation='relu', input_dim=30))  # Adds the first hidden layer with 32 neurons, ReLU activation, and input dimension 30.
model.add(Dense(16, activation='relu'))  # Adds a second hidden layer with 16 neurons and ReLU activation.

# Output Layer
model.add(Dense(1, activation='sigmoid'))  # Adds the output layer with 1 neuron and sigmoid activation for binary classification.

# Model Summary
model.summary()  # Prints a summary of the model architecture.

# Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compiles the model with Adam optimizer, binary cross-entropy loss, and accuracy metric.

# Model Training
history = model.fit(new_x_train_resampled, y_train_resampled, epochs=5, validation_split=0.2) #Trains the model for 5 epochs with 20% validation split

#Model Prediction
y_log = model.predict(new_x_test) #Predict the probability of each class
y_pred = np.where(y_log>0.5,1,0) #Convert probabilities to class labels (0 or 1) based on a threshold of 0.5

#Model Evaluation
from sklearn.metrics import accuracy_score #Import the accuracy_score function
accuracy_score(y_test,y_pred) #Calculate the accuracy of the model on the test data

# Plotting Training History
history.history  # Access the training history (loss and accuracy values).

plt.plot(history.history['loss'])  # Plots the training loss.
plt.plot(history.history['val_loss'])  # Plots the validation loss.
plt.title('Loss')
plt.show()

plt.plot(history.history['accuracy'])  # Plots the training accuracy.
plt.plot(history.history['val_accuracy'])  # Plots the validation accuracy.
plt.title('Accuracy')
plt.show()