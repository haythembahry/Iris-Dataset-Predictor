# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    # Press the green button in the gutter to run the script.

# Press the green button in the gutter to run the script.

import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title('Iris Dataset Predictor')

iris = datasets.load_iris()
X = iris.data
Y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

sepal_length_min, sepal_length_max, sepal_length_mean = X[:, 0].min(), X[:, 0].max(), X[:, 0].mean()
sepal_width_min, sepal_width_max, sepal_width_mean = X[:, 1].min(), X[:, 1].max(), X[:, 1].mean()
petal_length_min, petal_length_max, petal_length_mean = X[:, 2].min(), X[:, 2].max(), X[:, 2].mean()
petal_width_min, petal_width_max, petal_width_mean = X[:, 3].min(), X[:, 3].max(), X[:, 3].mean()

sepal_length = st.slider('Sepal Length', float(sepal_length_min), float(sepal_length_max), float(sepal_length_mean))
sepal_width = st.slider('Sepal Width', float(sepal_width_min), float(sepal_width_max), float(sepal_width_mean))
petal_length = st.slider('Petal Length', float(petal_length_min), float(petal_length_max), float(petal_length_mean))
petal_width = st.slider('Petal Width', float(petal_width_min), float(petal_width_max), float(petal_width_mean))

if st.button('Predict'):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = clf.predict(input_data)
    predicted_class = iris.target_names[prediction[0]]
    st.write('Predicted Iris Flower Type:', predicted_class)



