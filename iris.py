
# We are building a simple machine leanring web app in python.
# To start the prog. open/run iris.py and then conda activate sandbox(virtual env.) then streamlit run iris.py
# Importing the libraries:
import streamlit as st 
import pandas as pd 
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier


# First header 

st.write("""

# Simple Iris Flower Prediction App

This app predicts the **Iris flower** type!

""")

 # does exactly what it says: this is the side bar header.         
st.sidebar.header('user Input Parameters')


# Below is the custom function used to accept all of the four input parameters from the sidebar and it will create a pandas dataframe.
# The input parameters will be obtained from the sidebar. 
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) # (after the sepal length the numbers represent min.value , max.value and current selected value respectivetly )
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Section header name and the corresponding table below it . 
st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

clf = RandomForestClassifier() # Here we are creating a clasifier varible comprising of RFC. 
clf.fit(X, Y) # here we have applied the classifier to build a training model using as input arguments the X and Y data matrices. 

prediction = clf.predict(df) # Here we are making the prediction.
prediction_proba = clf.predict_proba(df) # giving you the prediction probability. 

st.subheader('Class labels and their corresponding index number') # simple print out of class label and their corresponding index number.
st.write(iris.target_names)

st.subheader('Prediction') # these are giving us the prediction.
st.write(iris.target_names[prediction]) # Which here is the class label of either setosa , versicolor and virginica. 
#st.write(prediction)

st.subheader('Prediction Probability') # Telling us what is the probability of being in one of the three classes.
st.write(prediction_proba)