import streamlit as st
import pandas as pd 
from tensorflow.keras.models import load_model
import pickle

st.title('Passanger survival chances in Titanic journey')

pclass=st.slider('Enter the Passenger class for the user', 1,3)
sex = st.selectbox('Enter the passenger Gender', ['Male', 'Female'])
sibsp = st.slider('Enter the Passenger total num of Sbling & Spouce', 1,8)
parch = st.slider('Enter Passenger total number of Parents & Child', 0,6)
fare = st.number_input('Enter the fare of the Passanger')
embarked = st.selectbox('Enter the passenger station from where they started the journey', ['Chebourg','QueensTown', 'Southampton'])

data = pd.DataFrame([{'Pclass':pclass, 'Sex':sex, 'SibSp':sibsp, 'Parch':parch, 'Fare':fare, "Embarked":embarked}])

if st.button('Load'):
    st.write(data)

model = load_model('model.h5')

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

with open('OneHotEncoder.pkl','rb') as file:
    ohe = pickle.load(file)
    
with open('scale.pkl', 'rb') as file:
    sc = pickle.load(file)

data['Sex'] = le.fit_transform(data['Sex']) # Label encoding only Sex column

embarked = ohe.transform(data[['Embarked']]) # ont hot encoding for only Embarked column

embarked = pd.DataFrame(embarked, columns=ohe.get_feature_names_out())

data = pd.concat([data.drop(columns=['Embarked']),embarked],axis=1)

data[['Pclass', 'SibSp','Parch','Fare']] = sc.transform(data[['Pclass', 'SibSp','Parch','Fare']]) # scaler encoding other 4 columns

y = model.predict(data)

y= y[0][0]

def chance(p):
    if p>0.5:
        return "Will Survive"
    else:
        return "no, not going to survive"
    
if st.button('Load 2'):
    st.write('Probability of passenger surviaval chance',y)
    st.write(chance(y))