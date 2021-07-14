import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
pickle_in = open("celebal_model.pkl","rb")
model=pickle.load(pickle_in)
dataset= pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,1:10].values
 
# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

def predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary):
  output= model.predict([[CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary]])
  print("Customer will leave =", output)
  if output==[1]:
    prediction="Customer will Leave"
  else:
    prediction="Customer will not Leave"
  print(prediction)
  return prediction
def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Celebal ML Internship</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;">Group - 8, Task-1 Project</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer Prediction")
    Age = st.number_input('Insert a Age',18,60)
    CreditScore= st.number_input('Insert a CreditScore',400,1000)
    HasCrCard = st.number_input('Insert a HasCrCard 0 For No 1 For Yes',0,1)
    Tenure = st.number_input('Insert a Tenure',0,20)
    Balance = st.number_input('Insert a Balance',0)
    Gender = st.number_input('Insert 0 For Male 1 For Female ',0,1)
    Geography= st.number_input('Insert Geography 0 For France 1 For Spain',0,1)
    IsActiveMember= st.number_input('Insert a IsActiveMember 0 For No 1 For Yes',0,1)
    EstimatedSalary= st.number_input('Insert a EstimatedSalary',0)
    
    
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.subheader("Developed by Team - 8")
      st.subheader("Celebal ML internship")

if __name__=='__main__':
  main()
   
