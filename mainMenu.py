import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split



df = " "

def mainMenu():
    global df
st.header("Loan Default Prediction")
data = st.file_uploader("Upload dataset:",type=['csv','xlsx'], key="MainMenu")
if data is not None:
    st.success("Data successfully loaded")
    
if data is not None:
        if data is not None:
            df = pd.read_csv(data, ';')
            st.dataframe(df)
    
        
        st.sidebar.subheader("Set inputs for Prediction")      
        #if st.sidebar.checkbox('Select Multiple Columns'):
        new_data = st.multiselect('Select preferred colunmn features',df.columns)
        df1=df[new_data]
        st.dataframe(df1)    
            
        df.drop(['account_check_status','purpose','savings','personal_status_sex',
                 'other_debtors','property','other_installment_plans','housing',
                 'job','telephone','foreign_worker','present_res_since'],axis=1,inplace=True)    
        le_CreditHistory = LabelEncoder()
        df['CreditHistory'] = le_CreditHistory.fit_transform(df['CreditHistory'])
        df["CreditHistory"].unique()
    
        le_Employment = LabelEncoder()
        df['Employment'] = le_Employment.fit_transform(df['Employment'])
        df["Employment"].unique()
    
        CreditHistory = (
        "all credits at this bank paid back duly",
        "critical account/ other credits existing (not at this bank)",
        "delay in paying off in the past",
        "existing credits paid back duly till now",
        "no credits taken/ all credits paid back duly",
        )
    
        Employment = (
            "unemployed",
            "... < 1 year ",
            "1 <= ... < 4 years",
            "4 <= ... < 7 years",
            ".. >= 7 years",
            )
        
        CreditHistory = st.selectbox("Credit History",CreditHistory)
        Employment = st.selectbox("Number of years employed",Employment)
    
        Age = st.slider("Age",0,75,18)
        CreditAmount = st.number_input("Credit Amount Outstanding",min_value=0,max_value=50000,step=1)
        duration_in_month = st.sidebar.slider("Duration of the credit in months",0,100,1)
        installment_as_income_perc = st.sidebar.slider("Outstanding Instalments",0,5,1)
        Dependents = st.sidebar.slider("Number of Depedents",0,10,0)
        credits_this_bank = st.sidebar.slider("Number of Active accounts",0,5,1)
        
        X = df.drop(columns=['Default'])
        y = df['Default']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
        
      
        X = np.array([[CreditHistory,CreditAmount,Employment,Age,duration_in_month,installment_as_income_perc,Dependents,credits_this_bank]])
        X[:,0] = le_CreditHistory.transform(X[:,0])
        X[:,2] = le_Employment.transform(X[:,2])
        X = X.astype(int)
        svc = SVC(kernel='rbf', gamma='auto')
        svc.fit(X_train, y_train) 
        y_pred = svc.predict(X_test)
        
        st.subheader("0 = Default and 1 = Non-Default")
        ok = st.button("Prediction")
        if ok:
            st.write("The client will",y_pred[1])  
    
        #accuracy=accuracy_score(y_test,y_pred)
        #st.write('Accuracy',accuracy)
        #st.write("Classifier report:",classification_report(y_test, y_pred))
        
        #st.subheader("Confusion matrix")
        #confusion_matrix(y_test, y_pred)
        #st.write(confusion_matrix(y_test, y_pred))

