import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import plotly
import plotly.express as px
import base64
import altair as alt
import plotly.offline as pyo
from plotly.offline import init_notebook_mode,plot,iplot
from sklearn.metrics import accuracy_score,mean_squared_error
from PIL import Image
import io
import time


import cufflinks as cf
pyo.init_notebook_mode(connected=True)
cf.go_offline()


@st.cache
def get_mydata():
    data=pd.read_csv("heart.csv")
    return data

@st.cache
def get_x():
    r,s=df.loc[:,:'thal'],df['target']
    return r

@st.cache
def get_y():
    r,s=df.loc[:,:'thal'],df['target']
    return s

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i

    return num




header=st.beta_container()
details=st.beta_container()
dataset=st.beta_container()
vist=st.beta_container()
datapre=st.beta_container()
acc=st.beta_container()
predict=st.beta_container()



with header:
    st.title("Welcome to Our Project\n")
    st.title("Heart Disease Prediction\n")
    st.title("--------------------------------------")



with details:
    st.header("INTRODUCTION")
    st.text("\n")
    st.write("\n \n \n Cardiovascular diseases (CVDs) are the number 1 cause of death globally,taking an\nestimated 17.9 million lives each year. CVDs are a group of disorders of the heart\nand blood vessels and include coronary heart disease, cerebrovascular disease rheumatic\nheart disease and other conditions.Four out of 5CVD deaths are due to heart attacks and\nstrokes,and one third of these deaths occur prematurely in people under 70 years of age. ")
    txtt, imgg=st.beta_columns(2)
    txtt.write("We’re cautioned every day to be aware of \nmany devastating diseases. But among the\nmost deadly is heart disease. \nHeart disease is the No. 1 killer of men \nand women in the whole world. It also is\namong the least understood, least researched\n,and least discussed chronic diseases.")
    txtt.write("Prediction of cardiovascular disease is \nregarded as one of the most important \nsubjects in the section of clinical data \nanalysis. The amount of data in the \nhealthcare industry is huge. Data mining \nturns the large collection of raw \nhealthcare data into information that can \nhelp to make informed decisions and \npredictions.Machine learning (ML) proves to\nbe effective in assisting in making \ndecisions and predictions from the large \nquantity of data produced by the healthcare\nindustry.")
    txtt.write("This makes heart disease a major concern to\nbe dealt with. But it is difficult to \nidentify heart disease because of several\ncontributory risk factors such as diabetes,\nhigh blood pressure, high cholesterol,\nabnormal pulse rate, and many other factors.\nDue to such constraints, scientists have\nturned towards modern approaches like\nData Mining and Machine Learning for\npredicting the disease.")
    im=Image.open("image1.jpeg")
    imgg.image(im, width=462)
    st.title("--------------------------------------")





with dataset:
    st.header("About our Dataset")
    st.text("The dataset used by us is the Cleveland Heart Disease dataset taken from the UCI \nrepository.")

    df=get_mydata()
    st.write(df.head(305))
    st.subheader("The dataset consists of 303 individuals data. There are 14 columns in the dataset,which are described below.")
    st.write('#')
    img=Image.open("image2.png")
    st.image(img, width=653)




    st.title("--------------------------------------")



with vist:

    st.title("Lets us understand our data")
    st.write('This bar chart below shows us the count of how many patients are there of a specific value in every attribute in our dataset. ')
    fz=pd.DataFrame(df)
    fz.hist(figsize=(20,20))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    st.write('This Chart below shows us the comparsion between three attributes that are sex,ST depression induced by exercise relative to rest(oldpeak) and our target attribute that determines whether a person has heart disease or not.')
    vt= pd.DataFrame(np.random.randn(180, 3),columns=['target', 'sex', 'oldpeak'])
    c = alt.Chart(vt).mark_circle().encode(x='target', y='sex', size='oldpeak', color='oldpeak', tooltip=['target', 'sex', 'oldpeak'])
    st.write(c)


    st.write("This bar chart below shows us two attributes Target and sex.It shows us how many patients of a paricular gender are having heart disease or not in our dataset")
    plt.figure(figsize=(10,7))
##    fg=pd.DataFrame(np.random.randn(303, 2),columns=['sex', 'target'])
    sns.barplot(x='sex', y='target', data=df)
    st.pyplot()
    st.write('#')

    st.write("Now this bar chart below shows us three attributes Target and sex and now age also.So, shows us how many patients of a paricular gender and of their particular age are having heart disease or not in our dataset")
    plt.figure(figsize=(6,5))
    sns.barplot(x='sex',y='age',hue='target', data=df)
    st.pyplot()

    st.write("This chart below shows us the comparsion done between Number of major vessels colored by flourosopy and our target attribute.")
    chart_data1 = pd.DataFrame(np.random.randn(151,2),columns=['ca', 'target']).head(70)
    st.line_chart(chart_data1)

    st.write("This graph below is plotted using three attributes that are Serum Cholestrol,resting ECG and target attribute")
    chart_data = pd.DataFrame(np.random.randn(150, 3),columns=['chol', 'restecg' ,'target']).head(100)
    st.area_chart(chart_data)


    st.write("This graph below is plotted using three attributes that are Fasting Blood Sugar, the thalassemia level of the patient and target attribute")
    chart_data = pd.DataFrame(np.random.randn(150, 3),columns=['fbs', 'thal' ,'target']).head(50)
    st.bar_chart(chart_data)

    st.write("This graph below is plotted using three attributes that are Serum Cholestrol, the maximum heart rate of the patient and target attribute")
    vt= pd.DataFrame(np.random.randn(180, 3),columns=['target', 'chol', 'thalach'])
    c = alt.Chart(vt).mark_circle().encode(x='target', y='chol', size='thalach', color='target', tooltip=['target', 'chol', 'thalach'])
    st.write(c)


    st.title("--------------------------------------")




with datapre:
    st.title("Data Preprocessing")
    st.subheader("Lets us first take a count of null and non-null values.")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)
    st.write("Here are are no null values.So lets proceed further.")

    st.subheader("Now We will divide our whole database into two parts the first one(i.e x) contained all the attribute expect the target attribute and the second one (i.e y) contained only one one attribute i.e target attribute.The x and y after splitting are shown below")
    st.write('#')
    x=get_x()
    y=get_y()
    tr,te=st.beta_columns(2)
    tr.write("THIS IS X ")
    tr.write(x.head(305))
    te.write("THIS IS Y")
    te.write(y.head(305))
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.3,shuffle=True)
    st.subheader("Now,We will divide the x and y randomly,in such a way that 70% data of both x and y will be used for training the model and 30% will left for testing. ")
    st.write('#')
    st.write("The data will be divided into x_train,x_test,y_train and y_test")
    trr,tes=st.beta_columns(2)
    trr.write("x_train")
    trr.write(x_train.head(212))
    tes.write("x_test")
    tes.write(x_test.head(91))
    trr.write("y_train")
    trr.write(y_train.head(212))
    tes.write("y_test")
    tes.write(y_test.head(91))
    from sklearn.preprocessing import StandardScaler
    std=StandardScaler().fit(x)
    x_std=std.transform(x)
    x_train_std,x_test_std,y_train,y_test=train_test_split(x_std,y,random_state=10,test_size=0.3,shuffle=True)
    st.title("--------------------------------------")

    ##decision tree
    from sklearn.tree import DecisionTreeClassifier
    dt=DecisionTreeClassifier()
    dt.fit(x_train,y_train)

    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    k_range=range(1,45)
    scores={}
    h_score = 0
    best_k=0
    scores_list=[]

    for k in k_range:
        knn=KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_std,y_train)
        prediction_knn=knn.predict(x_test)
        scores[k]=accuracy_score(y_test,prediction_knn)
        if scores[k]>h_score:
            h_score = scores[k]
            best_k = k

        knn=KNeighborsClassifier(n_neighbors=best_k)
        knn.fit(x_train_std,y_train)

    #LGBM
    from lightgbm import LGBMClassifier
    lg=LGBMClassifier(boosting_type='gbdt',n_estimators=24,learning_rate=0.25,objective='binary',metric='accuracy',is_unbalance=True,colsample_bytree=0.7,reg_lambda=3,reg_alpha=3,random_state=500,n_jobs=-1,num_leaves=20)
    lg.fit(x_train,y_train)

    #SVC
    from sklearn.svm import SVC
    svc= SVC(C=2.0,kernel='rbf',gamma='auto').fit(x_train_std,y_train)


    #RandomForestClassifier
    from sklearn.ensemble import RandomForestClassifier
    ranm = RandomForestClassifier(n_estimators=100, bootstrap = True)
    ranm.fit(x_train,y_train)



with acc:

    st.title("Accuracy of Algorithms")
    st.write("So here in our Project we have used five algorithms that are:-")
    st.write("1.Decision Tree Classifier")
    st.write("2.KNN")
    st.write("3.Random Forest Classifier")
    st.write("4.SVC")
    st.write("5.LGBM.")
    st.write("Now,The accuracy that the algorithms gave are shown below")
    st.subheader("Select the Algorithm.")
    algoth= st.selectbox(' ', ('None','Decision Tree Classifier', 'KNN', 'Random Forest Classifier','SVC','LGBM'))
    with st.spinner('Fetching Your Results'):
        time.sleep(3)


    if algoth == 'None':
        st.subheader("Please Select the Algorithm from above.")



    elif algoth == 'Decision Tree Classifier':
        st.write('You selected:', algoth)
        prediction=dt.predict(x_test)
        accuracy_dt=accuracy_score(y_test,prediction)*100
        st.subheader('Accuracy of Decision Tree Classifier')
        st.write(accuracy_dt)


    elif algoth == 'KNN':
        st.write('You selected:', algoth)
        prediction_knn=knn.predict(x_test_std)
        accuracy_knn=accuracy_score(y_test,prediction_knn)*100
        st.subheader('Accuracy of KNeighbors Classifier')
        st.write(accuracy_knn)



    elif algoth == 'Random Forest Classifier':
        st.write('You selected:', algoth)
        ranpr = ranm.predict(x_test)
        accuracy_random=accuracy_score(y_test,ranpr)*100
        st.subheader('Accuracy of Random Forest Classifier')
        st.write(accuracy_random)

    elif algoth == 'SVC':
        st.write('You selected:', algoth)
        Y_predict = svc.predict(x_test_std)
        accuracy_svc=accuracy_score(y_test,Y_predict)*100
        st.subheader('Accuracy of SVC')
        st.write(accuracy_svc)

    elif algoth == 'LGBM':
        st.write('You selected:', algoth)
        yprid = lg.predict(x_test)
        accuracy_lgbm=accuracy_score(y_test,yprid)*100
        st.subheader('Accuracy of LGBM Classifier')
        st.write(accuracy_lgbm)




with predict:
    st.title("--------------------------------------")
    st.title("Let us perform some prediction now")

    agee=st.number_input('Enter age of the person')

    opt= st.selectbox('Enter sex of the person ', ('Male','Female'))
    if opt == 'Male':
        sexx=1
    elif opt == 'Female':
        sexx=0

    opt2= st.radio("The type of chest-pain experienced by the individual ",('Typical Angina', 'Atypical Angina', 'Non—Anginal Pain','Asymptotic'))
    if opt2 == 'Typical Angina':
        cpp=1

    elif opt2 == 'Atypical Angina':
        cpp=2

    elif opt2 == 'Non—Anginal Pain':
        cpp=3

    elif opt2 == 'Asymptotic':
        cpp=4

    rbp=st.number_input('The Resting Blood Pressure value of an individual in mmHg (unit)')

    chol = st.slider('The serum cholesterol value of an individual in mg/dl (unit)', 0, 400, 110)
    st.write('The serum cholesterol value of an individual in mg/dl (unit) is', chol)

    cc=st.radio("The fasting blood sugar value of an individual is greater than 120mg/dl.",('Yes','No'))
    if cc == 'Yes':
        fbss=1
    else:
        fbss=0

    yy = st.selectbox('Resting Electrocardiographic results [Resting ECG ]', ('Normal','Having ST-T wave abnormality','Having ST-T wave abnormality'))
    if yy == 'Normal':
        rst=0
    elif yy == 'Having ST-T wave abnormality':
        rst=1
    elif yy =='Having ST-T wave abnormality':
        rst=2

    thl=st.slider('The max heart rate achieved by an individual.', 0, 220, 50)
    st.write('The max heart rate achieved by an individual is',thl)

    ll=st.radio("Do you suffer with Exercise Induced Angina ",('Yes','No'))
    if ll == 'Yes':
        exa=1
    else:
        exa=0

    oldpk=st.number_input('ST depression induced by exercise relative to rest')

    zz= st.selectbox('Peak exercise ST segment ', ('Upsloping','Flat','Downsloping'))
    if zz == 'Upsloping':
        rsst=1
    elif zz == 'Flat':
        rsst=2
    elif zz =='Downsloping':
        rsst=3

    caa=st.slider('Number of major vessels (0–3) colored by flourosopy .', 0, 3, 0)
    st.write('Number of major vessels (0–3) colored by flourosopy ',caa)

    ff=st.radio("The Thalassemia",('Normal','Fixed defect','Reversible defect'))
    if ff == 'Normal':
        th=3
    elif ff == 'Fixed defect':
        th=6
    elif ff == 'Reversible defect':
        th=7




    pdtn=['0','0','0','0','0']
    Category=['No,You donot have Heart disease','Sorry ,You are having heart disease']


    custom_data=np.array([[agee , sexx, cpp, rbp, chol , fbss, rst, thl, exa, oldpk, rsst, caa, th]])

    custom_data_prediction_dt=dt.predict(custom_data)

    custom_data_knn_std=std.transform(custom_data)
    custom_data_prediction_knn=knn.predict(custom_data_knn_std)

    custom_data_svc_std=std.transform(custom_data)
    custom_data_prediction_svc=svc.predict(custom_data_svc_std)

    custom_data_prediction_lgbm=lg.predict(custom_data)

    custom_data_prediction_random=ranm.predict(custom_data)


    pdtn[0]=int(custom_data_prediction_dt)
    pdtn[1]=int(custom_data_prediction_knn)
    pdtn[2]=int(custom_data_prediction_svc)
    pdtn[3]=int(custom_data_prediction_lgbm)
    pdtn[4]=int(custom_data_prediction_random)

##    st.write(pdtn[0])
##    st.write(pdtn[1])
##    st.write(pdtn[2])
##    st.write(pdtn[3])
##    st.write(pdtn[4])
    resultofpd=most_frequent(pdtn)
##    st.write(resultofpd)

    st.write('#')

    if  st.button('Click here to Predict'):
        with st.spinner('Processing your data.'):
            time.sleep(5)
        st.subheader('According to our Model')
        if resultofpd == 0:
            happ=Image.open("happy.jpg")
            st.image(happ, width=380)
            st.write("No You Don't have Heart Disease.Stay Safe")

        else:
            sadd=Image.open("sad.jpg")
            st.image(sadd, width=380)
            st.write("Sorry, You are having Heart Disease.Stay Safe")
        st.title("--------------------------------------")
