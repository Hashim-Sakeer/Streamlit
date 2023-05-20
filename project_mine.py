import streamlit as st
import openai
from streamlit_chat import message
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder  
import joblib

openai.api_key=st.secrets["api_secret"]

def generate_response(x):
    completions=openai.Completion.create(
        engine="text-davinci-003",
        prompt=x,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message=completions.choices[0].text
    return message


def preprocess_data(df):
   
    df = df.fillna(0)

    label_encoder = LabelEncoder()
    for column in df.select_dtypes(include="object"):
        df[column] = label_encoder.fit_transform(df[column])
    
    
    df = pd.get_dummies(df) 
    
    df_normalized = (df - df.min()) / (df.max() - df.min())  # Min-Max scaling
    
    return df_normalized


def train_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model = LinearRegression().fit(X, y)
    return model


def home():
    st.subheader(
        """Upload a csv dataset for cleaning , label encoding , Onehot Encoding , Normalising and train a linear regression model""")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        preprocessed_data = preprocess_data(df)
        model = train_model(preprocessed_data)
    
    
        joblib.dump(model, "trained_model.pkl")
        preprocessed_data.to_csv("preprocessed_data.csv", index=False)

        st.write("click here for downloading preprocessed data:")

        st.download_button(
             label="Download data",
            data=open("preprocessed_data.csv", "rb").read(),
            file_name="preprocessed_data.csv",
            mime="text/csv"
        )   
    
        st.write("click here for downloading the model:")

        st.download_button(
            label="Download Model",
            data=open("trained_model.pkl", "rb").read(),
            file_name="trained_model.pkl",
            mime="application/octet-stream"
        )

    st.write("""You can chat with the bot in the help 
    centre page for asking questions and clearing doubts.""")

if 'outputs' not in st.session_state:
    st.session_state['outputs']=[]

if 'user_inputs' not in st.session_state:
    st.session_state['user_inputs']=[]

def help():    
    st.title("chatbot")


    
    def get_text():
        input_text=st.text_input("you:","")
        return input_text

    user_input=get_text()
    if user_input:
        output=generate_response(user_input)

        st.session_state.user_inputs.append(user_input)
        st.session_state.outputs.append(output)

    if st.session_state['outputs']:
        for i in range(len(st.session_state['outputs'])-1,-1,-1):
            message(st.session_state["outputs"][i],key=str(i))
            message(st.session_state["user_inputs"][i], is_user=True,key=str(i) + '_user')
menu=['home','help centre']
choice=st.sidebar.selectbox('',menu)
if choice=="home":
    home()

if choice=="help centre":
    help()