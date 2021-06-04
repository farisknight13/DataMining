import streamlit as st
import plotly.figure_factory as ff
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# This code is different for each deployed app.
CURRENT_THEME = "light"
IS_DARK_THEME = False

# This code is the same for each deployed app.
col1, col2, col3 = st.beta_columns(3)
pred_btn = col2.image("lip.png",width=200)

#Title
st.markdown("<h1 style='text-align: center; font-size: 65px;'>My Lipstick Texture</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>เลือกคำตอบที่เป็นตัวของคุณเองมากที่สุดเพียงคำตอบเดียว</h2>", unsafe_allow_html=True)

THEMES = [
    "light",
    "dark",
    "green",
    "blue",
]
GITHUB_OWNER = "streamlit"


st.markdown('<style>.p, ol, ul, dl {margin: 0px 0px 1rem;padding: 0px;font-size: 20px;font-weight: 400;}</style>', unsafe_allow_html=True)
st.error("""คำถามข้อที่ 1 : โทนสีลิปสติกที่คุณเลือกใช้ ?""")

st.markdown('<style>.st-af {font-size: 20px;}</style>', unsafe_allow_html=True)
q1_name = ['โทนสีส้ม','โทนสีแดง','โทนสีชมพู','โทนสีนู้ด','โทนสีน้ำตาล']

st.markdown('<style>.css-145kmo2 {font-size: 1.0rem;color: rgb(38, 39, 48);margin-bottom: 0.4rem;}</style>', unsafe_allow_html=True)
page_1 = st.radio('เลือกเพียง 1 คำตอบ',q1_name)


st.error("""คำถามข้อที่ 2 : ความมันของเนื้อลิปสติกที่คุณเลือกใช้ ?""")
page_2 = st.select_slider("น้อย -> มาก", [1, 2, 3, 4, 5])

st.error("""คำถามข้อที่ 3 : ราคาของลิปสติกตามเนื้อลิปที่เลือกด้านบน ?""")
q3_name = ['น้อยกว่า 200 บาท','201-500 บาท','501-800 บาท','มากกว่า 800 บาท']
page_3 = st.radio('เลือกเพียง 1 คำตอบ',q3_name)

st.error("""คำถามข้อที่ 4 : ช่วงอายุของคุณ ?""" )
q4_name = ['น้อยกว่า 15 ปี','15-24 ปี','25-35 ปี','มากกว่า 35 ปี']
page_4 = st.radio('เลือกเพียง 1 คำตอบ',q4_name)

st.error("""คำถามข้อที่ 5 :โทนสีผิวของคุณ ?""")
q5_name = ['ผิวขาวอมชมพู','ผิวขาวเหลือง','ผิวสองสี','ผิวแทน','ผิวคล้ำ']
page_5 = st.radio('เลือกเพียง 1 คำตอบ',q5_name)

st.error("""คำถามข้อที่ 6 : ลักษณะผิวริมฝีปากของคุณ ?""")
q6_name = ['ริมฝีปากสีคล้ำ','ริมฝีปากอมชมพู','ริมฝีปากซีด']
page_6 = st.radio('เลือกเพียง 1 คำตอบ',q6_name)

st.error("""คำถามข้อที่ 7 : มีปัญหาปากแห้ง แตก หรือเป็นขุยหรือไม่ ?""")
q7_name = ['มี','ไม่มี']
page_7 = st.radio('เลือกเพียง 1 คำตอบ',q7_name)

st.error("""คำถามข้อที่ 8 : สไตล์ของเสื้อผ้าที่คุณเลือกใส่ให้เข้ากับสีลิปสติก ?""")
q8_name = ['วินเทจ','มินิมอล','อเมริกัน','เกาหลี','สตรีท']
page_8 = st.radio('เลือกเพียง 1 คำตอบ',q8_name)

st.error("""คำถามข้อที่ 9 : สไตล์การแต่งหน้าของคุณกับเนื้อลิปสติกที่เลือกใช้ด้านบน ?""")
q9_name = ['สไตล์ธรรมชาติ','สไตล์ฝรั่ง','สไตล์เกาหลี','สไตล์ญีุ่่ปุ่น']
page_9 = st.radio('เลือกเพียง 1 คำตอบ',q9_name)

st.error("""คำถามข้อที่ 10 : ความถี่ในการเติมลิปสติกโดยประมาณระหว่างวันของคุณ ?""")
q10_name = ['น้อย (ไม่เติมระหว่างวัน)','ปานกลาง (1-2 ครั้ง)','มาก (มากกว่า 2 ครั้ง)']
page_10 = st.radio('เลือกเพียง 1 คำตอบ',q10_name)

st.error("""คำถามข้อที่ 11 : การใช้สีในการทาลิปสติกของคุณ ?""")
q11_name = ['ผสมสี','สีเดียว']
page_11 = st.radio('เลือกเพียง 1 คำตอบ',q11_name)

st.error("""คำถามข้อที่ 12 : ลักษณะการทาลิปสติกของคุณ ?""")
q12_name = ['ทาแบบเต็มปาก','ทาแบบเบลอขอบปาก','ทาแบบเกินขอบปาก']
page_12 = st.radio('เลือกเพียง 1 คำตอบ',q12_name)

st.error("""คำถามข้อที่ 13 : จากเนื้อลิปสติกที่คุณเลือกใช้ คุณใช้เพื่อออกไปทำกิจกรรมใด ?""")
q13_name = ['เพื่อไปเที่ยว','เพื่อไปออกกำลังกาย','เพื่อไปเรียน','เพื่อไปออกงานสำคัญ','เพื่อไปทำงาน']
page_13 = st.radio('เลือกเพียง 1 คำตอบ',q13_name)

st.error("""คำถามข้อที่ 14 : ช่วงเวลาที่คุณใช้ทำกิจกรรมดังกล่าว ? """)
q14_name = ['กลางวัน','กลางคืน']
page_14 = st.radio('เลือกเพียง 1 คำตอบ',q14_name)

data = {'2': page_1,
        '3': page_3,
        '4': page_4,
        '5': page_5,
        '6': page_6,
        '7': page_7,
        '8': page_9,
        '9': page_8,
        '10': page_13,
        '11': page_14,
        '12': page_10,
        '13': page_11,
        '14': page_2,
        '15': page_12}
features = pd.DataFrame(data, index=[0])
#st.write(features)

lipstick_raw = pd.read_csv('https://raw.githubusercontent.com/farisknight13/DataMining/main/dataset_lipstick.csv')
lipstick = lipstick_raw.drop(columns=['1'])
df = pd.concat([features,lipstick],axis=0)

def feature_extraction(featMat):
    encode = ['2','3','4','5','6','7','8','9','10','11','12','13','15']
    for col in encode:
        dummy = pd.get_dummies(featMat[col], prefix=col)
        featMat = pd.concat([featMat,dummy], axis=1)
        del featMat[col]
    featMat = featMat[:1] # Selects only the first row (the user input data)
    featMat = featMat.iloc[:,[3,4,6,7,8,9,10,12,25,27,34,38,39,40,41,42,43,45]]
    return  featMat

load_model = pickle.load(open('LipsStick_clf_finish_final_superfinal.pkl', 'rb'))
df = feature_extraction(df)

#col1, col2, col3, col4 = st.beta_columns(4)
#predict_btn = col4.button("PREDICT")

st.markdown('<style>.css-13nzvf {display: inline-flex;-webkit-box-align: center;align-items: center;-webkit-box-pack: center;justify-content: center;font-weight: 400;padding: 0.25rem 19.7rem;border-radius: 0.25rem;margin: 0px;line-height: 1.6;color: inherit;width: auto;background-color: rgb(254, 247, 246);border: 1px solid rgba(38, 39, 48, 0.1);}</style>', unsafe_allow_html=True)
predict_btn = st.button("PREDICT")

if (predict_btn) :
    prediction = load_model.predict(df)
    #st.write(prediction)
    if (prediction == '1'):
        img = st.image("matte.jpg",width=697)
    elif (prediction == '2'):
        img = st.image("cream.jpg",width=697)
    elif (prediction == '3'):
        img = st.image("sheer.jpg",width=697)
    elif (prediction == '4'):
        img = st.image("liquid.jpg",width=697)
    elif (prediction == '5'):
        img = st.image("tint.jpg",width=697)
    elif (prediction == '6'):
        img = st.image("blam.jpg",width=697)
    elif (prediction == '7'):
        img = st.image("liner.jpg",width=697)

