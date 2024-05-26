
import json
import pickle
from pprint import pprint
import json

from flask import Flask,request,app,jsonify,url_for,render_template
#from flask import Flask,jsonify,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)
app=application
## Load the model
model=pickle.load(open('artifacts/model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        req_data = json.loads(request.data.decode('utf-8'))
        req_data = req_data.get('data')
        data=CustomData(
            V1=req_data.get('V1'),
            V2=req_data.get('V2'),
            V3=req_data.get('V3'),
            V4=req_data.get('V4'),
            V5=req_data.get('V5'),
            V6=req_data.get('V6'),
            V7=req_data.get('V7'),
            V8=req_data.get('V8'),
            V9=req_data.get('V9'),
            V10=req_data.get('V10'),
            V11=req_data.get('V11'),
            V12=req_data.get('V12'),
            V13=req_data.get('V13'),
            V14=req_data.get('V14'),
            V15=req_data.get('V15'),
            V16=req_data.get('V16'),
            V17=req_data.get('V17'),
            V18=req_data.get('V18'),
            V19=req_data.get('V19'),
            V20=req_data.get('V20'),
            V21=req_data.get('V21'),
            V22=req_data.get('V22'),
            V23=req_data.get('V23'),
            V24=req_data.get('V24'),
            V25=req_data.get('V25'),
            V26=req_data.get('V26'),
            V27=req_data.get('V27'),
            V28=req_data.get('V28'),
            V29=req_data.get('V29'),
            V30=req_data.get('V30'),
            V31=req_data.get('V31'),
            V32=req_data.get('V32'),
            V33=req_data.get('V33'),
            V34=req_data.get('V34'),
            V35=req_data.get('V35'),
            V36=req_data.get('V36'),
            V37=req_data.get('V37'),
            V38=req_data.get('V38'),
            V39=req_data.get('V39'),
            V40=req_data.get('V40'),
            V41=req_data.get('V41'),
            V42=req_data.get('V42'),
            V43=req_data.get('V43'),
            V44=req_data.get('V44'),
            V45=req_data.get('V45'),
            V46=req_data.get('V46'),
            V47=req_data.get('V47'),
            V48=req_data.get('V48'),
            V49=req_data.get('V49'),
            V50=req_data.get('V50'),
            V51=req_data.get('V51'),
            V52=req_data.get('V52'),
            V53=req_data.get('V53'),
            V54=req_data.get('V54'),
            V55=req_data.get('V55'),
            V56=req_data.get('V56'),
            V57=req_data.get('V57'),
            V58=req_data.get('V58'),
            V59=req_data.get('V59'),
            V60=req_data.get('V60'),
            V61=req_data.get('V61'),
            V62=req_data.get('V62'),
            V63=req_data.get('V63'),
            V64=req_data.get('V64'),
            V65=req_data.get('V65'),
            V66=req_data.get('V66'),
            V67=req_data.get('V67'),
            V68=req_data.get('V68'),
            V69=req_data.get('V69'),
            V70=req_data.get('V70'),
            V71=req_data.get('V71'),
            V72=req_data.get('V72'),
            V73=req_data.get('V73'),
            V74=req_data.get('V74'),
            V75=req_data.get('V75'),
            V76=req_data.get('V76'),
            V77=req_data.get('V77'),
            V78=req_data.get('V78'),
            V79=req_data.get('V79'),
            V80=req_data.get('V80'),
            V81=req_data.get('V81'),
            V82=req_data.get('V82'),
            V83=req_data.get('V83'),
            V84=req_data.get('V84'),
            V85=req_data.get('V85'),
            V86=req_data.get('V86')
        )
        pred_df=pd.DataFrame([req_data])
        pred_df_list = list(pred_df)
        if pred_df.empty : print('Dataframe empty')
        print("Pred_df datatype=",type(pred_df),"&&  Pred_df=",pred_df_list)
        print("Before Prediction")
        ordered_cols = [f"V{i}" for i in range(1, 86)]
        pred_df.columns = ordered_cols 
        predict_pipeline=PredictPipeline()
        print("Mid Prediction") 
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        print(results[0])
        return jsonify(results[0])
        #return render_template('home.html',results=results[0])
        #return render_template('home.html',prediction_text="The prediction is {}".format(results[0]))

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        return render_template('home.html')
    else:
        #request.form = json.loads(request.data.decode('utf-8'))
        data_front=CustomData(
            V1=request.form.get('V1'),
            V2=request.form.get('V2'),
            V3=request.form.get('V3'),
            V4=request.form.get('V4'),
            V5=request.form.get('V5'),
            V6=request.form.get('V6'),
            V7=request.form.get('V7'),
            V8=request.form.get('V8'),
            V9=request.form.get('V9'),
            V10=request.form.get('V10'),
            V11=request.form.get('V11'),
            V12=request.form.get('V12'),
            V13=request.form.get('V13'),
            V14=request.form.get('V14'),
            V15=request.form.get('V15'),
            V16=request.form.get('V16'),
            V17=request.form.get('V17'),
            V18=request.form.get('V18'),
            V19=request.form.get('V19'),
            V20=request.form.get('V20'),
            V21=request.form.get('V21'),
            V22=request.form.get('V22'),
            V23=request.form.get('V23'),
            V24=request.form.get('V24'),
            V25=request.form.get('V25'),
            V26=request.form.get('V26'),
            V27=request.form.get('V27'),
            V28=request.form.get('V28'),
            V29=request.form.get('V29'),
            V30=request.form.get('V30'),
            V31=request.form.get('V31'),
            V32=request.form.get('V32'),
            V33=request.form.get('V33'),
            V34=request.form.get('V34'),
            V35=request.form.get('V35'),
            V36=request.form.get('V36'),
            V37=request.form.get('V37'),
            V38=request.form.get('V38'),
            V39=request.form.get('V39'),
            V40=request.form.get('V40'),
            V41=request.form.get('V41'),
            V42=request.form.get('V42'),
            V43=request.form.get('V43'),
            V44=request.form.get('V44'),
            V45=request.form.get('V45'),
            V46=request.form.get('V46'),
            V47=request.form.get('V47'),
            V48=request.form.get('V48'),
            V49=request.form.get('V49'),
            V50=request.form.get('V50'),
            V51=request.form.get('V51'),
            V52=request.form.get('V52'),
            V53=request.form.get('V53'),
            V54=request.form.get('V54'),
            V55=request.form.get('V55'),
            V56=request.form.get('V56'),
            V57=request.form.get('V57'),
            V58=request.form.get('V58'),
            V59=request.form.get('V59'),
            V60=request.form.get('V60'),
            V61=request.form.get('V61'),
            V62=request.form.get('V62'),
            V63=request.form.get('V63'),
            V64=request.form.get('V64'),
            V65=request.form.get('V65'),
            V66=request.form.get('V66'),
            V67=request.form.get('V67'),
            V68=request.form.get('V68'),
            V69=request.form.get('V69'),
            V70=request.form.get('V70'),
            V71=request.form.get('V71'),
            V72=request.form.get('V72'),
            V73=request.form.get('V73'),
            V74=request.form.get('V74'),
            V75=request.form.get('V75'),
            V76=request.form.get('V76'),
            V77=request.form.get('V77'),
            V78=request.form.get('V78'),
            V79=request.form.get('V79'),
            V80=request.form.get('V80'),
            V81=request.form.get('V81'),
            V82=request.form.get('V82'),
            V83=request.form.get('V83'),
            V84=request.form.get('V84'),
            V85=request.form.get('V85'),
            V86=request.form.get('V86')
        )
        # pred_df=pd.DataFrame([data_front])
        # pred_df_list = list(pred_df)
        # if pred_df.empty : print('Dataframe empty')
        # print("Pred_df datatype=",type(pred_df),"&&  Pred_df=",pred_df_list)
        # print("Before Prediction")
        # #ordered_cols = [f"V{i}" for i in range(1, 86)]
        # #pred_df.columns = ordered_cols 
        # predict_pipeline=PredictPipeline()
        # print("Mid Prediction") 
        # results=predict_pipeline.predict(pred_df)
        # print("after Prediction")
        # print(results[0])
        # #return jsonify(results[0])
        # #return render_template('home.html',results=results[0])
        # return render_template('home.html',prediction_text="The prediction is {}".format(results[0]))

        pred_df=data_front.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        print(results[0])
        return render_template('home.html',prediction_text="The prediction is {}".format(results[0]))
        

if __name__=="__main__":
    app.run(host="0.0.0.0")
