
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

app=Flask(__name__)
## Load the model
rf_model=pickle.load(open('artifacts/model.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        req_data = json.loads(request.data.decode('utf-8'))
        req_data = req_data.get('data') # send req now this will work
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
        #return jsonify({"results":results})
        return render_template('home.html',results=results[0])
if __name__=="__main__":
    #app.run(host="0.0.0.0")
    app.run(host="0.0.0.0",debug=True, port=5000)





# @app.route('/predict_api',methods=['POST'])
# def predict_api():
#     data=request.json['data']
#     # print(data)
#     # print(np.array(list(data.values())).reshape(1,-1))
#     # new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
#     # output=regmodel.predict(new_data)
#     # print(output[0])
#     # return jsonify(output[0])
#     new_data = list(map(float, data[].split()))
#     output=rf_model.predict(new_data)
#     print(output[0])
#     # print(f"Original Data \n : {data}")
#     # data = [data]
#     # print(f"Transform Data \n : {data}")
#     # data_df = pd.DataFrame(data)
#     # print(data_df.dtypes)
#     # output=rf_model.predict(data)
#     # print(output[0])
    

#     return jsonify(output[0])
# if __name__=="__main__":
#     app.run(debug=True)