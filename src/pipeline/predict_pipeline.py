import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts",'model.pkl')
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")

            # Check if all required columns are present in the input data
            # required_columns = preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
            # print("Required columns :",required_columns)
            # print("Feature columns:",features.columns) 
            # missing_columns = set(required_columns) - set(features.columns)
            # if missing_columns:
            #     raise ValueError(f"Missing columns in dataset: {missing_columns}")
            
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
                V1 : int,
                V2: int,
                V3: int,
                V4 : int,
                V5 : int,
                V6 : int,
                V7: int,
                V8: int,
                V9: int,
                V10: int,
                V11: int,
                V12: int,
                V13: int,
                V14: int,
                V15: int,
                V16: int,
                V17: int,
                V18: int,
                V19: int,
                V20: int,
                V21: int,
                V22: int,
                V23: int,
                V24: int,
                V25: int,
                V26: int,
                V27: int,
                V28: int,
                V29: int,
                V30: int,
                V31: int,
                V32: int,
                V33: int,
                V34: int,
                V35: int,
                V36: int,
                V37: int,
                V38: int,
                V39: int,
                V40: int,
                V41: int,
                V42: int,
                V43: int,
                V44: int,
                V45: int,
                V46: int,
                V47: int,
                V48: int,
                V49: int,
                V50: int,
                V51: int,
                V52: int,
                V53: int,
                V54: int,
                V55: int,
                V56: int,
                V57: int,
                V58: int,
                V59: int,
                V60: int,
                V61: int,
                V62: int,
                V63: int,
                V64: int,
                V65: int,
                V66: int,
                V67: int,
                V68: int,
                V69: int,
                V70: int,
                V71: int,
                V72: int,
                V73: int,
                V74: int,
                V75: int,
                V76: int,
                V77: int,
                V78: int,
                V79: int,
                V80: int,
                V81: int,
                V82: int,
                V83: int,
                V84: int,
                V85: int,
                V86: int):
            self.V1 = V1
            self.V2 = V2
            self.V3 = V3
            self.V4 = V4
            self.V5 = V5
            self.V6 = V6
            self.V7 = V7
            self.V8 = V8
            self.V9 = V9
            self.V10 = V10
            self.V11 = V11
            self.V12 = V12
            self.V13 = V13
            self.V14 = V14
            self.V15 = V15
            self.V16 = V16
            self.V17 = V17
            self.V18 = V18
            self.V19 = V19
            self.V20 = V20
            self.V21 = V21
            self.V22 = V22
            self.V23 = V23
            self.V24 = V24
            self.V25 = V25
            self.V26 = V26
            self.V27 = V27
            self.V28 = V28
            self.V29 = V29
            self.V30 = V30
            self.V31 = V31
            self.V32 = V32
            self.V33 = V33
            self.V34 = V34
            self.V35 = V35
            self.V36 = V36
            self.V37 = V37
            self.V38 = V38
            self.V39 = V39
            self.V40 = V40
            self.V41 = V41
            self.V42 = V42
            self.V43 = V43
            self.V44 = V44
            self.V45 = V45
            self.V46 = V46
            self.V47 = V47
            self.V48 = V48
            self.V49 = V49
            self.V50 = V50
            self.V51 = V51
            self.V52 = V52
            self.V53 = V53
            self.V54 = V54
            self.V55 = V55
            self.V56 = V56
            self.V57 = V57
            self.V58 = V58
            self.V59 = V59
            self.V60 = V60
            self.V61 = V61
            self.V62 = V62
            self.V63 = V63
            self.V64 = V64
            self.V65 = V65
            self.V66 = V66
            self.V67 = V67
            self.V68 = V68
            self.V69 = V69
            self.V70 = V70
            self.V71 = V71
            self.V72 = V72
            self.V73 = V73
            self.V74 = V74
            self.V75 = V75
            self.V76 = V76
            self.V77 = V77
            self.V78 = V78
            self.V79 = V79
            self.V80 = V80
            self.V81 = V81
            self.V82 = V82
            self.V83 = V83
            self.V84 = V84
            self.V85 = V85
            self.V86 = V86

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "V1":[self.V1],
                "V2":[self.V2],
                "V3":[self.V3],
                "V4":[self.V4],
                "V5":[self.V5],
                "V6":[self.V6],
                "V7":[self.V7],
                "V8":[self.V8],
                "V9":[self.V9],
                "V10":[self.V10],
                "V11":[self.V11],
                "V12":[self.V12],
                "V13":[self.V13],
                "V14":[self.V14],
                "V15":[self.V15],
                "V16":[self.V16],
                "V17":[self.V17],
                "V18":[self.V18],
                "V19":[self.V19],
                "V20":[self.V20],
                "V21":[self.V21],
                "V22":[self.V22],
                "V23":[self.V23],
                "V24":[self.V24],
                "V25":[self.V25],
                "V26":[self.V26],
                "V27":[self.V27],
                "V28":[self.V28],
                "V29":[self.V29],
                "V30":[self.V30],
                "V31":[self.V31],
                "V32":[self.V32],
                "V33":[self.V33],
                "V34":[self.V34],
                "V35":[self.V35],
                "V36":[self.V36],
                "V37":[self.V37],
                "V38":[self.V38],
                "V39":[self.V39],
                "V40":[self.V40],
                "V41":[self.V41],
                "V42":[self.V42],
                "V43":[self.V43],
                "V44":[self.V44],
                "V45":[self.V45],
                "V46":[self.V46],
                "V47":[self.V47],
                "V48":[self.V48],
                "V49":[self.V49],
                "V50":[self.V50],
                "V51":[self.V51],
                "V52":[self.V52],
                "V53":[self.V53],
                "V54":[self.V54],
                "V55":[self.V55],
                "V56":[self.V56],
                "V57":[self.V57],
                "V58":[self.V58],
                "V59":[self.V59],
                "V60":[self.V60],
                "V61":[self.V61],
                "V62":[self.V62],
                "V63":[self.V63],
                "V64":[self.V64],
                "V65":[self.V65],
                "V66":[self.V66],
                "V67":[self.V67],
                "V68":[self.V68],
                "V69":[self.V69],
                "V70":[self.V70],
                "V71":[self.V71],
                "V72":[self.V72],
                "V73":[self.V73],
                "V74":[self.V74],
                "V75":[self.V75],
                "V76":[self.V76],
                "V77":[self.V77],
                "V78":[self.V78],
                "V79":[self.V79],
                "V80":[self.V80],
                "V81":[self.V81],
                "V82":[self.V82],
                "V83":[self.V83],
                "V84":[self.V84],
                "V85":[self.V85],
                "V86":[self.V86]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)