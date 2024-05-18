import sys
import os
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self,train_df):
        '''
        This function is responsible for data transformation
        
        '''
        try:
            target_column = 'V86'
            all_columns = train_df.columns
            categorical_columns = ['V1','V4','V5','V6','V44']
            numerical_columns = [col for col in all_columns if col not in categorical_columns and col != target_column]

            logging.info(f"Removal of correlated columns from preprocessing object starts")
            correlated_columns = self.correlated_columns(train_df.drop(columns=[target_column]), threshold=0.8)
            numerical_columns = [col for col in numerical_columns if col not in correlated_columns]
            logging.info(f"Removal of correlated columns preprocessing object ends")

            print("categorical columns",categorical_columns)
            print("numerical columns", numerical_columns)
            
            num_pipeline= Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler",StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder(min_frequency =  500, 
                                                 handle_unknown = 'ignore',
                                                 drop = 'first'))          
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                transformers=[
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def correlated_columns(self,dataset, threshold):
        col_corr = set()  
        corr_matrix = dataset.corr()
        try:
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if (corr_matrix.iloc[i, j]) > threshold: 
                        colname = corr_matrix.columns[i]  
                        col_corr.add(colname)
            return col_corr
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read train and test data completed")

            '''
            Removal of Correlated columns
            '''
            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object(train_df)
            logging.info(f"Removal of correlated columns starts")
            correlated_df = train_df.drop(['V1','V4','V5','V6','V44'],axis=1)

            correlated_columns = self.correlated_columns(correlated_df,0.8)
            print("correlated_columns= ",correlated_columns)
            train_df = train_df.drop(correlated_columns,axis=1)
            test_df = test_df.drop(correlated_columns,axis=1)
            logging.info(f"Removal of correlated columns ends")

            target_column_name='V86'
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)