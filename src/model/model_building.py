import pandas as pd 
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import Pipeline
from logger.logger import get_logger
from src.feature.feature_eng import column_preprocessor
import pickle
import os

logger=get_logger(__name__)

def load_data(file_path:str)->pd.DataFrame:
    try:
        logger.info(f'loading data from {file_path}')
        return pd.read_csv(file_path)
    except FileNotFoundError as e:
        logger.error(f'file not found at {file_path}: {e}')
        raise
    except Exception as e:
        logger.error(f'error loading data from {file_path}: {e}')
        raise


# Churn Yes → 1, No → 0 (target variable)
def encode_target(series):    
    return (series == 'Yes').astype(int)


def spliting_data(data:pd.DataFrame)->tuple:
    try:
        logger.info('starting data splitting')
        x=data.drop('churn',axis=1)
        y=encode_target(data['churn'])
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
        logger.info('data splitting completed successfully')
        return x_train,x_test,y_train,y_test
    except Exception as e:
        logger.error(f'error while splitting data: {e}')
        raise

def model_building(x_train:pd.DataFrame,y_train:pd.DataFrame)->Pipeline:
    try:
        logger.info('starting model building')
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
        model=LogisticRegression(max_iter=2000,tol=0.01,class_weight='balanced',C=0.3359818286283781,solver='lbfgs')
        model_pipe=Pipeline(steps=[('preprocessor',column_preprocessor()),('model_pipe',model)])
        cv_score=cross_val_score(model_pipe,x_train,y_train,cv=skf,scoring='recall')
        model_pipe.fit(x_train,y_train)
        logger.info(f'model building completed successfully with cv recall score: {cv_score.mean()}')
        return model_pipe
    except Exception as e:
        logger.error(f'error while building the model: {e}')


def save_model(model: Pipeline, save_model_path: str) -> None:
    try:
        logger.info(f'saving model at {save_model_path}')
        os.makedirs(save_model_path, exist_ok=True)
        pickle.dump(model, open(os.path.join(save_model_path, 'model.pkl'), 'wb'))
        logger.info('model saved successfully')
    except Exception as e:
        logger.error(f'error while saving model: {e}')
        raise        

def save_split_data(x_train:pd.DataFrame,x_test:pd.DataFrame,y_train:pd.Series,y_test:pd.Series,save_data_path:str)->None:
        try:
            logger.info(f'saving the split data at :{save_data_path}')        
            save_data_final_path=os.path.join(save_data_path,'proccessed/split')
            os.makedirs(save_data_final_path,exist_ok=True)
            x_train.to_csv(os.path.join(save_data_final_path,'x_train.csv'),index=False)
            x_test.to_csv(os.path.join(save_data_final_path,'x_test.csv'),index=False)
            y_train.to_csv(os.path.join(save_data_final_path,'y_train.csv'),index=False)
            y_test.to_csv(os.path.join(save_data_final_path,'y_test.csv'),index=False)
            logger.info(f'split data saved successfully at :{save_data_final_path}')
        except Exception as e:
            logger.error(f'an error accured while saving the split data:{e}')
            raise    

def main():
    final_data=load_data('data/processed/feature_eng_data.csv')    
    x_train,x_test,y_train,y_test=spliting_data(final_data)    
    model_pipe=model_building(x_train,y_train)
    save_model(model_pipe,'models')
    save_split_data(x_train,x_test,y_train,y_test,'data/processed')