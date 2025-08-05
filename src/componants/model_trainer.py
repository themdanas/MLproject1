import os
import sys
from src.exeption import CustomException
from src.logger import logging
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils import save_object
from src.utils import evalute_models

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,)



@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        #self.model= None


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info('Split training and test input data')
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            params = {
                "Random Forest": {"n_estimators": 100, "max_depth": 5},
                "Decision Tree": {"max_depth": 5, "min_samples_split": 2},
                "Gradient Boosting": {"n_estimators": 100, "learning_rate": 0.1},
                "Linear Regression": {},
                "XGBRegressor": {"max_depth": 5, "learning_rate": 0.1},
                "CatBoosting Regressor": {"iterations": 100, "learning_rate": 0.1},
                "AdaBoost Regressor": {"n_estimators": 100, "learning_rate": 0.1}
            }
            
            model_report:dict=evalute_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,params=params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise Exception("No Best Model Found")
            logging.info(f"Best Model Found , Model Name: {best_model_name} , R2 Score: {best_model_score}")            
            
            #preprocessing_obj = load_object(file_path=self.model_trainer_config.preprocessor_path)
            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)