import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import xgboost as xgb
from pandasai.llm import OpenAI
llm = OpenAI()
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
import os
import uuid
import json
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/prediction_api")
async def prediction_api(email,query,file: UploadFile = File(...)):
  print(file.filename)
  with open(email+".csv", "wb") as file_object:
      file_object.write(file.file.read())
            
  def load_data(file_path):
      return pd.read_csv(file_path)
  
  def dynamic_preprocessing(data):
      # Automatically detect numerical and categorical columns
      numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
      categorical_cols = data.select_dtypes(include=['object', 'category']).columns
  
      # Define preprocessing for numerical columns ( missing values)
      numerical_transformer = Pipeline(steps=[
          ('imputer', SimpleImputer(strategy='mean')),
          ('scaler', StandardScaler())
      ])
  
      # Define preprocessing for categorical columns ( missing values)
      categorical_transformer = Pipeline(steps=[
          ('imputer', SimpleImputer(strategy='most_frequent')),
          ('onehot', OneHotEncoder(handle_unknown='ignore'))
      ])
  
      # Combine preprocessing steps
      preprocessor = ColumnTransformer(
          transformers=[
              ('num', numerical_transformer, numerical_cols),
              ('cat', categorical_transformer, categorical_cols)
          ])
  
      return preprocessor
  
  def train_and_evaluate(data, target_column):
      # Split data into features and target
      X = data.drop(target_column, axis=1)
      y = data[target_column]
  
      # Splitting the data into training and test sets
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
      # Get dynamic preprocessing pipeline
      preprocessor = dynamic_preprocessing(X_train)
  
      # Define the model
      model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=300,max_depth=10, learning_rate=0.1,colsample_bytree=0.9, random_state=42)
  
      # Bundle preprocessing and modeling code in a pipeline
      pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('model', model)])
  
      # Fit the model
      pipeline.fit(X_train, y_train)
  
      # Predictions
      preds_test = pipeline.predict(X_test)
  
      # Evaluate the model
      mae = mean_absolute_error(y_test, preds_test)
      # Evaluate the model using R-squared
      r2 = r2_score(y_test, preds_test)
  
  
      rows = data.sample(n=4)
      prediction = pipeline.predict(rows.drop(columns=['sell_qty']))
  
      return mae,r2,prediction,rows,pipeline
  
  # testing the model training with varius datasets for prediction
  data = load_data(email+".csv")
  sdf = SmartDataframe(df, config={"llm": llm})
  target_column = sdf.chat(f"{query} using model training. so what column are in x and wich colunm need to in y")
  target_column = target_column.split(":")[-1]
  return mae,r2,prediction,rows,pipeline= train_and_evaluate(data, target_column)
