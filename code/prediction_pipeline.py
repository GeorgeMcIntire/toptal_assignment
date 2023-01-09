#Imports
import joblib
import json
import pandas as pd
import numpy as np

#Load in data
path = "../data/verify.json"
with open(path) as f:
    data = json.load(f)
    
#Convert to pandas dataframe
df = pd.json_normalize(data)

#Grab the sites columns
sites = df["sites"]

#Load in the countvectorizer object which is used to feature engineer the sites data
cv = joblib.load("../model/countvec.joblib")
#Transform sites data using cv
sites_enc = cv.transform(sites)
sites_enc = pd.DataFrame(sites_enc.toarray(), columns=cv.get_feature_names())
df.drop("sites", axis = 1, inplace=True)

#Prepare the rest of the data to fit the feature engineering process derived in the notebooks
df["software_language"] = df.locale.apply(lambda x:x[:2])
df['software_country'] = df.locale.apply(lambda x:x[-2:])
df.drop("locale", axis =1 , inplace=True)
df["country"] = df.location.apply(lambda x:x.split('/')[0])
df["city"] = df.location.apply(lambda x:x.split('/')[1])
df.drop("location", axis = 1, inplace=True)
df["hour"] = df.time.apply(lambda x:int(x[:2]))
drops = ["time", "date", 'city']
df.drop(drops, axis = 1, inplace=True)
df["russia_software"] = np.where(df.software_country=='RU', "RU", "Other")
df.drop(['software_country',"software_language"] , axis = 1, inplace=True)

#Combining the sites and df data
sites_enc.columns = sites_enc.columns.str.replace(".", "_")
df = pd.concat([df, sites_enc], axis = 1)
site_cols = sites_enc.columns

#Load in feature engineering pipeline which one hot encodes and normalizes the data.
feat_eng_pipe = joblib.load("../model/feature_engineer_pipeline.pkl")
pred_data = feat_eng_pipe.transform(df)
cat_cols = ['browser', 'os', 'gender', 'country', 'hour', "russia_software"]
cat_col_names = feat_eng_pipe.named_transformers_["cat"]["ohe"].get_feature_names(cat_cols).tolist()
column_names = cat_col_names + site_cols.tolist()
pred_data = pd.DataFrame(data=pred_data, columns=column_names)
pred_data.drop("browser_Safari", axis = 1, inplace=True)

#Next step of the feature engineering process is to remove the unnecessary features from pred_data using select
select = joblib.load("../model/feature_selector.pkl")
pred_data = select.transform(pred_data)

#Data is now ready for prediction and now load in the model
model = joblib.load("../model/Random_Forest_model.pkl")
#Make predictions
preds = model.predict(pred_data)
#Save predictions to data directory as result.csv
preds_df = pd.DataFrame(index=df.index, data = preds)
preds_df.to_csv("../data/result.csv")
