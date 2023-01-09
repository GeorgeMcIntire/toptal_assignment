# TopTal Assignment

This repo contains the full materials for my TopTal screening assignment submission.

The README page displays the directory of all the files used and created by this assignment.


### Directory

```
toptal_assignment
│   README.md
│   requirements.txt
└───code
│   │   EDA.ipynb
│   │   Modeling.ipynb
│   │   prediction_pipeline.py
│   │   utils.py   
└───data
│   │   dataset.json
│   │   processed_data.csv
│   │   result.csv
│   │   sites_encoded.csv
│   │   verify.json
└───model
│   │   Random_Forest_model.pkl
│   │   countvec.joblib
│   │   feature_engineer_pipeline.pkl
│   │   feature_selector.pkl
│   │   sites_vectorizer.pkl
```






## Run Locally

Clone the project

```bash
  https://github.com/GeorgeMcIntire/toptal_assignment.git
```

Go to the project directory

```bash
  cd toptal_assignment
```

Start up virtual environment

```bash
  virtualenv -p python3 envname
```

Activate virtual environment

```bash
  source envname/bin/activate
```

Run the following command in terminal to install the required packages
```bash
  pip3 install -r requirements.txt
```


## Usage/Examples
Make predictions on the `verify.json` data.
```python
python3 prediction_pipeline.py
```

