import sys
import Dataset as ds
from BaseModel import BaseModel
from ensemble import Ensemble
from regression import Regression
from svr import SVMRegression
from randomforest import RandomForest
from dnn import DNN

def tc6471_30():
    return ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET30_EXCLUDES, y_cols=ds.BET30)

def tc6471_60():
    return ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET60_EXCLUDES, y_cols=ds.BET60)

def tc6471_90():
    return ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET90_EXCLUDES, y_cols=ds.BET90)

def tc6471_120():
    return ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET120_EXCLUDES, y_cols=ds.BET120)

def tc6471_150():
    return ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET180, y_cols=ds.BET150)

def tc6471_180():
    return ds.load(ds.BLANKS_ZIP_FILE_704, y_cols=ds.BET180)


def run(model_name, testcase, function):
     
    (x_train, x_test, y_train, y_test) = getattr(sys.modules[__name__], 'tc'+testcase)()
   
    models = {
        "regression" : Regression(),
        "svr" : SVMRegression(),
        "rf": RandomForest(x_train.shape[1]),
        "dnn" : DNN(x_train.shape[1],y_train.shape[1]),
        "gradient" : Ensemble(BaseModel.GRADIENT),
        "ada" : Ensemble(BaseModel.ADA),
        "stack" : Ensemble(BaseModel.STACK)
    }
    model = models.get(model_name)
    
    if function == "save" : 
        model.train(x_train, x_test, y_train, y_test, save=True),
    elif function == "train" : 
        model.train(x_train, x_test, y_train, y_test),
    elif function == "load" : 
        model.load( '', x_test, y_test)
    

if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])