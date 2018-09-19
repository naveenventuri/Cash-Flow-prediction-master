import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import configparser

config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

ALL_FILE_704 = config['DEFAULT']['ALL_FILE_704']
NO_BLANKS_FILE_704 = config['DEFAULT']['NO_BLANKS_FILE_704']
NO_BLANKS_ZIP_FILE_704 = config['DEFAULT']['NO_BLANKS_ZIP_FILE_704']
BLANKS_ZIP_FILE_704 = config['DEFAULT']['BLANKS_ZIP_FILE_704']
ZIP_CLUSTERS_FILE_704 = config['DEFAULT']['ZIP_CLUSTERS_FILE_704']
PERSON_CLUSTERS_FILE_704 = config['DEFAULT']['PERSON_CLUSTERS_FILE_704']

NO_BLANKS_QUERY = 'AI_IsBB == 1 and AI_HasBBRecord == 1 and AI_HasCreditHistory ==1 and ( bet_30!=0 or bet_60!=0 or bet_90!=0 or bet_120!=0 or bet_150!=0 or bet_180!=0 )'
BLANKS_QUERY = 'AI_IsBB == 1 and AI_HasBBRecord == 1 and ( bet_30!=0 or bet_60!=0 or bet_90!=0 or bet_120!=0 or bet_150!=0 or bet_180!=0 )'

ZIP_EXCLUDES = ['IS_CURR_CUST','AI_IsBB', 'AI_HasBBRecord', 'AI_HasCreditHistory', 'DECEASED', 'ZIP_CODE_5', 'CAPITAL_DEFICIT_ROOT', 'CAPITAL_SURPLUS_ROOT', 'EST_INVOICE_TOT_6M_SQR', 'EST_INVOCIE_TOT_6M_ROOT', 'PUBLIC_REMARK_24M_SQR', 'PUBLIC_REMARK_24M_ROOT' , 'PUBLIC_Remark_12M_ROOT']
PID_EXCLUDES = ['PersonID','AI_HasPopRecord','AccountStatus']
EXCLUDE_COLS = ['PersonID','IS_CURR_CUST','AI_IsBB', 'AI_HasBBRecord', 'AI_HasCreditHistory', 'DECEASED']
BET30 = ['bet_30']
BET60 = ['bet_60']
BET90 = ['bet_90']
BET120 = ['bet_120']
BET150 = ['bet_150']
BET180 = ['bet_180']
BET120_EXCLUDES = BET150+BET180
BET90_EXCLUDES = BET120_EXCLUDES + BET120 
BET60_EXCLUDES = BET90_EXCLUDES + BET90
BET30_EXCLUDES = BET60_EXCLUDES + BET60

def load_data(file_path='', query='', exclude_cols=[]):
    
    print('Loading data...')
    df = pd.read_csv(file_path)
    
    print("Total columns : ", df.columns)
    print("Query: " , query)
    if query != '':
        df = df.query(query)
    
    if len(exclude_cols) != 0:
        df = df.drop(exclude_cols,axis=1)
    
    df = df.replace('N',0)
    df = df.replace('Y',1)
    df = df.replace('.*T.*',1,regex=True)
    df = df.replace('.*F.*',0, regex=True)
    df = df.replace('K',0)
    df = df.replace('M',1)
    df = df.replace('',-1)
        
    df = df.replace(r'\s+',0,regex=True).replace('',-1)
    df = df.fillna(-1)
    
    #for i,j in np.transpose(np.nonzero(np.isfinite(np.array(df, dtype=float)))):
    #    df.iloc[i:i,j:j]=0
    
    for col in df:
        if df[col].dtype == 'object':
            print("Object Column : ", col)
    
    print("Final Columns : ", df.columns)
    print("Total records : ", df.shape)

    return df

def load(file_path='', test_size=0.2, query='', exclude_cols=[], y_cols=[], scale=False, randomize=True):
    
    x = load_data(file_path=file_path, query=query, exclude_cols=exclude_cols)
    
    if randomize:
        x = x.sample(frac=1)
        
    y = []
    if len(y_cols) != 0:
        x.set_index(y_cols)
        y = x[y_cols]
        x = x.drop(y_cols, axis=1)
        
    if scale:
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(x)
        x = min_max_scaler.transform(x)
    
    print("X cols:" , x.columns)
    print("Y cols:" , y.columns)
    
    return train_test_split(x, y, test_size=test_size)   
