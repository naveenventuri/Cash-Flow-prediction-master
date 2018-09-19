import numpy as np
import pandas as pd
import Dataset as ds
 
# 5474 ds, with zip clusters
def prepare0704_no_blank_zip():

    df = ds.load_data(ds.ALL_FILE_704, ds.NO_BLANKS_QUERY, ds.ZIP_EXCLUDES)
    
    df1 = pd.read_csv(ds.ZIP_CLUSTERS_FILE_704)
             
    df2 =  pd.concat([df, df1], axis=1, join='inner')
    print(df2.shape)
            
    df2.to_csv(ds.NO_BLANKS_ZIP_FILE_704)
 
# 6471 ds with zip clusters
def prepare0704_blank_zip():
     
     df = ds.load_data(ds.ALL_FILE_704, ds.BLANKS_QUERY, ds.ZIP_EXCLUDES)
     
     df1 = pd.read_csv(ds.ZIP_CLUSTERS_FILE_704)
              
     df2 =  pd.concat([df, df1], axis=1, join='inner')
     
     df2 = df2.drop(df.columns[df.columns.str.contains('ZIP_CODE',case = False)], axis=1)
     
     print(df2.columns)
     
     #df2.drop(df2.columns[df2.columns.str.contains('unnamed',case = False)] , axis=1)
     
     print(df2.shape)
     
     df1 = pd.read_csv(ds.PERSON_CLUSTERS_FILE_704)
     df =  pd.concat([df1, df2], axis=1, join='inner')
     
     df = df.drop(['PersonID'], axis=1)
     #df.drop(df.columns[df.columns.str.contains('unnamed',case = False)] , axis=1)
     
     df.to_csv(ds.BLANKS_ZIP_FILE_704)
 
#5474 ds 
def prepare0704_no_blank():

    df = ds.load_data(ds.ALL_FILE_704, ds.NO_BLANKS_QUERY, ds.ZIP_EXCLUDES)
    df.to_csv(ds.NO_BLANKS_FILE_704)

# method to map a person id to a psuedo id
def map_personid(file_path=ds.ALL_FILE_704, save_file='c:/cashflow-prediction/temp.csv'):
    
    print('Loading data...')
    df = pd.read_csv(file_path)
    
    df.setIndex(['PersonID'])
    newdf = df[['PersonID']].copy()
    psudoIds = df['PersonID']
    psudoIds = psudoIds.rename(columns = {'PersonID':'PsuedoID'})
    
    for i in range(len(personIds)):
        psudoIds[i] = i
      
    newdf.add(psudoIds)
    newdf.to_csv(save_file)
    
    df['PersonID'] = psudoIds
    df.to_csv(file_path+".csv")
 
# method to join dataset with cluster numbers
def join_outputs(file1=ds.ALL_FILE_704, file2=ds.ZIP_CLUSTERS_FILE_704, mapping_file='c:/cashflow/temp.csv', to_file='c:/cashflow/out.csv'):
    
    print('Loading data...')
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    
    df4 =  pd.concat([df1, df2], axis=1, join='inner')
    df5 = pd.concat([df3,df4], axis=1, join='outer')
    
    df5.to_csv(to_file)

prepare0704_blank_zip()
