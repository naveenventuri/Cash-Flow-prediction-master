from MetricsHelper import gen_metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from mlxtend.regressor import StackingRegressor
from BaseModel import BaseModel as Parent
import Dataset as ds
from dnn import DNN
from ensemblednn import EnsembleDNN

class Ensemble(Parent):

    def __init__(self, name, input_dim = 0, output_dim = 0):
        Parent.__init__(self, name)
        
        switcher = {
            
            self.GRADIENT : GradientBoostingRegressor(random_state=0),
            
            self.ADA : AdaBoostRegressor(random_state=0) ,
            
            self.STACK : StackingRegressor(regressors=[RandomForestRegressor(),
		                           DecisionTreeRegressor(),
		                           KNeighborsRegressor(),LinearRegression(),DNN(input_dim, output_dim), AdaBoostRegressor(random_state=0), GradientBoostingRegressor(random_state=0)
                           ],meta_regressor=EnsembleDNN(7,1))               
        }
        
        self.model = switcher.get(name)
	
    def train(self, x_train, x_test, y_train, y_test, save=False):
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        
        print('Train...:', self.name)
        self.model.fit(x_train, y_train)
    
        gen_metrics(self.model.predict(x_test), y_test)
        
        if save:
            self.save('')

def bagging(x_train, x_test, y_train, y_test):
    
    for base_estimator in [RandomForestRegressor(),
                           DecisionTreeRegressor(),
                           KNeighborsRegressor(),LinearRegression()
                           ]:
    
        model = BaggingRegressor(base_estimator=base_estimator,random_state=0).fit(x_train, y_train) 
    
        gen_metrics(model.predict(x_test), y_test)

def run():
    
    (x_train, x_test, y_train, y_test) = ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET30_EXCLUDES, y_cols=ds.BET30)
    
    model = Ensemble(Ensemble.STACK,x_train.shape[1],y_train.shape[1])
  
    model.train(x_train, x_test, y_train, y_test, save=False)
  
if __name__ == '__main__':
    run()
