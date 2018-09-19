import Dataset as ds
from sklearn.ensemble import RandomForestRegressor
from BaseModel import BaseModel as Parent

class RandomForest(Parent):

    def __init__(self, depth):
        Parent.__init__(self, self.RANDOMFOREST)
        self.model = RandomForestRegressor(max_depth=depth, random_state=0,bootstrap=True, criterion='mse',max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=2,min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,oob_score=False, verbose=0, warm_start=False)

    def train(self, x_train, x_test, y_train, y_test, save=False):
        Parent.train(self, x_train, x_test, y_train, y_test, save=save)
        
        print("Decision Path: ", self.model.decision_path(x_test))
    
def run():

    (x_train, x_test, y_train, y_test) = ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET30_EXCLUDES, y_cols=ds.BET30)
    
    model = RandomForest(x_train.shape[1])
    
    model.train( x_train, x_test, y_train, y_test, save=False)

if __name__ == '__main__':
    run()