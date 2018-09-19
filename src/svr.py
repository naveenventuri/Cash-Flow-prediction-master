from sklearn.svm import SVR
import Dataset as ds
from BaseModel import BaseModel as Parent

class SVMRegression(Parent):

    def __init__(self, degrees=2):
        Parent.__init__(self, self.SVR)
        Parent.model = SVR(C=1.0, epsilon=0.001, cache_size=200, coef0=0.0, degree=degrees, gamma='auto', kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)

def run():

    model = SVMRegression()
    (x_train, x_test, y_train, y_test) = ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET30_EXCLUDES, y_cols=ds.BET30)
    model.train( x_train, x_test, y_train, y_test, save=False)
    

if __name__ == '__main__':
    run()