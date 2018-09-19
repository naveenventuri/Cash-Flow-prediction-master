from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import MultiTaskLassoCV
from sklearn.pipeline import make_pipeline
from BaseModel import BaseModel as Parent
import Dataset as ds

class Regression(Parent):

    def __init__(self, degrees=1):
        Parent.__init__(self, self.REGRESSION)
        print("Degrees :", degrees)
        Parent.model = make_pipeline(PolynomialFeatures(degrees, interaction_only=False), MultiTaskLassoCV(eps=0.01,n_alphas=20,max_iter=500,normalize=True,cv=5))
        
def run(degrees=2):
   
    model = Regression()
    
    (x_train, x_test, y_train, y_test) = ds.load(ds.BLANKS_ZIP_FILE_704,  exclude_cols=ds.BET180, y_cols=ds.BET150)
    model.train( x_train, x_test, y_train, y_test)
    model.save(Parent.REGRESSION_BLANKS + "_150")
    
def test_degrees(degree_min = 1, degree_max = 3):
    
    for d in range(degree_min,degree_max+1):

        model = Regression(d)
        model.train( ds.load(ds.NO_BLANKS_FILE_704, exclude_cols=ds.BET30_EXCLUDES, y_cols=ds.BET30) )


if __name__ == '__main__':
    run()