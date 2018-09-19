from Dataset import load
from MetricsHelper import gen_metrics
import pickle
from keras.models import Sequential
import configparser

config = configparser.ConfigParser()
config._interpolation = configparser.ExtendedInterpolation()
config.read('config.ini')

class BaseModel:

    model = None
    
    MODEL_HOME = config['MODEL']['MODEL_HOME']
    
    REGRESSION_NOBLANKS = config['MODEL']['REGRESSION_NOBLANKS']
    SVR_NOBLANKS = config['MODEL']['SVR_NOBLANKS']
    RF_NOBLANKS = config['MODEL']['RF_NOBLANKS']
    DNN_NOBLANKS = config['MODEL']['DNN_NOBLANKS']
    GRADIENT_NOBLANKS = config['MODEL']['GRADIENT_NOBLANKS']
    ADA_NOBLANKS = config['MODEL']['ADA_NOBLANKS']
    STACK_NOBLANKS = config['MODEL']['STACK_NOBLANKS']
    
    REGRESSION_BLANKS = config['MODEL']['REGRESSION_BLANKS']
    SVR_BLANKS = config['MODEL']['SVR_BLANKS']
    RF_BLANKS = config['MODEL']['RF_BLANKS']
    DNN_BLANKS = config['MODEL']['DNN_BLANKS']
    GRADIENT_BLANKS = config['MODEL']['GRADIENT_BLANKS']
    ADA_BLANKS = config['MODEL']['ADA_BLANKS']
    STACK_BLANKS = config['MODEL']['STACK_BLANKS']
    
    REGRESSION, SVR, RANDOMFOREST, DNN, GRADIENT, ADA, STACK = 1,2,3,4,5,6,7
    name = None

    models = {
        REGRESSION : REGRESSION_BLANKS,
        SVR : SVR_BLANKS,
        RANDOMFOREST : RF_BLANKS,
	DNN : DNN_BLANKS,
	GRADIENT : GRADIENT_BLANKS,
        ADA : ADA_BLANKS,
        STACK : STACK_BLANKS
    }
    
    def __init__(self, model_name):
        self.name = model_name
        
    def train(self, x_train, x_test, y_train, y_test, save=False):
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
        
        print('Train...:', self.name)
        self.model.fit(x_train, y_train)
    
        gen_metrics(self.model.predict(x_test), y_test)
        
        if save:
            self.save('')

    def predict(self, x_test):
        
        print('Predict...')
        return self.model.predict(x_test)
    
    def save(self, file_path=''):
        
        if file_path == '':
            file_path = self.models.get(self.name)
        
        print(file_path)
        
        # save the model to disk
        pickle.dump(self.model, open(file_path, 'wb'))

    def load(self, file_path='', x_test=[], y_test=[]):
        
        print(file_path)
        
        if file_path == '':
            file_path = self.models.get(self.name)
        
        print(file_path)
        
        self.model = pickle.load(open(file_path, 'rb'))
        
        if len(x_test) > 0:
            result = self.model.score(x_test, y_test)
            print("Result :>>", result)
