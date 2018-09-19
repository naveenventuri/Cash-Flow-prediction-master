from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
import Dataset as ds
from BaseModel import BaseModel as Parent
from MetricsHelper import gen_metrics
import types
import tempfile
import keras.models
from sklearn.base import BaseEstimator, ClassifierMixin

class DNN(Parent, BaseEstimator, ClassifierMixin):

    input_dim, output_dim = 0, 0
    
    def __init__(self, input_dim, output_dim):
        Parent.__init__(self, self.DNN)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = Sequential()
        self.model.add(Dense(128, activation='relu', input_dim=self.input_dim))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(output_dim=self.output_dim, activation='relu'))
        
        #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='mean_absolute_error',optimizer='rmsprop',metrics=['accuracy'])
        
    def train(self, x_train, x_test, y_train, y_test, save=False):
        print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
   
        print('Train...')
        self.model.fit(x_train, y_train,batch_size=100,epochs=500,validation_data=[x_test, y_test])
       
        gen_metrics(self.model.predict(x_test), y_test)
        
        if save:
            self.save('')
    
    def fit(self, x_train, y_train):
           
        print('Train...')
        self.model.fit(x_train, y_train,batch_size=100,epochs=200)
 
    def save(self, file_path):
        
        if file_path == '':
            file_path = self.models.get(self.name)
        
        # save the model to disk
        keras.models.save_model(self.model, file_path, overwrite=True)
    
    def load(self, file_path, x_test=[], y_test=[]):
        
        if file_path == '':
            file_path = self.models.get(self.name)
        
        self.model = keras.models.load_model(file_path)
            
        if len(x_test) > 0:
            gen_metrics ( self.model.predict(x_test) , y_test)

def run():
     
    (x_train, x_test, y_train, y_test) = ds.load(ds.BLANKS_ZIP_FILE_704, exclude_cols=ds.BET30_EXCLUDES, y_cols=ds.BET30)
    
    model = DNN(x_train.shape[1],y_train.shape[1]) 
    
    model.train(x_train, x_test, y_train, y_test, save=True)

if __name__ == '__main__':
    run()