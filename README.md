# cashflow-prediction: Cashflow prediction models for Sergel

This library contains set of models for predicting the cashflow in debt recovery scenarios of Sergel. It contains different models like regression, svm regression, random forest, deep neural network and gradient ensemble, ada ensemble and stacking ensemble. 

# Datasets


## Example Usage

Models can be tried with simple commands as below.

```python regression.py
   python svr.py
   python randomforest.py
   python dnn.py
   python ensemble.py
```
The default main (entry point) methods of these modules are configured to take default/most recent, accurate dataset and train the model. The entry points can be modified quickly to change the dataset or to make it save the model. Alternatively the models can be initialized from already pickled model parameters. 
