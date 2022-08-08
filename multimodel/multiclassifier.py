import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

class MultiClassifierNetwork(BaseEstimator, ClassifierMixin):
    '''
    class_labels : list of ints
    '''
    def __init__(self, 
                classifier_list, 
                class_labels=None,
                feature_list=None,
                categorical_list=None,
                random_state=None,
                ):
        self.classifier_list = classifier_list
        self.random_state = random_state
        self.feature_list = feature_list
        self.categorical_list = categorical_list

        if class_labels == None:
            self.class_labels = list(range(len(classifier_list)))
        else:
            self.class_labels = class_labels      
        

    def fit(self, X, y):
        y = pd.DataFrame(y)
        if self.feature_list == None:
            if isinstance(X, pd.DataFrame):
                self.feature_list = [X.columns.to_list()] * len(self.class_labels)
            else:
                raise TypeError
                    

        for i in self.class_labels:
            self.classifier_list[i].set_params(random_state=self.random_state)
            _X_i = X[self.feature_list[i]] # Select feature list
            _replace_y = {self.class_labels[i]:1}
            _replace_y.update(zip([x for x in self.class_labels if x!=i],[0 for x in [x for x in self.class_labels if x!=i]]))
            _y_i = y.replace(_replace_y) # Set selected class=1, others=0 (OneVsRest)

            if self.categorical_list is not None:
                ## For use with LightGBM
                self.classifier_list[i].fit(_X_i,_y_i.values.ravel(), feature_name=self.feature_list[i], categorical_feature=self.categorical_list[i])

            else:
                self.classifier_list[i].fit(_X_i,_y_i.values.ravel())

        return self
    
    def predict(self, X, threshold=None):
        _output = pd.DataFrame()
        for i in self.class_labels:
            _y_pred = self.classifier_list[i].predict_proba(X[self.feature_list[i]])
            _output['Clf'+str(i)] = _y_pred[:,1]

        for i in range(len(_output)):
            _output.iloc[i,:] = _output.iloc[i,:].divide(_output.sum(axis=1)[i]).to_numpy()
       
        _output = _output.to_numpy()
        _output = np.argmax(_output, axis=1)

        return _output


    def predict_proba(self, X):
        _output = pd.DataFrame()
        for i in self.class_labels:
            _y_pred = self.classifier_list[i].predict_proba(X[self.feature_list[i]])
            _output['Clf'+str(i)] = _y_pred[:,1]

        
        # for i in range(len(_output)):
        #     _output.iloc[i,:] = _output.iloc[i,:].divide(_output.sum(axis=1)[i]).to_numpy()
     
        _output = _output.to_numpy()

        return _output