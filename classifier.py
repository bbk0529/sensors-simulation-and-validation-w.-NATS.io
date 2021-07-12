from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from abc import abstractmethod, ABC

class Classifier(ABC): 
    @abstractmethod
    def validate() :
        pass

class SmoothingAndPredict(Classifier):    
    def correct(self, X_raw, y_raw, graph=False) : 
        y = y_raw.copy()
        X = X_raw.copy()
        while True :  
            reg = LinearRegression().fit(X.T, y)                
            score = reg.score(X.T,y)
            pred = reg.predict(X.T)
            
            error = pred - y 
            error_mean = np.mean(error)
            error_std = np.std(error)            
            
            idx_boolean = (error >= error_mean +  max(1.96 * error_std, 3)) | (error <= error_mean -  max(1.96 * error_std, 3))
            print("\tFitting by regression model")
            print("\t\t- R^2: {}, mean: {}, std: {}".format(round(score,3), round(error_mean,3), round(error_std,3)))
            idx = np.where(idx_boolean == True )[0]
            y[idx] = pred[idx] 
            if len(idx) == 0 :            
                break
        return y
    
    def validate(self, y, X) :                               
        # y = copy.deepcopy(ts_data[station])        
        # X = copy.deepcopy(ts_data[neighbor[station]])
        corrected_y = self.correct(X,y)                                    
        suspected_timesteps = sorted(np.where(abs(y - corrected_y)>3)[0])                   
        print("Found noise by model", suspected_timesteps)       
        print(y-corrected_y)                        
        print("\n")            




class SpatialComparision(Classifier) :
    def validate(self, y, X):
        
        neighbor_data = pd.DataFrame(X.T)
        ts_data = y
        v_col = 'value'

        ts_data = pd.DataFrame(y, columns = [v_col])
        

        const_rel_dw = 0.1
        const_rel_up = 0.1
        size = len(ts_data)
        
        const_abs = np.nan
        value_min = -np.inf
        value_max = np.inf
        
        aux_matrix = np.zeros((size, 3))
        aux_matrix[:, 0] = neighbor_data.mean(axis=1).abs() * const_rel_dw
        aux_matrix[:, 1] = neighbor_data.mean(axis=1).abs() * const_rel_up
        aux_matrix[:, 2] = [const_abs] * size
        aux_matrix[:, 0] = np.nanmax(aux_matrix[:, [0, 2]], axis=1)
        aux_matrix[:, 1] = np.nanmax(aux_matrix[:, [1, 2]], axis=1)
        aux_matrix[:, 2] = neighbor_data.std(axis=1) * 2
        aux_matrix[:, 0] = np.nanmax(aux_matrix[:, [0, 2]], axis=1)
        aux_matrix[:, 1] = np.nanmax(aux_matrix[:, [1, 2]], axis=1)

        ts_data = ts_data.assign(
            min=neighbor_data.min(axis=1) - aux_matrix[:, 0],
            max=neighbor_data.max(axis=1) + aux_matrix[:, 1],
        )
        event_time_index = ts_data[
            ((value_min <= ts_data[v_col]) & (ts_data[v_col] <= value_max))
            & ((ts_data[v_col] < ts_data["min"]) | (ts_data[v_col] > ts_data["max"]))
        ].index
        print("Found noise by model", event_time_index)

