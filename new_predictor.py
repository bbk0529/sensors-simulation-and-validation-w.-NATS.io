from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import pickle
import copy
from datetime import datetime


from abc import abstractmethod, ABC


class Classifier(ABC): 
    @abstractmethod
    def validate() :
        pass

class SpatialComparision(Classifier) :
    def validate(self, ts_data: np.ndarray, neighbor: np.ndarray, station: int):
        v_col = 'value'
        neighbor_data = pd.DataFrame(ts_data[neighbor[station]].T)
        ts_data = ts_data[station]
        ts_data = pd.DataFrame(ts_data, columns = [v_col])
        

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
        return [str(station) + "_" + str(x) for x in event_time_index]

class SmoothingAndPredict(Classifier):    
    def correct(self, X_raw, y_raw, graph=False) : 
        y = copy.deepcopy(y_raw)
        X = copy.deepcopy(X_raw)                
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
    
    def validate(self, ts_data, neighbor, station) :                       
        y = ts_data[station]
        X = ts_data[neighbor[station]]
        # y = copy.deepcopy(ts_data[station])        
        # X = copy.deepcopy(ts_data[neighbor[station]])
        corrected_y = self.correct(X,y)                                    
        suspected_timesteps = sorted(np.where(abs(ts_data[station] - corrected_y)>3)[0])                   
        return [str(station) + '_' + str(idx) for idx in suspected_timesteps]


class EnsembleRegression(Classifier) :
    def __init__(self, n_regressors: int, n_variables: int, eps: float, decision_boundary: float, global_search = False) :
        self._n_regressors = n_regressors
        self._n_variables = n_variables
        self._eps = eps
        self._decision_boundary = decision_boundary
        self._global_search = global_search

    def validate(self, ts_data, neighbor, station) :
        result = np.array([], dtype='int')
        for i in range(self._n_regressors) : 
            if self._global_search : 
                idx = np.random.choice(range(len(ts_data)), size=self._n_variables, replace=False)
            else : 
                idx = np.random.choice(neighbor[station], size=self._n_variables, replace=False)
            
            y = ts_data[station] 
            X = ts_data[idx]
            reg = LinearRegression().fit(X.T, y)
            # print(reg.score(X.T, y))
            original = ts_data[station]
            predict = reg.predict(X.T)
            # predict = reg.intercept_ + np.dot(X.T, reg.coef_)
            ix = np.where(abs(predict - original) > self._eps)    
            result = np.append(result, ix)    

        unique, counts = np.unique(result, return_counts=True)
        RESULT = dict(zip(unique, counts))
        # print(RESULT)
        # return [str(station) + '_' + str(k) for k, v in sorted(RESULT.items(), reverse=True, key=lambda item: item[1]) if v>(n_regressors/3*2)]
        return [str(station) + '_' + str(k) for k, v in RESULT.items() if v>(self._n_regressors * self._decision_boundary)]

    def validate_and_predict(self, ts_data, neighbor, station):    
        width = ts_data.shape[1]
        PRED = np.zeros((self._n_regressors, width))
        SCORE = np.zeros(self._n_regressors)
        for j in range(self._n_regressors) :             
            idx_variables = np.random.choice(range(self._n_variables), replace=False, size=self._n_variables)
            y = ts_data[station]
            X = ts_data[neighbor[station]][idx_variables]
            reg = LinearRegression().fit(X.T,y)
            score = reg.score(X.T, y)
            pred = reg.predict(ts_data[neighbor[station]][idx_variables].T)            
            PRED[j] = pred
            SCORE[j] = score
        SCORE = SCORE / np.sum(SCORE)
        predict = np.dot(PRED.T, SCORE)
        original = ts_data[station]
        ix = np.where(abs(predict - original) > self._eps)
        if len(ix[0]) <= 0 : 
            return None
        else : 
            return [ix, pred[ix]]
    


class Data(ABC):
    def __init__(self, p_noise_stations, p_noise_timesteps, min_noises, max_noises, both_side):
        self._create_neighor_list(self._metadata[:, 1:], self._k)                                
        self.ts_data, self.lst_station_timestep = self.add_noise(
            self.ts_rawdata, p_noise_stations=p_noise_stations, p_noise_timesteps = p_noise_timesteps, min_noises=min_noises, max_noises=max_noises, both_side = both_side
        )
        
    def _create_neighor_list(self, metadata: np.ndarray, k: int): 
        #neighbor list
        self._dist_matrix = distance_matrix(metadata, metadata)
        self._neighbor = self._dist_matrix.argsort()[:, 1:k+1]        

    def add_noise(self, ts_data, p_noise_stations: float, p_noise_timesteps: float, min_noises, max_noises, both_side = True) :
        n_stations = ts_data.shape[0]
        n_timesteps = ts_data.shape[1]
        self.dic_timesteps ={}
        matrix_noises = np.zeros(ts_data.shape)
        picked_stations = np.random.choice(range(n_stations), size= round(p_noise_stations * n_stations), replace=False)
        picked_stations.sort()
        self._picked_stations = picked_stations
        for s in picked_stations : 
            pick_timesteps = np.random.choice(range(n_timesteps), size= round(p_noise_timesteps * n_timesteps), replace=False)
            
            if both_side : 
                noises = np.append(
                    np.random.rand(round(len(pick_timesteps)/2)) * max_noises, 
                    - np.random.rand(len(pick_timesteps) - round(len(pick_timesteps)/2)) * max_noises
                )
            else : 
                noises =  np.random.rand(len(pick_timesteps)) * max_noises            
            
            if min_noises == max_noises : 
                noises[(0 <= noises) & (noises < min_noises) ] = min_noises
                noises[(0 >= noises) & (noises > -min_noises) ] = -min_noises
            else : 
                noises[(0 <= noises) & (noises < min_noises) ] = np.random.choice(range(min_noises,max_noises))
                noises[(0 >= noises) & (noises > -min_noises) ] = np.random.choice(range(-max_noises, -min_noises))
            matrix_noises[s, pick_timesteps] = noises
            
            self.dic_timesteps[s] = sorted(pick_timesteps)
        idx = np.where(abs(matrix_noises)>0)
        self.matrix_noises = matrix_noises
        return ts_data + matrix_noises, [str(x[0]) + '_' + str(x[1]) for x in np.array(idx).T]
    

class Tempearture_DWD2(Data) : 
    def __init__(self, n_stations: int, n_timesteps: int, k: int=5, p_noise_stations: float=0.1, p_noise_timesteps: float=0.1, min_noises=3, max_noises=10, both_side=True, idx_timesteps = None) : 
        df = pickle.load(open('dwd_data_wo_suspected(test).p','rb'))       
        df_metadata = pickle.load(open('metadata.p', 'rb'))                    
        if idx_timesteps  : 
            idx_rawdata = idx_timesteps
        else :
            idx_rawdata = np.random.choice(range(0, df.shape[1] - n_timesteps))        
        print(idx_rawdata)
        self._metadata = df_metadata.values[:n_stations]        
               

        self.ts_rawdata = self._preprocess_data(df[:n_stations, idx_rawdata:idx_rawdata+n_timesteps])
        self._k = k          
        super().__init__(p_noise_stations, p_noise_timesteps, min_noises, max_noises, both_side)

    def _preprocess_data(self, ts_data: np.ndarray) : 
        df = pd.DataFrame(ts_data)
        df[df<=-40] = np.nan
        df[df>50] = np.nan
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        return df.values


class Tempearture_DWD(Data) : 
    def __init__(self, n_stations: int, n_timesteps: int, k: int=5, p_noise_stations: float=0.1, p_noise_timesteps: float=0.1, min_noises=3, max_noises=10, both_side=True, idx_timesteps = None) : 
        df = pickle.load(open('data_dvd_reduced.p','rb'))
        df.columns = df.columns.map(lambda x : datetime.strptime(str(x), '%Y%m%d%H%M') )
        # df[df == -999] = np.NaN
        # df = df.dropna()

        df_metadata = pickle.load(open('metadata.p', 'rb'))                    
        if idx_timesteps  : 
            idx_rawdata = idx_timesteps
        else :
            idx_rawdata = np.random.choice(range(0, df.shape[1] - n_timesteps))        
        print(idx_rawdata)
        self._metadata = df_metadata.values[:n_stations]        
        
        

        self.ts_rawdata = self._preprocess_data(df.values[:n_stations, idx_rawdata:idx_rawdata+n_timesteps])
        self._k = k          
        super().__init__(p_noise_stations, p_noise_timesteps, min_noises, max_noises, both_side)

    def _preprocess_data(self, ts_data: np.ndarray) : 
        df = pd.DataFrame(ts_data)
        df[df<=-40] = np.nan
        df[df>50] = np.nan
        df = df.fillna(method='bfill')
        df = df.fillna(method='ffill')
        return df.values

class HuberRegressor(Classifier) : 
    def validate (self, ts_data, neighbor, station): 
        from sklearn.linear_model import HuberRegressor
        model = HuberRegressor(max_iter=10)
        X = ts_data[neighbor[station]]
        y = ts_data[station]
        model.fit(X.T, y)
        suspected_timesteps = sorted(np.where(abs(model.predict(X.T) - y) > 3)[0])
        return [str(station) + '_' + str(idx) for idx in suspected_timesteps]

class Executor() : 
    def __init__ (self, data: Data, classifier: Classifier) : 
        self.data = data
        self.classifier = classifier        
        
    def validate(self, station):
        return self.classifier.validate(self.data.ts_data, self.data._neighbor, station)


    def evaluate_validator(self) : 
        RESULT = []
        STAT = {'TP': 0, 'TN':0, 'FP':0, 'FN':0 }
        start = time.time()
        
        for i in range(len(self.data.ts_data)) :
            result = self.validate(i)
            RESULT = RESULT + result
            for r in result : 
                if r in self.data.lst_station_timestep :
                    STAT['TP'] = STAT.get('TP') + 1
                else :
                    STAT['FP'] = STAT.get('FP') + 1
        
        for a in self.data.lst_station_timestep :
            if a not in RESULT:
                STAT['FN'] = STAT.get('FN') + 1
        STAT['runtime'] = round(time.time() - start,4)
        lst_suspcted_stations = set([int(station.split('_')[0]) for station in  RESULT])
        
        toggle_lst_stations = {}

        for station in self.data._picked_stations : 
            toggle_lst_stations[station] = False 
        
        
        lst_false_positive = set()
        for s in lst_suspcted_stations : 
            if s in self.data._picked_stations :
                toggle_lst_stations[s]  = True 
            else : 
                lst_false_positive.add(s)
        STAT['lst_false_positive'] = lst_false_positive
        STAT['toggle_lst_stations'] = toggle_lst_stations


        STAT['n_stations'] = self.data.ts_data.shape[0]
        STAT['n_timestep'] = self.data.ts_data.shape[1]
        STAT['n_tsdata'] = self.data.ts_data.size
        STAT['p_noises'] = len(self.data.lst_station_timestep) / self.data.ts_data.size 
        STAT['TN'] = self.data.ts_data.size - STAT['TP'] - STAT['FP'] - STAT['FN']
        STAT['precision'] = round(STAT['TP'] / (STAT['TP'] + STAT['FP']),3)
        STAT['recall'] = round(STAT['TP'] / (STAT['TP'] + STAT['FN']),3)
        STAT['f1'] = round(2 * STAT['precision'] * STAT['recall'] / (STAT['precision'] + STAT['recall']),3)
        self.result = RESULT        
        return STAT


def test(p_noise_stations, p_noise_timesteps, min_noises, max_noises) : 
    RESULT = [] 
    n_stations = 456 #292, 456
    n_timesteps= 100
    k = min(round(n_timesteps/3),10)
    smoothpredictor = SmoothingAndPredict()
    for idx_start in range(0,52600, 100) :     
        data = Tempearture_DWD(n_stations, n_timesteps, k, p_noise_stations, p_noise_timesteps, min_noises, max_noises, True, idx_start)    
        executorNew = Executor(data, smoothpredictor)
        result = executorNew.evaluate_validator()
        RESULT.append([result['precision'], result['recall'], result['f1'], result['runtime']])        
    return np.array(RESULT)
        

import sys
import pickle
if __name__ == "__main__" :     
    p_noise_stations = sys.argv[1] 
    p_noise_timesteps = sys.argv[2]
    min_noises = sys.argv[3]
    max_noises = sys.argv[4]

    RESULT = test(p_noise_stations, p_noise_timesteps, min_noises, max_noises)
    filename = sys.argv[1:].join('')
    file = open()

    

