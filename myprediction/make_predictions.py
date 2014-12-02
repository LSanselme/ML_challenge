import os
import pickle
# pandas/sklearn/numpy imports
import pandas as pd
from pandas.tseries.offsets import Day, Hour, Minute, Week
from pandas import to_datetime, DataFrame
import numpy as np 
from sklearn import preprocessing, linear_model, ensemble, tree
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
CACHE_DIR = 'cache/'

__all__ = [
    'make_prediction',
    'fill_reality',
    'calcul_performance',
    'prepare_for_training'
]

def training(iden, Charg, Temps, use_cache_trainingset, test, verbose):
    ''' Return the prediction function, 
    for a given site iden, history Charg and temperature Temps'''
    if use_cache_trainingset:
        if test:
            X = pickle.load(open(CACHE_DIR+"X_test_"+iden+".p", "rb"))
        else:
            X = pickle.load(open(CACHE_DIR+"X_"+iden+".p", "rb"))
    else:
        X = DataFrame(Charg[iden])
        X = X.dropna(how='any')
        X['dayofweek'] = X.index.dayofweek
        X['Temps'] = Temps[iden].ix[X.index]
        X['fracday'] = X.index.minute/60.+X.index.hour
        X['lastminutes'] = X[iden].ix[X.index-10*Minute()].values
        X['yesterday'] = X[iden].ix[X.index-Day()].values
        X['yesterdaybis'] = X[iden].ix[X.index-Day()-10*Minute()].values
        X['lastweek'] = X[iden].ix[X.index-Week()].values
        X['lastweekbis'] = X[iden].ix[X.index-Week()-10*Minute()].values
        if test:
            pickle.dump(X, open(CACHE_DIR+"X_test_"+iden+".p", "wb" ) )
        else:
            pickle.dump(X, open(CACHE_DIR+"X_test_"+iden+".p", "wb" ) )
    X = X.dropna(how='any')
    y = X[iden]
    X = X.drop(iden, 1)
    scalerX = preprocessing.StandardScaler().fit(X)
    ##############################
    clf = linear_model.SGDRegressor(alpha = 0.000001,n_iter=3000)
    ##############################
    clf.fit(scalerX.transform(X), y)
    if verbose:
        print('Function for '+iden+' computed.')
    return(lambda x :clf.predict(scalerX.transform(x)))

def make_prediction(iden, Charg, Temps, Prediction, mypreds, verbose):
    same_ID = Prediction[Prediction['ID']==iden]
    for strdatej in same_ID.DATE:
        datej = pd.to_datetime(strdatej, format='%d/%m/%Y %H:%M')
        X_dayofweek = datej.dayofweek
        X_Temps = Temps[iden].ix[datej]
        X_fracday = datej.minute/60.+datej.hour
        X_lastminutes = Charg[iden][datej-10*Minute()]
        X_yesterday   = Charg[iden][datej-Day()]
        X_yesterdaybis   = Charg[iden][datej-Day()-10*Minute()]
        X_lastweek    = Charg[iden][datej-Week()]
        X_lastweekbis    = Charg[iden][datej-Week()-10*Minute()]
        X = [X_dayofweek, X_Temps, X_fracday,
             X_lastminutes,
             X_yesterday, X_yesterdaybis,
             X_lastweek, X_lastweekbis]
        res = mypreds[iden]([X])
        Charg[iden][datej] = res
        good_index = same_ID[same_ID['DATE'] == \
                             datej.strftime(format='%d/%m/%Y %H:%M')].index[0]
        Prediction.set_value(good_index, 'ESTIMATION', res)
    return(Prediction)


def make_reality(iden, Charg_orig, Reality):
    same_ID = Reality[Reality['ID']==iden]
    for strdatej in same_ID.DATE:
        datej = to_datetime(strdatej, format='%d/%m/%Y %H:%M')
        good_index = same_ID[same_ID['DATE'] == datej.strftime(format='%d/%m/%Y %H:%M')].index[0]
        res0 = Charg_orig[iden][datej]
        Reality.set_value(good_index, 'ESTIMATION', res0)
    return Reality


def fill_reality(Prediction, Charg_orig, compute_reality):
    if compute_reality:
        Reality = Prediction.copy()
        for iden in Charg_orig.columns:
            print('Computing real values for '+iden)
            Reality = make_reality(iden, Charg_orig, Reality)
        pickle.dump(Reality, open(CACHE_DIR+"reality.p", "wb"))
    else:
        print('Retrieving real values from cache')
        Reality = pickle.load(open(CACHE_DIR+"reality.p", "rb"))
    return Reality

def calcul_performance(Reality, Prediction, Poids, selec='all'):
    if selec != 'all':
        Reality = Reality[Reality.ID == selec]
        Prediction = Prediction[Prediction.ID == selec]
        Poids = Poids[Poids['Id site'] == selec]
    R = Reality.ESTIMATION.values
    P = Prediction.ESTIMATION.values
    W = Poids.Poids.values
    itsok = np.where((abs(R-P)/R != np.inf))
    R = R[itsok]
    P = P[itsok]
    W = W[itsok]
    N1 = (abs(R-P)/R*W).sum()/W.sum()*100.
    print('N1 = {0}%'.format(N1))
    return R, P, W, N1
