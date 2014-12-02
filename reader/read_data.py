# python imports
import os
from time import strftime, strptime
# numpy/pandas imports
import numpy as np
from pandas import read_csv, to_datetime
from pandas.tseries.offsets import Day, Minute

import pickle


__all__ = [
    'mydata',
    'select_for_test',
]

DATA_DIR = 'data'
POIDS_CSV = os.path.join(DATA_DIR, 'Poids.csv')
CHARGE_CSV = os.path.join(DATA_DIR, 'charge.csv')
TEMPS_CSV = os.path.join(
    DATA_DIR, 'Donnees_Ecometering_Temperature-par-site.csv')
DESCR_CSV = os.path.join(
    DATA_DIR, 'Donnees_Ecometering_Description_sites.csv')


def mydata(verbose, use_cache_data, test):
    if use_cache_data:
        if test:
            result = pickle.load(open("cache/data_test.p", "rb"))
        else:
            result = pickle.load(open("cache/data.p", "rb"))
        if verbose:
            print('Data retrieved from cache. Sample : ')
            print(echant(result[6]))
    else:
        poids_orig, temps, charg_orig, descr = read_data(verbose=verbose)
        if test:
            charg, poids = select_for_test(charg_orig, poids_orig, ndays=10)
        else:
            charg, poids = charg_orig, poids_orig
        sites_id = charg.columns
        print('Loading data: Done.')
        prediction = poids.copy()
        prediction.columns = ['ID', 'DATE', 'ESTIMATION']
        result = (prediction, sites_id, poids_orig, temps, charg_orig,\
                  descr, charg, poids, temps)
        if test:
            pickle.dump( result, open("cache/data_test.p", "wb"))
        else:
            pickle.dump( result, open("cache/data.p", "wb"))
        if verbose:
            print('Data retrieved from source. Sample : ')
            print(echant(charg))
    return(result)


def echant(dataframe, n=5, m = 5):
    return dataframe.T.head(n).T.head(m)

def change_format(date, f_in='%m/%d/%Y %H:%M', f_out='%d/%m/%Y %H:%M'):
    return strftime(f_out, strptime(date, f_in))

def shift_date(strdate, ndays=10, format='%d/%m/%Y %H:%M'):
    """
    Shift a given standard 
    format date by ndays
    """
    date = to_datetime(strdate, format=format)
    date -= ndays*Day()
    return date.strftime(format)

def read_data(verbose=False):
    """
    Read the files and return:
    Poids, Temps, Charge, Descr
    data+'Poids.csv'
    data+'Donnees_Ecometering_Temperature-par-site.csv'
    data+'charge.csv'
    data+'Donnees_Ecometering_Description_sites.csv'
    and a dataframe ready for Prediction
    """

    print('Retrieve the weights')
    Poids = read_csv(POIDS_CSV, sep=';')
    Poids.Date = Poids.Date.apply(change_format)
    
    print('Retrieve the temperature')
    Temps = read_csv(TEMPS_CSV, sep=';')
    Temps.Jour = Temps.Jour.apply(change_format)
    Temps.index = to_datetime(Temps['Jour'].values, format='%d/%m/%Y %H:%M')
    Temps = Temps.drop('Jour', 1)
    Temps = Temps.resample('10min', fill_method='ffill')
    if verbose:
        print(echant(Temps, n=10, m=20))
    
    print('Retrieve the charge')
    Charg = read_csv(CHARGE_CSV, sep=';')
    Charg.index = to_datetime(Charg['DATE_LOCAL'].values, format='%d/%m/%Y %H:%M')
    Charg = Charg.drop('DATE_LOCAL', 1)
    if verbose:
        print(echant(Charg, n=10, m = 20))
        
    print('Retrieve the description')
    Descr = read_csv(DESCR_CSV, sep=';')

    return Poids, Temps, Charg, Descr

def select_for_test(Charg, Poids, ndays=10):
    '''
    Creation of a frame for the Poids 
    and the prediction, translated by ndays=10 days.
    '''
    Poids_test = Poids.copy()
    for j in Poids_test.index:
        strdate = Poids_test.Date.ix[j]
        strdatebis = (to_datetime(strdate, format='%d/%m/%Y %H:%M')-10*Day()).strftime('%d/%m/%Y %H:%M')
        Poids_test.Date.ix[j] = strdatebis
    Charg_test = Charg.copy()
    for iden in Charg.columns:
        if iden == 'ID24':
            print('ID24 taken into account properly in the test case.')
            continue 
        else:
            id_fin = Charg[iden].dropna().index[-1]
            Charg_test[iden][id_fin-ndays*Day()+Minute():] = np.NaN
    return Charg_test, Poids_test
