from typing import Any
from venv import logger
import optuna
import sklearn
import logging
import argparse
import dynaconf
import numpy as np
import pandas as pd
from dynaconf import Dynaconf
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer, accuracy_score

RANDOM_STATE = 121

def get_logger(log_file: str = 'log.log') -> logging.Logger:
    logger = logging.getLogger(__name__)

    format_str = '%(asctime)s:%(module)s:%(filename)s:%(funcName)s:%(levelname)s%(message)s'
    formatter = logging.Formatter(format_str)

    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    fileHandler = logging.FileHandler(filename=log_file, mode='a')
    fileHandler.setFormatter(format)

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.DEBUG)

    return logger

def random_forest( 
        params: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray
        ):
    
    classifier = RandomForestClassifier(random_state = RANDOM_STATE, 
                                        max_depth = params['max_depth'], 
                                        min_samples_leaf = params['min_samples_leaf'],        
                                        n_estimators = params['n_estimators'], n_jobs=-1)
    
    performance = cross_val_score(classifier, 
                                  X_train, 
                                  y_train, 
                                  cv=5, 
                                  scoring=make_scorer(f1_score, average='macro'), 
                                  n_jobs=-1)
        
    return performance.sum()/len(performance)

def rf_objective(trail: Any,
                 X_train: np.ndarray, 
                 y_train: np.ndarray):
    
    params = {'max_depth': trail.suggest_int('max_depth', 3, 30),
              'n_estimators' : trail.suggest_int('n_estimators', 3, 110),
              'min_samples_leaf': trail.suggest_int('min_samples_leaf', 10, 100)
             }
    
    return random_forest(params, X_train, y_train)

def mlp(params: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray
        ):
    
    hidden_layer_sizes = (params['n_layer1'], params['n_layer2'])
    
    clf = MLPClassifier(solver=params['solver'],
                        activation=params['activation'],
                        learning_rate_init=round(params['learning_rate_init'],3),
                        alpha=1e-5,
                        hidden_layer_sizes=hidden_layer_sizes,
                        max_iter=1000,
                        random_state=RANDOM_STATE)

    performance = cross_val_score(clf, 
                                  X_train, 
                                  y_train, 
                                  cv=5, 
                                  scoring=make_scorer(f1_score, average='macro'))

    return performance.sum()/len(performance)

def mlp_objective(trail: Any, 
                  X_train: np.ndarray, 
                  y_train: np.ndarray):
    
    params = {
        'learning_rate_init': trail.suggest_uniform('learning_rate_init', 0.001, 0.1),
        'solver': trail.suggest_categorical('solver', ["adam", "sgd", "lbfgs"]),
        'activation': trail.suggest_categorical('activation', ['logistic', 'tanh', 'relu']),
        'n_layer1' : trail.suggest_int('n_layer1', 5, 25),
        'n_layer2' : trail.suggest_int('n_layer2', 5, 25)
        }
    
    return mlp(params, X_train, y_train)

def get_performance(
        classifier: sklearn.ensemble._forest.RandomForestClassifier | sklearn.neural_network._multilayer_perceptron.MLPClassifier,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        df_usr: pd.core.frame.DataFrame,
        settings: dynaconf.base.LazySettings,
        columns_exclude: set,
        scaler: sklearn.preprocessing._data.StandardScaler,
        logger
        ) -> None:
    
    y_pred = classifier.predict(X_train)
    logger.info(f'best f1 score on train: {f1_score(y_train, y_pred, average="macro")}')

    y_pred = classifier.predict(X_test)
    logger.info(f'best f1 score on test: {f1_score(y_test, y_pred, average="macro")}')

    X_usr = df_usr[df_usr.columns.difference(columns_exclude.union(settings.feature_groups.categorical))].values
    X_binary_usr = df_usr[settings.feature_groups.categorical].values
    y_usr = df_usr[settings.feature_groups.prediction].values
    X_usr_cat = np.concatenate((scaler.transform(X_usr), X_binary_usr), axis=1)

    y_pred = classifier.predict(X_usr_cat)
    logger.info(f'best f1 score on usr: {f1_score(y_usr, y_pred, average="macro")}')
    logger.info(f'best f1 score on usr: {accuracy_score(y_usr, y_pred)}')

def main(args):
    logger.info(args)
    settings_file = './Data/models.yaml'
    
    settings = Dynaconf(settings_file=settings_file)

    df_features = pd.read_csv(settings.files.features)
    df_humanForecasts = pd.read_csv(settings.files.humanForecasts)

    df_features_non_user = df_features[~df_features.doi.isin(df_humanForecasts.doi)]
    df_features_user = df_features[df_features.doi.isin(df_humanForecasts.doi)]

    columns_exclude = set(settings.feature_groups.meta_features).union(\
                        set(settings.feature_groups.prediction),
                        set(settings.feature_groups[args.ablation_feature]))
    
    X_nums = df_features_non_user[df_features_non_user.columns.difference(columns_exclude.union(settings.feature_groups.categorical))].values
    X_categorical = df_features_non_user[settings.feature_groups.categorical].values
    y = df_features_non_user[settings.feature_groups.prediction].values

    scaler = StandardScaler()
    scaler.fit(X_nums)
    X = np.concatenate((scaler.transform(X_nums), X_categorical), axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = 0.2, 
                                                        random_state = RANDOM_STATE)    

    if args.model.upper() == 'RF':
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: rf_objective(
                                                    trail=trial, 
                                                    X_train=X_train, 
                                                    y_train=y_train
                                                    ),
                        n_trials=args.n_trials
                        )
        params = study.best_params
        
        classifier = RandomForestClassifier(random_state = RANDOM_STATE, 
                                            max_depth = study.best_params['max_depth'], 
                                            min_samples_leaf = params['min_samples_leaf'],
                                            n_estimators=study.best_params['n_estimators'],
                                            n_jobs=-1)
        
    elif args.model.upper() == 'MLP':
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: mlp_objective(
                                                    trail=trial, 
                                                    X_train=X_train, 
                                                    y_train=y_train
                                                    ),
                        n_trials=args.n_trials
                        )
        logger.info('chamma')
        params = study.best_params
        hidden_layer_sizes = (params['n_layer1'], params['n_layer2'])
        
        classifier = MLPClassifier(solver=params['solver'],
                            activation=params['activation'],
                            learning_rate_init=round(params['learning_rate_init'],3),
                            alpha=1e-5,
                            hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=1000,
                            random_state=RANDOM_STATE)
        
    else:
        raise NotImplementedError  
    
    logger.info(study.best_trial)
   
    classifier.fit(X_train, y_train)
    get_performance(
        classifier = classifier,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        df_usr=df_features_user,
        settings=settings,
        columns_exclude=columns_exclude,
        scaler=scaler,
        logger=logger) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', 
                        type=str, 
                        default='RF',
                        help='model for tuning. possible options: RF, MLP')

    parser.add_argument('--n_trials',
                        type=int,
                        default=1500,
                        help='total number of iterations for parameter search')
    
    parser.add_argument('--ablation_feature',
                        type=str,
                        default='baseline',
                        help='feature groups to be removed for ablation study')

    args = parser.parse_args()
    main(args)