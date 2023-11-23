from src.settings import yaml_to_dict
from typing import List
import optuna
from src.signals_optimization import EarlyStoppingExceeded, early_stopping_opt, save_parameters

from src.normalizations import apply_normalizations, convert_to_weights
import pandas as pd
import yaml
import src.settings as settings
from src.signals_creation import *
from src.signals_evaluation import compute_metrics
from src import logging_config
import logging

logger = logging.getLogger(__name__)

config_values = settings.config_values
optim_parameters = config_values["signals_optimizations"]
metric_to_optimize = optim_parameters["param_to_optimize"]
normalization_ranges = optim_parameters["commons"]["normalization_choice"]
optuna_study_direction = optim_parameters["optuna_study_direction"]
train_data = settings.train_data
returns = settings.returns
SIGNAL_PARAMETERS_FILENAME = "best_parameters.yaml"
SIGNAL_PARAMETERS_FILENAME_OUTPUT = "best_parameters_signal_combined.yaml"
SIGNAL_SCORES_FILENAME = "best_signal_combined_scores.yaml"
OPTUNA_EARLY_STOPPING = optim_parameters["n_trials_before_callback"]
N_TRIALS = optim_parameters["n_trials"]
WEIGHT_MIN = 0.2
WEIGHT_MAX = 1.0
WEIGHT_STEP = 0.01

def collect_all_signals() -> dict:
    """
    Function that collect all signals with their best params

    Returns:
        dict: dict of pd.DataFrame of signals 
    """
    logging.info("Collection of all signals")
    signals_parameters = yaml_to_dict(SIGNAL_PARAMETERS_FILENAME)
    all_signals = {}
    for signal_name in signals_parameters.keys():
        all_signals[signal_name] = compute_signal(signal_name, train_data, **signals_parameters[signal_name])
    return all_signals


def combination_of_signals(weights: dict, normalization_choice: int) -> pd.DataFrame:
    """Function that will combined all signals together with a certain weight

    Args:
        weights (dict): weights to apply to each signal
        normalization_choice (int): choice of the normalization to apply to each signal

    Returns:
        pd.DataFrame: signal built as a combination of signal
    """
    signal_names = list(ALL_SIGNALS.keys())
    if (len(weights) != len(ALL_SIGNALS)):
        logging.error(f'There are {len(weights)} weights but {len(ALL_SIGNALS)} signals')
    else:
        for i in range(len(ALL_SIGNALS)):
            signal_name = list(ALL_SIGNALS)[i]
            signal_df = ALL_SIGNALS[signal_name].copy()
            #-- creation of the dataframe on the first occurence
            if i == 0:
                global signals_weighted
                signals_weighted = weights[signal_name] * signal_df
            else:
                signals_weighted = signals_weighted + weights[signal_name] * signal_df
    
    signals_normalized = apply_normalizations(signals_weighted, normalization_choice)
    final_signal = convert_to_weights(signals_normalized)
    return final_signal

def objective(trial):
    normalization_choice = trial.suggest_int("normalization_choice",
                                             normalization_ranges["min"],
                                             normalization_ranges["max"],
                                             normalization_ranges["step"]
                                             )
    all_trial_params = {}
    for signal_name in ALL_SIGNALS.keys():
        trial_param = trial.suggest_float(
                name = signal_name,
                low = WEIGHT_MIN,
                high = WEIGHT_MAX,
                step = WEIGHT_STEP
                )
        all_trial_params[signal_name] = trial_param
    signal = combination_of_signals(all_trial_params, normalization_choice)
    metrics = compute_metrics(signal, returns)
    return metrics[metric_to_optimize]

def check_if_improvement(dict_of_metrics: dict) -> bool:
    """
    Return true if there is an improvement of the metric studied
    :param dict_of_metrics: metrics of the current study
    :return: True if there is an improvement
    """
    previous_scores = yaml_to_dict(SIGNAL_SCORES_FILENAME)
    improvement = False
    if previous_scores is not None:
        if ((dict_of_metrics[metric_to_optimize] > float(previous_scores[metric_to_optimize]))
            and (optuna_study_direction.lower() == "maximize")):
            improvement = True
        if ((dict_of_metrics[metric_to_optimize] < float(previous_scores[metric_to_optimize]))
            and (optuna_study_direction.lower() == "minimize")):
            improvement = True
    else:
        improvement = True
    return improvement

def save_scores(dict_of_metrics: dict):
    """
    Function that saves metrics of the best combination of parameters of a signal, in a yaml file
    :param dict_of_metrics: metrics of the signal like sharpe ratio or daily pnl
    :return: None
    """
    if "pnl_series" in dict_of_metrics:
        del dict_of_metrics["pnl_series"]
    if 'turnover_series' in dict_of_metrics:
        del dict_of_metrics["turnover_series"]
    best_scores_dict = {k: str(round(v, 4)) if isinstance(v, float)
                                                else v for k, v in
                                                dict_of_metrics.items()}

    with open(SIGNAL_SCORES_FILENAME, 'w') as outfile:
        yaml.dump(best_scores_dict, outfile, default_flow_style=False)

def save_parameters(dict_of_parameters: dict):
    """
    Function that saves in a yaml file best parameters of a certain signal
    :param dict_of_parameters: best parameters of the signal
    :return: None
    """
    with open(SIGNAL_PARAMETERS_FILENAME_OUTPUT, 'w') as outfile:
        yaml.safe_dump(dict_of_parameters, outfile, default_flow_style=False)

def optimize_combined_signal():
    global ALL_SIGNALS
    
    #-- we keep the signals in the same order
    ALL_SIGNALS = dict(sorted(collect_all_signals().items()))
    study = optuna.create_study(direction = optuna_study_direction)
    try:
        study.optimize(objective, n_trials = N_TRIALS, callbacks=[early_stopping_opt])
    except EarlyStoppingExceeded:
        print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPPING}')
    
    normalization_choice = study.best_params["normalization_choice"]
    temp=study.best_params.copy()
    best_params_dict = {k: round(v, 4) if isinstance(v, float)
                                                else v for k, v in
                                                temp.items()}
    
    del temp["normalization_choice"]
    signal = combination_of_signals(temp, normalization_choice)
    metrics = compute_metrics(signal, returns)
    if check_if_improvement(metrics):
        logging.info("Signal has improved")
        save_scores(metrics)
        save_parameters(best_params_dict)
    else:
        logging.info("Signal hasn't improved")


            
            
            