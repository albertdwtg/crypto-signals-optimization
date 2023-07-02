import pandas as pd
import optuna
from src.signals_evaluation import compute_metrics
from src.signals_creation import *
import src.settings as settings
from typing import Union
import yaml
from src.settings import yaml_to_dict

config_values = settings.config_values
optim_parameters = config_values["signals_optimizations"]
lag_ranges = optim_parameters["commons"]["lag"]
normalization_ranges = optim_parameters["commons"]["normalization_choice"]
metric_to_optimize = optim_parameters["param_to_optimize"]
n_trials = optim_parameters["n_trials"]
optuna_study_direction = optim_parameters["optuna_study_direction"]
train_data = settings.train_data
returns = settings.returns

def params_to_optimize_with_trial(trial, signal_name:str):
    """
    Function that returns parameters to try for a signal,
    based on signal name
    :param trial: trail object of optuna
    :param signal_name: name of the signal we want to optimize
    :return: tuple of a dict of parameters and the trial
    """
    signal_parameters = optim_parameters[signal_name]
    param_names = signal_parameters.keys()
    all_trial_params = {}
    for param in param_names:
        if signal_parameters[param]["type"] == "int":
            trial_param = trial.suggest_int(
                param,
                signal_parameters[param]["min"],
                signal_parameters[param]["max"],
                signal_parameters[param]["step"]
                )
        if signal_parameters[param]["type"] == "float":
            trial_param = trial.suggest_float(
                param,
                signal_parameters[param]["min"],
                signal_parameters[param]["max"],
                signal_parameters[param]["step"]
                )
        all_trial_params[param] = trial_param
    return all_trial_params, trial


def objective(trial) -> Union[float, int]:
    """
    Objective function of the optuna study
    :param trial: trail object of optuna
    :return: metric that we want to optimize
    """
    lag = trial.suggest_int("lag",
                            lag_ranges["min"],
                            lag_ranges["max"],
                            lag_ranges["step"]
                            )
    normalization_choice = trial.suggest_int("normalization_choice",
                                             normalization_ranges["min"],
                                             normalization_ranges["max"],
                                             normalization_ranges["step"]
                                             )
    params, trial = params_to_optimize_with_trial(trial, signal_name_to_optimize)

    params.update({"lag": lag, "normalization_choice": normalization_choice})
    signal = compute_signal(signal_name_to_optimize, train_data, **params)
    metrics = compute_metrics(signal, returns)
    return metrics[metric_to_optimize]

def optimize_signal(signal_name: str) -> optuna.study.study.Study:
    """
    Function that optimizes a signal based on its name
    :param signal_name: name of the signal we want to optimize
    :return: study of all trials
    """
    global signal_name_to_optimize
    signal_name_to_optimize = signal_name
    study = optuna.create_study(direction = optuna_study_direction)
    study.optimize(objective, n_trials = n_trials)

    signal = compute_signal(signal_name_to_optimize, train_data, **study.best_params)
    metrics = compute_metrics(signal, returns)

    save_parameters(signal_name, study.best_params)
    save_scores(signal_name, metrics)

    return study

def save_parameters(signal_name: str, dict_of_parameters: dict):
    """
    Function that saves in a yaml file best parameters of a certain signal
    :param signal_name: Name of the signal that we are saving its parameters
    :param dict_of_parameters: best parameters of the signal
    :return: None
    """
    best_parameters_file = yaml_to_dict("best_parameters.yaml")
    if signal_name not in best_parameters_file:
        best_parameters_file[signal_name] = {}
    best_parameters_file[signal_name] = dict_of_parameters
    with open('best_parameters.yaml', 'w') as outfile:
        yaml.safe_dump(best_parameters_file, outfile, default_flow_style=False)

def save_scores(signal_name: str, dict_of_metrics: dict):
    """
    Function that saves metrics of the best combination of parameters of a signal, in a yaml file
    :param signal_name: name of the signal we are dealing with
    :param dict_of_metrics: metrics of the signal like sharpe ratio or daily pnl
    :return: None
    """
    best_scores_file = yaml_to_dict("best_signals_scores.yaml")
    if signal_name not in best_scores_file:
        best_scores_file[signal_name] = {}
    if "pnl_series" in dict_of_metrics:
        del dict_of_metrics["pnl_series"]
    if 'turnover_series' in dict_of_metrics:
        del dict_of_metrics["turnover_series"]
    best_scores_file[signal_name] = {k: str(round(v, 4)) if isinstance(v, float)
                                                else v for k, v in
                                                dict_of_metrics.items()}

    with open('best_signals_scores.yaml', 'w') as outfile:
        yaml.dump(best_scores_file, outfile, default_flow_style=False)