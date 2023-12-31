import src.settings as settings
settings.init()
config_values = settings.config_values
historical_data = settings.historical_data
train_data = settings.train_data
returns = settings.returns

import yaml
import warnings
warnings.filterwarnings("ignore")
from src.data_collection import collect_coins_data, load_data
from src.signals_creation import compute_signal, get_returns_data, get_rsi
from src.signals_evaluation import compute_metrics
from src.normalizations import convert_to_weights
from src.visualisation import plot_pnl
from src.signals_optimization import optimize_signal, optimize_all_signals
from src.signals_combined import *

# liste = [0.5] * 29
# combination_of_signals(liste)
optimize_combined_signal()