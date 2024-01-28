from .cross_validation import exec_cross_validation
import optuna
from ..utils.log_config import setup_logging

result_logger, _ = setup_logging()

def set_trial_params(trial, params):
    optuna_params = {}
    for key, value in params.items():
        if isinstance(value, dict):
            if 'type' in value and 'args' in value:
                param_type = value['type']
                if param_type == 'log_float':
                    optuna_params[key] = trial.suggest_float(key, *value['args'], log=True)
                elif param_type == 'float':
                    optuna_params[key] = trial.suggest_float(key, *value['args'])
                elif param_type == 'int':
                    optuna_params[key] = trial.suggest_int(key, *value['args'])
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
        else:
            # 'type' と 'args' キーがない場合、そのままの値を使用
            optuna_params[key] = value

    return optuna_params


def create_objective(model, dataset, metric, params, valuation_index):
    def objective(trial):
        # 探索するハイパーパラメータの範囲を定義
        optuna_params = set_trial_params(trial, params)
        # 交差検証を実行し、性能評価の指標を計算
        average_index = exec_cross_validation(model, dataset, metric, optuna_params, valuation_index)

        return average_index
    return objective


def exec_optuna(model, dataset, metric, params, valuation_index, objective):
    study = optuna.create_study(direction=objective)
    objective = create_objective(model, dataset, metric, params, valuation_index)
    study.optimize(objective, n_trials=100)
    result_logger.info(f'Best params:{study.best_params}')
    result_logger.info(f'Best score:{study.best_value}')
    return