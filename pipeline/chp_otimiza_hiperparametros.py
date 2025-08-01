# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das bibliotecas necessárias
import optuna  # Biblioteca para otimização de hiperparâmetros
import xgboost as xgb  # Biblioteca para modelos baseados em gradient boosting
from sklearn.model_selection import train_test_split  # Ferramenta para divisão dos dados em treino e validação
from sklearn.metrics import mean_squared_error  # Métrica para avaliação do modelo
import logging  # Biblioteca para logging de informações
import yaml

# Carrega a configuração do pipeline a partir do arquivo YAML
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Função para otimização de hiperparâmetros do modelo XGBoost
# Veja a definição no Capítulo 10 do curso
def chp_hyperparameter_tuning(X_train, y_train, random_state):
    
    # Definição da função objetivo para o Optuna realizar a otimização
    def objective(trial):
        
        # Definição do espaço de busca dos hiperparâmetros
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Número de árvores no modelo
            'max_depth': trial.suggest_int('max_depth', 3, 10),  # Profundidade máxima das árvores
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),  # Taxa de aprendizado
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Fração de amostras utilizadas para treino de cada árvore
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Fração de colunas utilizadas por árvore
            'gamma': trial.suggest_float('gamma', 0, 0.5),  # Parâmetro de regularização para redução de divisões desnecessárias
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),  # Regularização L1 (lasso)
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)  # Regularização L2 (ridge)
        }
        
        # Criação do modelo XGBoost com os hiperparâmetros sugeridos
        model = xgb.XGBRegressor(objective="reg:squarederror", **params, random_state=config["training"]["random_state"])
        
        # Divisão dos dados em treino e validação
        X_train_split, X_valid_split, y_train_split, y_valid_split = train_test_split(X_train, 
                                                                                      y_train, 
                                                                                      test_size=config["training"]["test_size"], 
                                                                                      random_state=config["training"]["random_state"])
        
        # Treinamento do modelo
        model.fit(X_train_split, y_train_split)
        
        # Predição dos valores no conjunto de validação
        y_pred = model.predict(X_valid_split)
        
        # Cálculo do erro quadrático médio
        rmse = mean_squared_error(y_valid_split, y_pred)
        
        return rmse  # Retorna o erro como métrica de avaliação do Optuna

    # Criação do estudo para minimizar o erro RMSE
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=config["training"]["random_state"]))
    
    # Execução da otimização dos hiperparâmetros com um número definido de tentativas
    study.optimize(objective, n_trials=10)

    # Retorna os melhores hiperparâmetros encontrados
    return study.best_params
