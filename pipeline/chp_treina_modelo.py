# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das bibliotecas necessárias
import xgboost as xgb  # Biblioteca para modelos baseados em gradient boosting
from sklearn.linear_model import LinearRegression  # Modelo de regressão linear
from sklearn.tree import DecisionTreeRegressor  # Modelo de árvore de decisão para regressão
from sklearn.ensemble import RandomForestRegressor  # Modelo de floresta aleatória para regressão

# Função para treinar múltiplos modelos de regressão
def chp_treina_modelo(X_train, y_train, config):
    
    # Dicionário contendo os modelos a serem treinados (em nosso caso sempre o mesmo algoritmo, mas outros poderiam ser usados)
    models = {"XGBoost": xgb.XGBRegressor(objective="reg:squarederror", random_state=config["training"]["random_state"])}
    
    # Dicionário para armazenar os modelos treinados
    trained_models = {}
    
    # Loop para treinar cada modelo do dicionário
    for name, model in models.items():
        model.fit(X_train, y_train)   # Treina o modelo com os dados fornecidos
        trained_models[name] = model  # Armazena o modelo treinado no dicionário
    
    # Retorna os modelos treinados
    return trained_models
