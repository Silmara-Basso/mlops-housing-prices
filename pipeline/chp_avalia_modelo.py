# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das métricas para avaliação de modelos
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Importa funções para cálculo de erro e desempenho

# Função para avaliar o desempenho do modelo
def chp_avalia_modelo(y_test, y_pred):
    
    # Cálculo do erro absoluto médio (MAE), que mede a média das diferenças absolutas entre os valores reais e previstos
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cálculo do erro quadrático médio (MSE), que mede o erro ao elevar ao quadrado as diferenças entre os valores reais e previstos
    mse = mean_squared_error(y_test, y_pred)
    
    # Cálculo da raiz do erro quadrático médio (RMSE), que torna o erro comparável às unidades dos valores reais
    rmse = mse ** 0.5
    
    # Cálculo do coeficiente de determinação (R²), que mede a qualidade do ajuste do modelo aos dados
    r2 = r2_score(y_test, y_pred)
    
    # Retorna um dicionário com as métricas calculadas
    return {"MAE": mae, "RMSE": rmse, "R2 Score": r2}
