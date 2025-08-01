# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das bibliotecas necessárias
import sys  # Biblioteca para manipulação do sistema
import os  # Biblioteca para manipulação de diretórios e arquivos
import pandas as pd  # Biblioteca para manipulação de dados
import joblib  # Biblioteca para salvar e carregar modelos e objetos
import warnings
warnings.filterwarnings('ignore')

# Adiciona o diretório raiz ao sys.path para facilitar a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importação da função de engenharia de atributos utilizada no treinamento
from pipeline.chp_engenharia_atributos import chp_seleciona_atributos

# Carregamento do modelo e dos artefatos necessários para a inferência
model = joblib.load("artefatos/optimized_xgboost_model.pkl")  # Carrega o modelo treinado
scaler = joblib.load("artefatos/scaler.pkl")  # Carrega o escalador de normalização
imputer_num = joblib.load("artefatos/imputer_num.pkl")  # Carrega o imputador de valores ausentes

# Obtém as features esperadas pelo modelo e pelo imputador
expected_features = list(model.feature_names_in_)
imputer_features = list(imputer_num.feature_names_in_)

# Função para realizar a inferência do modelo
def chp_inferencia(input_data):
    
    input_df = pd.DataFrame(input_data, index=[0])
    
    print("\n>>> DataFrame de entrada antes do pré-processamento:")
    print(input_df)
    
    # Garante que todas as colunas esperadas pelo imputador estejam presentes
    input_df = input_df.reindex(columns=imputer_features, fill_value=0)
    
    # Aplica o imputador para preencher valores ausentes
    if imputer_num:
        input_df[imputer_features] = imputer_num.transform(input_df[imputer_features])
    
    # Aplica a engenharia de atributos
    input_df, _ = chp_seleciona_atributos(input_df)
    
    # Reorganiza as colunas para corresponder às esperadas pelo modelo
    input_df = input_df.reindex(columns=expected_features, fill_value=0)
    
    # Aplica a normalização dos dados antes de enviar ao modelo
    if scaler:
        input_df = pd.DataFrame(scaler.transform(input_df), columns=expected_features)
    
    print("\n>>> DataFrame final enviado ao modelo:")
    print(input_df)
    
    # Faz a previsão usando o modelo treinado
    prediction_scaled = model.predict(input_df)
    
    # Cria um array para aplicar a inversão da normalização
    prediction_array = [[0] * len(expected_features)]  
    prediction_array[0][0] = prediction_scaled[0]  
    
    # Converte a previsão para o formato original
    prediction = scaler.inverse_transform(prediction_array)[0][0]  
    
    return prediction

# Função principal para execução via linha de comando
def main():
    input_data = {
        "total_rooms": 3,
        "total_bedrooms": 2,
        "population": 1500,
        "households": 5,
        "median_income": 54300.0
    }
    
    prediction = chp_inferencia(input_data)
    print("\n>>> Previsão do modelo:", prediction)

# Executa a função principal quando o script for rodado diretamente
if __name__ == "__main__":
    main()
