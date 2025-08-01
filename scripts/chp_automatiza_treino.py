# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das bibliotecas necessárias
import os  # Biblioteca para manipulação de diretórios e arquivos
import sys  # Biblioteca para manipulação do sistema
import yaml  # Biblioteca para carregamento de arquivos YAML
import joblib  # Biblioteca para salvar e carregar modelos e objetos
import logging  # Biblioteca para geração de logs
import logging.config  # Biblioteca para configuração de logs
import pandas as pd  # Biblioteca para manipulação de dados
import xgboost as xgb  # Biblioteca para modelos baseados em gradient boosting
from sklearn.model_selection import train_test_split  # Ferramenta para divisão dos dados em treino e teste
from sklearn.impute import SimpleImputer  # Ferramenta para tratamento de valores ausentes
from sklearn.preprocessing import StandardScaler  # Ferramenta para normalização de dados

# Adiciona o diretório raiz ao sys.path para facilitar a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importação das funções do pipeline
from pipeline.chp_preprocessa_dados import chp_carrega_dados, chp_preprocessa_dados
from pipeline.chp_engenharia_atributos import chp_seleciona_atributos
from pipeline.chp_treina_modelo import chp_treina_modelo
from pipeline.chp_avalia_modelo import chp_avalia_modelo
from pipeline.chp_otimiza_hiperparametros import chp_hyperparameter_tuning
from pipeline.chp_salva_artefatos import chp_salva_artefatos

# Carrega a configuração de logging a partir do arquivo YAML
with open("config/logging.yaml", "r") as file:
    logging.config.dictConfig(yaml.safe_load(file))

logger = logging.getLogger(__name__)

# Carrega a configuração do pipeline a partir do arquivo YAML
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Função principal do pipeline
def main():
    try:
        logger.info("Carregando e Pré-processando Dados...")

        # Carrega os dados a partir do caminho especificado na configuração
        df = chp_carrega_dados(config["data"]["raw_path"])

        # Remove a coluna 'ocean_proximity' se ela estiver presente
        if "ocean_proximity" in df.columns:
            df = df.drop(columns=["ocean_proximity"])

        # Define as variáveis independentes (X) e a variável dependente (y)
        X = df.drop(columns=["median_house_value"])
        y = df["median_house_value"]

        # Pré-processa os dados
        X_cleaned, _, scaler = chp_preprocessa_dados(X)

        logger.info("Ajustando Imputadores Para Valores Ausentes...")
        num_cols = X_cleaned.select_dtypes(include=["int64", "float64"]).columns.tolist()

        imputer_num = SimpleImputer(strategy="median")
        
        if num_cols:
            X_cleaned[num_cols] = imputer_num.fit_transform(X_cleaned[num_cols])

        # Seleciona os atributos mais relevantes
        X_selected, _ = chp_seleciona_atributos(X_cleaned)

        # Divide os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X_selected, 
                                                            y, 
                                                            test_size=config["training"]["test_size"], 
                                                            random_state=config["training"]["random_state"])

        logger.info("Treinando Modelo...")
        trained_models = chp_treina_modelo(X_train, y_train, config)

        logger.info("Ajuste de Hiperparâmetros Para o Melhor Modelo...")
        best_params = chp_hyperparameter_tuning(X_train, y_train, config["training"]["random_state"])
        optimized_model = xgb.XGBRegressor(objective="reg:squarederror", **best_params, random_state=config["training"]["random_state"])
        optimized_model.fit(X_train, y_train)

        # Faz as previsões
        y_pred = optimized_model.predict(X_test)

        # Avalia o modelo
        metricas = chp_avalia_modelo(y_test, y_pred)

        # Log
        logger.info(f"Métricas de Avaliação do Modelo: {metricas}")

        # Define os caminhos para salvar o modelo e os artefatos
        model_path = config["model"]["path"]
        scaler_path = config["model"]["scaler_path"]
        label_encoders_path = config["model"].get("label_encoders_path", None)

        # Salva os artefatos do modelo
        if label_encoders_path:
            chp_salva_artefatos(optimized_model, scaler, {}, model_path, scaler_path, label_encoders_path)
        else:
            chp_salva_artefatos(optimized_model, scaler, {}, model_path, scaler_path, "artefatos/label_encoders.pkl")

        # Salva o imputador numérico
        joblib.dump(imputer_num, "artefatos/imputer_num.pkl")

        # Salva os dados de treino e teste em arquivos CSV
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        train_data.to_csv(config["data"]["processed_path"] + '/dados_treino.csv', index=False)
        test_data.to_csv(config["data"]["processed_path"] + '/dados_teste.csv', index=False)

        logger.info("Treinamento Concluído com Sucesso!")

    except Exception as e:
        logger.exception("Ocorreu uma exceção durante o treinamento: %s", str(e))

# Executa a função principal quando o script for rodado diretamente
if __name__ == "__main__":
    main()
