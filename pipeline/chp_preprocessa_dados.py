# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das bibliotecas necessárias
import pandas as pd  # Biblioteca para manipulação de dados em DataFrames
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Ferramentas para padronização e codificação de dados
from sklearn.impute import SimpleImputer  # Ferramenta para tratamento de valores ausentes

# Função para carregar os dados a partir de um arquivo CSV
def chp_carrega_dados(file_path):
    return pd.read_csv(file_path)  # Retorna um DataFrame com os dados lidos

# Função para pré-processamento dos dados
def chp_preprocessa_dados(df, label_encoders=None, scaler=None, training=True):
    
    # Inicializa dicionários para armazenar os codificadores de rótulos e o escalador caso não sejam fornecidos
    if label_encoders is None:
        label_encoders = {}
    if scaler is None:
        scaler = StandardScaler()  # Define um escalador para normalizar os dados numéricos

    # Identifica colunas numéricas e categóricas
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()  # Lista colunas numéricas
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()  # Lista colunas categóricas

    # Tratamento de valores ausentes para colunas numéricas
    if num_cols:
        imputer_num = SimpleImputer(strategy="median")  # Usa a mediana para imputação
        df[num_cols] = imputer_num.fit_transform(df[num_cols]) if training else imputer_num.transform(df[num_cols])

    # Tratamento de valores ausentes para colunas categóricas
    if cat_cols:
        imputer_cat = SimpleImputer(strategy="most_frequent")  # Usa o valor mais frequente para imputação
        df[cat_cols] = imputer_cat.fit_transform(df[cat_cols]) if training else imputer_cat.transform(df[cat_cols])

    # Codificação das variáveis categóricas usando Label Encoding
    for feature in cat_cols:
        if feature not in label_encoders:  # Se a feature ainda não foi codificada
            label_encoders[feature] = LabelEncoder()  # Cria um codificador para a feature
            df[feature] = label_encoders[feature].fit_transform(df[feature]) if training else label_encoders[feature].transform(df[feature])
        else:
            df[feature] = label_encoders[feature].transform(df[feature])  # Aplica a transformação usando o codificador existente

    # Normalização dos dados numéricos usando StandardScaler
    if num_cols:
        df[num_cols] = scaler.fit_transform(df[num_cols]) if training else scaler.transform(df[num_cols])

    # Retorna o DataFrame pré-processado, os codificadores de rótulos e o escalador
    return df, label_encoders, scaler
