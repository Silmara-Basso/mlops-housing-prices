# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação da biblioteca necessária
import pandas as pd  # Biblioteca para manipulação de dados em DataFrames

# Função para criação de novos atributos e remoção de colunas desnecessárias
# Veja a definição no Capítulo 10 do curso
def chp_seleciona_atributos(df):
    
    # Criação de novos atributos com base em relações entre colunas existentes
    df['rooms_per_household'] = df['total_rooms'] / df['households']  # Número de quartos por domicílio
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']  # Proporção de quartos por total de cômodos
    df['population_per_household'] = df['population'] / df['households']  # Número de pessoas por domicílio

    # Definição das colunas a serem removidas após a criação dos novos atributos
    features_to_drop = ['total_rooms', 'total_bedrooms', 'population']
    
    # Remoção das colunas especificadas
    df = df.drop(columns=features_to_drop)

    # Retorna o DataFrame atualizado e a lista de colunas removidas
    return df, features_to_drop
