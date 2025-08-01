# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das bibliotecas necessárias
import joblib  # Biblioteca para salvar e carregar modelos e objetos
import os  # Biblioteca para manipulação de diretórios e arquivos

# Função para salvar o modelo e os artefatos de pré-processamento
def chp_salva_artefatos(model, scaler, label_encoders, model_path, scaler_path, label_encoders_path):
    
    # Garante que o diretório de destino existe, criando-o se necessário
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Salva o modelo treinado no caminho especificado
    joblib.dump(model, model_path)
    
    # Salva o escalador usado para normalização dos dados
    joblib.dump(scaler, scaler_path)
    
    # Salva os codificadores de rótulos usados para transformar variáveis categóricas
    joblib.dump(label_encoders, label_encoders_path)
    
    # Exibe uma mensagem indicando que os artefatos foram salvos com sucesso
    print("Modelo e Artefatos de Pré-Processamento Salvos com Sucesso!")
