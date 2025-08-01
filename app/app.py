# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

# Importação das bibliotecas necessárias
import sys  # Biblioteca para manipulação do sistema
import os  # Biblioteca para manipulação de diretórios e arquivos
from flask import Flask, request, render_template  # Importa Flask para criar a aplicação web
import joblib  # Biblioteca para carregar modelos salvos
import pandas as pd  # Biblioteca para manipulação de dados
import warnings
warnings.filterwarnings('ignore')

# Adiciona o diretório raiz ao sys.path para facilitar a importação dos módulos
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importação da função de inferência
from scripts.chp_inferencia_app import chp_inferencia

# Inicializa a aplicação Flask
app = Flask(__name__)

# Define a rota principal da aplicação
@app.route('/')
def home():
    return render_template('index.html')  # Renderiza a página inicial

# Define a rota para fazer previsões
@app.route('/predict', methods=['POST'])
def predict():
    
    # Obtém os dados enviados pelo formulário na requisição
    data = request.form.to_dict()
    
    # Converte os valores numéricos para float, mantendo 'ocean_proximity' como string
    data = {k: [float(v)] if k != 'ocean_proximity' else [v] for k, v in data.items()}
    
    # Exibe os dados recebidos no console para depuração
    print("Received data:", data)
    
    # Realiza a previsão com os dados processados
    prediction = chp_inferencia(data)
    
    # Renderiza a página com o resultado da previsão
    return render_template('index.html', prediction_text=f'Valor Previsto da Casa: ${prediction:,.2f}')

# Executa a aplicação Flask no modo debug se o script for executado diretamente
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)

