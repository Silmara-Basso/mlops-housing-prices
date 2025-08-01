# mlops-housing-prices
Lab de MLOps com modelo para prever o preço de uma casa na California


Estou usando o dataset baseado no disponível no link abaixo:

https://www.kaggle.com/datasets/camnugent/california-housing-prices

# Automação das Operações de CI/CD no Pipeline de Machine Learning e IA

## Estrutura do Projeto
```
MLOPS-HOUSING-PRICES/
│── .github/
│   └── workflows/
│       └── mlops-pipeline.yml    			# Arquivo do GitHub Actions para automação
│
├── artefatos/                    			# Armazena modelos treinados e objetos de pré-processamento
│   ├── optimized_xgboost_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── imputer_num.pkl
│
├── config/                       			# Arquivos de configuração
│   ├── config.yaml               			# Hiperparâmetros de modelo e pré-processamento
│   ├── logging.yaml              			# Configurações de log
│
├── dados/                         			# Dados brutos e processados
│   ├── brutos/
│   │   ├── dataset.csv
│   ├── processados/
│   │   ├── dados_treino.csv
│   │   ├── dados_teste.csv
│
├── pipeline/                          		# Código-fonte para pipeline de ML modularizado
│   ├── __init__.py
│   ├── chp_preprocessa_dados.py     		# Manipulando valores ausentes, codificação, dimensionamento
│   ├── chp_engenharia_atributos.py    		# Seleção de recursos, transformação
│   ├── chp_otimiza_hiperparametros.py  	# RandomizedSearchCV para ajuste de hiperparâmetros
│   ├── chp_treina_modelo.py         		# Treinamento de modelo de ML
│   ├── chp_avalia_modelo.py         		# Avaliação de modelo e análise residual
│   ├── chp_salva_artefatos.py         		# Salvar modelo e artefatos de pré-processamento
│
├── scripts/                      			# Scripts de automação
│   ├── chp_automatiza_treino.py        	# Automatiza o pipeline de treinamento
│   ├── chp_automatiza_inferencia.py    	# Automatiza o pipeline de inferência (previsões)
│   ├── chp_inferencia_app.py    			# Usado para previsões na app
│
├── app/                          			# Deploy do modelo via app web
│   ├── app.py                    			# Endpoint de inferência
│   ├── templates/
│       └── index.html            			# Página HTML para inputs do usuário
│
├── tests/                        			# Testes unitários para o Pipeline CI/CD
│   ├── chp_testa_preprocessamento.py
│   ├── chp_testa_engenharia_atributos.py
│   ├── chp_testa_treinamento_modelo.py
│
├── Dockerfile                    			# Dockerfile para criação de container
├── requirements.txt              			# Dependências
├── LEIAME.txt                     			# Documentação do projeto
```


O objetivo de um Pipeline CI/CD (Continuous Integration / Continuous Deployment) é automatizar o treinamento, testes e deploy do modelo de Machine Learning, garantindo que cada alteração seja testada e implantada corretamente.

🔹 O que o Pipeline irá fazer?

CI (Continuous Integration)

- Rodar testes unitários (pytest)
- Rodar verificações de estilo de código (flake8)
- Testar build do container (Docker)

CD (Continuous Deployment)

- Treinar (e retreinar) o modelo automaticamente
- Salvar os artefatos treinados 
- Atualizar e fazer deploy da API Flask
- Fazer deploy do container no Docker Hub
- Deploy em AWS

### Abra o terminal ou prompt de comando, navegue até a pasta com os arquivos e execute o comando abaixo para criar um ambiente virtual:

python3 -m venv chpvenv
source chpvenv/bin/activate

### Instale o pip e as dependências:

pip install pip
pip install -r requirements.txt 

### Execute os os comandos abaixo:

brew install libomp
python scripts/chp_automatiza_treino.py
python scripts/chp_automatiza_inferencia.py
python -m pytest
python app/app.py
docker build -t chp-mlops .
docker run -d -p 5002:5002 --name chp-env-mlops chp-mlops

clique no link da app no Docker

![Imagem App](/images/app.png)

### No GitHub 
````
Settings > Secrets and variables > Actions:
DOCKER_USERNAME: seu usuário Docker Hub.
DOCKER_PASSWORD: senha ou token do Docker Hub.
SERVER-HOST: endereço IP ou domínio do servidor onde será feito o deploy (pode ser em um provedor de Cloud Computing).
SERVER-USER: usuário SSH com permissão para acessar o servidor.
SSH_PRIVATE-KEY: chave privada SSH correspondente ao usuário acima para
acesso remoto ao servidor.
````

### Use os comandos abaixo para desativar o ambiente virtual e remover o ambiente:

deactivate