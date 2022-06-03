import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, r2_score


class TimeSeriesPredictor:
    ## Modelo para previsões
    __model: ExponentialSmoothing

    ## Lista com codigo dos indicadores que serão avaliados
    __indicators_codelist: list

    ## Lista para os nomes por extenso dos indicadores. É usado
    ## para colocar o título nos gráficos usados no matplotlib
    __indicators_namelist: dict 

    ## Arrays com os conjuntos de treino e teste, bem como as respostas preditas pelo modelo.
    ## *_data se refere aos valores observados na série. Essas são as variáveis mais importantes
    ## desse conjunto abaixo. *_years são os anos que foram usados nos períodos de treino, validação e teste,
    ## e eles são usados principalmente para construir os gráficos com matplotlib
    __training_data:    np.empty(0, dtype=np.double)
    __training_years:   np.empty(0, dtype=np.double)
    __validation_data:  np.empty(0, dtype=np.double)
    __validation_years: np.empty(0, dtype=np.double)
    __testing_data:     np.empty(0, dtype=np.double)
    __testing_years:    np.empty(0, dtype=np.double)
    __predictions_testing:     np.empty(0, dtype=np.double)
    __predictions_validation:  np.empty(0, dtype=np.double)
    ## O path absoluto do CSV com os dados
    __dataset_filepath: str

    ## Ano que a serie inicia.
    __tseries_start_year: int

    ## Ano que a serie finaliza.
    __tseries_end_year: int

    ## Porcentagem de dados para treino de cada indicador
    __percentage_train: float

    ## Porcentagem de dados de validação. É definido como uma parte
    ## do inteiro, logo, 10% de validação é igual aos próximos 10% 
    ## depois do fim do período de treino
    __percentage_validation: float

    '''
    Instancia um preditor de series temporais
    '''
    def __init__(self):
        self.__model = ExponentialSmoothing

        self.__training_data    = np.empty(0, dtype=np.double)
        self.__training_years   = np.empty(0, dtype=np.double)
        self.__validation_data  = np.empty(0, dtype=np.double)
        self.__validation_years = np.empty(0, dtype=np.double)
        self.__testing_data     = np.empty(0, dtype=np.double)
        self.__testing_years    = np.empty(0, dtype=np.double)
        self.__predictions_testing    = np.empty(0, dtype=np.double)
        self.__predictions_validation = np.empty(0, dtype=np.double)

        self.__runtime_metrics     = {}
        self.__indicators_namelist = {}
        

    '''
    Gets e Sets :)
    '''
    def set_tseries_start_year(self, start_year: int):
        self.__tseries_start_year = start_year

    def get_tseries_start_year(self):
        return self.__tseries_start_year
        
    def set_tseries_end_year(self, end_year: int):
        self.__tseries_end_year = end_year

    def get_tseries_end_year(self):
        return self.__tseries_end_year

    def set_indicators_codelist(self, indicators_codelist):
        self.__indicators_codelist = indicators_codelist

    def get_indicators_codelist(self):
        return self.__indicators_codelist

    def set_percentage_train(self, percentage: int):
        self.__percentage_train = percentage/100
    
    def get_percentage_train(self):
        return self.__percentage_train

    def set_percentage_validation(self, percentage: int):
        self.__percentage_validation = percentage/100

    def get_percentage_validation(self):
        return self.__percentage_validation

    def set_dataset_filepath(self, filepath: str):
        self.__dataset_filepath = filepath

    def get_dataset_filepath(self):
        return self.__dataset_filepath

    def get_training_data(self):
        return self.__training_data

    def get_training_years(self):
        return self.__training_years

    def get_validation_data(self):
        return self.__validation_data
    
    def get_validation_years(self):
        return self.__validation_years

    def get_testing_data(self):
        return self.__testing_data

    def get_testing_years(self):
        return self.__testing_years

    def get_predictions_testing(self):
        return self.__predictions_testing

    def get_predictions_validation(self):
        return self.__predictions_validation

    def get_indicators_namelist(self):
        return self.__indicators_namelist

    def set_runtime_metric(self, metric_name: str, measured_time: float):
        self.__runtime_metrics[metric_name] = f"{measured_time:.6f} s"
    
    def get_runtime_metrics(self):
        return self.__runtime_metrics


    '''
    Essa função separa os valores para um indicador no dataset entre treino e teste 
    de acordo com self.__percentage_train
    '''
    def split_train_test_val(self, indicator_code):

        ## Início to cronômetro para separação
        start = time.perf_counter()

        ## Abrindo o CSV
        df = pd.read_csv(self.get_dataset_filepath(), sep=";")

        ## Pega a coluna do dataset que tem as informações desse indicador
        df_row_indicator = df.loc[df['Indicator Code'] == indicator_code]

        ## Armazena o nome do indicador em uma lista para evitar ter que abrir várias vezes o CSV
        self.__indicators_namelist[indicator_code] = df_row_indicator['Indicator Name'].values[0]

        ## Indica qual o ultimo ano para treinamento de acordo com a porcentagem de treino
        last_year_training = int((self.get_tseries_end_year() - self.get_tseries_start_year())*self.get_percentage_train() + self.get_tseries_start_year())
        
        ## Obtendo o conjunto de dados para treinamento. Esse conjunto vai desde o inicio da serie temporal
        ## ate o ultimo ano de treinamento.
        tmp = df_row_indicator.loc[:, f"{self.get_tseries_start_year()}":f"{last_year_training}"]

        ## Armazenando os valores de treino
        self.__training_data   = np.asarray([float(x) for x in tmp.values[0]])
        self.__training_years  = np.asarray([datetime(int(x), 1, 1) for x in tmp.keys()])

        last_year_validation = int((self.get_tseries_end_year() - self.get_tseries_start_year())*self.get_percentage_validation() + last_year_training)
        tmp = df_row_indicator.loc[:, f"{last_year_training}":f"{last_year_validation}"]

        ## Obtendo o conjunto de dados de validação
        self.__validation_data   = np.asarray([float(x) for x in tmp.values[0]])
        self.__validation_years  = np.asarray([datetime(int(x), 1, 1) for x in tmp.keys()])

        ## Obtendo agora o conjunto de dados de teste, que vai desde o ultimo ano de treino ate o fim
        ## da série temporal.
        tmp = df_row_indicator.loc[:, f"{last_year_validation}":f"{self.get_tseries_end_year()}"]

        ## Armazenando os valores de teste
        self.__testing_data   = np.asarray([float(x) for x in tmp.values[0]])
        self.__testing_years  = np.asarray([datetime(int(x), 1, 1) for x in tmp.keys()])

        end   = time.perf_counter()
        self.set_runtime_metric("Separação dos dados de teste e treino", end - start)
        

    '''
    Treina o modelo com os dados em self.get_training_data(). Deve ser chamada
    somente depois de self.split_train_test_val()
    '''
    def train_model(self):
        start = time.perf_counter()

        self.__model = ExponentialSmoothing(
                                            ## Os valores observados na serie
                                            endog=self.get_training_data(),

                                            ## As datas referentes ao treino. É importante para o predict() conseguir prever 
                                            ## valores para anos específicos
                                            dates=self.get_training_years(),

                                            ## O espaçamento entre as datas. Os dados são anuais e começam no inicio de 
                                            ## cada ano, então usamos "AS". "AS" = Anual Start
                                            freq="AS",

                                            initialization_method="estimated",
                                            ## A série tem uma trend aditiva
                                            trend="add"
                                            ).fit(smoothing_level=0.8)

        end   = time.perf_counter()

        self.set_runtime_metric("Etapa de treinamento do modelo", end-start)


    '''
    Faz a previsão dos dados no conjunto de validação. Essa função deve ser chamada
    somente depois de self.split_train_test_val() e de self.train_model().
    Os dados de validação são usados para ajuste dos parâmetros do modelo
    '''
    def predict_validation_data(self):
        start = time.perf_counter()

        first_year, last_year = self.get_validation_years()[0], self.get_validation_years()[-1]

        ## Salvando os valores previstos entre os anos first_year e last_year
        self.__predictions_validation = self.__model.predict(start=first_year, end=last_year)

        print(f"MAE Score na validação = {mean_absolute_error(self.get_validation_data(), self.get_predictions_validation())}")
        print(f"R2 Score na validação  = {r2_score(self.get_validation_data(), self.get_predictions_validation())}", end="\n\n")

        end   = time.perf_counter()

        self.set_runtime_metric("Predição dos próximos valores na série", end - start)


    '''
    Faz a previsão dos dados no conjunto de teste. Essa função deve ser chamada 
    somente depois de self.split_train_test_val() e de self.train_model()
    '''
    def predict_testing_data(self):
        start = time.perf_counter()

        first_year, last_year = self.get_testing_years()[0], self.get_testing_years()[-1]

        ## Salvando os valores previstos entre os anos first_year e last_year
        self.__predictions_testing = self.__model.predict(start=first_year, end=last_year)

        end   = time.perf_counter()

        self.set_runtime_metric("Predição dos próximos valores na série", end - start)


    '''
    Plota os gráficos com as linhas dos valores de treino, dos valores previstos 
    e dos valores observados. É altamente recomendado chamar apenas essa função
    e deixar ela orquestrar a chamada das outras.
    '''
    def plot_indicators(self):
        start = time.perf_counter()

        for indicator_code in self.get_indicators_codelist():
            self.split_train_test_val(indicator_code)
            self.train_model()
            self.predict_testing_data()

            indicator_name = self.get_indicators_namelist()[indicator_code]

            print(f"Métricas de avaliação para a previsão de '{indicator_name}'")

            print(f"MAE Score no teste = {mean_absolute_error(self.get_testing_data(), self.get_predictions_testing())}")
            print(f"R2 Score no teste  = {r2_score(self.get_testing_data(), self.get_predictions_testing())}", end="\n\n")

            plt.title(f"{indicator_name}", fontsize=18)
            plt.plot(self.get_training_years(), self.get_training_data(), c="green", label="Treino", linewidth=2.5)
            plt.plot(self.get_validation_years(), self.get_validation_data(), c="darkorange", label="Validação", linewidth=2.5)
            plt.plot(self.get_testing_years(), self.get_testing_data(), c="blue", label="Teste", linewidth=2.5)
            plt.plot(self.get_testing_years(), self.get_predictions_testing(), c="red", label="Previsão", linewidth=2.5)

            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.legend(fontsize=14)
            plt.show()

            plt.clf()

        end   = time.perf_counter()
        self.set_runtime_metric("Geração dos gráficos dos indicadores", end - start)