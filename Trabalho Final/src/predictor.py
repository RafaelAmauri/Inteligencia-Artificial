import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TimeSeriesPredictor:
    ## Modelo para previsões
    __model: ExponentialSmoothing

    ## Lista com codigo dos indicadores que serão avaliados
    __indicators_codelist: list

    ## Arrays com os conjuntos de treino e teste, bem como as respostas preditas pelo modelo.
    ## *_data se refere aos valores observados na série. Essas são as variáveis mais importantes
    ## desse conjunto abaixo. *_years são os anos que foram usados nos períodos de teste e treino,
    ## e eles são usados principalmente para construir os gráficos com matplotlib
    __training_data:    np.empty(0, dtype=np.float64)
    __training_years:   np.empty(0, dtype=np.float64)
    __testing_data:     np.empty(0, dtype=np.float64)
    __testing_years:    np.empty(0, dtype=np.float64)
    __predictions:      np.empty(0, dtype=np.float64)
    
    ## O path absoluto do CSV com os dados
    __dataset_filepath: str

    ## Ano que a serie inicia.
    __tseries_start_year: int

    ## Ano que a serie finaliza.
    __tseries_end_year: int

    ## Porcentagem de dados para treino de cada indicador
    __percentage_train: int

    ## Lista para os nomes por extenso dos indicadores. É usado
    ## para colocar o título nos gráficos usados no matplotlib
    __indicators_namelist: dict 


    '''
    Instancia um preditor de series temporais
    '''
    def __init__(self):
        self.__model = ExponentialSmoothing

        self.__training_data    = np.empty(0, dtype=np.float64)
        self.__training_years   = np.empty(0, dtype=np.float64)
        self.__testing_data     = np.empty(0, dtype=np.float64)
        self.__testing_years    = np.empty(0, dtype=np.float64)
        self.__predictions      = np.empty(0, dtype=np.float64)

        self.__runtime_metrics     = {}
        self.__indicators_namelist = {}
        

    ## Gets e Sets :)
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

    def set_dataset_filepath(self, filepath: str):
        self.__dataset_filepath = filepath

    def get_dataset_filepath(self):
        return self.__dataset_filepath

    def get_training_data(self):
        return self.__training_data

    def get_training_years(self):
        return self.__training_years

    def get_testing_data(self):
        return self.__testing_data

    def get_testing_years(self):
        return self.__testing_years

    def get_predictions(self):
        return self.__predictions

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
    def split_train_test(self, indicator_code):
        self.__answers_test  = np.empty(0, dtype=np.int32)
        self.__answers_train = np.empty(0, dtype=np.int32)

        ## Início to cronômetro para separação
        start = time.perf_counter()

        ## Abrindo o CSV
        df = pd.read_csv(self.get_dataset_filepath())

        ## Pega a coluna do dataset que tem as informações desse indicador
        df_row_indicator = df.loc[df['Indicator Code'] == indicator_code]

        ## Armazena o nome do indicador em uma lista para evitar ter que abrir várias vezes o CSV
        self.__indicators_namelist[indicator_code] = df_row_indicator['Indicator Name']

        ## Indica qual o ultimo ano para treinamento de acordo com a porcentagem de treino
        last_year_training = int((self.get_tseries_end_year() - self.get_tseries_start_year())*self.get_percentage_train() + self.get_tseries_start_year())

        ## Obtendo o conjunto de dados para treinamento. Esse conjunto vai desde o inicio da serie temporal
        ## ate o ultimo ano de treinamento
        tmp = df_row_indicator.loc[:, f"{self.get_tseries_start_year()}":f"{last_year_training}"]

        # Armazenando os valores 

        ## !! Trocar os nomes, porque endog da funcao so recebe uma serie, não um par <value> e <predicted> !!
        self.__training_data   = np.concatenate(tmp.values, axis=0)
        self.__training_years  = np.asarray(tmp.keys())

        tmp = df_row_indicator.loc[:, f"{last_year_training}":f"{self.get_tseries_end_year()}"]

        self.__testing_data   = np.concatenate(tmp.values, axis=0)
        self.__testing_years  = np.asarray(tmp.keys())

        end   = time.perf_counter()
        self.set_runtime_metric("Separação dos dados de teste e treino", end - start)
        

    '''
    Treina o modelo com os dados em self.get_training_data(). Deve ser chamada
    somente depois de self.split_train_test()
    '''
    def train_model(self):
        start = time.perf_counter()

        self.__model = ExponentialSmoothing(
                                            endog=self.get_training_data(),
                                            initialization_method='estimated',
                                            trend='add'
                                            ).fit()

        end   = time.perf_counter()

        self.set_runtime_metric("Etapa de treinamento do modelo", end-start)


    '''
    Faz a previsão dos dados no conjunto de teste. Essa função deve ser chamada 
    somente depois de self.split_train_test() e de self.train_model()
    '''
    def predict_testing_data(self):
        start = time.perf_counter()
        self.__predictions = self.__model.forecast(steps=len(self.get_testing_data()))
        end   = time.perf_counter()

        self.set_runtime_metric("Predição dos próximos valores na série", end - start)


    def plot_indicators(self):
        start = time.perf_counter()

        for indicator_code in self.get_indicators_codelist():
            self.split_train_test(indicator_code)
            self.train_model()
            self.predict_testing_data()

            plt.title(f"{self.get_indicators_namelist()[indicator_code]}")
            plt.plot(self.get_training_years(), self.get_training_data(), c="green", label="Treino")
            plt.plot(self.get_testing_years(), self.get_testing_data(), c="blue", label="Teste")
            plt.plot(self.get_testing_years(), self.get_predictions(), c="red", label="Previsão")
            plt.legend()
            plt.show()

            plt.clf()

        end   = time.perf_counter()
        self.set_runtime_metric("Geração dos gráficos dos indicadores", end - start)