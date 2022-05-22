import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing


class TimeSeriesPredictor:
    # Modelo para previsões
    __model: ExponentialSmoothing

    # Lista com codigo dos indicadores que serão avaliados
    __indicators_codelist: list

    ## Arrays com os conjuntos de treino e teste, bem como as respostas
    __values_train:    np.empty(0, dtype=np.float64)
    __values_test:     np.empty(0, dtype=np.float64)
    __predictions:   np.empty(0, dtype=np.float64)
    __answers_train: np.empty(0, dtype=np.int32)
    __answers_test:  np.empty(0, dtype=np.int32)
    
    # O path absoluto do CSV com os dados
    __dataset_filepath: str

    # Ano que a serie inicia.
    __tseries_start_year: int

    # Ano que a serie finaliza.
    __tseries_end_year: int

    # Porcentagem de dados para treino de cada indicador
    __percentage_train: int


    '''
    Instancia um preditor de series temporais
    '''
    def __init__(self):
        self.__model = ExponentialSmoothing

        self.__data_train    = np.empty(0, dtype=np.float64)
        self.__data_test     = np.empty(0, dtype=np.float64)
        self.__predictions   = np.empty(0, dtype=np.float64)
        self.__answers_train = np.empty(0, dtype=np.int32)
        self.__answers_test  = np.empty(0, dtype=np.int32)

        self.__runtime_metrics = {}
        

    # Gets e Sets :)
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

    def get_values_train(self):
        return self.__values_train

    def get_answers_train(self):
        return self.__answers_train

    def get_values_test(self):
        return self.__values_test

    def get_answers_test(self):
        return self.__answers_test

    def get_predictions(self):
        return self.__predictions

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

        df = pd.read_csv(self.get_dataset_filepath())

        df_row_indicator = df.loc[df['Indicator Code'] == indicator_code]
        indicator_name = df_row_indicator['Indicator Name']

        ## Indica qual o ultimo ano para treinamento de acordo com a porcentagem de treino
        last_year_training = int((self.get_tseries_end_year() - self.get_tseries_start_year())*self.get_percentage_train() + self.get_tseries_start_year())

        ## Obtendo o conjunto de dados para treinamento. Esse conjunto vai desde o inicio da serie temporal
        ## ate o ultimo ano de treinamento
        tmp = df_row_indicator.loc[:, f"{self.get_tseries_start_year()}":f"{last_year_training}"]

        # Armazenando os valores 

        ## !! Trocar os nomes, porque endog da funcao so recebe uma serie, não um par <value> e <predicted> !!
        self.__values_train  = np.asarray(tmp.keys())
        self.__answers_train = np.asarray(tmp.values)

        tmp = df_row_indicator.loc[:, f"{last_year_training+1}":f"{self.get_tseries_end_year()}"]
        self.__values_test  = np.asarray(tmp.keys())
        self.__answers_test = np.asarray(tmp.values)

        print(self.__values_train)
        print(self.__values_test)
        

    '''
    Treina o modelo com as imagens em self.get_images_train(). Deve ser chamada
    somente depois de self.split_train_test()
    '''
    def train_model(self):
        start = time.perf_counter()
        self.__model.fit(self.get_images_train(), self.get_answers_train())
        end   = time.perf_counter()

        self.set_runtime_metric("Etapa de treinamento do modelo", end-start)


    '''
    Faz a previsão das classes BIRADS das imagens de teste. Essa função deve ser chamada 
    somente depois de self.split_train_test() e de self.train_model()
    '''
    def predict_test_images(self):
        start = time.perf_counter()
        self.__predictions = self.__model.predict(self.get_images_test())
        end   = time.perf_counter()

        self.set_runtime_metric("Identificação das imagens de teste", end - start)


'''
import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


class TimeSeriesPredictor():

    # O preditor
    __predictor: ExponentialSmoothing

    # O CSV com os dados
    __dataset_file: str

    # Lista com codigo dos indicadores que serão avaliados
    __indicators_codelist: list

    # Ano que o treinamento inicia. Deve ser maior que 1960
    __start_date_train:int

    # Ano que o treinamento finaliza. Deve ser menor que 2014
    __end_date_train:int


    def __init__(self, dataset_file, indicators_codelist, start_date, end_date):
        self.__dataset_file = dataset_file
        self.__indicators_codelist = indicators_codelist

        self.__start_date_train = start_date
        self.__end_date_train   = end_date


    
    def plot_indicators(self):
        df = pd.read_csv(self.__dataset_file)

        for indicator_code in self.__indicators_codelist:

            # Get row of indicator
            df_indicator = df.loc[df['Indicator Code'] == indicator_code]
            indicator_name = df_indicator['Indicator Name']

            anos = []
            valores = []
            
            # Adicionar valores para aquele indicador entre start_date e end_date
            for i in range(self.__start_date_train, self.__end_date_train):
                anos.append(i)
                valores.append(df_indicator[f"{i}"].values[0])

            
            plt.title(f"{indicator_name.values[0]}")
            plt.plot(anos, valores)
            plt.show()

            plt.clf()


    def predict_indicators(self):
        df = pd.read_csv(self.__dataset_file)

        for indicator_code in self.__indicators_codelist:

            # Get row of indicator
            df_indicator = df.loc[df['Indicator Code'] == indicator_code]
            indicator_name = df_indicator['Indicator Name']

            anos = []
            valores = []
            
            # Adicionar valores para aquele indicador entre start_date e end_date
            for i in range(self.__start_date_train, self.__end_date_train):
                anos.append(i)
                valores.append(df_indicator[f"{i}"].values)
        
            print(f"VALORES: {valores}\n\n")

            self.__predictor = ExponentialSmoothing(endog=valores, trend='mul', initialization_method="heuristic").fit().forecast(steps=10)
            
            print(self.__predictor)

            outros = []
            outros_2 = []
            for i in range(self.__end_date_train, 2014):
                outros.append(i)
                outros_2.append(df_indicator[f"{i}"].values)

            plt.plot(anos, valores, c="b")
            plt.plot(outros,outros_2, c="green")
            plt.plot(outros,self.__predictor, c="r")
            plt.show()

            plt.clf()

'''