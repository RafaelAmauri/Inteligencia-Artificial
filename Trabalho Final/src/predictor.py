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