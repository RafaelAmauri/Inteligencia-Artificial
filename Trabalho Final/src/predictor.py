import pandas as pd

import matplotlib.pyplot as plt


class Predictor():

    # O CSV com os dados
    __dataset_file: str

    # Lista com codigo dos indicadores que serão avaliados
    __indicators_codelist: list


    def __init__(self, dataset_file, indicators_codelist):
        self.__dataset_file = dataset_file
        self.__indicators_codelist = indicators_codelist

    
    def plot_indicators(self):
        df = pd.read_csv(self.__dataset_file)

        for indicator_code in self.__indicators_codelist:

            # Get row of indicator
            df_indicator = df.loc[df['Indicator Code'] == indicator_code]
            indicator_name = df_indicator['Indicator Name']

            anos = []
            valores = []
            for i in range(1960, 2014):
                anos.append(i)
                valores.append(df_indicator[f"{i}"].values[0])
                
            print(valores)
            
            plt.title(f"{indicator_name.values[0]}")
            plt.plot(anos, valores)
            plt.show()

            plt.clf()