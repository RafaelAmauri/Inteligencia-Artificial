import predictor

pred = predictor.TimeSeriesPredictor()

pred.dataset_filepath = "./assets/BR.csv"
pred.indicators_codelist = ['SP.POP.TOTL', 'SP.RUR.TOTL.ZS', 'SP.URB.TOTL.IN.ZS']

pred.percentage_train = 83
pred.percentage_validation = 2
pred.tseries_start_year = 1960
pred.tseries_end_year = 2020



pred.plot_indicators()

'''
## Exemplo que printa os valores para as métricas de avaliação no conjunto de validação.
## Esse foi o código usado para ajustar os parâmetros do modelo.

for indicator_code in pred.indicators_codelist:
    pred.split_train_test_val(indicator_code)

    print(f"Anos de treino = {pred.training_years}", end="\n\n")
    print(f"Anos de validação = {pred.validation_years}", end="\n\n")
    print(f"Anos de teste = {pred.testing_years}", end="\n\n")
    
    pred.train_model()
    print(f"Métricas de avaliação para a previsão de '{indicator_code}'")
    pred.predict_validation_data()
'''
