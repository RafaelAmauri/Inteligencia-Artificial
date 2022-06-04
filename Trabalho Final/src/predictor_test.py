import predictor

pred = predictor.TimeSeriesPredictor()

pred.set_dataset_filepath("/home/rafael/PUC/PUC-Minas/Inteligencia-Artificial/Trabalho Final/assets/BR.csv")
pred.set_indicators_codelist(['SP.POP.TOTL', 'SP.RUR.TOTL.ZS', 'SP.URB.TOTL.IN.ZS'])

pred.set_percentage_train(83)
pred.set_percentage_validation(2)
pred.set_tseries_start_year(1960)
pred.set_tseries_end_year(2020)

'''
Testes de validação para ajustar os parâmetros do modelo!!

for indicator_code in pred.get_indicators_codelist():
    pred.split_train_test_val(indicator_code)

    print(f"Anos de treino = {pred.get_training_years()}")
    print(f"Anos de validação = {pred.get_validation_years()}")
    print(f"Anos de teste = {pred.get_testing_years()}")
    
    pred.train_model()
    print(f"Métricas de avaliação para a previsão de '{indicator_code}'")
    pred.predict_validation_data()
'''

pred.plot_indicators()
