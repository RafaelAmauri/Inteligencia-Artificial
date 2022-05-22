import predictor

pred = predictor.TimeSeriesPredictor()

pred.set_dataset_filepath("/home/rafael/PUC/PUC-Minas/Inteligencia-Artificial/Trabalho Final/assets/BR.csv")
pred.set_indicators_codelist(['SP.POP.TOTL', 'SP.RUR.TOTL'])
pred.set_percentage_train(90)
pred.set_tseries_start_year(1960)
pred.set_tseries_end_year(2014)

pred.plot_indicators()