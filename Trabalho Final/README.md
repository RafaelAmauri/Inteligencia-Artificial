# Trabalho Final de IA

Desenvolvedores:

[Lucas Santiago](https://github.com/LucasSnatiago "Lucas Santiago")

Rafael Amauri(Eu :D)

[Thiago Henriques](https://github.com/ThiagoHN "Thiago Henriques")

## DISCLAIMER
O programa foi feito para ser utilizado no Linux. Embora alguns testes mostraram que ele roda sem problemas no Windows,
o grupo não teve interesse nem incentivo para dar suporte à versão Windows. É fortemente recomendado utilizar no Linux!

## Como utilizar ?

O dataset já vem incluso no repositório, mas caso tenha algum problema e ele se corromper ou tiver outro problema, o dataset pode ser
baixado diretamente do website do Banco Mundial por meio do link https://datacatalog.worldbank.org/search/dataset/0037652/Health%20Nutrition%20and%20Population%20Statistics

Para instalar as dependências do programa basta rodar:

```
pip3 install -r requirements.txt
```

Para executar o programa, rode:

```
cd src/
python3 main.py
```

## Dica de utilização

Tenha em mente que a main.py só executa algumas das várias funções que foram implementadas para o classificador. Se tiver interesse, veja abaixo algumas das principais funções usadas.

### Essenciais:

    @dataset_filepath.setter
    predictor.dataset_filepath       -> Recebe o filepath do dataset.

    @indicators_codelist.setter
    predictor.indicators_codelist    -> Recebe uma lista com os indicadores a serem utilizados.

    @percentage_train.setter
    predictor.percentage_train       -> Define a porcentagem utilizada para treino.

    @percentage_validation.setter
    predictor.percentage_validation  -> Define a porcentagem de dados utilizada para validação.

    @tseries_start_year.setter
    predictor.tseries_start_year     -> Define o ano que a serie temporal inicia. Para o dataset incluso aqui é 1960, mas 
                                        pode ser qualquer valor desde que não seja antes de 1960 e depois de 2020.

    @tseries_end_year.setter
    predictor.tseries_end_year       -> Define o ano que a serie temporal acaba. O dataset incluso vai até 2020, então 
                                        tem que ser um valor entre 1960 e 2020.

    predictor.plot_indicators()      -> Faz as operações de separar os dados de acordo com as porcentagens, treina
                                        o modelo para cada indicador, faz as previsões, printa os valores para as
                                        métricas de avaliação para a previsão e retorna uma imagem do plot dos valores
                                        de treino, validação e teste para cada indicador na sua tela. É altamente recomendado
                                        que se deixe essa função aqui orquestrar tudo e chamar as outras, basta só definir os
                                        valores para todos atributos acima e chamar predictor.plot_indicators() que ela já faz tudo!
### Debug:
    @property
    predictor.training_years    -> Retorna os anos que serão usados para treino.

    @property
    predictor.training_data     -> Retorna os valores para um indicador nos anos de treino.

    @property
    predictor.testing_years     -> Retorna os anos que serão usados para teste.

    @property
    predictor.testing_data      -> Retorna os valores para um indicador nos anos de teste.

    @property
    predictor.validation_years  -> Retorna os anos que serão usados para validação.

    @property
    predictor.validation_data   -> Retorna os valores para um indicador nos anos de validação.

    @property
    predictor.train_model() -> Treina o modelo

    predictor.split_train_test_val(indicator_code) -> Separa os dados em treino, teste e validação de acordo com as porcentagens definidas antes


EXISTEM MUITO MAIS FUNCIONALIDADES, PRINCIPALMENTE MUITOS GETTERS E SETTERS! OLHAR predictor.py!!
