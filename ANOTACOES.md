 matriz é muito esparsa -> funk svd

# Primeira submissão
TEST_SIZE = 0.2 
RANDOM_STATE = 0  
N_FACTORS = 100 
N_EPOCHS = 20 
BIAS = False

RMSE: 1.5213
MAE:  1.1152

Sem usar contexto

Kaggle result - 0.31892

# Segunda submissão
TEST_SIZE = 0.3 
RANDOM_STATE = 0  
N_FACTORS = 80 
N_EPOCHS = 10 
BIAS = True

RMSE: 1.5034
MAE:  1.1017

Sem usar contexto

Kaggle result - 0.33023

# Terceira submissão
TEST_SIZE = 0.2
RANDOM_STATE = 0
N_FACTORS = 50
N_EPOCHS = 15
BIAS = True

RMSE: 1.4989
MAE:  1.0948

Sem usar contexto

Kaggle result - 0.32736

# Quarta submissão
TEST_SIZE = 0.3
RANDOM_STATE = 0  # ensure reproducibility
N_FACTORS = 80 
N_EPOCHS = 25 
BIAS = True

RMSE: 1.5378
MAE:  1.1299

Sem usar contexto

Kaggle result - 0.31470

# Quinta submissão
TEST_SIZE = 0.15
RANDOM_STATE = 0  # ensure reproducibility
N_FACTORS = 120 
N_EPOCHS = 15 
BIAS = True 

RMSE: 1.5087
MAE:  1.1027

Sem usar contexto

Kaggle result - 0.32474

# Sexta submissão
FACTORS = 150
EPOCHS = 25
LR = 0.005
REG = 0.02
USE_BIAS = True

rating = (
    0.5
    * fullPredictions[i]
    * 0.45
    * itemInfo["imdbVotes"]
    * 0.02
    * itemInfo["Metascore"]
    * 0.02
    * itemInfo["rtRating"]
    * 0.03
    * itemInfo["imdbRating"]
    + 6 * itemInfo["Awards"]
)

Kaggle result - 0.66453

# Setimo submissão
FACTORS = 150
EPOCHS = 25
LR = 0.005
REG = 0.02
USE_BIAS = True

rating = (
    0.9
    * fullPredictions[i]
    * 0.05
    * itemInfo["imdbVotes"]
    * 0.02
    * itemInfo["Metascore"]
    * 0.02
    * itemInfo["rtRating"]
    * 0.03
    * itemInfo["imdbRating"]
    + 15 * itemInfo["Awards"]
)

Kaggle result - 0.66461

### -> Comentário; Parametros de soma ponderada foram menos significativos

# Oitava submissão
FACTORS = 80
EPOCHS = 15
LR = 0.01 # Learning rate
REG = 0.02
USE_BIAS = True

rating = (
    0.9
    * fullPredictions[i]
    * 0.05
    * itemInfo["imdbVotes"]
    * 0.02
    * itemInfo["Metascore"]
    * 0.02
    * itemInfo["rtRating"]
    * 0.03
    * itemInfo["imdbRating"]
    + 15 * itemInfo["Awards"]
)

Kaggle result - 0.66421

# Nona submissão
FACTORS = 50
EPOCHS = 10
LR = 0.01 # Learning rate
REG = 0.09
USE_BIAS = False


rating = (
    1
    * fullPredictions[i]
    * 0.00
    * itemInfo["imdbVotes"]
    * 0.00
    * itemInfo["Metascore"]
    * 0.00
    * itemInfo["rtRating"]
    * 0.00
    * itemInfo["imdbRating"]
    + 0 * itemInfo["Awards"]
)

Kaggle result - 0.10481

# Nona submissão
FACTORS = 50
EPOCHS = 10
LR = 0.01 # Learning rate
REG = 0.09
USE_BIAS = False

rating = (
    1
    * fullPredictions[i]
    * 0.00
    * itemInfo["imdbVotes"]
    * 0.00
    * itemInfo["Metascore"]
    * 0.00
    * itemInfo["rtRating"]
    * 0.00
    * itemInfo["imdbRating"]
    + 0 * itemInfo["Awards"]
)

Kaggle result - 0.66421

# Decima submissao
FACTORS = 100
EPOCHS = 10
LR = 0.01 # Learning rate
REG = 0.09
USE_BIAS = True

WEIGHT_PREDICTION = 0.85
WEIGHT_IMDB_VOTES = 0.1
WEIGHT_METASCORE = 0.05
WEIGHT_RT_RATING = 0.02
WEIGHT_IMDB = 0.03
WEIGHT_BIAS_AWARDS = 6

Kaggle result - 0.66494

# Submissao onze

FACTORS = 80
EPOCHS = 5
LR = 0.001 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.85
WEIGHT_IMDB_VOTES = 0.1
WEIGHT_METASCORE = 0.05
WEIGHT_RT_RATING = 0.02
WEIGHT_IMDB = 0.03
WEIGHT_BIAS_AWARDS = 1

Kaggle result - 0.66594

# Decima segunda submissao

FACTORS = 150
EPOCHS = 45
LR = 0.001 # Learning rate
REG = 0.01
USE_BIAS = False

WEIGHT_PREDICTION = 0.85
WEIGHT_IMDB_VOTES = 0.1
WEIGHT_METASCORE = 0.05
WEIGHT_RT_RATING = 0.02
WEIGHT_IMDB = 0.03
WEIGHT_BIAS_AWARDS = 1

Kaggle result - 0.67614

# Decima terceira submissao
FACTORS = 20
EPOCHS = 2
LR = 0.09 # Learning rate
REG = 0.5
USE_BIAS = False

WEIGHT_PREDICTION = 0.9
WEIGHT_IMDB_VOTES = 0.05
WEIGHT_METASCORE = 0.05
WEIGHT_RT_RATING = 0.02
WEIGHT_IMDB = 0.03
WEIGHT_BIAS_AWARDS = 50

Kaggle result - 0.67655

# Decima quarta submissao
FACTORS = 500
EPOCHS = 1
LR = 0.00000001 # Learning rate
REG = 0.9
USE_BIAS = False

WEIGHT_PREDICTION = 0.1
WEIGHT_IMDB_VOTES = 0.85
WEIGHT_METASCORE = 0.05
WEIGHT_RT_RATING = 0.02
WEIGHT_IMDB = 0.03
WEIGHT_BIAS_AWARDS = 1

Kaggle result - 0.58017

# Decima quinta submissao
FACTORS = 500
EPOCHS = 1
LR = 0.05 # Learning rate
REG = 0.01
USE_BIAS = False

WEIGHT_PREDICTION = 1
WEIGHT_IMDB_VOTES = 0
WEIGHT_METASCORE = 0
WEIGHT_RT_RATING = 0
WEIGHT_IMDB = 0
WEIGHT_BIAS_AWARDS = 0

Kaggle result - 0.10481

# Decima sexta submissao
FACTORS = 80
EPOCHS = 15
LR = 0.05 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 1
WEIGHT_IMDB_VOTES = 0
WEIGHT_METASCORE = 0
WEIGHT_RT_RATING = 0
WEIGHT_IMDB = 0
WEIGHT_BIAS_AWARDS = 0

Kaggle result - 0.10

# Decima setima submissao

FACTORS = 150
EPOCHS = 25
LR = 0.05 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 1
WEIGHT_IMDB_VOTES = 0
WEIGHT_METASCORE = 0
WEIGHT_RT_RATING = 0
WEIGHT_IMDB = 0
WEIGHT_BIAS_AWARDS = 0

Kaggle result - 0.10

# Decima oitava submissao

FACTORS = 100
EPOCHS = 25
LR = 0.05 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.9
WEIGHT_IMDB_VOTES = 0
WEIGHT_METASCORE = 0
WEIGHT_RT_RATING = 0
WEIGHT_IMDB = 0.1
WEIGHT_BIAS_AWARDS = 1

Kaggle result - 0.15

# Decima nona submissao

FACTORS = 100
EPOCHS = 25
LR = 0.01 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.8
WEIGHT_IMDB_VOTES = 0.1
WEIGHT_METASCORE = 0
WEIGHT_RT_RATING = 0
WEIGHT_IMDB = 0.1
WEIGHT_BIAS_AWARDS = 2

Kaggle result - 0.15


# 20 submissao

FACTORS = 150
EPOCHS = 25
LR = 0.005 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.8
WEIGHT_IMDB_VOTES = 0.1
WEIGHT_METASCORE = 0
WEIGHT_RT_RATING = 0
WEIGHT_IMDB = 0.1
WEIGHT_BIAS_AWARDS = 2

Kaggle result - 0.15

# 21 submissao

FACTORS = 150
EPOCHS = 25
LR = 0.005 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.2
WEIGHT_IMDB_VOTES = 0.2
WEIGHT_METASCORE = 0.2
WEIGHT_RT_RATING = 0.2
WEIGHT_IMDB = 0.2
WEIGHT_BIAS_AWARDS = 2

Kaggle result - 0.66

# 22 submissao

FACTORS = 150
EPOCHS = 25
LR = 0.005 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.3
WEIGHT_IMDB_VOTES = 0
WEIGHT_METASCORE = 0.3
WEIGHT_RT_RATING = 0.4
WEIGHT_IMDB = 0.
WEIGHT_BIAS_AWARDS = 2

Kaggle result - 0.15

### -> Sem usar o IMDB fica pior de mais

# 23 submissao

FACTORS = 150
EPOCHS = 25
LR = 0.005 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.3
WEIGHT_IMDB_VOTES = 0.001
WEIGHT_METASCORE = 0.3
WEIGHT_RT_RATING = 0.398
WEIGHT_IMDB = 0.001
WEIGHT_BIAS_AWARDS = 2

Kaggle result - 0.66

### Só com um pouquinho de IMDB melhora de mais

# 24 submissao

FACTORS = 150
EPOCHS = 25
LR = 0.005 # Learning rate
REG = 0.01
USE_BIAS = True

WEIGHT_PREDICTION = 0.3
WEIGHT_IMDB_VOTES = 0.00001
WEIGHT_METASCORE = 0.3
WEIGHT_RT_RATING = 0.39998
WEIGHT_IMDB = 0.00001
WEIGHT_BIAS_AWARDS = 2

Kaggle result - 0.66

### Só com um pouquinho de IMDB melhora de mais
