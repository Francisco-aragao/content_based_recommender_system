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

# 25 submissao

Final weights: {'prediction': 0.2793055550484167, 'imdb_votes': 0.006201343845505356, 'metascore': 0.22029817550012423, 'rt_rating': 0.2773619814751889, 'imdb': 0.009635311564216958, 'bias_awards': 1.991592899542149}

FACTORS = 1
EPOCHS = 95
LR = 0.005  # Learning rate for SVD
REG = 0.01
USE_BIAS = True
LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 100

Kaggle 0.10

# 26 submissao

Final weights: {'prediction': 0.221468924660898, 'imdb_votes': 0.46525713686143955, 'metascore': 0.21142391692127435, 'rt_rating': 0.29147650196533653, 'imdb': 0.17362110624397378, 'bias_awards': 0.556359991335989}

FACTORS = 1
EPOCHS = 95
LR = 0.005  # Learning rate for SVD
REG = 0.01
USE_BIAS = True
LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 100

Kaggle 0.10

# 27 submissao

Final weights: {'prediction': 0.5599591367037845, 'imdb_votes': 0.3048172195087188, 'metascore': 0.056150155204912205, 'rt_rating': 0.13198081025955166, 'imdb': 0.12798465450269414, 'bias_awards': 0.8194830135364868}

FACTORS = 50
EPOCHS = 50
LR = 0.005  # Learning rate for SVD
REG = 0.01
USE_BIAS = True
LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 100

kaggle 0.10

# 28 submissao

Final weights: {'prediction': 0.07391042401151084, 'imdb_votes': 0.013761603357143094, 'metascore': 0.11284649887756952, 'rt_rating': 0.27463590355618334, 'imdb': 0.4693288194743187, 'bias_awards': 0.9782429775633855}

FACTORS = 100
EPOCHS = 50
LR = 0.005  # Learning rate for SVD
REG = 0.01
USE_BIAS = True
LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 100

kaggle 0.10

# 29 submissao

FACTORS = 150
EPOCHS = 20
LR = 0.01  # Learning rate for SVD
USE_BIAS = True
BATCH_SIZE = 30

LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 100

Final weights: {'prediction': 0.12999025406517856, 'imdb_votes': -0.03924541876976724, 'metascore': -0.006649615533715257, 'rt_rating': -0.020631423289098755, 'imdb': 0.9910646691504967, 'bias_awards': 0.009865093967845035}

kaggle 0.10


# 30 submissao

FACTORS = 150
EPOCHS = 20
LR = 0.01  # Learning rate for SVD
USE_BIAS = True
BATCH_SIZE = 30

LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 100

Final weights: {'prediction': 0.15920407260401465, 'imdb_votes': 0.006128541013855994, 'metascore': 0.13483262813945196, 'rt_rating': -0.05412821844780214, 'imdb': 0.7251987659448155, 'bias_awards': 0.8695016299686347}

kaggle 0.10

# 32 submissao
Final weights: {'prediction': 0.24123956556466264, 'imdb_votes': 0.36707063709804616, 'metascore': 0.08013705708822978, 'rt_rating': 0.3193497867001439, 'imdb': 0.28736783318312126, 'bias_awards': 0.38114184590437733}
FACTORS = 150
EPOCHS = 25
LR = 0.05  # Learning rate for SVD
USE_BIAS = True
REG = 0.05

LEARNING_RATE_WEIGHTS = 0.002
GRADIENT_EPOCHS = 70

kaggle 0.10

# 34 submissao

Final weights: {'prediction': 0.05292737099789775, 'imdb_votes': 0.08943101829605517, 'metascore': 0.16857429562482432, 'rt_rating': 0.22193172725216848, 'imdb': 0.5280767245518214, 'bias_awards': 0.630901980534699}

FACTORS = 80
EPOCHS = 25
LR = 0.005
REG = 0.02
USE_BIAS = True

LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 40

kaggle 0.49

# 35

Final weights: {'prediction': 0.2686978656383046, 'imdb_votes': 0.3071337179154058, 'metascore': 0.09455936574646003, 'rt_rating': 0.08375788323237696, 'imdb': 0.526435016386344, 'bias_awards': 0.2661165131610226}

FACTORS = 100
EPOCHS = 25
LR = 0.005
REG = 0.02
USE_BIAS = True

LEARNING_RATE_WEIGHTS = 0.005
GRADIENT_EPOCHS = 40

kaggle 0.55

# 36

Final weights: {'prediction': 0.2548789284649461, 'imdb_votes': 0.3705062916993015, 'metascore': 0.060486385393324474, 'rt_rating': 0.06093673937437485, 'imdb': 0.06821818093813317, 'bias_awards': 3.984780605225248}

FACTORS = 150
EPOCHS = 25
LR = 0.005 
REG = 0.02
USE_BIAS = True

LEARNING_RATE_WEIGHTS = 0.005
GRADIENT_EPOCHS = 10

kaggle 0.60

# 37 - Final

Final weights: {'prediction': 0.26257357790769936, 'imdb_votes': 0.38876356778703797, 'metascore': 0.0664125220031843, 'rt_rating': 0.06355173165087688, 'imdb': 0.06678509482692568, 'bias_awards': 3.99321965771369}

FACTORS = 150
EPOCHS = 25
LR = 0.005 
REG = 0.02
USE_BIAS = True

LEARNING_RATE_WEIGHTS = 0.001
GRADIENT_EPOCHS = 10

kaggle 0.60