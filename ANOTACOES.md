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

AINDA NÃO SUBMETI NO KAGGLE