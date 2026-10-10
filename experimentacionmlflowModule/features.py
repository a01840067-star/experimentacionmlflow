import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from experimentacionmlflowModule.config import ORIGIN_MAP, ID_COL, TARGET

def preparar_datos(df):
    df_clean = df.copy()
    df_clean['origin'] = df_clean['origin'].map(ORIGIN_MAP)
    df_clean = pd.get_dummies(
        df_clean, columns=['origin'], prefix='origin', drop_first=True, dtype=int
        )
    X = df_clean.drop(columns=[ID_COL, TARGET ])
    y = df_clean['mpg']
    return X, y

def dividir(X, y, test_size, seed):
    mpg_bins = pd.qcut(y, q=5, labels=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=mpg_bins
    )   
    return  X_train, X_test, y_train, y_test
    
def imputar_horsepower(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    con_hp = X_train['horsepower'].notna()
    imputador_hp = LinearRegression()
    imputador_hp.fit(X_train.loc[con_hp, ['displacement']], X_train.loc[con_hp, 'horsepower'])

    for X_set in (X_train, X_test):
        faltan = X_set['horsepower'].isna()
        if faltan.any():
            X_set.loc[faltan, 'horsepower'] = imputador_hp.predict(X_set.loc[faltan, ['displacement']]).round()
    
    return X_train, X_test
   