
def preparardatos():
    df_clean = df.copy()
    df_clean['origin'] = df_clean['origin'].map({1: 'USA', 2: 'Europa', 3: 'Japón'})
    df_clean = pd.get_dummies(df_clean, columns=['origin'], prefix='origin', drop_first=True, dtype=int)
    df_clean.head()
    print("Registros y columnas:", df_clean.shape)
    df_clean.info()
def dividir():
