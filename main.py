# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth

# Antes das APIs
colunas = [
 'saw_sizecharts',
 'saw_account_upgrade',
 'detail_wishlist_add',
 'saw_delivery',
 'account_page_click',
 'checked_returns_detail',
 'device_mobile',
 'promo_banner_click',
 'device_tablet',
 'loc_uk',
 'sort_by',
 'returning_user',
 'image_picker',
 'device_computer',
 'closed_minibasket_click',
 'saw_homepage',
 'list_size_dropdown',
 'basket_add_list',
 'basket_add_detail',
 'basket_icon_click',
 'sign_in'
]

# Criação de uma app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'admin'
app.config['BASIC_AUTH_PASSWORD'] = '123'

#Habilitando autenticação na app
basic_auth = BasicAuth(app)

def create_model():
    print('Iniciando carga dos dados')
    df_train = pd.read_csv('data/raw/dataset_train.csv')
    df_train.drop(columns='UserID', inplace=True)
    print('df_train.shape:', df_train.shape)

    print('Preparação dos dados')
    #Reduzindo o dataset de treino:
    df_train = df_train[colunas].copy()

    print('Modelagem...')

    #Definição das variáveis X e y:
    X = df_train.loc[:,df_train.columns[0]:df_train.columns[len(df_train.columns)-2]].values
    y = df_train['ordered'].values

    #Definindo estratégia de Over Sample
    oversample = RandomOverSampler(sampling_strategy='minority')
    X_samp, y_samp = oversample.fit_resample(X, y)
    
    X_train, X_valid, y_train, y_valid = \
    train_test_split(X_samp,
                     y_samp,
                     test_size=0.25, random_state=42)
    
    model = Sequential()
    num_classes = df_train['ordered'].nunique()

    model.add(Dense(42, activation = 'relu', input_dim=21))
    model.add(Dropout(0.2))
    model.add(Dense(42, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units = num_classes-1, activation = 'sigmoid'))
    model.summary()

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    EPOCHS = 10
    BATCH_SIZE = 40

    model.fit(X_train, y_train,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        verbose = 1,
        validation_data=(X_valid, y_valid))

    return model.predict(X_train)

def load_model(file_name = 'over_model_mlp.pkl'):
    return pickle.load(open(file_name, "rb"))

# model = create_model()
model = load_model('models/over_model_mlp.pkl')

# Nova rota - receber um CPF como parâmetro
@app.route('/score/', methods=['POST'])
@basic_auth.required
def score():
    # Pegar o JSON da requisição
    dados = request.get_json()
    payload = [dados[col] for col in colunas]
    score = np.round(np.float64(model.predict(np.array( [payload,] ))[0]), 3)
    status = 'ASSIDUO'
    if score < 0.7:
        status = 'NAO ASSIDUO'

    return jsonify(score=score, status=status)

# Rota padrão
@app.route('/')
def home():
    return 'API de análise de retenção de clientes'

# Rota padrão
@app.route('/auth/<cpf>', methods=['POST'])
@basic_auth.required
def test_auth(cpf):
    return { 'cpf': cpf }

# Subir a API
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')