---
layout: post
title: "Detector de idioma usando Python"
subtitle: ""
date: 2021-05-09 00:00:00 -0400
background: '/img/posts/2021-05-09-detector-idioma-machine-learning/2021-05-09-detector-idioma-machine-learning-1.jpeg'
categories: posts/machine-learning
tags: [python, pandas, numpy, scikit-learn]
---

La detección de idioma es una tarea del procesamiento de lenguaje natural en donde necesitamos identificar el idioma de un texto o documento. Usar aprendizaje automático (**machine learning**) para la identificación de un idioma era una labor difícil hace algunos años debido a que no había una gran cantidad de datos sobre idiomas, pero la disponibilidad de datos en la actualidad ha facilitado las cosas. Por lo tanto, ahora tenemos muchos modelos poderosos disponibles para la identificación de lenguajes. Entonces, si quieres aprender como entrenar un modelo para la detección de idiomas, este artículo es para ti.
![traductor de Google](/img/posts/2021-05-09-detector-idioma-machine-learning/2021-05-09-detector-idioma-machine-learning-1.png)

# Detección de lenguaje
Como humanos, podemos identificar fácilmente los idiomas que conocemos. Sin embargo, identificar el resto de idiomas existentes nos es casi imposible. Es aquí cuando la detección de mensajes puede ser usada. El Traductor de Google es uno de los más populares traductores en el mundo. También incluye un modelo de aprendizaje automático para detectar el idioma que tu puedes usar si no sabes que idioma es el que deseas traducir.
La parte más importante de entrenar un modelo para la detección de idiomas son los datos. Cuantos más datos tengamos sobre cada idioma, más preciso será el rendimiento de nuestro modelo en tiempo real. El conjunto de datos que usaremos contiene datos en 22 idiomas diferentes y 1000 frases en cada uno de los idiomas, por lo que será un dataset apropiado para entrenar un detector de idiomas con aprendizaje automático usando Python.

# Detección de lenguaje usando Python
Empezamos la detección de lenguaje con aprendizaje automático importando las librerías de Python necesarias y el conjunto de datos:
```python
in [1]:     import pandas as pd
            import numpy as np
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import MultinomialNB

            data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
            print(data.head())
```
```python
out [1]:       Text                                               language 
            0  klement gottwaldi surnukeha palsameeriti ning ...  Estonian 
            1  sebes joseph pereira thomas  på eng the jesuit...   Swedish 
            2  ถนนเจริญกรุง อักษรโรมัน thanon charoen krung เ...          Thai 
            3  விசாகப்பட்டினம் தமிழ்ச்சங்கத்தை இந்துப் பத்திர...             Tamil 
            4  de spons behoort tot het geslacht haliclona en...     Dutch 
```  

Veamos si este conjunto de datos contiene algún valor `null`:
```python
in [2]:     data.isnull().sum()
```
```python
out [2]:    Text        0
            language    0
            dtype: int64
```  
Ahora, veamos todos los idiomas presentes en nuestro conjunto de datos:
```python
in [3]:     data["language"].value_counts()
```
```python
out [3]:    Japanese      1000
            Arabic        1000
            Thai          1000
            Turkish       1000
            Latin         1000
            Indonesian    1000
            Portugese     1000
            English       1000
            Swedish       1000
            Estonian      1000
            Tamil         1000
            Romanian      1000
            Korean        1000
            Russian       1000
            Persian       1000
            Chinese       1000
            Dutch         1000
            Urdu          1000
            Hindi         1000
            Spanish       1000
            Pushto        1000
            French        1000
            Name: language, dtype: int64
```

El conjunto de datos contiene 22 idiomas con 1000 frases cada uno. Este es un conjunto de datos bastante balanceado y sin valores faltantes, por lo que podemos decir que este conjunto de datos está completamente listo para ser usado para entrenar un modelo de aprendizaje automático.

# Modelo para detección de idioma
A continuación, vamos a dividir nuestra data en un set de entranamiento y un set de prueba:
```python
in [4]:     x = np.array(data["Text"])
            y = np.array(data["language"])
            cv = CountVectorizer()
            X = cv.fit_transform(x)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
Como este es un problema de clasificación multiclase, usaremos el algoritmo Naive Bayes Multinomial para entrenar el modelo de detección de idiomas ya que este algoritmo tiene un desempeño muy bueno en problemas basados en clasificación multiclase:
```python
in [5]:     modelo = MultinomialNB()
            modelo.fit(X_train, y_train)
            modelo.score(X_test, y_test)
```
```python
out [5]:    0.953168044077135
```
Finalmente, usaremos este modelo para detectar el idioma de una frase ingresada por el usuario:
```python
in [6]:     usuario = input("Escribe una frase: ")
            data = cv.transform([usuario]).toarray()
            pred = modelo.predict(data)
            print(pred)
```
```python
out [6]:    Escribe una frase: prueba detector de idiomas
            ['Spanish']
```
Como podemos ver, el modelo trabaja bien. Una cosa a resaltar es que **este modelo solamente puede detectar los idiomas que están presentes en el conjunto de datos.**    

Puedes descargar el cuaderno de Jupyter del projecto [aquí](https://drive.google.com/file/d/1204ZuFnzfVu4jo1wvxEADgnOBNG1RFwW/view?usp=sharing)