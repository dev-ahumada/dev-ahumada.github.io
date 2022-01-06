---
layout: post
title: "Detectar fake news con Machine Learning usando Python"
subtitle: ""
date: 2021-11-04 00:00:00 -0400
background: "/img/posts/2021-05-23-detectar-fake-news-machine-learning/2021-05-23-detectar-fake-news-machine-learning-bckgrd.jpeg"
categories: posts/machine-learning
tags: [python, numpy, pandas, scikit-learn]
---
Las fake news son uno de lo mayores problemas de las redes sociales e incluso de los sitios de noticias. Por lo tanto, usar Machine Learning para la detección de fake news es una tarea muy compleja.

# Detección de Fake News
Las fake news generan desinformación. La mayoría de las veces, difundir noticias falsas sobre las creencias políticas y religiosas de una comunidad puede provocar disturbios y violencia. Entonces, para detectar fake news, debemos encontrar relaciones entre los titulares de fake news para poder entrenar un modelo de Machine Learning que pueda decirnos si una información en particular es falsa o real simplemente observando el titular de la noticia. 

# Detección de Fake News usando Python
El [conjunto de datos](https://drive.google.com/file/d/1JTBaMn56PH8bKebqMXyNKCSCiRSxgyZ-/view?usp=sharing) que usaremos para la tarea de detectar fake news contiene datos sobre el título de la noticia, el contenido de la noticia y una columna llamada `label` que nos indica si la noticia es falsa o real. Por lo tanto, podemos usar este conjunto de datos para encontrar relaciones entre los titulares de noticias falsas y reales para comprender qué tipo de titulares se encuentran en la mayoría de las fake news. Para empezar, importamos las librerías de Python y el conjunto de datos que necesitamos:
```python
in [1]:     import pandas as pd
            import numpy as np
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import MultinomialNB

            data = pd.read_csv("news.csv")
            print(data.head())
```
```
out [1]:       Unnamed: 0                                              title                                                text   label
            0  8476                             You Can Smell Hillary’s Fear   Daniel Greenfield, a Shillman Journalism Fello...    FAKE
            1  10294       Watch The Exact Moment Paul Ryan Committed Pol...   Google Pinterest Digg Linkedin Reddit Stumbleu...    FAKE
            2  3608              Kerry to go to Paris in gesture of sympathy   U.S. Secretary of State John F. Kerry said Mon...    FAKE
            3  10142       Bernie supporters on Twitter erupt in anger ag...   — Kaydee King (@KaydeeKing) November 9, 2016 T...    FAKE
            4  875          The Battle of New York: Why This Primary Matters   It's primary day in New York and front-runners...    REAL
```
Este conjunto de datos es muy grande pero, afortunadamente, no tiene valores faltantes. Por lo tanto, usaremos no necesitamos hacer ningun pre-procesamiento de los datos. Tomaremos la columna  `title` como nuestra variable de entrada para entrenar un modelo de Machine Learning y la columna `label` será nuestra variable a predecir:
```python
in [2]:     X = np.array(data["title"])
            y = np.array(data["label"])

            cv = CountVectorizer()
            X = cv.fit_transform(X)
```
Ahora, vamos a separar el conjunto de datos en conjuntos de entrenamiento y de prueba. Después, usaremos el algoritmo de Naive Bayes Multinomial para entrenar nuestro modelo de detección de fake news:
```python
in [3]:     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = MultinomialNB()
            modelo.fit(X_train, y_train)
            print(modelo.score(X_test, y_test))
```
```
out [3]:    0.8074191002367798
```
Ahora probemos este modelo. Para probarlo, primero escribimos el título de cualquier noticia que encontremos en Google News para ver lo que predice:
```python
in [4]:     titulo = "Biden administration grants temporary protected status to Haitians living in U.S."
            data = cv.transform([titulo]).toarray()
            print(modelo.predict(data))
```
```
out [4]:    ['REAL']  
```  
Finalmente, vamos a escribir una fake news al azar para ver si el modelo la predice como falsa:
```python
in [5]:     titulo = "Chlorine products can cure coronavirus"
            data = cv.transform([titulo]).toarray()
            print(modelo.predict(data))
```
```  
out [5]:    ['FAKE']
```
Así es como podemos entrenar un modelo de Machine Learning para la tarea de detección de fake news utilizando Python.  

Puedes descargar el cuaderno del projecto [aquí](https://drive.google.com/file/d/1iRo1j9Ib-9_01ld1wrnGcIftFZ9IwaBy/view?usp=sharing)