---
layout: post
title: "Detector de Spam usando Python"
subtitle: ""
date: 2021-09-10 00:00:00 -0400
background: '/img/posts/2021-05-06-detector-spam/2021-05-06-detector-spam-machine-learning-4.png'
categories: posts/machine-learning
tags: [python, pandas, numpy, scikit-learn]
---

# Detección de spam
La detección de spam es una de las principales aplicaciones que las grandes compañías tratan de mejorar para sus clientes. Gmail de Google es un buen ejemplo de una aplicación en donde el detector de spam trabaja bien para proteger a sus usuarios mediante alertas de spam.  

Cada vez que envías detalles personales en cualquier plataforma, como tu correo electrónico o tu número telefónico, estás permitiendo a esas plataformas comercializar sus productos anunciándolos mediante el envío de correos electrónicos o enviando mensajes directamente a tu número telefónico. Esto da como resultado una gran cantidad de alertas y notificaciones de spam en tu bandeja de entrada. Aquí es en donde tiene lugar la tarea de detección de spam.  

La detección de spam consiste en detectar los mensajes o correos electrónicos no deseados mediante la comprensión del contenido del texto, de modo que sólo pueda recibir notificaciones sobre los mensajes o correos electrónicos que son relevantes para el usuario. Si se encuentran mensajes o correos no deseados, se transfieren automáticamente a una carpeta de spam y nunca se le notifican dichas alertas al usuario. Esto ayuda a mejorar la experiencia del usuario, ya que muchas alertas de spam pueden resultar molestas.
  
---

## Detección de spam usando Python
A continuación, veremos como entrenar un modelo de aprendizaje automático (**machine learning**) para detectar spam usando Python. Comenzamos esta tarea importando las librerías de Python y el conjunto de datos que necesitaremos.
```python
in [1]:     import pandas as pd
            import numpy as np
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.model_selection import train_test_split
            from sklearn.naive_bayes import MultinomialNB

            data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/SMS-Spam-Detection/master/spam.csv", encoding='latin-1')
            data.head()
```
```
out [1]:        class       message                                             Unnamed:2   Unnamed:3   Unnamed:4  
            0 	ham 	    Go until jurong point, crazy.. Available only ... 	NaN 	    NaN 	    NaN
            1 	ham 	    Ok lar... Joking wif u oni...                       NaN 	    NaN 	    NaN
            2 	spam 	    Free entry in 2 a wkly comp to win FA Cup fina... 	NaN 	    NaN 	    NaN
            3 	ham 	    U dun say so early hor... U c already then say... 	NaN 	    NaN 	    NaN
            4 	ham 	    I don't think he goes to usf, he lives aro...       NaN 	    NaN 	    NaN  
```
  

En esta conjunto de datos, las únicas columnas que necesitamos para entrenar nuestro modelo para la detección de spam son `class` y `message`, así que seleccionamos estas dos columnas como el nuevo conjunto de datos:
```python
in [2]:     data = data[["class", "message"]]
```
```
out [2]:        class 	    message
            0 	ham 	    Go until jurong point, crazy.. Available only ...
            1 	ham 	    Ok lar... Joking wif u oni...
            2 	spam 	    Free entry in 2 a wkly comp to win FA Cup fina...
            3 	ham 	    U dun say so early hor... U c already then say...
            4 	ham 	    Nah I don't think he goes to usf, he lives aro... 
```
Ahora, debemos separar este dataset en un set de entrenamiento y un set de prueba para poder entrenar nuestro modelo para detectar spam:
```python
in[3]:      x = np.array(data["message"])
            y = np.array(data["class"])
            cv = CountVectorizer()
            X = cv.fit_transform(x) 
            X_train, X_test, y_train, x_test = train_test_split(X, y, test_size=0.33, random_state=42)

            clf = MultinomialNB()
            clf.fit(X_train, y_train)
```
```
out [3]:    MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
```
Finalmente, probemos este modelo tomando la entrada de un usuario para detectar si es spam o no:
```python
in [4]:     mensaje = input("Escribe un mensaje: ")
            data = cv.transform([mensaje]).toarray()
            print(clf.predict(data))
```
```
out [4]:    Escribe un mensaje: Time is running out. Save 50%
            ['spam']
```
---
> **NOTA**: Debido a que el conjunto de datos para este projecto está en idioma inglés, el mensaje que proporcionará el usuario final deberá estar escrito en este idioma.  

---

Así es como se puede usar el aprendizaje automático para entrenar un modelo para detectar si un correo electrónico o un mensaje es spam.  Un detector de spam detecta mensajes o correos no deseados al comprender el contenido del texto, de este modo, usted solo recibirá notificaciones sobre mensajes o correos electrónicos que sean muy importantes.

Puedes descargar el cuaderno de este projecto [aquí](https://drive.google.com/file/d/1Lgt3dR_uhDPAwlh_tLgdlVP_YaRL8yq_/view?usp=sharing)