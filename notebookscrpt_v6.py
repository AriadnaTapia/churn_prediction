#!/usr/bin/env python
# coding: utf-8

# # Modelo de Predicción de Tasa de Cancelación de Cliente para Interconnect

# # Introducción
# 
# Interconnect, un operador de telecomunicaciones, busca pronosticar la tasa de cancelación de clientes con el fin de implementar estrategias de retención basadas en promociones y planes especiales.
# 
# Se requiere desarrollar un modelo predictivo que, utilizando el comportamiento de clientes anteriores, permita identificar a aquellos que podrían decidir terminar su contrato con la compañía. Este es un caso de clasificación binaria.
# 
# El objetivo principal es predecir si un cliente cancelará su contrato, seleccionando el modelo con mejor rendimiento. La evaluación se realizará principalmente mediante la métrica AUC-ROC, complementada por la exactitud, para determinar qué tan preciso es el modelo.
# 

# ### Inicialización
# 
# Para iniciar el desarrollo del análisis, comenzaremos importando las librerías que utilizaremos en la preparación de datos y la construcción del modelo.
# 
# Las principales librerías que se utilizarán son:
# 
# - **pandas**: permite la lectura y manipulación de datos en forma de tablas (DataFrames).
# - **NumPy**: facilita operaciones numéricas y manejo de arreglos multidimensionales.
# - **Seaborn**: se utilizará para visualizar y explorar patrones en los datos.
# - **scikit-learn**: proporciona herramientas para crear y evaluar modelos de machine learning.

# In[119]:


import warnings
# Suppress all FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# ## Carga de archivos
# 
# Procedemos con la carga de las bases de datos que nos proporcionan información de diferentes fuentes. A continuación se describen los archivos disponibles:
# 
# - `contract.csv`: información del contrato.
# - `personal.csv`: datos personales del cliente.
# - `internet.csv`: información sobre los servicios de Internet.
# - `phone.csv`: información sobre los servicios telefónicos.
# 
# Para la carga de los datos, se utilizará la función `read_csv` de la librería pandas, indicando como argumento la ubicación de cada archivo.
# 
# Guardaremos los datos en variables con los siguientes nombres:
# - `df_contract`
# - `df_personal`
# - `df_internet`
# - `df_phone`

# In[2]:


df_contract = pd.read_csv('/datasets/final_provider/contract.csv')
df_personal = pd.read_csv('/datasets/final_provider/personal.csv')
df_internet = pd.read_csv('/datasets/final_provider/internet.csv')
df_phone = pd.read_csv('/datasets/final_provider/phone.csv')


# # Exploración inicial de los datos
# 
# Ahora que hemos cargado los datasets, es importante validar la información que contienen.
# 
# Examinaremos cada uno de ellos utilizando los siguientes métodos de pandas:
# - `head`: para visualizar las primeras filas y tener una idea general de los datos.
# - `shape`: para conocer la cantidad de filas y columnas.
# - `info()`: para revisar los tipos de datos y la presencia de valores nulos.
# - `describe()`: para obtener estadísticas básicas de las variables numéricas.
# 
# Este análisis preliminar nos permitirá identificar si es necesario realizar correcciones, limpieza o un tratamiento adicional de los datos antes de continuar con el preprocesamiento.
# Una vez que todos los datasets estén limpios y preparados, se integrarán en un único dataset maestro que consolide la información de cada cliente.

# ### Exploración estructural y corrección de datos para el dataset de contratos.

# In[3]:


df_contract.head()


# In[4]:


df_contract.info()


# In[5]:


df_contract.describe()


# In[6]:


df_contract.shape


# #### Transformación y corrección de datos
# 
# Después de revisar la estructura de los datos con los métodos `shape`, `info()`, `head()` y `describe()`, se realizarán los siguientes pasos:
# 
# - Cambiar los nombres de las columnas a minúsculas para estandarizar, se creara la función column_snake_case.
# - Corregir los tipos de datos en las columnas:
#   - Columnas de fechas (`begindate` y `enddate`) al formato datetime.
#   - Columna `totalcharges` a tipo numérico (float).
# - Validar si existen valores nulos (`NA`) en las columnas.
# - Verificar la presencia de registros duplicados.
# 
# **Observaciones preliminares:**
# - El dataset contiene 8 columnas y 7043 registros.
# - Según `info()`, no se observaron valores ausentes de forma evidente, pero se realizará una validación adicional para confirmar.
# - Los valores numéricos presentan rangos coherentes en `monthlycharges`, aunque se debe revisar `totalcharges` por posibles inconsistencias en su formato.

# In[7]:


#Creación de función y conversión de nombres de columnas a minisculas.

def column_snake_case(df):
    df.columns = (
        df.columns
        .str.replace(r'([a-z])([A-Z])', r'\1_\2', regex=True)
        .str.lower()
    )
    return df

column_snake_case(df_contract)
print(df_contract.columns)


# In[8]:


#Conversión de datatype de la columna 'begin_date' y 'end_date'
dates_col_contract = ['begin_date', 'end_date']
for columna in dates_col_contract:
    df_contract[columna] = pd.to_datetime(df_contract[columna], errors='coerce')

df_contract[['begin_date', 'end_date']].info()


# In[9]:


#Coversión de 'total_charges' a tipo númerico.
df_contract['total_charges'] = pd.to_numeric(df_contract['total_charges'], errors='coerce')
df_contract[['total_charges']].info()


# In[10]:


#Validación de datos nulos (NA)
print(df_contract.isnull().sum())


# Observamos que existen 5174 valores nulos, sin embargo, son clientes activos. Más adelante esta columna se convertira en una columna binaria para identificar activos y bajas.

# In[11]:


# Vamos a analizar los valores nulos en 'total_charges' para decidir cómo imputarlos.
df_contract[df_contract['total_charges'].isnull()]


# Podemos observar que los datos nulos en la columna `total_charges` corresponden a clientes que recién comenzaron su contrato con la compañía, por lo que aún no tienen un importe facturado. Por este motivo, se imputarán con 0.

# In[12]:


#Imputar valores NA con 0 en columna 'totalcharges'
df_contract['total_charges'] = df_contract['total_charges'].fillna(0)
df_contract['total_charges'].isnull().sum()


# In[13]:


#Validación de duplicados
df_contract.duplicated().sum()


# Ahora con los datos correctos analizaremos la información correcta con .describe()

# In[14]:


df_contract.describe()


# Hallazgos de la exploración del dataset de contratos
# 
# Después de la limpieza y conversión de datos, se identificaron las siguientes características:
# 
# - No existen valores nulos en `monthly_charges` y `total_charges`.
# - Se imputaron con 0 los registros recientes que no tenían cargos acumulados.
# - Los rangos de `monthly_charges` y `total_charges` son coherentes con la operación de la compañía.
# - La dispersión de `total_charges` es alta, lo que indica que hay clientes con distinta antigüedad y niveles de facturación.
# 
# Los datos quedaron listos para integrarse posteriormente con los demás datasets.

# ### Exploración estructural para el dataset de personal.

# In[15]:


df_personal.head()


# In[16]:


df_personal.shape


# In[17]:


df_personal.info()


# In[18]:


df_personal.describe()


# #### Transformación y corrección de datos
# 
# En este dataset no será necesario cambiar los tipos de datos, ya que todos se encuentran correctamente definidos.
# 
# Se realizarán las siguientes acciones:
# - Estandarizar los nombres de las columnas a formato `snake_case`.
# - Verificar si existen valores ausentes o nulos en alguna columna y, en caso de encontrarlos, analizar la mejor forma de imputación.
# - Revisar si existen registros duplicados.
# 

# In[19]:


#Conversión de nombres de columnas a minisculas.
column_snake_case(df_personal)
print(df_personal.columns)


# In[20]:


#Validación de datos nulos (NA)
print(df_personal.isnull().sum())


# In[21]:


#Validación de duplicados
df_personal.duplicated().sum()


# Hallazgos de la exploración del dataset de datos personales
# 
# Después de la exploración de datos, se identificaron las siguientes características:
# 
# - El tipo de dato (`dtype`) de cada columna es correcto, por lo que no fue necesario realizar cambios.
# - No se encontraron valores ausentes ni nulos.
# - No hay registros duplicados.
# - La información obtenida con `describe()` es coherente: el dataset contiene 7043 registros, y en la columna `senior_citizen` el valor mínimo es 0 y el máximo es 1, lo que confirma que ya es una variable binaria.
# 
# Este dataset está listo para integrarse al dataset maestro.
# 

# ### Exploración estructural para el dataset de internet.

# In[22]:


df_internet.head()


# In[23]:


df_internet.info()


# In[24]:


df_internet.shape


# In[25]:


df_internet.describe()


# #### Transformación y corrección de datos.
# Después de realizar la exploración inicial (`head()`, `info()`, `shape()` y `describe()`), se identificaron las siguientes características:
# 
# - Todas las columnas tienen el tipo de dato `object`, lo que es adecuado ya que representan variables categóricas.
# - La mayoría de las columnas contienen valores "Yes" y "No", por lo que serán más fáciles de procesar con codificación one-hot en etapas posteriores.
# - El dataset contiene 5,517 registros, un número menor al de los otros datasets revisados previamente, por lo que será necesario considerar este detalle al momento de hacer la unión final.
# - La información obtenida con `describe()` muestra que las categorías son consistentes y no hay valores inesperados.
# 
# A continuación, se realizarán estas acciones:
# 
# - Estandarizar los nombres de las columnas a formato `snake_case`.
# - Verificar si existen valores nulos o ausentes y definir la estrategia de imputación si es necesario.
# - Revisar si hay registros duplicados.
# 

# In[26]:


#Conversión de nombres de columnas a minisculas.
column_snake_case(df_internet)
print(df_internet.columns)


# In[27]:


#Validación de datos nulos (NA)
print(df_internet.isnull().sum())


# In[28]:


#Validación de duplicados
df_internet.duplicated().sum()


# Hallazgos finales del dataset de servicios de Internet
# 
# Después de la exploración y limpieza del dataset, se identificaron las siguientes características complementarias:
# 
# - Todas las columnas tienen tipos de datos correctos y son variables categóricas.
# - No se encontraron valores nulos, ausentes ni registros duplicados.
# - Los nombres de las columnas fueron estandarizados al formato `snake_case`.
# 
# **Nota importante:** Este dataset contiene menos registros (5,517) que los otros datasets revisados anteriormente. Esto se debe a que no todos los clientes cuentan con servicios de Internet. Será necesario tener en cuenta este detalle al momento de realizar la unión final de los datos.

# ### Exploración estructural para el dataset de phone.

# In[29]:


df_phone.head()


# In[30]:


df_phone.shape


# In[31]:


df_phone.info()


# In[32]:


df_phone.describe()


# #### Transformación y corrección de datos.
# 
# Después de realizar la exploración inicial (`head()`, `info()`, `shape()` y `describe()`), se identificaron las siguientes características:
# 
# - Las columnas tienen tipos de datos correctos y no es necesario realizar cambios en este aspecto.
# - El dataset contiene 6,361 registros, por lo que también tiene menos datos que los dos primeros datasets analizados.
# - La columna `multiple_lines` es categórica con valores "Yes" y "No".
# 
# A continuación, se realizarán estas acciones:
# 
# - Estandarizar los nombres de las columnas al formato `snake_case`.
# - Verificar si existen valores nulos o ausentes y definir la estrategia de imputación en caso necesario.
# - Revisar si hay registros duplicados.
# 
# **Nota:** Será importante tener presente la diferencia en la cantidad de registros al momento de unir este dataset con los demás.
# 

# In[33]:


#Conversión de nombres de columnas a estructura snake_case.
column_snake_case(df_phone)
print(df_phone.columns)


# In[34]:


#Validación de datos nulos (NA)
print(df_phone.isnull().sum())


# In[35]:


#Validación de duplicados
df_phone.duplicated().sum()


# Hallazgos finales del dataset de servicios telefónicos
# 
# Después de la exploración y limpieza del dataset, se identificaron las siguientes características:
# 
# - No se encontraron valores nulos, ausentes ni registros duplicados.
# - Los nombres de las columnas fueron estandarizados al formato `snake_case`.
# 
# **Nota importante:** Este dataset contiene menos registros (6,361) que algunos de los otros datasets revisados. Esto se debe a que no todos los clientes cuentan con servicios telefónicos, por lo que será necesario considerar este detalle al realizar la unión final de los datos.
# 

# # Unión de los datasets.
# 
# En este paso integraremos los cuatro datasets en un único DataFrame que consolide toda la información disponible sobre los clientes.
# 
# Para conservar todos los registros de clientes, se realizará una unión utilizando la tabla de contratos como base principal. Esto permite mantener a todos los clientes en el dataset final, incluso si algunos no cuentan con servicios de Internet o servicios telefónicos.
# 
# La unión se hará sobre la columna `customer_id`, que actúa como identificador único en los cuatro datasets. El método de unión elegido será el de tipo `left`, de modo que se conserven todos los registros de la tabla principal (`contract`) y se añadan los datos de las otras tablas donde haya coincidencias.
# 
# Después de la unión, se revisará si existen valores nulos resultantes y se definirá la estrategia para su imputación.
# 

# In[36]:


df_master = df_contract.merge(df_personal, on="customer_id", how="left")
df_master = df_master.merge(df_internet, on="customer_id", how="left")
df_master = df_master.merge(df_phone, on="customer_id", how="left")


# In[37]:


df_master.head()


# In[38]:


df_master.info()


# In[39]:


#Validación de datos nulos (NA)
print(df_master.isnull().sum())


# #### Transformación y corrección de datos
# 
# Después de unir los datasets, se identificaron valores nulos en las columnas relacionadas con los servicios de Internet y telefónicos. Estos valores corresponden a clientes que no tienen contratado alguno de estos servicios y, por lo tanto, no aparecen en los datasets originales de dichas categorías.
# 
# La estrategia de imputación será la siguiente:
# 
# - Para todas las columnas binarias relacionadas con los servicios de Internet y teléfono (por ejemplo, `online_security`, `online_backup`, `tech_support`, `multiple_lines`, etc.), se imputará el valor `"No"` indicando que el cliente no utiliza estos servicios.
# - Para la columna `internet_service`, que contiene el nombre del tipo de servicio contratado, se imputará el valor `"Sin servicio"` para diferenciar claramente los casos donde no hay servicio de Internet.
# 
# De este modo, se estandarizarán los datos sin eliminar registros y se mantendrá la información coherente con la situación real de cada cliente.

# In[40]:


#Creo una lista de las columnas sin servicio.
no_service = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "multiple_lines"
]


# In[41]:


#Creamos un bucle para imputar los valores ausentes con "No" para las columnas sin servicio que contienen datos de Si o No. 
for columna in no_service:
    df_master[columna] = df_master[columna].fillna("No")


# In[42]:


#Imputamos valores de la columna 'internet_service' con "Sin servicio"
df_master['internet_service'].fillna('Sin Servicio', inplace=True)


# In[43]:


df_master.isnull().sum()


# In[44]:


df_master.info()


# ## Creación de variable de antigüedad del cliente
# La columna begin_date representa la fecha en la que el cliente comenzó su contrato con la compañía.
# 
# En este caso, se transformará begin_date en una nueva variable que representa la antigüedad del cliente, calculando la diferencia en días entre una fecha de corte fija y la fecha de inicio.
# 
# La fecha de corte elegida será el 1 de febrero de 2020, ya que es el punto de referencia.
# 
# Esta nueva característica puede resultar valiosa, ya que es razonable suponer que los clientes con mayor antigüedad podrían tener menor probabilidad de cancelar su contrato, por costumbre, satisfacción o permanencia a largo plazo.

# In[45]:


cut_date = pd.to_datetime('2020-02-01') #Creamos una variable para convertir la fecha de corte 2 de febrero a date type. 
df_master['seniority_date'] = cut_date - df_master['begin_date'] #Calculamos antigüedad de acuerdo a la fecha de inicio del contrato.

df_master['seniority_days'] = df_master['seniority_date'].dt.days #Convertimos la antüedad a días.
df_master.sample(10)


# In[46]:


#Eliminamos las columnas begin_date y seniority_date que previamente creamos para dejar unicamente la variable numerica de antigüedad en días. 
df_master = df_master.drop(['begin_date', 'seniority_date'], axis=1)


# ## Distribución de cancelaciones a lo largo del tiempo
# 
# Antes de transformar la columna `end_date`, se realizará una visualización para explorar la distribución temporal de las cancelaciones de contrato.
# 
# El objetivo de esta gráfica es identificar posibles patrones en las fechas en las que los clientes han cancelado sus servicios, como concentraciones inusuales en determinados días o periodos. Este análisis puede aportar información valiosa sobre el comportamiento de cancelación, la estacionalidad o incluso eventos específicos que hayan influido en la pérdida de clientes.
# 
# Solo se tomarán en cuenta las fechas válidas, excluyendo los valores nulos (`NaT`), ya que estos corresponden a clientes que aún siguen activos.
# 

# In[47]:


# Agrupamos por día de cancelaciones para saber el número de cancelaciones.
inactive_clients = df_master.groupby(['end_date'])['customer_id'].count().reset_index()
#Renombramos columnas
inactive_clients.columns = ['end_effective_date', 'inact_client_count']


# In[48]:


inactive_clients.columns


# In[49]:


plt.figure(figsize=(10,5))
plt.plot(inactive_clients['end_effective_date'], inactive_clients['inact_client_count'], marker='o', color='steelblue')
plt.title('Número de bajas por fecha', color='dimgray')
plt.xlabel('Fecha', color='midnightblue')
plt.ylabel('Clientes inactivos', color='dimgray')
plt.xticks(rotation=60)
plt.tick_params(axis='x', colors='dimgray')
plt.tick_params(axis='y', colors='dimgray')
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Hallazgos del análisis temporal de cancelaciones
# 
# La visualización de la columna `end_date` permitió identificar el comportamiento de las cancelaciones de clientes a lo largo del tiempo.
# 
# Se observa que a inicios de octubre de 2019 se registraban aproximadamente 450 cancelaciones. A lo largo del mes, esta cifra fue en aumento, alcanzando su punto más alto el 1 de noviembre de 2019, con un pico de aproximadamente 485 cancelaciones.
# 
# Posteriormente, a partir de esa fecha, las cancelaciones muestran una tendencia descendente constante hasta el cierre del periodo observado (enero de 2020).
# 
# Este comportamiento sugiere que pudo haber ocurrido algún evento o condición particular en octubre que motivó un mayor número de cancelaciones, seguido de una estabilización en los meses siguientes.
# 

# Con estas acciones, se concluye la etapa de exploración y limpieza. Los datos se encuentran listos para iniciar el preprocesamiento y la construcción de los modelos de clasificación.

# # Preprocesamiento de datos.
# 
# Como primer paso del preprocesamiento, definiremos la variable objetivo que será utilizada para entrenar nuestro modelo de clasificación.
# 
# La columna `end_date` nos indica si un cliente ha cancelado su contrato. Por lo tanto, a partir de esta columna crearemos una nueva variable binaria llamada `exited`, que tomará los siguientes valores:
# 
# - `1` si el cliente ha cancelado (es decir, si `end_date` contiene una fecha).
# - `0` si el cliente sigue activo (es decir, si `end_date` es nulo o `NaT`).
# 
# Esta nueva variable será nuestro objetivo el cual el modelo intentará predecir en función de las características del cliente y los servicios que utiliza.

# In[50]:


df_master['exited'] = df_master['end_date'].notnull().astype(int)
print(df_master['exited'])


# Eliminación de columna `end_date`
# 
# Una vez creada la variable binaria `exited`, que representa si un cliente canceló o no su contrato, la columna `end_date` deja de ser necesaria para el análisis.
# 
# Por lo tanto, se elimina del dataset para evitar redundancia y asegurar que no sea utilizada como variable predictora en el modelo.
# 

# In[51]:


df_master = df_master.drop(['end_date'], axis=1)


# In[52]:


df_master.info()


# In[53]:


# En este punto creamos una copia del dataframe original, antes de procesar las variables categoricas.
df_mastercopy = df_master.copy()


# Para la conversión de todas las columnas categóricas con respuestas tipo "Yes" y "No" a valores binarios, utilizando la siguiente lógica:
# 
# "Yes" será reemplazado por 1
# 
# "No" será reemplazado por 0

# In[54]:


binary_columns = ['paperless_billing','partner','dependents','online_security','online_backup','device_protection','tech_support','streaming_tv','streaming_movies', 'multiple_lines']


# In[55]:


for column in binary_columns:
    df_master[column] = df_master[column].map({'Yes': 1, 'No': 0})


# In[56]:


#Validamos la reasignación
df_master[binary_columns].head()


# ## Transformación de variables categóricas con one-hot encoding
# 
# Algunas columnas del dataset contienen variables categóricas con más de dos categorías, por lo que no pueden ser transformadas con mapeo binario como las variables tipo `"Yes"`/`"No"`. Para hacerlas utilizables por los modelos de machine learning, aplicaremos **codificación one-hot**.
# 
# Las columnas seleccionadas para este proceso son:
# 
# - `type`: tipo de contrato del cliente  
# - `payment_method`: método de pago  
# - `gender`: género del cliente  
# - `internet_service`: tipo de servicio de Internet
# 
# Estas variables serán transformadas en columnas binarias, una por cada categoría distinta, lo que permitirá a los modelos interpretar sus valores correctamente.
# 
# Para evitar multicolinealidad en modelos lineales, se aplicará el argumento `drop_first=True`, eliminando una de las categorías de cada variable como referencia.

# In[57]:


# Convertir variables categóricas en variables dummy
category_columns = ['type','payment_method','gender','internet_service']
dummies = pd.get_dummies(df_master[category_columns], drop_first=True)


# In[58]:


df_master = pd.concat([df_master, dummies], axis=1)
df_master = df_master.drop(category_columns, axis=1) #Eliminamos columnas con strings


# ## Examen del Equilibrio de Clases
# Distribución de clases: Vamos a ver cuántos clientes se han ido (1) y cuántos no (0).
# 
# Visualización: Graficaremos la distribución de las clases para tener una visión clara de cuán equilibrada está la variable objetivo.

# In[59]:


# Ver la distribución de la clase objetivo 'exited'
class_distribution = df_master['exited'].value_counts(normalize=True)

# Imprimir la distribución
print("Distribución de clases en el target (exited):\n", class_distribution)


# In[60]:


class_distribution.plot(kind='bar', color=['palevioletred','thistle'])
plt.title('Distribución de la Clase Objetivo: Exited')
plt.xlabel('Exited (0 = No, 1 = Sí)')
plt.ylabel('Número de Clientes')
plt.xticks(rotation=0)
plt.show()


# Al analizar la distribución de la variable objetivo `exited`, se observa que:
# 
# - El 73.5% de los clientes **no cancelaron** su contrato.
# - El 26.5% de los clientes **sí cancelaron**.
# 
# Aunque no se trata de un desbalance extremo, es una diferencia considerable que podría sesgar los modelos hacia la clase mayoritaria si no se considera adecuadamente.
# 
# Por esta razón, se tendrán en cuenta métricas sensibles al desbalance (como AUC-ROC y F1-score) durante la evaluación. También se valorará el uso de técnicas de reequilibrio si el desempeño del modelo lo requiere.
# 

# In[61]:


df_master.head()


# # Segmentación de datos
# 
# A continuación, se dividirá el conjunto de datos en variables predictoras (`target`) y variable objetivo (`features`).  
# 
# La columna `exited` será usada como variable objetivo, ya que indica si un cliente canceló o no su contrato. El resto de las columnas serán utilizadas como características (`features`), excluyendo cualquier identificador como `customer_id`.
# 
# Se utilizará la función `train_test_split` para dividir los datos en tres partes:
# 
# - **70% para entrenamiento**
# - **15% para validación**
# - **15% para prueba final**
# 
# Esta división permite ajustar y comparar modelos con los datos de entrenamiento y validación, y reservar un conjunto independiente para la evaluación final del rendimiento del modelo. Para asegurar resultados reproducibles, se fijó el parámetro `random_state=12345`.

# In[62]:


# Separar las características (features) y el objetivo (target)
features = df_master.drop(['exited','customer_id'], axis=1)
target = df_master['exited']


# In[63]:


# División en 70% para entrenamiento y 30% para validación + prueba
features_train, features_temp, target_train, target_temp = train_test_split(features, target, test_size=0.3, random_state=12345)

# Dividir la parte de validación + prueba en 50% para validación y 50% para prueba (15% cada uno)
features_val, features_test, target_val, target_test = train_test_split(features_temp, target_temp, test_size=0.5, random_state=12345)

# Verificar las dimensiones de los conjuntos
print(f"Entrenamiento: {features_train.shape[0]} muestras")
print(f"Validación: {features_val.shape[0]} muestras")
print(f"Prueba: {features_test.shape[0]} muestras")


# # Preparación de modelos.

# ## Escalación de variables númericas.
# 
# Se aplicará escalamiento estándar (`StandardScaler`) a las variables numéricas:
# 
# - `monthly_charges`
# - `total_charges`
# - `seniority_days`
# 
# El escalador se ajusta con (`fit`) únicamente con los datos de entrenamiento para evitar fuga de información, y luego se aplicó (`transform`) a los conjuntos de entrenamiento, validación y prueba. Esto garantiza que todas las variables numéricas estén en la misma escala para el entrenamiento de modelos.
# 
# Para evitar advertencias generadas por asignaciones encadenadas durante la transformación de columnas en `pandas`, se desactivó el modo de advertencia temporalmente con:
# 
# ```python
# pd.options.mode.chained_assignment = None

# In[64]:


pd.options.mode.chained_assignment = None

numeric = ['monthly_charges', 'total_charges', 'seniority_days']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])
features_val[numeric] = scaler.transform(features_val[numeric])
features_test[numeric] = scaler.transform(features_test[numeric])


# ## Función para evaluar modelos
# 
# Se definió una función personalizada llamada `evaluate_model()` que permite evaluar fácilmente el rendimiento de cualquier modelo entrenado, utilizando tres conjuntos de datos:
# 
# - **Entrenamiento**
# - **Validación**
# - **Prueba**
# 
# Para cada conjunto, la función calcula las siguientes métricas de clasificación:
# 
# - **Accuracy (Exactitud)**: proporción de predicciones correctas.
# - **F1-score**: equilibrio entre precisión y recall, útil en casos de desbalance de clases.
# - **AUC-ROC**: capacidad del modelo para distinguir entre clases usando probabilidades.
# 
# Esta función facilita comparar de forma consistente distintos modelos y detectar posibles problemas como sobreajuste o bajo rendimiento general.
# 

# In[65]:


def evaluate_model(model, features_train, target_train, features_val, target_val, features_test, target_test):
    """
    Evalúa un modelo en los conjuntos de entrenamiento, validación y prueba.
    Calcula Accuracy, F1 y AUC-ROC para cada uno.
    """
    results = {}
    
    for name, X, y in [('Entrenamiento', features_train, target_train),
                       ('Validación', features_val, target_val),
                       ('Prueba', features_test, target_test)]:

        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, y_proba)

        print(f"\n{name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")

        results[name] = {'Accuracy': acc, 'F1-score': f1, 'AUC-ROC': auc}

    return pd.DataFrame(results).round(4)


# # Entranamiento y Evaluación de Modelos.

# ### Modelo 1: Regresión logistica
# Para comenzar la fase de modelado, entrenaremos un modelo de regresión logística utilizando las características previamente procesadas. 

# In[66]:


lr = LogisticRegression(random_state=12345)
lr.fit(features_train, target_train)
metrics_lr_all = evaluate_model(lr, features_train, target_train, features_val, target_val, features_test, target_test)


# Se calcularon las métricas **Accuracy**, **F1-score** y **AUC-ROC** para analizar su rendimiento en cada conjunto.
# Estos resultados indican un buen desempeño general y una adecuada generalización del modelo. Las métricas son consistentes entre los tres conjuntos, lo cual sugiere que el modelo no está sobreajustado, sin embargo, podemos buscar algunas mejoras o parametros que hagan que nuestros resultados sean mejores. 

# Evaluaremos la importancia de las características en el modelo de regresión logística.
# Usaremos los coeficientes para asignar a cada variable y entender su impacto en la predicción.
# Esto nos permitirá identificar qué variables influyen más en la cancelación de clientes.
# 
# En una regresión logística, los coeficientes indican la dirección e intensidad del impacto de cada variable en la probabilidad de que el evento ocurra:
# 
# Coeficientes positivos → Aumentan la probabilidad de salida.
# 
# Coeficientes negativos → Disminuyen la probabilidad de salida.

# In[67]:


coefs_lr = pd.Series(lr.coef_[0], index=features_train.columns)
coefs_lr = coefs_lr.sort_values(key=abs, ascending=False)
coefs_lr.head(20).plot(kind='barh', title='Coeficientes más importantes')


# In[68]:


features_top10_lr = ['type_Two year','internet_service_Sin Servicio','internet_service_Fiber optic','total_charges','type_One year','streaming_tv','streaming_movies','online_security','payment_method_Electronic check','paperless_billing']

features_train_top10lr = features_train[features_top10_lr]
features_val_top10lr = features_val[features_top10_lr]
features_test_top10lr = features_test[features_top10_lr]

lr_top7 = LogisticRegression(random_state=12345)
lr_top7.fit(features_train_top10lr, target_train)
metrics_lrtop7 = evaluate_model(lr_top7, features_train_top10lr, target_train, features_val_top10lr, target_val, features_test_top10lr, target_test)


# Tras comparar los resultados obtenidos al entrenar el modelo de regresión logística utilizando todas las variables versus solo las 10 variables más influyentes, se observa que los valores de las métricas son muy similares en ambos casos. Sin embargo, el modelo que incluye todas las variables muestra ligeramente mejores resultados en términos de precisión, F1-score y AUC-ROC en los tres subconjuntos (entrenamiento, validación y prueba).
# 
# Con base en estos hallazgos, se concluye que utilizar el conjunto completo de características es más beneficioso para este modelo. Por lo tanto, a continuación se evaluarán diferentes valores del hiperparámetro C usando todas las variables, con el objetivo de optimizar aún más su desempeño.

# #### Evaluación y optimización del modelo de Regresión Logística
# 
# Para mejorar el rendimiento del modelo de regresión logística, se evaluaran distintos valores del hiperparámetro C, que controla la fuerza de la regularización (inversamente proporcional).
# 
# Para cada valor de C, se calcularan seguiran calculando las tres métricas de desempeño en el conjunto de validación:
# 
# Accuracy: proporción de predicciones correctas.
# 
# F1-score: balance entre precisión y sensibilidad.
# 
# AUC-ROC: capacidad del modelo para diferenciar entre clases.
# 

# In[69]:


best_auc_lr = 0
best_model_lr = None
best_metrics_lr = {}

for c in [1, 5, 7, 10]:
    print(f"\n Evaluando modelo con C = {c}")
    
    model_lr_hipertop7 = LogisticRegression(random_state=12345, solver='liblinear', C=c)
    model_lr_hipertop7.fit(features_train, target_train)

    # Predicciones en validación
    val_pred = model_lr_hipertop7.predict(features_val)
    val_proba = model_lr_hipertop7.predict_proba(features_val)[:, 1]

    # Métricas
    acc_lr = accuracy_score(target_val, val_pred)
    f1_lr = f1_score(target_val, val_pred)
    auc_lr = roc_auc_score(target_val, val_proba)

    print(f"Accuracy: {acc_lr:.4f}, F1: {f1_lr:.4f}, AUC-ROC: {auc_lr:.4f}")

    # Guardar si es mejor
    if auc_lr > best_auc_lr:
        best_auc_lr = auc_lr
        best_model_lr = model_lr_hipertop7
        best_metrics_lr = {'C': c, 'Accuracy': acc_lr, 'F1': f1_lr, 'AUC': auc_lr}

print("\n Mejor modelo en validación en Regresion Logistica:")
for k, v in best_metrics_lr.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Evaluarlo completamente con la función
metrics_lr_hipertop7 = evaluate_model(best_model_lr,
               features_train, target_train,
               features_val, target_val,
               features_test, target_test)


# Al comparar el modelo de regresión logística base (con parámetros por defecto) contra versiones ajustadas mediante el hiperparámetro C, se identificó que el valor C = 10 ofrece el mejor rendimiento general.
# 
# AUC ROC paso de 0.8335 a 0.8338 en validación y de 0.8271 a 0.8273 en prueba. 
# 
# Este ajuste logró una ligera mejora en las métricas clave (especialmente AUC-ROC) en el conjunto de validación y prueba, en comparación con el modelo inicial.
# 
# Por tanto, se concluye que entrenar la regresión logística con todas las variables originales y C = 10 es la configuración óptima

# ### Modelo 2: DecisionTreeClassifier
# 
# Este modelo toma decisiones dividiendo los datos según las características más relevantes.

# In[70]:


#Entrenar modelo
dtree = DecisionTreeClassifier(max_depth=10, random_state=12345)
dtree.fit(features_train, target_train)
metrics_dtree = evaluate_model(dtree, features_train, target_train, features_val, target_val, features_test, target_test)


# En nuestro primer entrenamiento del modelo DecisionTreeClassifier con una profundidad máxima (max_depth) de 10, obtuvimos un rendimiento muy competitivo:
# 
# Entrenamiento: Accuracy de 0.9002, AUC-ROC de 0.9579 y F1-score de 0.7953, lo cual indica un modelo que se ajusta muy bien a los datos.
# 
# Validación y prueba: Las métricas también fueron altas (AUC-ROC > 0.83), lo que sugiere que el modelo generaliza bien, aunque podría estar comenzando a sobreajustarse ligeramente.

# #### Evaluación y optimización del modelo de Regresión Logística
# 
# Para buscar posibles mejoras, vamos a identificar las características más influyentes mediante el atributo feature_importances_.
# Con base en ello, seleccionaremos las variables más importantes y entrenaremos un nuevo modelo utilizando únicamente dichas características.

# In[71]:


# Mostrar importancia de características
importances = dtree.feature_importances_
feature_names = features_train.columns
sorted_idx = importances.argsort()

plt.figure(figsize=(8,6))
plt.barh(feature_names[sorted_idx], importances[sorted_idx], color='plum')
plt.xlabel("Importancia")
plt.title("Importancia de características")
plt.tight_layout()
plt.show()


# La visualización nos muestra que variables como total_charges, seniority_days y payment_method_Electronic check son las que más contribuyen a las decisiones del modelo.
# 
# Con base en este análisis, seleccionaremos las variables más importantes (top 7, importancia>0.5) y entrenaremos nuevamente el modelo únicamente con estas características.

# In[72]:


features_top7 = ['total_charges','seniority_days','payment_method_Electronic check','monthly_charges','type_Two year','internet_service_Fiber optic','type_One year']

features_train_top7 = features_train[features_top7]
features_val_top7 = features_val[features_top7]
features_test_top7 = features_test[features_top7]


# In[73]:


#Entrenar modelo con top 7 de variables.
dtree_top7 = DecisionTreeClassifier(max_depth=10, random_state=12345)
dtree_top7.fit(features_train_top7, target_train)
#Evaluación del modelo
metrics_dtree_top7 = evaluate_model(dtree_top7, features_train_top7, target_train, features_val_top7, target_val, features_test_top7, target_test)


# Después de entrenar un modelo con todas las variables, probamos una versión más simple usando solo las 7 características más importantes. Aunque el rendimiento fue muy similar, el modelo con 7 variables logró ligeras mejoras en las métricas de validación y prueba.
# 
# AUC-ROC: 0.8374 paso a 0.8495 para validación y de 0.8381 a 0.8446 en conjunto de prueba. 
# 
# Como es más simple y sigue dando buenos resultados, usaremos este conjunto de 7 variables para buscar los mejores hiperparámetros y así intentar mejorar aún más el modelo.

# In[74]:


# Hiperparámetros a evaluar
depths = [10, 15, 20, 25, 30]
leaves = [5, 10, 15, 20, 25]
weights = [None, 'balanced']

# Variables para guardar el mejor modelo y su puntuación
best_auc_dtree = 0
best_model_dtree = None
best_params_dtree = {}

# Bucle para explorar combinaciones
for d in depths:
    for l in leaves:
        for w in weights:
            print(f"Evaluando: max_depth={d}, min_samples_leaf={l}, class_weight={w}")

            model_dtree_top7hip = DecisionTreeClassifier(
                max_depth=d,
                min_samples_leaf=l,
                class_weight=w,
                random_state=12345
            )
            model_dtree_top7hip.fit(features_train_top7, target_train)
            preds = model_dtree_top7hip.predict(features_val_top7)
            proba = model_dtree_top7hip.predict_proba(features_val_top7)[:, 1]

            acc_dtree = accuracy_score(target_val, preds)
            f1_dtree = f1_score(target_val, preds)
            auc_dtree = roc_auc_score(target_val, proba)

            print(f"Accuracy: {acc_dtree:.3f}, F1: {f1_dtree:.3f}, AUC-ROC: {auc_dtree:.3f}")
            print("-"*40)

            if auc_dtree > best_auc_dtree:
                best_auc_dtree = auc_dtree
                best_model_dtree = model_dtree_top7hip
                best_params_dtree = {
                    'max_depth': d,
                    'min_samples_leaf': l,
                    'class_weight': w,
                    'accuracy': acc_dtree,
                    'f1': f1_dtree,
                    'auc': auc_dtree
                }

# Mostrar el mejor modelo
print("\nMejor combinación encontrada en Arbol de Decisión:")
for k, v in best_params_dtree.items():
    print(f"{k}: {v}")

# Evaluar el mejor modelo con la función completa
metrics_dtree_top7hip = evaluate_model(
    best_model_dtree,
    features_train_top7, target_train,
    features_val_top7, target_val,
    features_test_top7, target_test
)


# Después de probar diferentes configuraciones, el mejor rendimiento se logró al usar únicamente las 7 variables más importantes, combinadas con el ajuste de hiperparámetros (max_depth=20, min_samples_leaf=15, class_weight='balanced'), se logro una mejora clara:
# 
# Validación: AUC-ROC pasó de 0.8301 a 0.8839
# 
# Prueba: AUC-ROC mejoró de 0.8419 a 0.8637
# 
# También mejoraron el F1-score y accuracy de validación y prueba
# 
# Este modelo superó en AUC-ROC, F1 y accuracy tanto al modelo base como al que usaba todas las variables.

# ## Modelo 3: Random Forest
# 
# Tras evaluar la importancia de las variables en el modelo anterior, reutilizaremos las 7 características más influyentes para entrenar un modelo de Random Forest. Este algoritmo de ensamblado combina múltiples árboles de decisión para mejorar la capacidad predictiva y reducir el sobreajuste.
# 
# El objetivo es comparar su rendimiento con los modelos anteriores y explorar posibles mejoras mediante el ajuste de hiperparámetros.

# Entrenamiento inicial con Random Forest
# 
# Entrenaremos un modelo de Random Forest utilizando todas las variables disponibles, sin ajustes adicionales en sus hiperparámetros. Este primer entrenamiento servirá como línea base para comparar el rendimiento del modelo antes de realizar ajustes y optimizaciones.

# In[75]:


#Entrenar modelo
random_f = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=12345)
random_f.fit(features_train, target_train)

#Evaluación del modelo
metrics_ramdom_f = evaluate_model(random_f, features_train, target_train, features_val, target_val, features_test, target_test)


#  El modelo obtuvo un buen rendimiento general, destacando especialmente en el conjunto de entrenamiento con un AUC-ROC de 0.9674 y un F1-score de 0.7724. Sin embargo, se observa una ligera caída en las métricas sobre los conjuntos de validación y prueba, lo que sugiere que aún hay espacio para optimización.

# #### Evaluación y optimización del modelo de Regresión Logística
# 
# En esta segunda fase, reutilizaremos las 7 variables más importantes previamente identificadas para entrenar un modelo de Random Forest. Esta selección busca reducir la complejidad del modelo y optimizar el uso de recursos computacionales, sin sacrificar el rendimiento. Entrenaremos el modelo utilizando únicamente estas variables, manteniendo los parámetros básicos como punto de partida, para después evaluar si se obtienen mejoras en las métricas.
# 

# In[76]:


#Entrenar modelo con top 7 variables.
random_f_top7 = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=12345)
random_f_top7.fit(features_train_top7, target_train)
#Evaluación del modelo. 
metrics_random_f_top7 = evaluate_model(random_f_top7, features_train_top7, target_train, features_val_top7, target_val, features_test_top7, target_test)


# A pesar de la reducción de variables, el modelo mostró un excelente desempeño general, superando incluso al modelo entrenado con todas las variables.
# 
# **AUC ROC mejoro de 86.40 a 89.11 en conjunto de validación y de 0.8682 a 89.67 en el conjunto de prueba.**
# 
# Por ello, tomaremos este modelo como base para ajustar hiperparámetros en el siguiente paso.

# In[77]:


depths_rf = [8, 10, 12]
leaves_rf = [5, 10, 15]
best_auc_rf = 0
best_model_rf = None
best_params_rf = {}

for d in depths_rf:
    for l in leaves_rf:
        rf_hiper_top7 = RandomForestClassifier(
            n_estimators=150,
            max_depth=d,
            min_samples_leaf=l,
            random_state=12345
        )
        rf_hiper_top7.fit(features_train_top7, target_train)
        proba = rf_hiper_top7.predict_proba(features_val_top7)[:, 1]
        pred = rf_hiper_top7.predict(features_val_top7)
        auc_rf = roc_auc_score(target_val, proba)
        f1_rf = f1_score(target_val, pred)
        
        print(f"max_depth={d}, min_samples_leaf={l}, AUC={auc_rf:.4f}, F1={f1_rf:.4f}")
        
        if auc_rf > best_auc_rf:
            best_auc_rf = auc_rf
            best_model_rf = rf_hiper_top7
            best_params_rf = {'max_depth': d, 'min_samples_leaf': l, 'AUC': auc_rf, 'F1': f1_rf}

# Mostrar mejor combinación
print("\n Mejor combinación encontrada en Random Forest:")
for k, v in best_params_rf.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Evaluar el mejor modelo completamente
metrics_rf_hiper_top7 = evaluate_model(best_model_rf,
               features_train_top7, target_train,
               features_val_top7, target_val,
               features_test_top7, target_test)


# Después de realizar una búsqueda de hiperparámetros, encontramos que la mejor combinación fue max_depth=12 y min_samples_leaf=5.
# 
# El modelo resultante mostró un excelente rendimiento:
# 
# AUC-ROC en prueba: 0.8918, superando el umbral objetivo de 0.88.
# Accuracy en prueba: 0.8382 y F1-score: 0.6743, lo cual indica un buen equilibrio entre precisión y recall.
# 
# Las métricas de validación y prueba son bastante cercanas a las de entrenamiento, lo que sugiere que el modelo generaliza adecuadamente y no presenta signos evidentes de sobreajuste.
# Además de su buen desempeño, este modelo tiene la ventaja de ser más eficiente al estar entrenado únicamente con las variables más relevantes.
# Por estas razones, considero que esta versión ajustada del modelo de Random Forest es una de las más robustas hasta el momento.

# ## Modelo 4: LightGBM
# A continuación, entrenaremos un modelo con LightGBM, un algoritmo de boosting que utiliza árboles de decisión y está optimizado para eficiencia y velocidad. Es especialmente útil cuando se trabaja con grandes volúmenes de datos o conjuntos con muchas características.
# 
# El objetivo es comparar su rendimiento con los modelos anteriores y explorar posibles mejoras a través del análisis de características e hiperparámetros.

# **Entrenamiento inicial con todas las variables**
# 
# El primer entrenamiento se realizará utilizando todas las variables disponibles, sin ajustes adicionales en los hiperparámetros del modelo.
# Este entrenamiento servirá como línea base para comparar el rendimiento del modelo antes de aplicar técnicas de selección de características u optimización de parámetros.

# In[78]:


#Entrenar modelo
lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=12345)
lgbm.fit(features_train, target_train)

#Evaluación del modelo
metrics_lgbm = evaluate_model(lgbm, features_train, target_train, features_val, target_val, features_test, target_test)


# El modelo mostró un buen rendimiento, con un F1-score superior a 0.70 y un AUC-ROC por encima de 0.92 en el conjunto de prueba.
# 
# Sin embargo, se observa un ligero sobreajuste, evidenciado por el desempeño más alto en entrenamiento en comparación con validación y prueba. Se explorará si este comportamiento mejora al utilizar únicamente las 7 variables más relevantes.

# #### Evaluación y optimización del modelo de Regresión Logística
# 
# Se reutilizan las 7 características más influyentes identificadas previamente para entrenar el modelo LightGBM. El objetivo es evaluar si al reducir el número de variables, el modelo logra mejorar la generalización y reducir el sobreajuste, manteniendo o incluso superando su rendimiento en validación y prueba.
# 
# Se utilizarán los mismos hiperparámetros del primer experimento (n_estimators=100, learning_rate=0.1, max_depth=6) como base de comparación.

# In[79]:


# Entrenar modelo con top 7 variables
lgbm_top7 = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=12345)
lgbm_top7.fit(features_train_top7, target_train)

# Evaluación del modelo
metrics_lgbm_top7 = evaluate_model(
    lgbm_top7,
    features_train_top7, target_train,
    features_val_top7, target_val,
    features_test_top7, target_test)


# Al entrenar el modelo con las 7 variables más importantes, observamos un mejor desempeño en los conjuntos de validación y prueba respecto al modelo base que utilizó todas las variables. En particular:
# 
# El F1-score en prueba aumentó de 0.7269 a 0.7747, lo cual indica una mejora en el equilibrio entre precisión y recall.
# 
# El AUC-ROC en prueba también mejoró de 0.9242 a 0.9376, confirmando que el modelo es más eficaz diferenciando entre las clases.
# 
# Además, se observa una reducción del sobreajuste, ya que la diferencia entre las métricas de entrenamiento y prueba se volvió más estrecha.
# 
# Tomaremos este modelo como base para ajustar hiperparámetros en el siguiente paso.

# In[80]:


depths_lgbm = [6, 8, 10]
leaves_lgbm = [20, 25, 30]
learning_rates = [0.1, 0.05, 0.01]

best_auc_lgbm = 0
best_model_lgbm = None
best_params_lgbm = {}

for d in depths_lgbm:
    for l in leaves_lgbm:
        for lr in learning_rates:
            lgbm_top7hip = LGBMClassifier(
                n_estimators=200,              
                max_depth=d,
                min_samples_leaf=l,
                learning_rate=lr,
                random_state=12345
            )
            
            lgbm_top7hip.fit(features_train_top7, target_train)
            proba = lgbm_top7hip.predict_proba(features_val_top7)[:, 1]
            pred = lgbm_top7hip.predict(features_val_top7)
            auc_lgbm = roc_auc_score(target_val, proba)
            f1_lgbm = f1_score(target_val, pred)
        
            print(f"max_depth={d}, min_samples_leaf={l}, learning_rate={lr}, AUC={auc_lgbm:.4f}, F1={f1_lgbm:.4f}")
        
            if auc_lgbm > best_auc_lgbm:
                best_auc_lgbm = auc_lgbm
                best_model_lgbm = lgbm_top7hip
                best_params_lgbm = {
                    'max_depth': d,
                    'min_samples_leaf': l,
                    'learning_rate': lr,
                    'AUC': auc_lgbm,
                    'F1': f1_lgbm
                }

# Mostrar mejor combinación
print("\nMejor combinación encontrada en LightGBM:")
for k, v in best_params_lgbm.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Evaluar el mejor modelo completamente
metrics_lgbm_top7hip = evaluate_model(
    best_model_lgbm,
    features_train_top7, target_train,
    features_val_top7, target_val,
    features_test_top7, target_test
)


# Tras el ajuste de hiperparámetros, se entrenó un modelo de LightGBM con los siguientes parámetros:
# 
# - `max_depth`: 8  
# - `min_samples_leaf`: 25  
# - `learning_rate`: 0.1  
# 
# Este modelo obtuvo un excelente rendimiento en todos los conjuntos, destacando un AUC-ROC de **0.9480** y un F1-score de **0.8195** en el conjunto de prueba, lo que indica una muy buena capacidad de generalización.
# 
# Estos resultados reflejan un equilibrio sólido entre precisión y robustez, por lo que se selecciona este modelo como el más adecuado para el problema planteado.
# 

# ## Modelo 5: Catboost
# 
# Como modelo final entrenaremos un modelo **CatBoost**, un algoritmo de gradient boosting desarrollado para manejar de forma eficiente variables categóricas sin necesidad de codificación explícita. Este modelo es conocido por su robustez frente al sobreajuste y por requerir un preprocesamiento mínimo.
# 
# Este modelo es especialmente relevante para nuestro caso, ya que permite trabajar directamente con las variables categóricas del conjunto de datos sin transformaciones previas complejas. Esto puede ayudar a preservar relaciones importantes entre categorías y mejorar la precisión del modelo. Además, su rendimiento competitivo y facilidad de uso lo convierten en una excelente opción para problemas de clasificación como el presente.

# ### Preparación de datos para CatBoost
# 
# Para entrenar el modelo CatBoost, utilizaremos una copia del dataframe maestro que fue creada antes de procesar las variables categóricas, ya que este modelo puede manejar variables categóricas en su forma original (tipo object), sin necesidad de codificarlas manualmente.
# 
# En este paso:
# 
#     - Se separarán las características y la variable objetivo (exited).
# 
#     - Se dividirá el conjunto en entrenamiento, validación y prueba.
# 
#     - Se identificarán las columnas categóricas que serán indicadas al modelo.
# 
# Esta preparación permitirá aprovechar al máximo las capacidades internas de CatBoost para manejar datos categóricos de forma eficiente.

# In[81]:


# Definimos variables y objetivo.
features_cboost = df_mastercopy.drop(columns=['customer_id', 'exited'])
target_cboost = df_mastercopy['exited']

#Indetificamos columnas categoricas
cat_features = features_cboost.select_dtypes(include='object').columns.tolist()
cat_feature_indices = [features_cboost.columns.get_loc(col) for col in cat_features]

#Mostramos columnas categóricas reconocidas por CatBoost
print("Columnas categóricas:", cat_features)


# In[82]:


#separamos conjunto de entrenamiento, validación y prueba.
X_cb_train, X_cb_temp, y_cb_train, y_cb_temp = train_test_split(features_cboost, target_cboost, test_size=0.4, random_state=12345)
X_cb_val, X_cb_test, y_cb_val, y_cb_test = train_test_split(X_cb_temp, y_cb_temp, test_size=0.5, random_state=12345)


# ### Entrenamiento de modelo

# In[83]:


# Entrenamiento del modelo
cboost = CatBoostClassifier(verbose=200, auto_class_weights='Balanced', random_state=12345)
cboost.fit(X_cb_train, y_cb_train, cat_features=cat_feature_indices)

# Evaluación del modelo
metrics_cboost = evaluate_model(cboost, X_cb_train, y_cb_train, X_cb_val, y_cb_val, X_cb_test, y_cb_test)


# En este primer entrenamiento de CatBoost, sin ajustes específicos de hiperparámetros, se observó un desempeño aceptable en términos de AUC-ROC tanto en validación como en prueba (0.92 y 0.91 respectivamente). 
# Sin embargo, el F1-score muestra una caída en el conjunto de prueba (0.75), lo que indica que el modelo tiene dificultades para equilibrar precisión y recall en la predicción de cancelaciones.

# #### Evaluación y optimización del modelo CatBoost
# 
# Al igual que con los modelos anteriores, realizaremos un análisis de importancia de características para identificar las variables más influyentes. 
# 
# A partir de esta información, entrenaremos nuevamente el modelo utilizando únicamente las variables más relevantes, con el objetivo de mejorar su rendimiento y reducir el posible sobreajuste observado anteriormente.

# In[84]:


# Obtener las importancias
importances_cboost = cboost.feature_importances_
feature_namescb = X_cb_train.columns
sorted_idx_cb = importances_cboost.argsort()

top_n = 15
top_idx_cb = sorted_idx_cb[-top_n:]

plt.figure(figsize=(8,6))
plt.barh(feature_namescb[sorted_idx_cb], importances_cboost[sorted_idx_cb], color='plum')
plt.xlabel("Importancia Catboost")
plt.title("Importancia de características Catboost")
plt.tight_layout()
plt.show()


# Selección de características más importantes para CatBoost
# 
# Siguiendo la misma lógica empleada en modelos anteriores, se identificaron las variables más influyentes en el rendimiento del modelo CatBoost. A partir del análisis de importancia de características, se observó una diferencia significativa después de la séptima variable, lo que justifica limitar el entrenamiento a este subconjunto.
# 
# Las variables seleccionadas fueron:
# 
# - `seniority_days`
# - `total_charges`
# - `type`
# - `monthly_charges`
# - `internet_service`
# - `payment_method`
# - `dependents` 
# 
# Estas variables combinan atributos numéricos y categóricos, permitiendo aprovechar las capacidades nativas de CatBoost para manejar distintos tipos de datos sin necesidad de transformaciones adicionales.
# 
# A continuación, se procederá a entrenar el modelo con estas características y explorar la optimización de sus hiperparámetros.

# In[85]:


cbfeatures_top7 = ['seniority_days','total_charges','type', 'monthly_charges','internet_service','payment_method','dependents']

cb_features_train_top7 = X_cb_train[cbfeatures_top7]
cb_features_val_top7 = X_cb_val[cbfeatures_top7]
cb_features_test_top7 = X_cb_test[cbfeatures_top7]


# In[86]:


cat_feature_indices_top7 = [
    idx for idx, col in enumerate(cbfeatures_top7) if col in cat_features
]

# Entrenamiento del modelo con solo las 7 variables más importantes
cboost_top7 = CatBoostClassifier(verbose=200, auto_class_weights='Balanced', random_state=12345)

# Entrenar con el subconjunto reducido
cboost_top7.fit(cb_features_train_top7, y_cb_train, cat_features=cat_feature_indices_top7)

# Evaluación del modelo reducido
metrics_cboost_top7 = evaluate_model(
    cboost_top7,
    cb_features_train_top7, y_cb_train,
    cb_features_val_top7, y_cb_val,
    cb_features_test_top7, y_cb_test
)


# Tras comparar el rendimiento del modelo CatBoost utilizando **todas las variables** frente a una versión optimizada con las **7 características más importantes**, se observó que ambos enfoques ofrecen métricas sólidas y consistentes.
# 
# Aunque el modelo con todas las variables presentó un rendimiento ligeramente superior en entrenamiento y validación, la versión con solo 7 variables mantuvo un **AUC-ROC competitivo** en el conjunto de prueba, con un **F1-score y Accuracy igualmente elevados**. Además, esta versión reduce significativamente la complejidad del modelo.
# 
# >  **Decisión final:**  
# Se opta por continuar con el modelo entrenado con las **7 variables más relevantes**, priorizando la **simplicidad**, **eficiencia computacional** y **mejor capacidad de generalización**, sin sacrificar calidad en las predicciones.
# 
# A continuación, se procederá con el ajuste de hiperparámetros sobre este conjunto reducido para explorar mejoras adicionales en el rendimiento del modelo.
# 

# In[87]:


depths_cb = [4, 6, 8, 10]
learning_rate_valuescb = [0.01, 0.05, 0.1]
iterations_values_cb = [200, 500, 800]

best_auc_cb = 0
best_model_cb = None
best_params_cb = {}

# Bucle de búsqueda
for depth in depths_cb:
    for lr in learning_rate_valuescb:
        for iterat in iterations_values_cb:
            cb_hipertop7 = CatBoostClassifier(
                depth=depth,
                learning_rate=lr,
                iterations=iterat,
                auto_class_weights='Balanced',
                verbose=0,
                random_state=12345
            )
            
            cb_hipertop7.fit(cb_features_train_top7, y_cb_train, cat_features=cat_feature_indices_top7)
            y_val_pred_cbhip = cb_hipertop7.predict(cb_features_val_top7)
            y_val_proba_cbhip = cb_hipertop7.predict_proba(cb_features_val_top7)[:, 1]
            
            auc_cb = roc_auc_score(y_cb_val, y_val_proba_cbhip)
            f1_cb = f1_score(y_cb_val, y_val_pred_cbhip)
            
            if auc_cb > best_auc_cb:
                best_auc_cb = auc_cb
                best_model_cb = cb_hipertop7
                best_params_cb = {
                    'depth': depth,
                    'learning_rate': lr,
                    'iterations': iterat,
                    'AUC': auc_cb,
                    'F1': f1_cb
                }

print("\n Mejor combinación encontrada en CatBoost:")
for key, value in best_params_cb.items():
    print(f"{key}: {value}")

# Evaluar el mejor modelo completamente
metrics_cb_top7hip = evaluate_model(
    best_model_cb,
    cb_features_train_top7, y_cb_train,
    cb_features_val_top7, y_cb_val,
    cb_features_test_top7, y_cb_test
)


# Tras ajustar los hiperparámetros del modelo CatBoost utilizando como base las 7 variables más influyentes, se logró una mejora en el rendimiento. La combinación óptima (depth=4, learning_rate=0.1, iterations=800) permitió obtener métricas destacadas en los tres conjuntos.
# 
# En comparación con la versión anterior (sin ajuste de hiperparámetros), el AUC-ROC en el conjunto de prueba pasó de 0.9135 a 0.9320 y el F1-score de 0.7409 a 0.7917, demostrando una mejora tanto en la calidad de clasificación como en la estabilidad del modelo.
# 
# El modelo no solo alcanzó un AUC-ROC superior al 0.93 en validación y prueba, sino que también mantuvo buenos niveles de precisión y balance entre clases, según el F1-score. Esto refuerza la idea de que una selección adecuada de variables, puede mejorar la eficiencia del modelo sin comprometer su desempeño.

# # Selección del mejor modelo y validaciones finales
# 
# | Modelo               | Accuracy (Test) | F1-score (Test) | AUC-ROC (Test) |
# |----------------------|------------------|------------------|----------------|
# | Logistic Regression  | 0.7928           | 0.6011           | 0.8273         |
# | DecisionTreeClassifier| 0.7777          | 0.6959           | 0.8637         |
# | Random Forest         | 0.8382           | 0.6743           | 0.8918         |
# | LightGBM       | **0.9092**     | **0.8195**           | **0.9480**         |
# | CatBoost       | 0.8857       | 0.7917       | 0.9320     |
# 
# > *El modelo LightGBM, entrenado con las 7 variables más importantes y tras una búsqueda de hiperparámetros, obtuvo el mejor rendimiento general en todos los indicadores.*

# ## Matriz de Confusión del Modelo LightGBM (Top 7 variables)
# 
# A continuación se muestra la matriz de confusión del modelo LightGBM ajustado, utilizando únicamente las 7 variables más importantes. Esta visualización nos permite observar con mayor claridad cómo el modelo clasifica correctamente o incorrectamente a los clientes que cancelan (clase positiva) y a los que no cancelan (clase negativa).
# 
# El objetivo es verificar que el modelo no solo tenga buenas métricas generales, sino que también mantenga un buen equilibrio entre los falsos positivos y los falsos negativos, fundamentales en problemas de retención de clientes.

# In[98]:


# Generar predicciones del mejor modelo (LightGBM)
y_pred_lgbm = best_model_lgbm.predict(features_test_top7)

# Crear matriz de confusión
cm_lgbm = confusion_matrix(target_test, y_pred_lgbm)

# Visualizar
disp = ConfusionMatrixDisplay(confusion_matrix=cm_lgbm, display_labels=best_model_lgbm.classes_)
disp.plot(cmap='BuPu', values_format='d')
plt.title("Matriz de Confusión - LightGBM (Top 7 variables)")
plt.show()


# El modelo LightGBM con las 7 variables más importantes muestra un buen desempeño en la clasificación:
# 
# - **Verdaderos Negativos (TN):** 743 clientes fueron correctamente identificados como no canceladores.
# - **Falsos Positivos (FP):** Solo 21 clientes fueron incorrectamente clasificados como canceladores cuando en realidad no lo eran.
# - **Falsos Negativos (FN):** 75 clientes que sí cancelaron fueron clasificados erróneamente como no canceladores.
# - **Verdaderos Positivos (TP):** 218 clientes fueron correctamente clasificados como canceladores.
# 
# **Conclusión:**  
# El modelo logra un buen equilibrio entre sensibilidad y especificidad. Aunque aún hay falsos negativos (clientes que cancelan pero no fueron detectados), el número es razonablemente bajo considerando que estamos utilizando solo 7 variables. Esto sugiere que el modelo es eficiente y suficientemente preciso para tareas de predicción en escenarios reales.

# ## Validación cruzada del modelo LightGBM (Top 7 variables)
# 
# Para confirmar la estabilidad y capacidad de generalización del modelo LightGBM entrenado con las 7 variables más importantes, se aplica una validación cruzada estratificada con 5 particiones.
# 
# Esta técnica permite evaluar el rendimiento del modelo en distintos subconjuntos del conjunto de entrenamiento, utilizando como métrica principal el AUC-ROC. El objetivo es asegurar que los buenos resultados obtenidos no dependen únicamente de una partición específica de los datos.

# In[99]:


# Validación cruzada con 5 particiones, evaluando AUC-ROC
cv_scores = cross_val_score(
    best_model_lgbm,
    features_train_top7,
    target_train,
    cv=5,
    scoring='roc_auc'
)

print("AUC-ROC por fold:", cv_scores)
print("AUC-ROC promedio:", cv_scores.mean())


# Los resultados muestran un AUC-ROC promedio de 0.933, lo que respalda la solidez del modelo seleccionado y confirma que mantiene un rendimiento alto y estable ante diferentes divisiones del conjunto de datos.

# In[118]:


# Obtener probabilidades del modelo
y_proba_lgbm = best_model_lgbm.predict_proba(features_test_top7)[:, 1]

# Calcular errores absolutos
errores_abs = abs(target_test - y_proba_lgbm)

# Crear DataFrame para graficar
df_errores = pd.DataFrame({
    'Error absoluto': errores_abs,
    'Clase real': target_test.values
})

# Visualización con seaborn
plt.figure(figsize=(10, 6))
sns.histplot(data=df_errores, x='Error absoluto', hue='Clase real', bins=30, kde=True, palette='Set1', element="step", stat='count')
plt.title('Distribución de errores absolutos por clase real')
plt.xlabel('Error absoluto (|real - probabilidad|)')
plt.ylabel('Frecuencia')
plt.grid(True)
plt.tight_layout()
plt.show()


# La distribución de errores absolutos muestra que, en general, el modelo comete errores pequeños en la mayoría de las predicciones, especialmente en la clase 0 (clientes que no cancelan). Esto es esperable dada la distribución de clases en el dataset y el buen desempeño global del modelo.
# 
# Sin embargo, también se observa que la clase 1 (clientes que cancelan) presenta una dispersión ligeramente mayor en los errores. Esto indica que, aunque el modelo es estable, aún podría beneficiarse de ajustes finos o estrategias adicionales (por ejemplo, focal loss o técnicas de muestreo) si se desea mejorar específicamente la predicción de clientes propensos a cancelar.

# # Conclusion
# Tras evaluar el rendimiento de todos los modelos candidatos mediante métricas clave como AUC-ROC, Accuracy y F1-score, y confirmar su estabilidad mediante la matriz de confusión, LightGBM entrenado con las 7 variables más importantes ha demostrado ser el modelo más robusto y generalizable.
# 
# La matriz de confusión final refuerza esta decisión, mostrando una excelente capacidad para distinguir entre clientes que cancelan y los que no. 
# 
# Por tanto, este modelo se selecciona como el definitivo para el despliegue o siguientes etapas del proyecto.
