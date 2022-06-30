# **School dropout predictive model**

Using the database MEN ESTADÍSTICAS EN EDUCACIÓN EN PREESCOLAR, BÁSICA Y MEDIA POR MUNICIPIO provided by Datos abiertos, it is going to be build a predictive model to get the school dropout.


### **Imports**


```python
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
```

## **Database**


```python
df = pd.read_csv('MEN_ESTADISTICAS_EN_EDUCACION_EN_PREESCOLAR__B_SICA_Y_MEDIA_POR_MUNICIPIO.csv')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11219 entries, 0 to 11218
    Data columns (total 41 columns):
     #   Column                       Non-Null Count  Dtype  
    ---  ------                       --------------  -----  
     0   AÑO                          11219 non-null  int64  
     1   CÓDIGO_MUNICIPIO             11219 non-null  int64  
     2   MUNICIPIO                    11219 non-null  object 
     3   CÓDIGO_DEPARTAMENTO          11219 non-null  int64  
     4   DEPARTAMENTO                 11219 non-null  object 
     5   CÓDIGO_ETC                   11219 non-null  int64  
     6   ETC                          11219 non-null  object 
     7   POBLACIÓN_5_16               11213 non-null  float64
     8   TASA_MATRICULACIÓN_5_16      11104 non-null  float64
     9   COBERTURA_NETA               11108 non-null  float64
     10  COBERTURA_NETA_TRANSICIÓN    11167 non-null  float64
     11  COBERTURA_NETA_PRIMARIA      11128 non-null  float64
     12  COBERTURA_NETA_SECUNDARIA    11125 non-null  float64
     13  COBERTURA_NETA_MEDIA         11126 non-null  float64
     14  COBERTURA_BRUTA              11151 non-null  float64
     15  COBERTURA_BRUTA_TRANSICIÓN   11122 non-null  float64
     16  COBERTURA_BRUTA_PRIMARIA     11138 non-null  float64
     17  COBERTURA_BRUTA_SECUNDARIA   11131 non-null  float64
     18  COBERTURA_BRUTA_MEDIA        11092 non-null  float64
     19  TAMAÑO_PROMEDIO_DE_GRUPO     7572 non-null   float64
     20  SEDES_CONECTADAS_A_INTERNET  7768 non-null   float64
     21  DESERCIÓN                    11077 non-null  float64
     22  DESERCIÓN_TRANSICIÓN         10316 non-null  float64
     23  DESERCIÓN_PRIMARIA           10977 non-null  float64
     24  DESERCIÓN_SECUNDARIA         10949 non-null  float64
     25  DESERCIÓN_MEDIA              10485 non-null  float64
     26  APROBACIÓN                   11194 non-null  float64
     27  APROBACIÓN_TRANSICIÓN        11194 non-null  float64
     28  APROBACIÓN_PRIMARIA          11194 non-null  float64
     29  APROBACIÓN_SECUNDARIA        11165 non-null  float64
     30  APROBACIÓN_MEDIA             11118 non-null  float64
     31  REPROBACIÓN                  11133 non-null  float64
     32  REPROBACIÓN_TRANSICIÓN       11126 non-null  float64
     33  REPROBACIÓN_PRIMARIA         11122 non-null  float64
     34  REPROBACIÓN_SECUNDARIA       11113 non-null  float64
     35  REPROBACIÓN_MEDIA            11074 non-null  float64
     36  REPITENCIA                   11076 non-null  float64
     37  REPITENCIA_TRANSICIÓN        11060 non-null  float64
     38  REPITENCIA_PRIMARIA          11071 non-null  float64
     39  REPITENCIA_SECUNDARIA        11067 non-null  float64
     40  REPITENCIA_MEDIA             11080 non-null  float64
    dtypes: float64(34), int64(4), object(3)
    memory usage: 3.5+ MB
    


```python
df.columns
```




    Index(['AÑO', 'CÓDIGO_MUNICIPIO', 'MUNICIPIO', 'CÓDIGO_DEPARTAMENTO',
           'DEPARTAMENTO', 'CÓDIGO_ETC', 'ETC', 'POBLACIÓN_5_16',
           'TASA_MATRICULACIÓN_5_16', 'COBERTURA_NETA',
           'COBERTURA_NETA_TRANSICIÓN', 'COBERTURA_NETA_PRIMARIA',
           'COBERTURA_NETA_SECUNDARIA', 'COBERTURA_NETA_MEDIA', 'COBERTURA_BRUTA',
           'COBERTURA_BRUTA_TRANSICIÓN', 'COBERTURA_BRUTA_PRIMARIA',
           'COBERTURA_BRUTA_SECUNDARIA', 'COBERTURA_BRUTA_MEDIA',
           'TAMAÑO_PROMEDIO_DE_GRUPO', 'SEDES_CONECTADAS_A_INTERNET', 'DESERCIÓN',
           'DESERCIÓN_TRANSICIÓN', 'DESERCIÓN_PRIMARIA', 'DESERCIÓN_SECUNDARIA',
           'DESERCIÓN_MEDIA', 'APROBACIÓN', 'APROBACIÓN_TRANSICIÓN',
           'APROBACIÓN_PRIMARIA', 'APROBACIÓN_SECUNDARIA', 'APROBACIÓN_MEDIA',
           'REPROBACIÓN', 'REPROBACIÓN_TRANSICIÓN', 'REPROBACIÓN_PRIMARIA',
           'REPROBACIÓN_SECUNDARIA', 'REPROBACIÓN_MEDIA', 'REPITENCIA',
           'REPITENCIA_TRANSICIÓN', 'REPITENCIA_PRIMARIA', 'REPITENCIA_SECUNDARIA',
           'REPITENCIA_MEDIA'],
          dtype='object')




```python
df.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AÑO</th>
      <th>CÓDIGO_MUNICIPIO</th>
      <th>MUNICIPIO</th>
      <th>CÓDIGO_DEPARTAMENTO</th>
      <th>DEPARTAMENTO</th>
      <th>CÓDIGO_ETC</th>
      <th>ETC</th>
      <th>POBLACIÓN_5_16</th>
      <th>TASA_MATRICULACIÓN_5_16</th>
      <th>COBERTURA_NETA</th>
      <th>...</th>
      <th>REPROBACIÓN</th>
      <th>REPROBACIÓN_TRANSICIÓN</th>
      <th>REPROBACIÓN_PRIMARIA</th>
      <th>REPROBACIÓN_SECUNDARIA</th>
      <th>REPROBACIÓN_MEDIA</th>
      <th>REPITENCIA</th>
      <th>REPITENCIA_TRANSICIÓN</th>
      <th>REPITENCIA_PRIMARIA</th>
      <th>REPITENCIA_SECUNDARIA</th>
      <th>REPITENCIA_MEDIA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011</td>
      <td>5001</td>
      <td>Medellín</td>
      <td>5</td>
      <td>Antioquia</td>
      <td>3759</td>
      <td>Medellín</td>
      <td>386466.0</td>
      <td>108.73</td>
      <td>108.5</td>
      <td>...</td>
      <td>0.03</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.17</td>
      <td>4.57</td>
      <td>0.15</td>
      <td>3.26</td>
      <td>7.44</td>
      <td>2.95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011</td>
      <td>5002</td>
      <td>Abejorral</td>
      <td>5</td>
      <td>Antioquia</td>
      <td>3758</td>
      <td>Antioquia (ETC)</td>
      <td>4146.0</td>
      <td>97.81</td>
      <td>97.8</td>
      <td>...</td>
      <td>1.70</td>
      <td>0.00</td>
      <td>1.23</td>
      <td>2.96</td>
      <td>1.18</td>
      <td>0.89</td>
      <td>0.00</td>
      <td>0.85</td>
      <td>1.08</td>
      <td>1.23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011</td>
      <td>5004</td>
      <td>Abriaquí</td>
      <td>5</td>
      <td>Antioquia</td>
      <td>3758</td>
      <td>Antioquia (ETC)</td>
      <td>483.0</td>
      <td>88.61</td>
      <td>88.6</td>
      <td>...</td>
      <td>7.29</td>
      <td>0.00</td>
      <td>1.47</td>
      <td>14.66</td>
      <td>7.46</td>
      <td>1.69</td>
      <td>3.13</td>
      <td>1.47</td>
      <td>2.22</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011</td>
      <td>5021</td>
      <td>Alejandría</td>
      <td>5</td>
      <td>Antioquia</td>
      <td>3758</td>
      <td>Antioquia (ETC)</td>
      <td>702.0</td>
      <td>118.52</td>
      <td>118.5</td>
      <td>...</td>
      <td>3.58</td>
      <td>0.00</td>
      <td>2.16</td>
      <td>4.39</td>
      <td>8.04</td>
      <td>0.60</td>
      <td>0.00</td>
      <td>0.96</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011</td>
      <td>5030</td>
      <td>Amagá</td>
      <td>5</td>
      <td>Antioquia</td>
      <td>3758</td>
      <td>Antioquia (ETC)</td>
      <td>6631.0</td>
      <td>78.65</td>
      <td>78.7</td>
      <td>...</td>
      <td>8.99</td>
      <td>0.24</td>
      <td>6.73</td>
      <td>14.46</td>
      <td>7.45</td>
      <td>0.42</td>
      <td>0.00</td>
      <td>0.24</td>
      <td>0.91</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 41 columns</p>
</div>



## **Graphic analysis**


```python
#Fixing some data
df = df.replace({"DEPARTAMENTO": "Bogotá D.C."}, "Bogotá, D.C.")
df = df.replace({"DEPARTAMENTO": "Archipiélago de San Andrés. Providencia y Santa Catalina"}, "Archipiélago de San Andrés, Providencia y Santa Catalina")
df = df.replace({"DEPARTAMENTO": "Archipiélago de San Andrés, Providencia y Santa Catalina"}, "Islas")
```


```python
fig  = sns.pairplot(data = df)
```


    
![png](README_files/Model_9_0.png)
    



```python
fig, ax = plt.subplots(figsize=(35,35))       
fig1 = sns.heatmap(df.corr(),annot = True,linewidths=.5, cmap = 'Blues',ax = ax)
```


    
![png](README_files/Model_10_0.png)
    



```python
df_mod = df[['DEPARTAMENTO','APROBACIÓN_MEDIA','DESERCIÓN_MEDIA']]
fig1 = sns.heatmap(df_mod.corr(),annot = True, cmap = 'Blues')
```


    
![png](README_files/Model_11_0.png)
    



```python
f, ax = plt.subplots(figsize=(40, 40))
fig = sns.violinplot(data = df, x  ="DEPARTAMENTO", y = "DESERCIÓN_MEDIA",ax = ax)
```


    
![png](README_files/Model_12_0.png)
    


## **Filtering**


```python
cols = ['DEPARTAMENTO','DESERCIÓN_MEDIA','APROBACIÓN_MEDIA'];
df_model = df[cols]
df_model = df_model.dropna()
```

## **Getting dummy variables**


```python
dept = pd.get_dummies(df_model.DEPARTAMENTO, prefix='DEPARTAMENTO')
df_model = df_model.join(dept)
df_model.drop(['DEPARTAMENTO'], axis=1, inplace=True)
```

## **Creating output column**

It is going to be a classification model, so we are going to classify dropout in high dropout and low dropout. We are going to put high dropout when dropout is higher than 4 and low dropout when dropout is lower than 4.


```python
df_model['DESERCIÓN_MEDIA_CAT'] = df_model['DESERCIÓN_MEDIA'].map(lambda x: 1 if x>=4 else 0)
df_model = df_model.drop(['DESERCIÓN_MEDIA'], axis = 1)
```

## **Creating Model**


```python
# Defining input and output
X = df_model.drop(['DESERCIÓN_MEDIA_CAT'], axis = 1);
y = df_model.DESERCIÓN_MEDIA_CAT.copy();
```

## **Splitting data**


```python

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
```

## **Training model**


```python
model = LogisticRegression(max_iter=1000)
log_mod = model.fit(X_train, y_train)
```

## **Evaluating model**


```python
log_mod.score(X_val, y_val)
```




    0.7546148949713558



### **Cross-validation**


```python
succ = [];
for i in range(100):
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
  log_mod = LogisticRegression(max_iter=1000).fit(X_train, y_train);
  succ.append(log_mod.score(X_val, y_val));
sns.histplot(succ)
succ = np.array(succ);

```


    
![png](README_files/Model_28_0.png)
    



```python
np.mean(succ)
```




    0.7525238701464037




```python
np.var(succ)
```




    5.077357912611657e-05



# **Hypothesis**

The most rural municipalities have a higher dropout rate.


```python
#Database
df = pd.read_csv('MEN_ESTADISTICAS_EN_EDUCACION_EN_PREESCOLAR__B_SICA_Y_MEDIA_POR_MUNICIPIO.csv')
cols = ['DEPARTAMENTO','DESERCIÓN_MEDIA','APROBACIÓN_MEDIA'];
df_model = df[cols]
df_model = df_model.dropna()
#Fixing some data
df_model = df_model.replace({"DEPARTAMENTO": "Bogotá D.C."}, "Bogotá, D.C.")
df_model = df_model.replace({"DEPARTAMENTO": "Archipiélago de San Andrés. Providencia y Santa Catalina"}, "Archipiélago de San Andrés, Providencia y Santa Catalina")
df_model = df_model.replace({"DEPARTAMENTO": "Archipiélago de San Andrés, Providencia y Santa Catalina"}, "Islas")
# setting up the data
dept = pd.get_dummies(df_model.DEPARTAMENTO, prefix='DEPARTAMENTO')
df_model = df_model.join(dept)
df_model = df_model[ (df_model['DEPARTAMENTO'] == 'Guainía') | (df_model['DEPARTAMENTO'] == 'Vichada') | (df_model['DEPARTAMENTO'] == 'Vaupés') | (df_model['DEPARTAMENTO'] == 'Putumayo') | (df_model['DEPARTAMENTO'] == 'Bogotá, D.C.') | (df_model['DEPARTAMENTO'] == 'Islas') | (df_model['DEPARTAMENTO'] == 'Atlántico') | (df_model['DEPARTAMENTO'] == 'Caldas')]
df_model['Resultado'] = df_model['DEPARTAMENTO'].map(lambda x: 1 if (x=='Guainía' or x=='Vichada' or x=='Vaupés' or x=='Putumayo') else 0)
dept.head()
df_model.drop(['DEPARTAMENTO','DESERCIÓN_MEDIA'], axis=1, inplace=True)
X = df_model.drop(['Resultado'], axis = 1)
Y = df_model['Resultado']
y_pred = log_mod.predict(X)
```

## **Confusion matrix**


```python
sns.heatmap(confusion_matrix(Y, y_pred, labels=[1, 0])/len(Y), annot = True, cmap = 'Blues')
```




    <AxesSubplot:>




    
![png](README_files/Model_34_1.png)
    


## **Conclusion**

Based on the rurality data provided by the United Nations compared with the dropout rate and the predictive model, the hypothesis can be approved.
