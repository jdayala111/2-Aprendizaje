import utileria as ut
import bosque_aleatorio as ba
import os
import random

# Heart Disease Dataset
url = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
archivo = "datos/heart_disease.zip"
archivo_datos = "datos/processed.cleveland.data"
atributos = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
             'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

if not os.path.exists("datos"):
    os.makedirs("datos")
if not os.path.exists(archivo):
    ut.descarga_datos(url, archivo)
    ut.descomprime_zip(archivo)

datos = ut.lee_csv(
    archivo_datos,
    atributos=atributos,
    separador=","
)

for d in datos:
    for k in d.keys():
        if d[k] == "?":
            d[k] = None
    d['target'] = 1 if int(d['target']) > 0 else 0  # Convertimos target a binario
    for k in atributos[:-1]:  # Convertimos atributos a float
        if d[k] is not None:
            d[k] = float(d[k])

datos = [d for d in datos if None not in d.values()]

# Selecciona los atributos
target = 'target'
atributos.remove(target)

random.seed(42)
random.shuffle(datos)
N = int(0.8 * len(datos))
datos_entrenamiento = datos[:N]
datos_validacion = datos[N:]

num_arboles_list = [1, 5, 10, 20, 50]  # Número de árboles en el bosque
profundidades_list = [3, 5, 10]  # Profundidades de los árboles
variables_list = [5, 10, len(atributos) - 1]  


resultados = []
for M in num_arboles_list:
    for profundidad in profundidades_list:
        for num_vars in variables_list:
            bosque = ba.entrena_bosque(
                datos_entrenamiento, 
                target, 
                clase_default=0, 
                M=M, 
                max_profundidad=profundidad, 
                variables_seleccionadas=num_vars
            )
            pred_entrenamiento = ba.predice_bosque(bosque, datos_entrenamiento, tipo="clasificacion")
            pred_validacion = ba.predice_bosque(bosque, datos_validacion, tipo="clasificacion")

            error_entrenamiento = sum(1 for p, d in zip(pred_entrenamiento, datos_entrenamiento) if p != d[target]) / len(datos_entrenamiento)
            error_validacion = sum(1 for p, d in zip(pred_validacion, datos_validacion) if p != d[target]) / len(datos_validacion)

            resultados.append((M, profundidad, num_vars, error_entrenamiento, error_validacion))


print('M'.center(5) + 'Prof'.center(10) + 'Vars'.center(10) + 'Ein'.center(15) + 'Eout'.center(15))
print('-' * 50)
for M, profundidad, num_vars, error_entrenamiento, error_validacion in resultados:
    print(
        f'{M}'.center(5) 
        + f'{profundidad}'.center(10) 
        + f'{num_vars}'.center(10) 
        + f'{error_entrenamiento:.2f}'.center(15) 
        + f'{error_validacion:.2f}'.center(15)
    )
print('-' * 50 + '\n')

