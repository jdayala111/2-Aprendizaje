import random
from arboles_numericos import entrena_arbol

__author__ = "Ana Sofía Matti Ríos y Jesus David Ayala Morales"
__date__ = "Febrero 2025"

# 1) Separar los datos en subconjuntos con selección aleatoria con repetición (para M subconjuntos).
def generar_subconjuntos(datos, n):
    return random.choices(datos, k=n)

def entrena_bosque(datos, target, clase_default, M=None, max_profundidad=None, 
                   acc_nodo=1.0, min_ejemplos=0, variables_seleccionadas=None):
    
    if M is None:
        M = 10 # con 10 árboles en un bosque es suficiente para mejorar la generalización sin necesidad de una cantidad muy grande

    bosque = [] # ¡ para guardar los árboles ! q no se te olvide

    todos_los_atributos = list(datos[0].keys())
    todos_los_atributos.remove(target) # para quitar la clase a predecir


    for i in range (M):
        subconjunto = generar_subconjuntos(datos, len(datos)) # se genera el subconj aleatorio :p

        lim_variables = None
        if lim_variables is None:
            lim_variables = max(1, len(todos_los_atributos) // 2) # ** Por cada subconjunto, entrenar un árbol con un número limitado de variables en cada nodo

        if variables_seleccionadas is None:
            variables_seleccionadas = random.sample(todos_los_atributos, lim_variables)

        arbol = entrena_arbol(subconjunto, target, clase_default, max_profundidad,   # entrenamos el árbol :3
                              acc_nodo, min_ejemplos, variables_seleccionadas)
        bosque.append(arbol)

    return bosque

 #def predice_bosque(bosque, datos):
   # predicciones_arbol = [predice_arbol (arbol, datos) for arbol in bosque]
    #prediccion_instacia

