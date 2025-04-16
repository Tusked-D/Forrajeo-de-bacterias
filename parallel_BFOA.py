from copy import copy
from multiprocessing import Manager
import time
from bacteria import bacteria           
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastaReader import fastaReader      
import copy
import random

def run_algorithm(numeroDeBacterias, iteraciones, tumbo, dAttr, wAttr, hRep, wRep, maxSwim=5, stepSize=1):
   
    # Se leen las secuencias y nombres desde el archivo FASTA
    reader = fastaReader()  # Asegúrate de tener el módulo implementado
    secuencias = reader.seqs
    names = reader.names

    # Convertir las secuencias en listas de caracteres
    for i in range(len(secuencias)):
        secuencias[i] = list(secuencias[i])

    globalNFE = 0  # Contador global de evaluaciones
    manager = Manager()
    numSec = len(secuencias)
    poblacion = manager.list(range(numeroDeBacterias))
    names = manager.list(names)

    # Función para inicializar la población con copias independientes de cada secuencia
    def poblacionInicial():
        for i in range(numeroDeBacterias):
            bacterium = []
            for j in range(numSec):
                bacterium.append(copy.deepcopy(secuencias[j]))
            poblacion[i] = list(bacterium)

    # Instancia del operador bacteriano (la clase bacteria que incorpora las mejoras)
    operadorBacterial = bacteria(numeroDeBacterias)
    veryBest = [None, None, None]  # [índice, fitness, secuencias]
    start_time = time.time()

    poblacionInicial()

    for it in range(iteraciones):
        # Se aplica el operador tumbó (inserción de gaps aleatorios)
        operadorBacterial.tumbo(numSec, poblacion, tumbo)
        # Se alínean (cuadran) las secuencias de cada bacteria
        operadorBacterial.cuadra(numSec, poblacion)
        # **Aquí se llama al nuevo operador de quimiotaxis con adaptación dinámica y elitismo**
        operadorBacterial.quimiotaxis_con_adaptacion(numSec, poblacion)
        # Luego se generan la gran lista de pares a partir de la cual se evaluará el score
        operadorBacterial.creaGranListaPares(poblacion)
        # Se evalúa el score usando la función de Blosum (u otro criterio de evaluación)
        operadorBacterial.evaluaBlosum()
        # Se calculan las tablas de atracción y repulsión
        operadorBacterial.creaTablasAtractRepel(poblacion, dAttr, wAttr, hRep, wRep)
        # Se construye la tabla de interacción y se calcula el fitness global
        operadorBacterial.creaTablaInteraction()
        operadorBacterial.creaTablaFitness()
        globalNFE += operadorBacterial.getNFE()
        bestIdx, bestFitness = operadorBacterial.obtieneBest(globalNFE)

        # Actualizar la mejor solución encontrada globalmente
        if (veryBest[0] is None) or (bestFitness > veryBest[1]):
            veryBest[0] = bestIdx
            veryBest[1] = bestFitness
            veryBest[2] = copy.deepcopy(poblacion[bestIdx])


        # Se reemplaza la bacteria peor por una copia de la mejor
        operadorBacterial.replaceWorst(poblacion, veryBest[0])
        # Se reinician las listas internas (para la siguiente iteración de evaluación)
        operadorBacterial.resetListas(numeroDeBacterias)

    execution_time = time.time() - start_time

    return {
        "fitness": veryBest[1],
        "time": execution_time,
        "interaction": operadorBacterial.tablaInteraction[veryBest[0]],
        "blosum_score": operadorBacterial.blosumScore[veryBest[0]]
    }

def performance_analysis(runs=5):
    """
    Ejecuta el algoritmo 'runs' veces, guarda los resultados en un DataFrame y genera un CSV.
    """
    results = []
    parameters = {
        "numeroDeBacterias": 15,
        "iteraciones": 10,
        "tumbo": 50,
        "dAttr": 0.1,
        "wAttr": 0.002,
        "hRep": 0.1,
        "wRep": 0.001,
        "maxSwim": 5,
        "stepSize": 1
    }

    for run in range(runs):
        print(f"Ejecutando corrida {run + 1}/{runs}")
        result = run_algorithm(**parameters)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv("performance_results_2.csv", index=False)
    return df

def plot_results(df):
    """
    Genera gráficos que muestran las relaciones entre fitness, tiempo, interacción y BLOSUM score.
    """
    plt.figure(figsize=(12, 8))

    # Fitness vs Tiempo de Ejecución
    plt.subplot(2, 2, 1)
    plt.scatter(df['time'], df['fitness'])
    plt.xlabel('Tiempo de Ejecución (s)')
    plt.ylabel('Fitness')
    plt.title('Fitness vs Tiempo de Ejecución')

    # Fitness vs Interacción
    plt.subplot(2, 2, 2)
    plt.scatter(df['interaction'], df['fitness'])
    plt.xlabel('Interacción')
    plt.ylabel('Fitness')
    plt.title('Fitness vs Interacción')

    # Fitness vs BLOSUM Score
    plt.subplot(2, 2, 3)
    plt.scatter(df['blosum_score'], df['fitness'])
    plt.xlabel('BLOSUM Score')
    plt.ylabel('Fitness')
    plt.title('Fitness vs BLOSUM Score')

    # Distribución de Fitness
    plt.subplot(2, 2, 4)
    plt.hist(df['fitness'], bins=20)
    plt.xlabel('Fitness')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Fitness')

    plt.tight_layout()
    plt.savefig("performance_analysis.png")
    plt.show()

if __name__ == "__main__":
    # Ejecuta el análisis de desempeño con 5 corridas (ajusta runs si lo deseas)
    df = performance_analysis(runs=30)
    # Genera y muestra los gráficos
    plot_results(df)
