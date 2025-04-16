import pandas as pd
import matplotlib.pyplot as plt

def compare_fitness_line(csv1, csv2):
    # Cargar los CSV en DataFrames
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)

    # Verificar que ambos DataFrames tengan la columna "fitness"
    if 'fitness' not in df1.columns or 'fitness' not in df2.columns:
        raise ValueError("Ambos CSV deben tener la columna 'fitness'")

    # Extraer la columna "fitness". Se usará el índice de cada DataFrame para el eje X.
    fitness1 = df1['fitness']
    fitness2 = df2['fitness']

    # Crear la gráfica lineal
    plt.figure(figsize=(10, 6))
    plt.plot(fitness1.index, fitness1, label=csv1, marker='o')
    plt.plot(fitness2.index, fitness2, label=csv2, marker='o')
    plt.title("Comparación de Fitness (gráfica lineal)")
    plt.xlabel("Índice / Iteración")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csv_file1 = "performance_results_1.csv"
    csv_file2 = "performance_results_2.csv"
    compare_fitness_line(csv_file1, csv_file2)
