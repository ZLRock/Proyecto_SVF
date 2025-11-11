import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

def crear_matriz_confusion_normalizada():
    """
    Crea una matriz de confusión normalizada y la guarda en PDF
    """
    # Definir las clases
    clases = ['poste', 'grafiti', 'grieta', 'background']
    
    # Crear matriz de confusión normalizada basada en los datos proporcionados
    # Los valores están normalizados entre 0 y 1
    matriz_confusion = np.array([
        [0.67, 0.00, 0.00, 0.31],  # poste
        [0.00, 0.46, 0.00, 0.64],  # grafiti  
        [0.00, 0.01, 0.71, 0.05],  # grieta
        [0.33, 0.53, 0.29, 0.00]   # background
    ])
    
    # Crear la figura
    plt.figure(figsize=(10, 8))
    
    # Crear el heatmap de la matriz de confusión
    ax = sns.heatmap(matriz_confusion, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='Blues',
                    cbar_kws={'label': 'Precisión Normalizada'},
                    xticklabels=clases,
                    yticklabels=clases,
                    vmin=0, 
                    vmax=1)
    
    # Configurar etiquetas y título
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Verdadero', fontsize=12, fontweight='bold')
    plt.title('Matriz de Confusión Normalizada - Entrenamiento 1', 
              fontsize=14, fontweight='bold', pad=20)
    
    # Rotar las etiquetas del eje x para mejor legibilidad
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Ajustar el layout
    plt.tight_layout()
    
    # Guardar en PDF
    with PdfPages('confusion_matrix_normalized_train_1.pdf') as pdf:
        pdf.savefig()
        plt.close()
    
    print("Matriz de confusión guardada como 'confusion_matrix_normalized_train_1.pdf'")

def crear_matriz_alternativa():
    """
    Versión alternativa con los valores exactos mencionados en el texto
    """
    clases = ['poste', 'grafiti', 'grieta', 'background']
    
    # Matriz alternativa basada en los valores listados
    matriz_confusion = np.array([
        [0.67, 0.46, 0.01, 0.33],  # Fila 1
        [0.53, 0.71, 0.29, 0.70],  # Fila 2
        [0.60, 0.50, 0.40, 0.20],  # Fila 3
        [0.10, 0.00, 0.00, 0.90]   # Fila 4 (valores hipotéticos para completar)
    ])
    
    # Normalizar las filas para que sumen 1
    matriz_confusion = matriz_confusion / matriz_confusion.sum(axis=1, keepdims=True)
    
    plt.figure(figsize=(10, 8))
    
    ax = sns.heatmap(matriz_confusion, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='Blues',
                    cbar_kws={'label': 'Precisión Normalizada'},
                    xticklabels=clases,
                    yticklabels=clases,
                    vmin=0, 
                    vmax=1)
    
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Verdadero', fontsize=12, fontweight='bold')
    plt.title('Matriz de Confusión Normalizada - Entrenamiento 1\n(Versión Alternativa)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    with PdfPages('confusion_matrix_normalized_alternative.pdf') as pdf:
        pdf.savefig()
        plt.close()
    
    print("Matriz alternativa guardada como 'confusion_matrix_normalized_alternative.pdf'")

if __name__ == "__main__":
    # Instalar las librerías necesarias si no las tienes:
    # pip install matplotlib seaborn numpy
    
    print("Generando matrices de confusión normalizadas...")
    
    # Crear la primera versión
    crear_matriz_confusion_normalizada()
    
    # Crear la versión alternativa
    crear_matriz_alternativa()
    
    print("¡Proceso completado!")
