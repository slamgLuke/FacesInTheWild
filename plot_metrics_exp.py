import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_metrics(accuracy, precision, recall, f1, labels):
    # Extraer los resultados de los scores
    
    accuracy = accuracy
    precision = precision
    recall = recall
    f1 = f1

    # Crear un DataFrame para facilitar el uso de Seaborn
    num_combinaciones = len(accuracy)
    df = pd.DataFrame({
        "Metric": ["Accuracy"] * num_combinaciones + ["Precision"] * num_combinaciones + ["Recall"] * num_combinaciones + ["F1"] * num_combinaciones,
        "Score": np.concatenate([accuracy, precision, recall, f1]),
        "Labels": labels * 4,
    })

    # Plotear
    fig, ax = plt.subplots(figsize=(8, 6))

    # Barras
    sns.barplot(x="Labels", y="Score", hue="Metric", data=df, ax=ax, width=0.3)
    
    
    ax.set_ylim(0.5, 1)

    # Ajustar las etiquetas y el diseño del gráfico
    ax.set_xlabel("Labels", fontsize=14)
    ax.set_ylabel("Scores", fontsize=14)
    plt.xticks(rotation=75, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Añadir la leyenda
    ax.legend(title="Metric", title_fontsize='13', fontsize='12', framealpha=0.4, loc="upper right")
    
    # Líneas horizontales en el gráfico
    ax.grid(True, axis="y", ls="-", color="gray", alpha=0.4)
    
    plt.tight_layout()  # Ajustar automáticamente los márgenes de la figura
    plt.show()

# Resultados de experimentacion
triplet_nn =      [0.74, 0.79, 0.83, 0.81]
facenet_pytorch = [0.97, 0.98, 0.97, 0.99]
resnet50 =        [0.58, 0.58, 0.58, 0.58] 
resnet152 =       [0.69, 0.69, 0.69, 0.69]
alexnet =         [0.6,  0.6,   0.6, 0.6]

accuracy =  [0.74, 0.97, 0.58, 0.69, 0.6]   
precision = [0.79, 0.98, 0.58, 0.69, 0.6]  
recall =    [0.83, 0.97, 0.58, 0.69, 0.6] 
f1 =        [0.81, 0.99, 0.58, 0.69, 0.6]

labels = ["triplet_nn", "facenet_pytorch", "resnet50", "resnet152", "alexnet"]  # Ejemplo de etiquetas para cada combinación
plot_metrics(accuracy, precision, recall, f1, labels)