import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import pandas as pd
import numpy as np

def plot_roc_curve(y_true, y_pred):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curve ROC(Receiver Operating Characteristic)')
    plt.legend(loc="lower right")
    plt.show()



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
    
    # Ajustar los límites del eje y
    y_limit_inf = df["Score"].min()
    ax.set_ylim(y_limit_inf, 1)

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

# Ejemplo de uso
accuracy = [0.95, 0.92]  # Ejemplo de accuracy para cada combinación
precision = [0.90, 0.85]  # Ejemplo de precision para cada combinación
recall = [0.95, 0.92]  # Ejemplo de recall para cada combinación
f1 = [0.92, 0.88]  # Ejemplo de f1 para cada combinación

labels = ["Resnet", "Facenet"]  # Ejemplo de etiquetas para cada combinación
plot_metrics(accuracy, precision, recall, f1, labels)