import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from typing import List

def plot_uid_confusion_matrix(cm_array, labels, accuracy, title="Confusion Matrix"):
    """
    Plots a confusion matrix using Service Names and ensures labels are formatted correctly.
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use a heatmap with integer formatting for counts
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=labels, yticklabels=labels, cbar=False)
    
    ax.set_title(f"{title}\nOverall Accuracy: {accuracy:.2%}", pad=20)
    ax.set_xlabel('Predicted Service', labelpad=15)
    ax.set_ylabel('True Service', labelpad=15)

    # Rotate tick labels to prevent "eating" into the plot
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    plt.yticks(rotation=0)

    # Adjust layout to make room for long labels
    plt.tight_layout()
    
    return fig