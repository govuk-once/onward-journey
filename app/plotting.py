import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from typing import List

def plot_uid_confusion_matrix(cm: np.ndarray, uid_labels: List[str], accuracy: float, title: str):
    """
    Plots the confusion matrix, returns the Matplotlib figure object.
    """
    # Create the figure and axes objects explicitly (recommended practice)
    fig, ax = plt.subplots(figsize=(10, 8)) 
    
    cm_df = pd.DataFrame(cm, index=uid_labels, columns=uid_labels)
    
    # Use ax=ax to tell seaborn where to draw the heatmap
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=.5, linecolor='black', ax=ax)
    
    ax.set_title(f"{title}\nOverall Accuracy: {accuracy:.2%}")
    ax.set_xlabel('Predicted UID Label', fontsize=12)
    ax.set_ylabel('Actual UID Label', fontsize=12)
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    
    fig.tight_layout()
    
    # Returns the figure object
    return fig