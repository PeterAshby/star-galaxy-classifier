import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os

def eda(df, figurepath):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # A1. Color-Color: g-r vs r-i
    sns.scatterplot(data=df, x='psfMag_(g-r)', y='psfMag_(r-i)', hue='class',
                    palette={'Star': 'orange', 'Galaxy': 'blue'}, alpha=0.7, ax=axes[0])
    axes[0].set_title('g-r vs r-i')
    axes[0].set_xlabel('g - r')
    axes[0].set_ylabel('r - i')
    axes[0].grid(True)

    # A2. Color-Color: u-g vs g-r
    sns.scatterplot(data=df, x='psfMag_(u-g)', y='psfMag_(g-r)', hue='class',
                    palette={'Star': 'orange', 'Galaxy': 'blue'}, alpha=0.7, ax=axes[1])
    axes[1].set_title('u-g vs g-r')
    axes[1].set_xlabel('u - g')
    axes[1].set_ylabel('g - r')
    axes[1].grid(True)

    # C. Class Distribution
    sns.countplot(x='class', data=df, palette={'Star': 'orange', 'Galaxy': 'blue'},
                  hue='class', legend=False, ax=axes[2])
    axes[3].set_title('Class Distribution')
    axes[3].set_xlabel('Class')
    axes[3].set_ylabel('Count')

    # B. Magnitude Distribution (r-band)
    sns.histplot(data=df, x='psfMag_r', hue='class', kde=True, bins=50,
                 palette={'Star': 'orange', 'Galaxy': 'blue'}, stat='count', alpha=0.5, ax=axes[3])
    axes[2].set_title('r-band PSF Magnitude')
    axes[2].set_xlabel('PSF Magnitude')
    axes[2].set_ylabel('Count')

    # D. Concentration Index
    sns.histplot(data=df, x='concentration_r', hue='type', kde=True, bins=400,
                 palette={3: 'blue', 6: 'orange'}, element='step', ax=axes[4])
    axes[4].axvline(x=2.0, color='k', linestyle='--', label='Galaxy Cut')
    axes[4].set_title('Concentration Index (r-band)')
    axes[4].set_xlabel('r90 / r50')
    axes[4].set_ylabel('Count')
    axes[4].legend(title='SDSS Type')
    axes[4].grid(True)
    axes[4].set_xlim(0, 5)

    # E. PSF - Model Difference
    sns.histplot(data=df, x='psf-model', hue='type', kde=True, bins=200,
                 palette={3: 'blue', 6: 'orange'}, element='step', ax=axes[5])
    axes[5].axvline(x=0.145, color='k', linestyle='--', label='Galaxy Cut')
    axes[5].set_title('PSF - Model (r-band)')
    axes[5].set_xlabel('PSF - Model Mag')
    axes[5].set_ylabel('Count')
    axes[5].legend(title='SDSS Type')
    axes[5].grid(True)
    axes[5].set_xlim(-1, 2)

    plt.tight_layout()
    plt.savefig(figurepath)
def eda2(df, figurepath):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()

    # List of magnitude columns to plot
    magnitude_columns = [
        'psfMag_u', 'psfMag_g', 'psfMag_r', 'psfMag_i',
        'psfMag_z', 'modelMag_u', 'modelMag_g', 'modelMag_r',
        'modelMag_i', 'modelMag_z'
    ]

    # A. Plot histograms for each magnitude
    for i, col in enumerate(magnitude_columns):
        sns.histplot(data=df, x=col, hue='class', kde=True, bins=50,
                     palette={'Star': 'orange', 'Galaxy': 'blue'}, stat='count', alpha=0.5, ax=axes[i])
        axes[i].set_title(f'{col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.savefig(figurepath)
def plot_confusion_matrix(y_true, y_pred, ax=None, title='', labels=['Star', 'Galaxy']):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    save_path = os.path.abspath(os.path.join(__file__, '..', '..', '..', 'data', 'visualisations', f'{title}.png'))
    plt.savefig(save_path)
