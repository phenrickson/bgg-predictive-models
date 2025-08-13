import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

def plot_embeddings(df, 
                    embedding_cols=None, 
                    category_col=None, 
                    title='Embedding Visualization', 
                    filename='embeddings_tsne.png'):
    """
    Visualize high-dimensional embeddings using t-SNE.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing embeddings and optional categorical data
    embedding_cols : list of str, optional
        List of column names for embedding dimensions. 
        If None, assumes all columns starting with 'text__embed_'
    category_col : str, optional
        Column name for categorical coloring
    title : str, optional
        Title for the plot
    filename : str, optional
        Filename to save the plot
    
    Returns:
    --------
    tsne_result : numpy.ndarray
        The 2D t-SNE embedding coordinates
    """
    # Determine embedding columns if not specified
    if embedding_cols is None:
        embedding_cols = [col for col in df.columns if col.startswith('text__embed_')]
    
    # Extract embeddings
    embeddings = df[embedding_cols].values
    
    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_result = tsne.fit_transform(embeddings)
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    # If category column is provided, use it for coloring
    if category_col and category_col in df.columns:
        labels = df[category_col]
        unique_labels = np.unique(labels)
        
        # Use a colormap to distinguish categories
        cmap = plt.cm.get_cmap('tab20')
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                tsne_result[mask, 0], 
                tsne_result[mask, 1], 
                label=label, 
                color=cmap(i / len(unique_labels)),
                alpha=0.7
            )
        
        # Adjust legend
        plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'figures/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    
    return tsne_result

# Example usage (commented out)
# plot_embeddings(train_pandas, category_col='primary_category')
    else:
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.7)
    
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'figures/{filename}', bbox_inches='tight', dpi=300)
    plt.close()
    
    return result

# Example usage (commented out)
# plot_embeddings(train_pandas, category_col='primary_category')
