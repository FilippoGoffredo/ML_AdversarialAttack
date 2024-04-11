from preprocessing_main import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tabulate import tabulate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df_copy = df_normalized.copy()
labels = df_normalized['result']
df_feature = df_copy.drop('result', axis=1)

pca = PCA(random_state=15)
pca.fit(df_feature)
explained_variance = pca.explained_variance_ratio_
cumul_exp_var = np.cumsum(explained_variance)
perc_cumul_exp_var = cumul_exp_var * 100

#Plot
plt.figure(figsize=(8, 5))
plt.plot(perc_cumul_exp_var, marker='o')
plt.axhline(y=95, color='r', linestyle='--', label='95% Explained Variance')
plt.xlabel('# Principal Components (PCs)')
plt.ylabel('Cumulative Explained Variance [%]')
plt.xticks(np.arange(1, 48, 2), np.arange(1, 48, 2))
plt.grid(True)
plt.title(f'35 PCs explain {round(perc_cumul_exp_var[35], 2)}% of $\sigma^2$')
plt.legend()
plt.tight_layout()
plt.show()

#initialize the PCA with the best number of components
threshold = 95
n_components_threshold = np.argmax(perc_cumul_exp_var >= threshold) + 1

pca = PCA(n_components=n_components_threshold, random_state=15)
pca.fit(df_feature)
pca_result = pca.transform(df_feature)

# lets creating the reduced dataset
df_pca = pd.DataFrame(pca_result)
df_pca['labels'] = labels
num_components = df_pca.shape[1] - 1
new_column_names = [f'Component {i}' for i in range(1, num_components + 1)]
column_name_mapping = dict(zip(df_pca.columns[:-1], new_column_names))
df_pca.rename(columns=column_name_mapping, inplace=True)

print(tabulate(df_pca.head(), headers='keys', tablefmt='pretty'))


#Plotting t-SNE

#Defining learning rates (epsilon), perplexity and the number of iterations
learning_rates = [0.1, 0.2, 0.3]
perplexities = [5, 30, 50]
n_iterations = [250, 500, 1000]

# Iterating over learning rates, perplexities and iterations, finding the stable configuration
for lr in learning_rates:
    for perplexity in perplexities:
        for n_iter in n_iterations:
                model = TSNE(n_components=2, learning_rate=lr, perplexity=perplexity, n_iter=n_iter,
                             random_state=15)
                tsne_data = model.fit_transform(df_feature)

                tsne_df = pd.DataFrame(data=np.column_stack((tsne_data, labels)), columns=["Dim_1", "Dim_2", "label"])

                # Convert label column to integer
                tsne_df['label'] = tsne_df['label'].astype(int)

                # Plot the data with a unique label for each combination of lr, perplexity, and iteration
                label = f"LR={lr}_Perp={perplexity}_Iter={n_iter}"
                sns.FacetGrid(tsne_df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend(title=label)
                plt.title(f"TSNE Visualization - {label}")
                plt.show()






