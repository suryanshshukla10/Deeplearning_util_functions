def plot_label_distribution(df):
    """
    Plots the distribution of positive labels per disease.
    
    Args:
        df (pd.DataFrame):
    """
    label_counts = df.drop(columns='Patient_ID').sum().sort_values()
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='barh', color='skyblue')
    plt.title('Label Distribution per Disease', fontsize=16)
    plt.xlabel('Number of Patients with Disease')
    plt.ylabel('Disease')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def compute_class_freqs(labels):
    N = labels.shape[0]
    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies
    
    return positive_frequencies, negative_frequencies


def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    def weighted_loss(y_true, y_pred):
        loss = 0.0
        for i in range(len(pos_weights)):
            loss += K.mean(
                - pos_weights[i] * y_true[:, i] * K.log(y_pred[:, i] + epsilon)
                - neg_weights[i] * (1 - y_true[:, i]) * K.log(1 - y_pred[:, i] + epsilon)
            )
        return loss
    return weighted_loss
