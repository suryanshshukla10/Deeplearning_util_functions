def plot_class_distribution(labels, class_names=None, title="Class Distribution", rotation=90):
    class_indices = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
    unique_classes, counts = np.unique(class_indices, return_counts=True)
    class_names = class_names or [str(cls) for cls in unique_classes]

    plt.figure(figsize=(8, 6))
    plt.bar(class_names, counts)
    plt.xticks(rotation=rotation)
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel("Class")
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
