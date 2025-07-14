def true_negatives(y, pred, th=0.5):
    thresholded_preds = pred >= th
    TN = np.sum((y == 0) & (thresholded_preds == 0))
    return TN

def false_positives(y, pred, th=0.5):
    thresholded_preds = pred >= th
    FP = np.sum((y == 0) & (thresholded_preds == 1))
    return FP

def false_negatives(y, pred, th=0.5):
    thresholded_preds = pred >= th
    FN = np.sum((y == 1) & (thresholded_preds == 0))
    return FN

def get_accuracy(y, pred, th=0.5):
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def get_prevalence(y):
    prevalence = np.mean(y)
    return prevalence
 

def get_sensitivity(y, pred, th=0.5):
    TP = true_positives(y, pred, th)
    FN = false_negatives(y, pred, th)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    return sensitivity

def get_specificity(y, pred, th=0.5):
    TN = true_negatives(y, pred, th)
    FP = false_positives(y, pred, th)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    return specificity


def get_ppv(y, pred, th=0.5):
    TP = true_positives(y, pred, th)
    FP = false_positives(y, pred, th)
    PPV = TP / (TP + FP) if (TP + FP) > 0 else 0
    return PPV

def get_npv(y, pred, th=0.5):
    TN = true_negatives(y, pred, th)
    FN = false_negatives(y, pred, th)
    NPV = TN / (TN + FN) if (TN + FN) > 0 else 0
    return NPV

def single_class_dice_coefficient(y_true, y_pred, axis=(0, 1, 2), epsilon=0.00001):
    """
    Compute dice coefficient for single class.
    """
    dice_numerator = 2 * K.sum(y_true * y_pred, axis=axis)
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    dice_coefficient = (dice_numerator + epsilon) / (dice_denominator + epsilon)
    return dice_coefficient


def dice_coefficient(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    """
    Compute mean dice coefficient over all abnormality classes.
    """
    dice_numerator = 2 * K.sum(y_true * y_pred, axis=axis)
    dice_denominator = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis)
    dice_coefficient = K.mean((dice_numerator + epsilon) / (dice_denominator + epsilon))
    return dice_coefficient

def soft_dice_loss(y_true, y_pred, axis=(1, 2, 3), epsilon=0.00001):
    """
    Compute mean soft dice loss over all abnormality classes.
    """
    dice_numerator = 2 * K.sum(y_true * y_pred, axis=axis)
    dice_denominator = K.sum(K.square(y_true), axis=axis) + K.sum(K.square(y_pred), axis=axis)
    dice_loss = 1 - K.mean((dice_numerator + epsilon) / (dice_denominator + epsilon))
    return dice_loss


def compute_class_sens_spec(pred, label, class_num):
    """
    Compute sensitivity and specificity for a given class.
    """
    class_pred = pred[class_num]
    class_label = label[class_num]

    tp = np.sum((class_pred == 1) & (class_label == 1))
    tn = np.sum((class_pred == 0) & (class_label == 0))
    fp = np.sum((class_pred == 1) & (class_label == 0))
    fn = np.sum((class_pred == 0) & (class_label == 1))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity
