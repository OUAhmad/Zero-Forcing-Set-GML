import tensorflow as tf
import numpy as np

def my_softmax_cross_entropy_updated_new(preds, labels, distance, selected_nodes, epsilon = 0.0, new_term = 0.5):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    x = labels[:, 1][labels[:, 1]>0]
    solution_size = tf.cast(tf.reduce_sum(x), tf.int32)
    selected_nds_sum = tf.reduce_sum(selected_nodes)
    pred_ones = preds[:, 1::2]
    loss_appended = []
    loss_appended_2 = []
    
    # for pred in tf.transpose(pred_ones):
    for i in range(pred_ones.shape[1]):
        values, indices = tf.math.top_k(preds[:,2*(i)+1], solution_size)
        loss_appended.append( tf.math.reduce_sum(tf.gather(tf.gather(distance, indices=indices, axis=1), indices=indices)) )
        loss_appended_2.append(tf.math.reduce_sum(tf.math.multiply(tf.cast(selected_nodes, tf.float32), preds[:,2*(i)+1])))
    loss_dist = tf.reduce_min(tf.reduce_mean(loss_appended))
    loss_soln = tf.reduce_min(tf.reduce_mean(loss_appended_2))
    solution_size_1 = tf.cast(solution_size, tf.float32)
    loss_dist = tf.cond(tf.math.is_nan(loss_dist), lambda: 100.0 * solution_size_1, lambda: loss_dist)
    # return (1- epsilon - new_term) * tf.reduce_mean(loss) + epsilon* 6*tf.reduce_mean(loss_dist)/tf.cast(solution_size, tf.float32)**3 + new_term * tf.math.abs(loss_soln)/tf.cast(selected_nds_sum, tf.float32)
    return (1- epsilon - new_term) * tf.reduce_mean(loss) + epsilon* 6*tf.reduce_mean(loss_dist)/tf.cast(solution_size, tf.float32)**3 + new_term * tf.math.abs(loss_soln)
    # return tf.math.reduce_sum(tf.math.multiply(tf.cast(selected_nodes, tf.float32), preds[:,2*(i)+1]))/tf.cast(solution_size, tf.float32)



def my_softmax_cross_entropy_updated(preds, labels, distance, epsilon = 0.5):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    x = labels[:, 1][labels[:, 1]>0]
    solution_size = tf.cast(tf.reduce_sum(x), tf.int32)
    pred_ones = preds[:, 1::2]
    loss_appended = []
    # for pred in tf.transpose(pred_ones):
    for i in range(pred_ones.shape[1]):
        values, indices = tf.math.top_k(preds[:,2*(i)+1], solution_size)
        loss_appended.append( tf.math.reduce_sum(tf.gather(tf.gather(distance, indices=indices, axis=1), indices=indices)) )
    loss_dist = tf.reduce_min(tf.reduce_mean(loss_appended))
    solution_size_1 = tf.cast(solution_size, tf.float32)
    loss_dist = tf.cond(tf.math.is_nan(loss_dist), lambda: 100.0 * solution_size_1, lambda: loss_dist)

    # return (1-epsilon) * tf.reduce_mean(loss) + epsilon* 6*tf.reduce_mean(loss_dist)/tf.cast(tf.shape(preds)[0], tf.float32)**3
    return (1-epsilon) * tf.reduce_mean(loss) + epsilon* 6*tf.reduce_mean(loss_dist)/tf.cast(solution_size, tf.float32)**3
    # return (1-epsilon) * tf.reduce_mean(loss) + epsilon* 1/(6*tf.reduce_mean(loss_dist)/tf.cast(solution_size, tf.float32)**3)


def my_softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def my_accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)
