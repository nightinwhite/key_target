import tensorflow as tf
from tensorflow.python.platform.flags import FLAGS

def smooth_l1(x):
    l2 = 0.5 * (x**2.0)
    l1 = tf.abs(x) - 0.5
    condition = tf.less(tf.abs(x), 1.0)
    condition = tf.to_float(condition)
    re = condition * l2 + (1 - condition) * l1
    return re

def build_loss(pred_labels, pred_locs, anno_labels, anno_locs, anno_masks, anno_logist_length):
    with tf.variable_scope("Loss"):
        loss_alpha = FLAGS.loss_alpha
        pred_top_labels = tf.nn.softmax(pred_labels)
        pred_top_labels = tf.reduce_max(pred_top_labels, -1)
        positives_mask = 1 - anno_masks

        positives_num = tf.reduce_sum(positives_mask, axis=1)
        negatives_num = positives_num * FLAGS.negatives_scale
        negatives_num = tf.minimum(negatives_num, anno_logist_length*6)
        negatives_num = tf.to_int32(negatives_num)

        pred_negatives_top_labels = pred_top_labels * (anno_masks)
        pred_negatives_min_value = []
        for i in range(FLAGS.batch_size):
            tmp_pred_negatives_min_value, _ = tf.nn.top_k(pred_negatives_top_labels[i, :], negatives_num[i], True)
            pred_negatives_min_value.append(tmp_pred_negatives_min_value[-1])
        pred_negatives_min_value = tf.stack(pred_negatives_min_value)
        pred_negatives_min_value = tf.expand_dims(pred_negatives_min_value, -1)
        pred_negatives_mask = pred_negatives_top_labels - pred_negatives_min_value
        pred_negatives_mask = pred_negatives_mask >= 0
        pred_negatives_mask = tf.cast(pred_negatives_mask, tf.float32)

        active_mask = positives_mask + pred_negatives_mask

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_labels,
                                                                     labels=anno_labels) * active_mask
        class_loss = tf.reduce_sum(class_loss, axis=1) / (1e-5 + tf.reduce_sum(active_mask, axis=1))
        sum_class_loss = tf.reduce_mean(class_loss)
        loc_loss = tf.reduce_sum(smooth_l1(pred_locs - anno_locs), axis=2) * active_mask
        loc_loss = tf.reduce_sum(loc_loss, axis=1) / (1e-5 + tf.reduce_sum(active_mask, axis=1))* 10
        sum_loc_loss = tf.reduce_mean(loc_loss)
        total_loss = tf.reduce_mean(loss_alpha * class_loss + (1.0 - loss_alpha) * loc_loss) * 2
        acc = tf.reduce_sum(tf.to_float(tf.equal(tf.to_int32(tf.argmax(pred_labels, -1)), anno_labels))*(1 - anno_masks))
        acc = acc / tf.reduce_sum((positives_mask))
        return sum_class_loss, sum_loc_loss, total_loss, acc


def build_loss_v2(pred_labels, pred_locs, anno_labels, anno_locs, anno_masks, anno_logist_length):
    with tf.variable_scope("Loss"):
        loss_alpha = FLAGS.loss_alpha
        pred_top_labels = tf.nn.softmax(pred_labels)
        pred_top_labels = pred_top_labels[:, :, -1]
        positives_mask = 1 - anno_masks

        pred_negatives_top_labels = pred_top_labels * (anno_masks)
        pred_negatives_mask = pred_negatives_top_labels - 0.2
        pred_negatives_mask = pred_negatives_mask < 0
        pred_negatives_mask = tf.cast(pred_negatives_mask, tf.float32)

        # return pred_negatives_mask, pred_negatives_top_labels

        active_mask = positives_mask + pred_negatives_mask

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_labels,
                                                                    labels=anno_labels) * active_mask
        class_loss = tf.reduce_sum(class_loss, axis=1) / (1e-5 + tf.reduce_sum(active_mask, axis=1))
        sum_class_loss = tf.reduce_mean(class_loss)
        loc_loss = tf.reduce_sum(smooth_l1(pred_locs - anno_locs), axis=2) * positives_mask
        loc_loss = tf.reduce_sum(loc_loss, axis=1) / (1e-5 + tf.reduce_sum(positives_mask, axis=1)) * 10
        sum_loc_loss = tf.reduce_mean(loc_loss)
        total_loss = tf.reduce_mean(loss_alpha * class_loss + (1.0 - loss_alpha) * loc_loss) * 2
        acc = tf.reduce_sum(
            tf.to_float(tf.equal(tf.to_int32(tf.argmax(pred_labels, -1)), anno_labels)) * active_mask)
        acc = acc / tf.reduce_sum(active_mask)
        return sum_class_loss, sum_loc_loss, total_loss, acc


def test_build_loss(pred_labels, pred_locs, anno_labels, anno_locs, anno_masks, anno_logist_length):
    with tf.variable_scope("Loss"):
        loss_alpha = FLAGS.loss_alpha
        pred_top_labels = tf.nn.softmax(pred_labels)
        pred_top_labels = pred_top_labels[:, :, -1]
        positives_mask = 1 - anno_masks

        pred_negatives_top_labels = pred_top_labels * (anno_masks)
        pred_negatives_mask = pred_negatives_top_labels - 0.2
        pred_negatives_mask = pred_negatives_mask < 0
        pred_negatives_mask = tf.cast(pred_negatives_mask, tf.float32)

        return pred_negatives_mask, pred_negatives_top_labels

        active_mask = positives_mask + pred_negatives_mask

        class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_labels,
                                                                    labels=anno_labels) * active_mask
        class_loss = tf.reduce_sum(class_loss, axis=1) / (1e-5 + tf.reduce_sum(active_mask, axis=1))
        sum_class_loss = tf.reduce_mean(class_loss)
        loc_loss = tf.reduce_sum(smooth_l1(pred_locs - anno_locs), axis=2) * active_mask
        loc_loss = tf.reduce_sum(loc_loss, axis=1) / (1e-5 + tf.reduce_sum(active_mask, axis=1)) * 10
        sum_loc_loss = tf.reduce_mean(loc_loss)
        total_loss = tf.reduce_mean(loss_alpha * class_loss + (1.0 - loss_alpha) * loc_loss) * 2
        acc = tf.reduce_sum(
            tf.to_float(tf.equal(tf.to_int32(tf.argmax(pred_labels, -1)), anno_labels)) * (1 - anno_masks))
        acc = acc / tf.reduce_sum((positives_mask))
        return sum_class_loss, sum_loc_loss, total_loss, acc


def build_accuracy(pred_labels, anno_labels):
    with tf.variable_scope("Accuracy"):
        class_acc = tf.reduce_mean(tf.to_float(tf.equal(tf.to_int32(tf.argmax(pred_labels, -1)), anno_labels)))
        return class_acc






