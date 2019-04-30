import tensorflow as tf

'''
focal_loss实现
'''
def focal_loss(gamma=2.,alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true,y_pred):
        epsilon = 1.e-9
        y_true = tf.convert_to_tensor(y_true,tf.float32)
        y_pred = tf.convert_to_tensor(y_pred,tf.float32)

        model_out = tf.add(y_pred,epsilon)
        ce = tf.multiply(y_true,-tf.log(model_out))
        weight = tf.multiply(y_true,tf.pow(tf.subtract(1.,model_out),gamma))
        f1 = tf.multiply(alpha,tf.multiply(weight,ce))
        reduced_f1 = tf.reduce_max(f1,axis=1)
        return tf.reduce_mean(reduced_f1)
    return focal_loss_fixed

