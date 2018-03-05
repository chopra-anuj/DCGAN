import tensorflow as tf


class DenseLayer(object):

    def __init__(self,name,M1, M2, apply_batch_norm, f = tf.nn.relu):
        self.W = tf.get_variable("W_%s" %name, shape=(M1,M2), initializer=tf.random_normal_initializer(stddev=0.02))
        self.b = tf.get_variable("b_%s" %name, shape=(M2,), initializer=tf.zeros_initializer())

        self.f = f
        self.name = name
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]


    def forward(self,X , reuse, is_training):
        a = tf.matmul(X, self.W) + self.b

        if self.apply_batch_norm:
            a = tf.contrib.layers.batch_norm(a, decay = 0.9, updates_collections = None, epsilon = 1e-5,
                                             scale = True, is_training = is_training, reuse = reuse,
                                             scope = self.name)

        return self.f(a)


def tes():
    import numpy as np
    n = "anuj"
    m1 = 12
    m2 = 4
    abn = True
    X = np.random.randn(15,12).astype(np.float32)

    conv = DenseLayer(n,m1,m2,abn)
    print(conv.forward(X,False, False))

if __name__ == '__main__':
    tes()
