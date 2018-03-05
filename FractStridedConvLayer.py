import tensorflow as tf

class FractionallyStridedConvLayer:
    def __init__(self, name,mi,mo,output_shape,apply_batch_norm,filtersz = 5,stride = 2, f = tf.nn.relu):
        self.W = tf.get_variable("W_%s" %name, shape = (filtersz,filtersz, mo,mi),
                                                        initializer = tf.random_normal_initializer(stddev=0.02))

        self.b = tf.get_variable("b_%s" %name, shape=( mo,), initializer=tf.zeros_initializer())

        self.f = f
        self.stride = stride
        self.name = name
        self.output_shape = output_shape
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]


    def forward(self,X, reuse, is_training):
        conv_out = tf.nn.conv2d_transpose(value=X,filter = self.W, output_shape=self.output_shape,
                                          strides=[1,self.stride,self.stride,1])

        conv_out = tf.nn.bias_add(conv_out, self.b)
        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm( conv_out, decay = 0.9, updates_collections = None,
                                                     epsilon = 1e-5, scale = True, is_training = is_training,
                                                     reuse = reuse, scope = self.name)

        return self.f(conv_out)

def pert():
    import numpy as np
    n = "anuj"
    mi = 3
    mo = 4
    output_shape = [10,5,7,4]
    abn = True
    f = 2
    s = 1
    X = np.random.randn(5,15,15,3).astype(np.float32)

    conv = FractionallyStridedConvLayer(n,mi,mo,output_shape,abn,f,s)
    print(conv.forward(X,False, False))

if __name__ == '__main__':
    pert()