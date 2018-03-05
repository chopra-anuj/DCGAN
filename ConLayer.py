import tensorflow as tf

class ConvLayer(object):
    def __init__(self,name,mi,mo,apply_batch_norm,filtersz = 5, stride = 2, f = tf.nn.relu):
        self.W = tf.get_variable("W_%s"%name, shape=(filtersz,filtersz,mi,mo),
                                 initializer=(tf.truncated_normal_initializer(stddev=0.02)))

        self.b = tf.get_variable("b_%s"%name, shape=(mo,), initializer=tf.zeros_initializer())

        self.name = name
        self.f = f
        self.stride = stride
        self.apply_batch_norm = apply_batch_norm
        self.params = [self.W, self.b]


    def forward(self,X,reuse,is_training):
        conv_out = tf.nn.conv2d(X,self.W,strides=[1,self.stride,self.stride,1], padding="SAME")
        conv_out = tf.nn.bias_add(conv_out,self.b)

        if self.apply_batch_norm:
            conv_out = tf.contrib.layers.batch_norm(conv_out,decay = 0.9,updates_collections = None, epsilon = 1e-5,
                                                    scale = True, is_training = is_training, reuse = reuse,
                                                    scope = self.name)

        return self.f(conv_out)




def test():
    import numpy as np
    n = "anuj"
    mi = 3
    mo = 2
    abn = True
    f = 3
    s = 1
    X = np.random.randn(5,15,15,3).astype(np.float32)

    conv = ConvLayer(n,mi,mo,abn,f,s)
    print(conv.forward(X,False, False))

if __name__ == '__main__':
    test()