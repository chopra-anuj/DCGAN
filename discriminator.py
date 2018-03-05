import tensorflow as tf
import numpy as np

def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

class Discriminator:

    def __init__(self, image_length, latent_dims, batch_sz, num_colors):
        self.latent_dims = latent_dims
        self.img_length = image_length
        self.batch_sz = batch_sz
        self.num_colors = num_colors

    def build_discriminator(self, X, d_sizes, DenseLayer ,ConvLayer):
        with tf.variable_scope("discriminator") as scope:

            # build conv layers
            self.d_convlayers = []
            mi = self.num_colors
            dim = self.img_length
            count = 0
            for mo, filtersz, stride, apply_batch_norm in d_sizes['conv_layers']:
                # make up a name - used for get_variable
                name = "convlayer_%s" % count
                count += 1

                layer = ConvLayer(name, mi, mo, apply_batch_norm, filtersz, stride, lrelu)
                self.d_convlayers.append(layer)
                mi = mo
                print("dim:", dim)
                dim = int(np.ceil(float(dim) / stride))

            mi = mi * dim * dim
            # build dense layers
            self.d_denselayers = []
            for mo, apply_batch_norm in d_sizes['dense_layers']:
                name = "denselayer_%s" % count
                count += 1

                layer = DenseLayer(name, mi, mo, apply_batch_norm, lrelu)
                mi = mo
                self.d_denselayers.append(layer)

            # final logistic layer
            name = "denselayer_%s" % count
            self.d_finallayer = DenseLayer(name, mi, 1, False, lambda x: x)

            # get the logits
            logits = self.d_forward(X)

            # build the cost later
            return logits

    def d_forward(self, X, reuse=None, is_training=True):
        # encapsulate this because we use it twice
        output = X
        for layer in self.d_convlayers:
            output = layer.forward(output, reuse, is_training)
        output = tf.contrib.layers.flatten(output)
        for layer in self.d_denselayers:
            output = layer.forward(output, reuse, is_training)
        logits = self.d_finallayer.forward(output, reuse, is_training)
        return logits
