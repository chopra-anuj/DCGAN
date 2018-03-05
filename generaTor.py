
import tensorflow as tf
import numpy as np

class Generator(object):

    def __init__(self, image_length, latent_dims, batch_sz):
        self.latent_dims = latent_dims
        self.img_length = image_length
        self.batch_sz = batch_sz

    def build_generator(self, Z, g_sizes, DenseLayer, FractionallyStridedConvLayer):
        with tf.variable_scope("generator") as scope:
            dims = [self.img_length]
            dim = self.img_length
            for _a, _b, stride, _c in reversed(g_sizes["conv_layers"]):
                dim = int(np.ceil(float(dim) / stride))
                dims.append(dim)

            dims = list(reversed(dims))
            print("dims : ", dims)
            self.g_dims = dims

            mi = self.latent_dims
            self.g_denselayers = []
            count = 0

            for mo, apply_batch_norm in g_sizes["dense_layers"]:
                name = "g_denselayer_%s" % count
                count += 1
                layer = DenseLayer(name, mi, mo, apply_batch_norm)
                self.g_denselayers.append(layer)
                mi = mo

            mo = g_sizes["projection"] * dims[0] * dims[0]
            name = "g_denselayer_%s" % count
            layer = DenseLayer(name, mi, mo, not g_sizes["bn_after_project"])
            self.g_denselayers.append(layer)
            # 348

            mi = g_sizes["projection"]
            self.g_convlayers = []
            num_relus = len(g_sizes["conv_layers"]) - 1
            activation_functions = [tf.nn.relu] * num_relus + [g_sizes["output_activation"]]

            for i in range(len(g_sizes["conv_layers"])):
                name = "fs_convlayer_%s" % i
                mo, filtersz, stride, apply_batch_norm = g_sizes["conv_layers"][i]
                f = activation_functions[i]
                output_shape = [self.batch_sz, dims[i + 1], dims[i + 1], mo]
                print("line 146/363 \t", "mi:", mi, " mo:", mo, " output shape:", output_shape)

                layer = FractionallyStridedConvLayer(name, mi, mo, output_shape, apply_batch_norm, filtersz,
                                                     stride, f)
                self.g_convlayers.append(layer)
                mi = mo

            self.g_sizes = g_sizes
            return self.g_forward(Z)


    def g_forward(self, Z, reuse=None, is_training=True):
        output = Z
        for layer in self.g_denselayers:
            output = layer.forward(output, reuse, is_training)

        output = tf.reshape(output, [-1, self.g_dims[0], self.g_dims[0], self.g_sizes["projection"]])

        if self.g_sizes["bn_after_project"]:
            output = tf.contrib.layers.batch_norm(output, decay=0.9, updates_collections=None, epsilon=1e-5,
                                                  scale=True, is_training=is_training, reuse=reuse,
                                                  scope="bn_after_project")

        for layer in self.g_convlayers:
            output = layer.forward(output, reuse, is_training)

        return output

