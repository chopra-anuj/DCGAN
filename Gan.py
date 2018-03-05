import tensorflow as tf
from ConLayer import ConvLayer
import numpy as np
from DenseLayr import DenseLayer
from FractStridedConvLayer import FractionallyStridedConvLayer
from datetime import datetime
import utils
import scipy as sp
import matplotlib.pyplot as plt
from generaTor import Generator
from discriminator import Discriminator


BATCH_SIZE = 64
LEARNING_RATE = 0.0002
BETA1 = 0.5
EPOCHS = 2
SAVE_SAMPLE_PERIOD = 50

def lrelu(x, alpha = 0.2):
    return tf.maximum(alpha*x,x)

class DCGAN:

    def __init__(self, img_length, num_colors, d_sizes, g_sizes):
        self.img_length = img_length
        self.num_colors = num_colors
        self.latent_dims = g_sizes["z"]
        self.X = tf.placeholder(tf.float32, shape = (None,img_length,img_length,num_colors), name = "X")
        self.Z = tf.placeholder(tf.float32, shape=(None,self.latent_dims), name="Z")
        self.batch_sz = tf.placeholder(tf.int32, shape=(), name = "batch_sz")
        dnrt = Discriminator(self.img_length, self.latent_dims, 64, num_colors)
        gnrt = Generator(self.img_length, self.latent_dims, 64)
        logits = dnrt.build_discriminator(self.X, d_sizes, DenseLayer, ConvLayer)
        self.sample_images = gnrt.build_generator(self.Z, g_sizes, DenseLayer, FractionallyStridedConvLayer)

        with tf.variable_scope("discriminator") as scope:
            scope.reuse_variables()
            sample_logits = dnrt.d_forward(self.sample_images, True)

        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            self.sample_images_test = gnrt.g_forward(self.Z, reuse = True, is_training = False)

        d = tf.reduce_sum(self.sample_images - self.sample_images_test)+ 7.5
        print ("anuj", d)
        self.d_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits))
        self.d_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits= sample_logits,
                                                                   labels=tf.zeros_like(sample_logits))
        self.d_cost = tf.reduce_mean(self.d_cost_real) + tf.reduce_mean(self.d_cost_fake)

        self.g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = sample_logits,
                                                                             labels=tf.ones_like(sample_logits)))
        real_predictions = tf.cast(logits > 0, tf.float32)
        fake_predictions = tf.cast( sample_logits < 0, tf.float32)
        num_predictions = 2*BATCH_SIZE
        num_correct = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)
        self.d_accuracy = num_correct/num_predictions


        #optimizers
        self.d_params = [t for t in tf.trainable_variables() if t.name.startswith("d")]
        self.g_params = [t for t in tf.trainable_variables() if t.name.startswith("g")]

        self.d_train_op = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.d_cost,
                                                                                      var_list=self.d_params)
        self.g_train_op = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1).minimize(self.g_cost,
                                                                                      var_list=self.g_params)
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()
        self.sess.run(self.init_op)


    def fit(self, X):
        d_costs = []
        g_costs = []
        N = len(X)
        n_batches = N//BATCH_SIZE
        total_iters = 0

        for i in range(EPOCHS):
            print ("epoch:", i)
            np.random.shuffle(X)
            for j in range(n_batches):
                t0 = datetime.now()

                if type(X[0]) is str: #celeb dataset
                    batch = utils.files2images(X[j*BATCH_SIZE:(j+1)*BATCH_SIZE])

                else: #mnist dataset
                    batch = X[j*BATCH_SIZE:(j+1)*BATCH_SIZE]

                Z = np.random.uniform(-1,1,size=(BATCH_SIZE, self.latent_dims))

                # train discriminator

                t1, t2, t3, t4 = self.sess.run((self.d_cost_real,self.d_cost_fake, self.d_cost, self.d_accuracy),
                                                feed_dict={self.X:batch, self.Z:Z, self.batch_sz:BATCH_SIZE })

                print (t1)
                print ("_______")
                print (t2)
                _,d_cost, d_acc = self.sess.run((self.d_train_op, self.d_cost, self.d_accuracy),
                                                feed_dict={self.X:batch, self.Z:Z, self.batch_sz:BATCH_SIZE })
                d_costs.append(d_cost)

                #train generator
                _, g_cost1 = self.sess.run((self.g_train_op, self.g_cost),
                                           feed_dict={self.Z: Z, self.batch_sz: BATCH_SIZE})
                _, g_cost2 = self.sess.run((self.g_train_op, self.g_cost),
                                           feed_dict={self.Z:Z, self.batch_sz: BATCH_SIZE})
                g_costs.append((g_cost1+g_cost2)/2)

                print ("batch:",j+1,"/",n_batches,"\t dt", datetime.now()- t0,"\t d_accuracy", d_acc)
                total_iters += 1

                if total_iters % SAVE_SAMPLE_PERIOD == 0:
                    print ("saving a sample")
                    samples = self.sample(64)
                    d = self.img_length
                    if samples.shape[-1] == 1: #checking the number of colors, if one color, we want 2d image(NXN)
                        samples = samples.reshape(64,d,d)
                        flat_image = np.empty((8*d,8*d))
                        k = 0
                        for i in range(8):
                            for j in range(8):
                                #flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k]
                                flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k]
                                k += 1

                    else: #color = 3, ie - 3 D image, N x N x 3
                        flat_image = np.empty((8*d,8*d,3))
                        k = 0
                        for i in range(8):
                            for j in range(8):
                                flat_image[i*d:(i+1)*d, j*d:(j+1)*d] = samples[k]
                                k += 1

                    sp.misc.imsave("sample_new/samples_at_iter_%d.png" %total_iters, flat_image)

        plt.clf()
        plt.plot(d_costs, label = "discriminator cost")
        plt.plot(g_costs, label = "generator cost")
        plt.legend()
        plt.savefig("cost_vs_iteration.png")


    def sample(self,n):
        Z = np.random.uniform(-1,1,size = (n,self.latent_dims))
        samples = self.sess.run(self.sample_images_test, feed_dict={self.Z :Z, self.batch_sz:n})
        return samples



def mnist():
    X, Y = utils.get_mnist()
    print ("y shape", Y.shape, Y[0:10])
    Y = None
    X = X.reshape(len(X), 28,28, 1)
    dim = X.shape[1]
    colors = X.shape[-1]

    # mnist
    d_sizes = {"conv_layers":[(2,5,2,False), (64,5,2,True)], "dense_layers":[(1024,True)]}
    g_sizes = {"z":100, "projection": 128, "bn_after_project": False, "conv_layers": [(128,5,2,True),(colors,5,2,False)],
               "dense_layers":[(1024,True)],"output_activation": tf.sigmoid}

    gan = DCGAN(dim, colors, d_sizes, g_sizes)
    gan.fit(X)


if __name__ == '__main__':
    mnist()


