from __future__ import division
from __future__ import print_function
import os.path

import tensorflow as tf
from dataset import load_dataset
import sys
import numpy as np

dataset = load_dataset(sys.argv[1])
sample = dataset.sample_img()
input_dim = np.product(sample.shape)
# TODO: turn these into flags?
hidden_encoder_dim = 400
hidden_decoder_dim = 400
latent_dim = 20
lam = 0 # Scaling factor for L2 loss
n_steps = int(1e6) # 1m
batch_size = 100
PRINT_EVERY = 50
SAVE_EVERY = 500

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.001)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0., shape=shape)
  return tf.Variable(initial)

x = tf.placeholder("float", shape=[None, input_dim])
l2_loss = tf.constant(0.0)

W_encoder_input_hidden = weight_variable([input_dim,hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

W_encoder_hidden_mu = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu

W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim,latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar

# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.mul(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim,hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_summ = tf.scalar_summary("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.merge_all_summaries()

# add Saver ops
saver = tf.train.Saver()

with tf.Session() as sess:
  summary_writer = tf.train.SummaryWriter('experiment',
                                          graph=sess.graph)
  if os.path.isfile("save/model.ckpt"):
    print("Restoring saved parameters")
    saver.restore(sess, "save/model.ckpt")
  else:
    print("Initializing parameters")
    sess.run(tf.initialize_all_variables())

  for step in range(1, n_steps):
    batch = dataset.next_batch(batch_size)
    feed_dict = {x: batch}
    _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, step)

    if step % PRINT_EVERY == 0:
      print("Step {0} | Loss: {1}".format(step, cur_loss))
    if step % SAVE_EVERY == 0:
      print "Saving model to 'save/model.ckpt'"
      save_path = saver.save(sess, "save/model.ckpt")


