from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow_probability as tfp
tfd = tfp.distributions

__all__ = ['VAEEstimator', 'NAVAEEstimator']

def _vae_model_fn(n_hidden, features, labels, mode, encoder_fn, decoder_fn,
                  likelihood, latent_prior, learning_rate, config):

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    predict_task = None

    if (mode == tf.estimator.ModeKeys.PREDICT) and 'x' in features:
        x = features['x']
        predict_task = 'encode'
    else:
        x = labels

    if (mode == tf.estimator.ModeKeys.PREDICT) and 'z' in features:
        code = features['z']
        predict_task = 'decode'
    else:
        # posterior inference

        net = encoder_fn(x, is_training=is_training, scope='encoder')
        qz_mu = slim.fully_connected(net, n_hidden,
                                     activation_fn=None, scope='fc_qz_mu')
        qz_sigma = tfd.matrix_diag_transform(tfd.fill_triangular(
            slim.fully_connected(net, (n_hidden * (n_hidden + 1) // 2),
            activation_fn=None, scope='fc_qz_sigma')),
            transform=tf.nn.softplus)
        qz = tfd.MultivariateNormalTriL(loc=qz_mu, scale_tril=qz_sigma,
                                        name='code')
        pz = latent_prior(n_hidden)
        # Sample from the infered posterior
        code = qz.sample()

    if predict_task is 'encode':
        return tf.estimator.EstimatorSpec(mode=mode, predictions=code,
                                          export_outputs={'encode': tf.estimator.export.PredictOutput({'z': code})})

    # Generator
    predictions = decoder_fn(code, is_training=is_training, scope='generator')
    decoder_likelihood = likelihood(slim.flatten(predictions), features)

    if predict_task is 'decode':
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs={'decode': tf.estimator.export.PredictOutput({'x': predictions})})

    train_op = None
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.image('input',  tf.clip_by_value(labels,0,1))
        tf.summary.image('rec', tf.clip_by_value(predictions,0,1))
        tf.summary.image('diff', labels - predictions)

        # Compute KL divergence between code and prior distribution
        kl = tf.reduce_mean(qz.log_prob(code) - pz.log_prob(code), axis=0)
        tf.summary.scalar('kl_divergence', kl)
        tf.losses.add_loss(kl)

        rec_loss = -tf.reduce_mean(decoder_likelihood.log_prob(slim.flatten(labels)))
        tf.summary.scalar('reconstruction', rec_loss)
        tf.losses.add_loss(rec_loss)

        total_loss = tf.losses.get_total_loss()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss,
                                                                                global_step=tf.train.get_global_step())
    elif mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            "kl": tf.reduce_mean(qz.log_prob(code) - pz.log_prob(code),
                                axis=0),
            "log_p(x|z)": tf.reduce_mean(decoder_likelihood.log_prob(slim.flatten(labels)), axis=0)
        }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


class VAEEstimator(tf.estimator.Estimator):
    """An estimator for Variational Auto-Encoders (VAEs). Vanilla version.
    """

    def __init__(self,
                 n_hidden,
                 encoder_fn,
                 decoder_fn,
                 latent_prior,
                 likelihood,
                 learning_rate=0.001,
                 model_dir=None, config=None):
        """
        Args:
            encoder_fn: model function for the encoder
        """

        def _model_fn(features, labels, mode):
            return _vae_model_fn(n_hidden, features, labels, mode, encoder_fn, decoder_fn,
                                 likelihood, latent_prior, learning_rate, config)

        super(VAEEstimator, self).__init__(model_fn=_model_fn,
                                           model_dir=model_dir,
                                           config=config)


def _navae_model_fn(n_hidden, features, labels, mode, encoder_fn, decoder_fn,
                  likelihood, latent_prior, training_size, learning_rate, config):

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    predict_task = None

    x = labels

    if (mode == tf.estimator.ModeKeys.PREDICT) and 'z' in features:
        code = features['z']
        predict_task = 'decode'
    else:
        inds = features['inds']
        # Create parametrized posterior for entire training sample
        with tf.variable_scope("code"):
            qz_mu = tf.Variable(initial_value=np.zeros((training_size, n_hidden), dtype=tf.float32)
            sigma = tf.Variable(initial_value=np.ones((training_size, (n_hidden *(n_hidden +1) // 2))), dtype=tf.float32)
        qz_sigma = tfd.matrix_diag_transform(tfd.fill_triangular(sigma), transform=tf.nn.softplus)

        qz = tfd.MultivariateNormalTriL(loc=qz_mu, scale_tril=qz_sigma,
                                        name='code')
        pz = latent_prior(n_hidden)
        # Sample from the infered posterior
        code = qz.sample()

    # Generator
    predictions = decoder_fn(code, is_training=is_training, scope='generator')
    decoder_likelihood = likelihood(slim.flatten(predictions), features)

    if predict_task is 'decode':
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                          export_outputs={'decode': tf.estimator.export.PredictOutput({'x': predictions})})

    train_op = None
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.image('input',  tf.clip_by_value(labels, 0, 1))
        tf.summary.image('rec', tf.clip_by_value(predictions, 0, 1))
        tf.summary.image('diff', labels - predictions)

        # Compute KL divergence between code and prior distribution
        kl = tf.reduce_mean(qz.log_prob(code) - pz.log_prob(code), axis=0)
        tf.summary.scalar('kl_divergence', kl)
        tf.losses.add_loss(kl)

        rec_loss = -tf.reduce_mean(decoder_likelihood.log_prob(slim.flatten(labels)))
        tf.summary.scalar('reconstruction', rec_loss)
        tf.losses.add_loss(rec_loss)

        total_loss = tf.losses.get_total_loss()
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss,
                                                                                global_step=tf.train.get_global_step())
    elif mode == tf.estimator.ModeKeys.EVAL:

        eval_metric_ops = {
            "kl": tf.reduce_mean(qz.log_prob(code) - pz.log_prob(code),
                                axis=0),
            "log_p(x|z)": tf.reduce_mean(decoder_likelihood.log_prob(slim.flatten(labels)), axis=0)
        }

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                                      loss=total_loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)

class NAVAEEstimator(tf.estimator.Estimator):
    """An estimator for Non-Amortized Variational Auto-Encoders (NAVAE)
    """

    def __init__(self,
                 n_hidden,
                 encoder_fn,
                 decoder_fn,
                 latent_prior,
                 likelihood,
                 training_size,
                 learning_rate=0.001,
                 model_dir=None, config=None):
        """
        Args:
            encoder_fn: model function for the encoder
        """

        def _model_fn(features, labels, mode):
            return _navae_model_fn(n_hidden, features, labels, mode,
                                   encoder_fn, decoder_fn, likelihood,
                                   latent_prior, training_size, learning_rate, config)

        super(NAVAEEstimator, self).__init__(model_fn=_model_fn,
                                           model_dir=model_dir,
                                           config=config)
