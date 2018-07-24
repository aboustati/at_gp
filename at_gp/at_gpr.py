import tensorflow as tf

from gpflow import settings

from gpflow.conditionals import base_conditional
from gpflow.decors import name_scope, params_as_tensors
from gpflow.models import GPModel
from gpflow.likelihoods import Gaussian
from gpflow.params import DataHolder, Parameter
from gpflow.logdensities import multivariate_normal
from gpflow.transforms import Log1pe

class ATGPR(GPModel):
    """
    Adaptive Transfer Gaussian Process Regression

    This is an implemntation of the model from Cao et. al. 2010
    """

    def __init__(self,
                 X_target,
                 Y_target,
                 X_source,
                 Y_source,
                 kern,
                 source_likelihood_variance,
                 mean_function=None,
                 b=1.0,
                 mu=1.0,
                 **kwargs):
        """
        :param X_target: Input data matrix of size NxD
        :param Y_target: Output data matrix of size NxR
        :param X_source: Input data matrix of size NxD
        :param Y_source: Output data matrix of size NxR
        :param kern: GPflow kernel object
        :param mean_function: GPflow mean_function object
        :param b: Initial value for transfer param b (float)
        :param mu: Initial value for transfer param mu (float)
        :param kwargs: Additional keyword arguments
        """
        target_likelihood = Gaussian()

        super(ATGPR, self).__init__(X_target,
                                    Y_target,
                                    kern,
                                    target_likelihood,
                                    mean_function,
                                    **kwargs)

        self.source_likelihood = Gaussian(source_likelihood_variance)
        self.source_likelihood.variance.trainable = False
        self.X_source = DataHolder(X_source)
        self.Y_source = DataHolder(Y_source)

        self.b = Parameter(b, transform=Log1pe(), dtype=settings.float_type)
        self.mu = Parameter(mu, transform=Log1pe(), dtype=settings.float_type)

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct tensorflow fucntion to compute the marginal likelihood

        :returns: TF tensor
        """

        transfer_param = 2 * (1 / (1 + self.mu) ** self.b) - 1
        K_cross = transfer_param * self.kern.K(self.X_source, self.X)
        K_source = self.kern.K(self.X_source) \
                   + tf.eye(tf.shape(self.X_source)[0], dtype=settings.float_type)\
                   * self.source_likelihood.variance
        K_target = self.kern.K(self.X) \
                   + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type)\
                   * self.likelihood.variance

        m, C = base_conditional(K_cross, K_source, K_target, self.Y_source, full_cov=True)
        # L_source = tf.cholesky(K_source)
        # A = tf.matrix_triangular_solve(L_source, K_cross, lower=True)
        # A = tf.matrix_triangular_solve(tf.transpose(L_source), A, lower=True)
        # m = self.mean_function(self.X) + tf.matmul(A, self.Y_source, transpose_a=True)
        #
        # C = K_target - tf.transpose(tf.matmul(A, K_cross))
        m = self.mean_function(self.X) + m
        L = tf.cholesky(C)[0]

        logpdf = multivariate_normal(self.Y, m, L)

        return tf.reduce_sum(logpdf)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Construct a tensorflow function to compute predictions

        :param Xnew: Test data matrix of size NxD
        :param full_cov: If True, returns full covariance function
        :return: TF tensor of size NxR
        """
        transfer_param = 2 * (1 / (1 + self.mu) ** self.b) - 1

        y_target = self.Y - self.mean_function(self.X)
        y_source = self.Y_source - self.mean_function(self.X_source)
        y = tf.concat([y_target, y_source], axis=0)
        K_new_target = self.kern.K(self.X, Xnew)
        K_new_source = transfer_param * self.kern.K(self.X_source, Xnew)

        K_new = tf.concat([K_new_target, K_new_source], axis=0)

        K_cross = transfer_param * self.kern.K(self.X, self.X_source)
        K_source = self.kern.K(self.X_source) \
                   + tf.eye(tf.shape(self.X_source)[0], dtype=settings.float_type)\
                   * self.source_likelihood.variance
        K_target = self.kern.K(self.X) \
                   + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type)\
                   * self.likelihood.variance

        C = tf.concat([
            tf.concat([K_target, tf.transpose(K_cross)], axis=0),
            tf.concat([K_cross, K_source], axis=0)
        ], axis=1)

        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(K_new, C, Knn, y, full_cov=full_cov, white=False)
        return f_mean + self.mean_function(Xnew), f_var