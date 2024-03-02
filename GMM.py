import os

#our customized Gussian mixture model with fused lasso
import numpy as np
from scipy.stats import multivariate_normal

from scipy import io

from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters, _compute_precision_cholesky

from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from glmnet import ElasticNet




class GMM(GaussianMixture):
    def __init__(self,y, n_components=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super(GMM,self).__init__(
            n_components=n_components, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)
        
        self.Y= y
        
        
    
    """Customized m-step to fit fused lasso"""
    def _m_step(self, X, log_resp):   
        """M step.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, n_features = X.shape
        self.weights_, self.mu, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        
        # update lasso coefficient
        # print ("*************updata means by fused lasso now*****************")
        r_ic = np.exp(log_resp)
        
        for i in range(self.n_components):
            idx = np.where(np.argmax(r_ic,axis=1) == i)
            
            # print "len(idx):", len(idx[0])
            #ensure it can be fitted by fused lasso
            if len(idx[0])>(n_samples/(2*self.n_components)):
                # print "fused lasso used"
                data_X_i = X[idx[0]]
                data_Y_i = self.Y[idx[0]]
                n = len(data_X_i)
                p = n_features
                # print "lasso_n:",n
                # print "lasso_p:",p
                n, p = data_X_i.shape
                tuning_param = np.sqrt(n * np.log(p))
                fit = ElasticNet(alpha=1, l1_ratio=1, n_lambda=1000, lambda_min_ratio=1e-4)
                fit.fit(data_X_i, data_Y_i, lambda_path=np.array([tuning_param]))
                coef = fit.coef_[:, 0] / (1 + 2 * tuning_param)
                result = np.array(coef[-1])
                mu_i = np.multiply(result,np.mean(data_X_i,axis=0))
                if i == 0:
                    self.means_ = mu_i
                else:
                    self.means_ = np.vstack((self.means_, mu_i))
                
            else:
                # print "not enough data for fused lasso"
                if i == 0:
                    self.means_ = self.mu[i]
                else:
                    self.means_ = np.vstack((self.means_,self.mu[i]))
                
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type) 
  