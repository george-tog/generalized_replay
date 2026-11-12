from functools import partial
from warnings import warn

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd.scipy.stats import dirichlet
from autograd import hessian

from ssm.util import one_hot, logistic, relu, rle, ensure_args_are_lists, LOG_EPS, DIV_EPS
from ssm.regression import fit_multiclass_logistic_regression, fit_negative_binomial_integer_r
from ssm.stats import multivariate_normal_logpdf
from ssm.optimizers import adam, bfgs, lbfgs, rmsprop, sgd

from autograd import grad


class Transitions(object):
    def __init__(self, K, D, M=0):
        self.K, self.D, self.M = K, D, M

    @property
    def params(self):
        raise NotImplementedError

    @params.setter
    def params(self, value):
        raise NotImplementedError

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        pass

    def permute(self, perm):
        pass

    def log_prior(self):
        return 0

    def log_transition_matrices(self, data, input, mask, tag):
        raise NotImplementedError

    def transition_matrices(self, data, input, mask, tag):
        return np.exp(self.log_transition_matrices(data, input, mask, tag))

    def m_step(self, expectations, datas, inputs, masks, tags,
               optimizer="lbfgs", num_iters=1000, **kwargs):
        """
        If M-step cannot be done in closed form for the transitions, default to BFGS.
        """
        optimizer = dict(sgd=sgd, adam=adam, rmsprop=rmsprop, bfgs=bfgs, lbfgs=lbfgs)[optimizer]

        # Maximize the expected log joint
        def _expected_log_joint(expectations):
            elbo = self.log_prior()
            for data, input, mask, tag, (expected_states, expected_joints, _) \
                in zip(datas, inputs, masks, tags, expectations):
                log_Ps = self.log_transition_matrices(data, input, mask, tag)
                elbo += np.sum(expected_joints * log_Ps)
            return elbo

        # Normalize and negate for minimization
        T = sum([data.shape[0] for data in datas])
        def _objective(params, itr):
            self.params = params
            obj = _expected_log_joint(expectations)
            return -obj / T

        # Call the optimizer. Persist state (e.g. SGD momentum) across calls to m_step.
        optimizer_state = self.optimizer_state if hasattr(self, "optimizer_state") else None
        self.params, self.optimizer_state = \
            optimizer(_objective, self.params, num_iters=num_iters,
                      state=optimizer_state, full_output=True, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        warn("Analytical Hessian is not implemented for this transition class. \
              Optimization via Laplace-EM may be slow. Consider using an \
              alternative posterior and inference method.")
        obj = lambda x, E_zzp1: np.sum(E_zzp1 * self.log_transition_matrices(x, input, mask, tag))
        hess = hessian(obj)
        terms = np.array([-1 * hess(x[None,:], Ezzp1) for x, Ezzp1 in zip(data, expected_joints)])
        return terms

class StationaryTransitions(Transitions):
    """
    Standard Hidden Markov Model with fixed initial distribution and transition matrix.
    """
    def __init__(self, K, D, M=0):
        super(StationaryTransitions, self).__init__(K, D, M=M)
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

    @property
    def params(self):
        return (self.log_Ps,)

    @params.setter
    def params(self, value):
        self.log_Ps = value[0]

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]

    @property
    def transition_matrix(self):
        return np.exp(self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True))

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)
        return log_Ps[None, :, :]

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        K = self.K
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-32
        P = np.nan_to_num(P / P.sum(axis=-1, keepdims=True))

        # Set rows that are all zero to uniform
        P = np.where(P.sum(axis=-1, keepdims=True) == 0, 1.0 / K, P)
        log_P = np.log(P)
        self.log_Ps = log_P - logsumexp(log_P, axis=-1, keepdims=True)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))


class ConstrainedStationaryTransitions(StationaryTransitions):
    """
    Standard Hidden Markov Model with fixed transition matrix.
    Allows the user to specify some entries of the transition matrix to be zeros,
    in order to prohibit certain transitions.

    The user passes an array, `mask`, which must be the same size
    as the transition matrix. Entries of the mask which are zero
    correspond to entries in the transition matrix which will be
    fixed at zero.
    """
    def __init__(self, K, D, transition_mask=None, M=0):
        super(ConstrainedStationaryTransitions, self).__init__(K, D, M=M)
        Ps = self.transition_matrix
        if transition_mask is None:
            transition_mask = np.ones_like(Ps, dtype=bool)
        else:
            transition_mask = transition_mask.astype(bool)

        # Validate the transition mask. A valid mask must have be the same shape
        # as the transition matrix, contain only ones and zeros, and contain at
        # least one non-zero entry per row.
        assert transition_mask.shape == Ps.shape, "Mask must be the same size " \
            "as the transition matrix. Found mask of shape {}".format(transition_mask.shape)
        assert np.isin(transition_mask,[1,0]).all(), "Mask must contain only 1s and zeros."
        for i in range(transition_mask.shape[0]):
            assert transition_mask[i].any(), "Mask must contain at least one " \
                "nonzero entry per row."

        self.transition_mask = transition_mask
        Ps = Ps * transition_mask
        Ps /= Ps.sum(axis=-1, keepdims=True)
        self.log_Ps = np.log(Ps)
        self.log_Ps[~transition_mask] = -np.inf

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        super(ConstrainedStationaryTransitions, self).m_step(
            expectations,
            datas,
            inputs,
            masks,
            tags,
            **kwargs
        )
        assert np.allclose(self.transition_matrix[~self.transition_mask], 0,
                           atol=2 * LOG_EPS)
        self.log_Ps[~self.transition_mask] = -np.inf


class StickyTransitions(StationaryTransitions):
    """
    Upweight the self transition prior.

    pi_k ~ Dir(alpha + kappa * e_k)
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=100):
        super(StickyTransitions, self).__init__(K, D, M=M)
        self.alpha = alpha
        self.kappa = kappa

    def log_prior(self):
        K = self.K
        log_P = self.log_Ps - logsumexp(self.log_Ps, axis=1, keepdims=True)

        lp = 0
        for k in range(K):
            alpha = self.alpha * np.ones(K) + self.kappa * (np.arange(K) == k)
            lp += np.dot((alpha - 1), log_P[k])
        return lp

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        expected_joints = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
        expected_joints += self.kappa * np.eye(self.K) + (self.alpha-1) * np.ones((self.K, self.K))
        P = (expected_joints / expected_joints.sum(axis=1, keepdims=True)) + 1e-16
        assert np.all(P >= 0), "mode is well defined only when transition matrix entries are non-negative! Check alpha >= 1"
        self.log_Ps = np.log(P)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

class InputDrivenTransitions(StickyTransitions):
    """
    Hidden Markov Model whose transition probabilities are
    determined by a generalized linear model applied to the
    exogenous input.
    """
    def __init__(self, K, D, M, alpha=1, kappa=0, l2_penalty=0.0):
        super(InputDrivenTransitions, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

        # Parameters linking input to state distribution
        self.Ws = npr.randn(K, M)

        # Regularization of Ws
        self.l2_penalty = l2_penalty

    @property
    def params(self):
        return self.log_Ps, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.Ws = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.Ws = self.Ws[perm]

    def log_prior(self):
        lp = super(InputDrivenTransitions, self).log_prior()
        lp = lp + np.sum(-0.5 * self.l2_penalty * self.Ws**2)
        return lp

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        assert input.shape[0] == T
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        return np.zeros((T-1, D, D))

class RecurrentTransitions(InputDrivenTransitions):
    """
    Generalization of the input driven HMM in which the observations serve as future inputs
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0):
        super(RecurrentTransitions, self).__init__(K, D, M, alpha=alpha, kappa=kappa)

        # Parameters linking past observations to state distribution
        self.Rs = np.zeros((K, D))

    @property
    def params(self):
        return super(RecurrentTransitions, self).params + (self.Rs,)

    @params.setter
    def params(self, value):
        self.Rs = value[-1]
        super(RecurrentTransitions, self.__class__).params.fset(self, value[:-1])

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        super(RecurrentTransitions, self).permute(perm)
        self.Rs = self.Rs[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        T, _ = data.shape
        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))
        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        # Past observations effect
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        T, D = data.shape
        hess = np.zeros((T-1,D,D))
        vtildes = np.exp(self.log_transition_matrices(data, input, mask, tag)) # normalized probabilities
        Ez = np.sum(expected_joints, axis=2) # marginal over z from T=1 to T-1
        for k in range(self.K):
            vtilde = vtildes[:,k,:] # normalized probabilities given state k
            Rv = vtilde @ self.Rs
            hess += Ez[:,k][:,None,None] * \
                    ( np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
                    + np.einsum('ti, tj -> tij', Rv, Rv))
        return -1 * hess

class RecurrentOnlyTransitions(Transitions):
    """
    Only allow the past observations and inputs to influence the
    next state.  Get rid of the transition matrix and replace it
    with a constant bias r.
    """
    def __init__(self, K, D, M=0):
        super(RecurrentOnlyTransitions, self).__init__(K, D, M)

        # Parameters linking past observations to state distribution
        self.Ws = npr.randn(K, M)
        self.Rs = npr.randn(K, D)
        self.r = npr.randn(K)

    @property
    def params(self):
        return self.Ws, self.Rs, self.r

    @params.setter
    def params(self, value):
        self.Ws, self.Rs, self.r = value

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ws = self.Ws[perm]
        self.Rs = self.Rs[perm]
        self.r = self.r[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        log_Ps = np.dot(input[1:], self.Ws.T)[:, None, :]              # inputs
        log_Ps = log_Ps + np.dot(data[:-1], self.Rs.T)[:, None, :]     # past observations
        log_Ps = log_Ps + self.r                                       # bias
        log_Ps = np.tile(log_Ps, (1, self.K, 1))                       # expand
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)       # normalize

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)

    def neg_hessian_expected_log_trans_prob(self, data, input, mask, tag, expected_joints):
        # Return (T-1, D, D) array of blocks for the diagonal of the Hessian
        v = np.dot(input[1:], self.Ws.T) + np.dot(data[:-1], self.Rs.T) + self.r
        shifted_exp = np.exp(v - np.max(v,axis=1,keepdims=True))
        vtilde = shifted_exp / np.sum(shifted_exp,axis=1,keepdims=True) # normalized probabilities
        Rv = vtilde@self.Rs
        hess = np.einsum('tn, ni, nj ->tij', -vtilde, self.Rs, self.Rs) \
               + np.einsum('ti, tj -> tij', Rv, Rv)
        return -1 * hess


class RBFRecurrentTransitions(InputDrivenTransitions):
    """
    Recurrent transitions with radial basis functions for parameterizing
    the next state probability given current continuous data. We have,

    p(z_{t+1} = k | z_t, x_t)
        \propto N(x_t | \mu_k, \Sigma_k) \times \pi_{z_t, z_{t+1})

    where {\mu_k, \Sigma_k, \pi_k}_{k=1}^K are learned parameters.
    Equivalently,

    log p(z_{t+1} = k | z_t, x_t)
        = log N(x_t | \mu_k, \Sigma_k) + log \pi_{z_t, z_{t+1}) + const
        = -D/2 log(2\pi) -1/2 log |Sigma_k|
          -1/2 (x - \mu_k)^T \Sigma_k^{-1} (x-\mu_k)
          + log \pi{z_t, z_{t+1}}

    The difference between this and the recurrent model above is that the
    log transition matrices are quadratic functions of x rather than linear.

    While we're at it, there's no harm in adding a linear term to the log
    transition matrices to capture input dependencies.
    """
    def __init__(self, K, D, M=0, alpha=1, kappa=0):
        super(RBFRecurrentTransitions, self).__init__(K, D, M=M, alpha=alpha, kappa=kappa)

        # RBF parameters
        self.mus = npr.randn(K, D)
        self._sqrt_Sigmas = npr.randn(K, D, D)

    @property
    def params(self):
        return self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws

    @params.setter
    def params(self, value):
        self.log_Ps, self.mus, self._sqrt_Sigmas, self.Ws = value

    @property
    def Sigmas(self):
        return np.matmul(self._sqrt_Sigmas, np.swapaxes(self._sqrt_Sigmas, -1, -2))

    @ensure_args_are_lists
    def initialize(self, datas, inputs=None, masks=None, tags=None):
        # Fit a GMM to the data to set the means and covariances
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(self.K, covariance_type="full")
        gmm.fit(np.vstack(datas))
        self.mus = gmm.means_
        self._sqrt_Sigmas = np.linalg.cholesky(gmm.covariances_)

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.mus = self.mus[perm]
        self.sqrt_Sigmas = self.sqrt_Sigmas[perm]
        self.Ws = self.Ws[perm]

    def log_transition_matrices(self, data, input, mask, tag):
        assert np.all(mask), "Recurrent models require that all data are present."

        T = data.shape[0]
        assert input.shape[0] == T
        K, D = self.K, self.D

        # Previous state effect
        log_Ps = np.tile(self.log_Ps[None, :, :], (T-1, 1, 1))

        # RBF recurrent function
        rbf = multivariate_normal_logpdf(data[:-1, None, :], self.mus, self.Sigmas)
        log_Ps = log_Ps + rbf[:, None, :]

        # Input effect
        log_Ps = log_Ps + np.dot(input[1:], self.Ws.T)[:, None, :]
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        Transitions.m_step(self, expectations, datas, inputs, masks, tags, **kwargs)


# Allow general nonlinear emission models with neural networks
class NeuralNetworkRecurrentTransitions(Transitions):
    def __init__(self, K, D, M=0, hidden_layer_sizes=(50,), nonlinearity="relu"):
        super(NeuralNetworkRecurrentTransitions, self).__init__(K, D, M=M)

        # Baseline transition probabilities
        Ps = .95 * np.eye(K) + .05 * npr.rand(K, K)
        Ps /= Ps.sum(axis=1, keepdims=True)
        self.log_Ps = np.log(Ps)

        # Initialize the NN weights
        layer_sizes = (D + M,) + hidden_layer_sizes + (K,)
        self.weights = [npr.randn(m, n) for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [npr.randn(n) for n in layer_sizes[1:]]

        nonlinearities = dict(
            relu=relu,
            tanh=np.tanh,
            sigmoid=logistic)
        self.nonlinearity = nonlinearities[nonlinearity]

    @property
    def params(self):
        return self.log_Ps, self.weights, self.biases

    @params.setter
    def params(self, value):
        self.log_Ps, self.weights, self.biases = value

    def permute(self, perm):
        self.log_Ps = self.log_Ps[np.ix_(perm, perm)]
        self.weights[-1] = self.weights[-1][:,perm]
        self.biases[-1] = self.biases[-1][perm]

    def log_transition_matrices(self, data, input, mask, tag):
        # Pass the data and inputs through the neural network
        x = np.hstack((data[:-1], input[1:]))
        for W, b in zip(self.weights, self.biases):
            y = np.dot(x, W) + b
            x = self.nonlinearity(y)

        # Add the baseline transition biases
        log_Ps = self.log_Ps[None, :, :] + y[:, None, :]

        # Normalize
        return log_Ps - logsumexp(log_Ps, axis=2, keepdims=True)

    def m_step(self, expectations, datas, inputs, masks, tags, optimizer="adam", num_iters=100, **kwargs):
        # Default to adam instead of bfgs for the neural network model.
        Transitions.m_step(self, expectations, datas, inputs, masks, tags,
            optimizer=optimizer, num_iters=num_iters, **kwargs)


class NegativeBinomialSemiMarkovTransitions(Transitions):
    """
    Semi-Markov transition model with negative binomial (NB) distributed
    state durations, as compared to the geometric state durations in the
    standard Markov model.  The negative binomial has higher variance than
    the geometric, but its mode can be greater than 1.

    The NB(r, p) distribution, with r a positive integer and p a probability
    in [0, 1], is this distribution over number of heads before seeing
    r tails where the probability of heads is p. The number of heads
    between each tails is an independent geometric random variable.  Thus,
    the total number of heads is the sum of r independent and identically
    distributed geometric random variables.

    We can "embed" the semi-Markov model with negative binomial durations
    in the standard Markov model by expanding the state space.  Map each
    discrete state k to r new states: (k,1), (k,2), ..., (k,r_k),
    for k in 1, ..., K. The total number of states is \sum_k r_k,
    where state k has a NB(r_k, p_k) duration distribution.

    The transition probabilities are as follows. The probability of staying
    within the same "super state" are:

    p(z_{t+1} = (k,i) | z_t = (k,i)) = p_k

    and for 0 <= j <= r_k - i

    p(z_{t+1} = (k,i+j) | z_t = (k,i)) = (1-p_k)^{j-i} p_k

    The probability of flipping (r_k - i + 1) tails in a row in state k;
    i.e. the probability of exiting super state k, is (1-p_k)^{r_k-i+1}.
    Thus, the probability of transitioning to a new super state is:

    p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * P[k, j]

    where P[k, j] is a transition matrix with zero diagonal.

    As a sanity check, note that the sum of probabilities is indeed 1:

    \sum_{j=i}^{r_k} p(z_{t+1} = (k,j) | z_t = (k,i))
        + \sum_{m \neq k}  p(z_{t+1} = (m, 1) | z_t = (k, i))

    = \sum_{j=0}^{r_k-i} (1-p_k)^j p_k + \sum_{m \neq k} (1-p_k)^{r_k-i+1} * P[k, j]

    = p_k (1-(1-p_k)^{r_k-i+1}) / (1-(1-p_k)) + (1-p_k)^{r_k-i+1}

    = 1 - (1-p_k)^{r_k-i+1} + (1 - p_k)^{r_k-i+1}

    = 1.

    where we used the geometric series and the fact that \sum_{j != k} P[k, j] = 1.
    """
    def __init__(self, K, D, M=0, r_min=1, r_max=20):
        assert K > 1, "Explicit duration models only work if num states > 1."
        super(NegativeBinomialSemiMarkovTransitions, self).__init__(K, D, M=M)

        # Initialize the super state transition probabilities
        self.Ps = npr.rand(K, K)
        np.fill_diagonal(self.Ps, 0)
        self.Ps /= self.Ps.sum(axis=1, keepdims=True)

        # Initialize the negative binomial duration probabilities
        self.r_min, self.r_max = r_min, r_max
        self.rs = npr.randint(r_min, r_max + 1, size=K)
        # self.rs = np.ones(K, dtype=int)
        # self.ps = npr.rand(K)
        self.ps = 0.5 * np.ones(K)

        # Initialize the transition matrix
        self._transition_matrix = None

    @property
    def params(self):
        return (self.Ps, self.rs, self.ps)

    @params.setter
    def params(self, value):
        Ps, rs, ps = value
        assert Ps.shape == (self.K, self.K)
        assert np.allclose(np.diag(Ps), 0)
        assert np.allclose(Ps.sum(1), 1)
        assert rs.shape == (self.K)
        assert rs.dtype == int
        assert np.all(rs > 0)
        assert ps.shape == (self.K)
        assert np.all(ps > 0)
        assert np.all(ps < 1)
        self.Ps, self.rs, self.ps = Ps, rs, ps

        # Reset the transition matrix
        self._transition_matrix = None

    def permute(self, perm):
        """
        Permute the discrete latent states.
        """
        self.Ps = self.Ps[np.ix_(perm, perm)]
        self.rs = self.rs[perm]
        self.ps = self.ps[perm]

        # Reset the transition matrix
        self._transition_matrix = None

    @property
    def total_num_states(self):
        return np.sum(self.rs)

    @property
    def state_map(self):
        return np.repeat(np.arange(self.K), self.rs)

    @property
    def transition_matrix(self):
        if self._transition_matrix is not None:
            return self._transition_matrix

        As, rs, ps = self.Ps, self.rs, self.ps

        # Fill in the transition matrix one block at a time
        K_total = self.total_num_states
        P = np.zeros((K_total, K_total))
        starts = np.concatenate(([0], np.cumsum(rs)[:-1]))
        ends = np.cumsum(rs)
        for (i, j), Aij in np.ndenumerate(As):
            block = P[starts[i]:ends[i], starts[j]:ends[j]]

            # Diagonal blocks (stay in sub-state or advance to next sub-state)
            if i == j:
                for k in range(rs[i]):
                    # p(z_{t+1} = (.,i+k) | z_t = (.,i)) = (1-p)^k p
                    # for 0 <= k <= r - i
                    block += (1 - ps[i])**k * ps[i] * np.diag(np.ones(rs[i]-k), k=k)

            # Off-diagonal blocks (exit to a new super state)
            else:
                # p(z_{t+1} = (j,1) | z_t = (k,i)) = (1-p_k)^{r_k-i+1} * A[k, j]
                block[:,0] = (1-ps[i]) ** np.arange(rs[i], 0, -1) * Aij

        assert np.allclose(P.sum(1),1)
        assert (0 <= P).all() and (P <= 1.).all()

        # Cache the transition matrix
        self._transition_matrix = P

        return P

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        P = self.transition_matrix
        return np.tile(np.log(P)[None, :, :], (T-1, 1, 1))

    def m_step(self, expectations, datas, inputs, masks, tags, samples, **kwargs):
        # Update the transition matrix between super states
        P = sum([np.sum(Ezzp1, axis=0) for _, Ezzp1, _ in expectations]) + 1e-16
        np.fill_diagonal(P, 0)
        P /= P.sum(axis=-1, keepdims=True)
        self.Ps = P

        # Fit negative binomial models for each duration based on sampled states
        states, durations = map(np.concatenate, zip(*[rle(z_smpl) for z_smpl in samples]))
        for k in range(self.K):
            self.rs[k], self.ps[k] = \
                fit_negative_binomial_integer_r(durations[states == k], self.r_min, self.r_max)

        # Reset the transition matrix
        self._transition_matrix = None

class MultiMapBehaviorStickyTransitions(Transitions):
    """
    Transition model for multiple concatenated linear maps with self-transition parameter, SR-smoothed local dynamics
    and a fragmented mixture:

        A = (1 - eps) * [kappa * I + (1-kappa) * A_local] + eps * A_jump

    The SR is computed via an inverse.
    If use_sr=False and use_tcmsr=False, A_local is just the base_P (no SR smoothing).
    With use_sr=True and reverse_sr=True, A_local is the SR of the TRANSPOSED base policy --
    the same kernel swept backwards through space (see _reverse_sr).
    """
    def __init__(
        self,
        K, # number of states
        D, # observation
        seg_sizes=None,
        base_Ps=None, # block-local dynamics; list of matrices
        fixed_base_P=None, # full K x K local dynamics
        jump_P=None,
        use_sr=True,
        use_tcmsr=False,
        reverse_sr=False,
        eps=0.0,
        learn_eps=False,
        eps_eps=1e-8,
        eps_cap=0.1,
        kappa=0.3,
        kappa_eps=1e-8,
        learn_kappa=True,
        gamma=0.95,
        learn_gamma=True,
        gamma_eps=1e-4,
        gamma_cap=0.99,
        learn_omega=True,
        omega=0.5,
        omega_eps=1e-4,
        omega_cap=0.99,
        # zero_diag_P=False,
        gamma_prior_alpha=0.5,
        gamma_prior_beta=0.5,
        omega_prior_alpha=0.5,
        omega_prior_beta=0.5,
        eps_prior_alpha=0.5,
        eps_prior_beta=0.5,
        mstep_steps=25,
        mstep_lr=0.05,
        mstep_clip=5.0,
        alpha=0.001
    ):
        super().__init__(K, D, M=0)

        if fixed_base_P is not None and base_Ps is not None:
            raise ValueError("Provide either fixed_base_P or base_Ps, not both.")

        self.fixed_base_P = None if fixed_base_P is None else np.array(fixed_base_P, dtype=float)
        self.base_Ps = None
        if base_Ps is not None:
            self.base_Ps = [np.array(P, dtype=float) for P in base_Ps]

        # init checks
        if self.fixed_base_P is None and self.base_Ps is None:
            raise ValueError("Provide fixed_base_P or base_Ps.")
        if seg_sizes is None:
            if self.base_Ps is None:
                raise ValueError("seg_sizes required when base_Ps is not provided.")
            seg_sizes = [P.shape[0] for P in self.base_Ps]
        self.seg_sizes = [int(s) for s in seg_sizes]
        if len(self.seg_sizes) < 1:
            raise ValueError("seg_sizes must have at least one segment.")
        if any(s <= 1 for s in self.seg_sizes):
            raise ValueError("Each segment must have at least 2 states.")
        if sum(self.seg_sizes) != int(K):
            raise ValueError("sum(seg_sizes) must equal K.")
        if self.base_Ps is not None:
            if len(self.base_Ps) != len(self.seg_sizes):
                raise ValueError(
                    f"base_Ps length ({len(self.base_Ps)}) must match "
                    f"seg_sizes length ({len(self.seg_sizes)})."
                )

            for i, (P, s) in enumerate(zip(self.base_Ps, self.seg_sizes)):
                if P.shape != (s, s):
                    raise ValueError(
                        f"base_Ps[{i}] shape {P.shape} must be ({s}, {s}) "
                        f"for segment size {s}."
                    )


        self.jump_P = None if jump_P is None else np.array(jump_P, dtype=float)
        self.use_sr = bool(use_sr)
        self.use_tcmsr = bool(use_tcmsr)
        assert not (use_sr and use_tcmsr), "Cannot have both use_sr and use_tcmsr true"
        # Time-reversed SR: A_local becomes the SR of the transposed base policy, i.e. the
        # same kernel run backwards through space. No new learnable parameter -- it is the
        # forward kernel's mirror, so `params`, `_transition_objective` and `m_step` are
        # unchanged and a reversed kernel has exactly the parameter count of its forward twin.
        self.reverse_sr = bool(reverse_sr)
        # TCM-SR's omega term M.T @ M is symmetric, so transposing it is a no-op at omega=1
        # and mixes a forward SR with a symmetric term at omega<1 either way -- "reversed
        # TCM-SR" would silently not be reversed. Refuse rather than accept the combination.
        assert not (reverse_sr and use_tcmsr), "reverse_sr is meaningless with use_tcmsr"
        if reverse_sr and not use_sr:
            raise ValueError("reverse_sr requires use_sr: there is no SR to transpose.")
        # self.zero_diag_P = bool(zero_diag_P)

        if self.fixed_base_P is not None and self.fixed_base_P.shape != (K, K):
            raise ValueError("fixed_base_P is not (K, K)")
        if self.jump_P is not None and self.jump_P.shape != (K, K):
            raise ValueError("jump_P is not (K, K)")

        # learnable gamma - SR future horizon
        self.gamma_eps = float(gamma_eps)
        self.learn_gamma = bool(learn_gamma)
        self.gamma_cap = float(gamma_cap)
        # assigned through the property; gamma_cap must already be set for the inverse map
        self.gamma = float(gamma)

        # learnable kappa - self transitions
        self.learn_kappa = bool(learn_kappa)
        self.kappa_eps = kappa_eps
        self.kappa = float(kappa)
        kmax = 1.0 - self.kappa_eps
        k0 = np.clip(float(kappa), self.kappa_eps, kmax) # valid probability
        self.kappa_unconstrained = np.log(k0 / kmax) - np.log(1.0 - k0 / kmax)

        # learnable epsilon - fragmented dynamics
        self.eps_eps = float(eps_eps)
        self.learn_eps = bool(learn_eps)
        self.eps_cap = float(eps_cap)
        # assigned through the property so the constraint lives in exactly one place;
        # eps_cap must already be set for the inverse map.
        self.eps = float(eps)

        # learnable pre-experimental strength
        self.omega_eps = float(omega_eps)
        self.omega_cap = float(omega_cap)
        # assigned through the property; omega_cap must already be set for the inverse map
        self.omega = float(omega)
        self.learn_omega = bool(learn_omega and self.use_tcmsr)

        # priors
        self.gamma_prior_alpha = float(gamma_prior_alpha)
        self.gamma_prior_beta = float(gamma_prior_beta)
        self.eps_prior_alpha = float(eps_prior_alpha)
        self.eps_prior_beta = float(eps_prior_beta)
        self.omega_prior_alpha = float(omega_prior_alpha)
        self.omega_prior_beta = float(omega_prior_beta)

        # small optimizer for generalized EM
        self.mstep_steps = int(mstep_steps)
        self.mstep_lr = float(mstep_lr)
        self.mstep_clip = float(mstep_clip)

        # prior off-diagonal, within segment strength
        self.alpha = float(alpha)

        # Hard-zero mode: eps and alpha are exactly 0 and the transition logs carry no floor,
        # so an entry outside the kernel's structural support is impossible (log = -inf) rather
        # than merely improbable. Off by default; set after construction, like the project's
        # freeze_eps marker, so it survives deepcopy and pickling. Every reader uses
        # getattr(self, "hard_zero", False) so that objects pickled before it existed load.
        self.hard_zero = False
        self._hard_zero_support_cache = None

        self._transition_matrix = None

    @property
    def params(self):
        return (
            self.gamma_unconstrained,
            self.eps_unconstrained,
            self.kappa_unconstrained,
            self.omega_unconstrained,
        )
    @params.setter
    def params(self, value):
        self.gamma_unconstrained, self.eps_unconstrained, self.kappa_unconstrained, self.omega_unconstrained = value
        self._transition_matrix = None

    def permute(self, perm):
        if self.fixed_base_P is not None:
            self.fixed_base_P = self.fixed_base_P[np.ix_(perm, perm)]
        if self.base_Ps is not None:
            # Only valid if perm preserves block structure
            raise NotImplementedError("permute is not supported for base_Ps without explicit block mapping.")
        if self.jump_P is not None:
            self.jump_P = self.jump_P[np.ix_(perm, perm)]
        self._hard_zero_support_cache = None
        self._transition_matrix = None

    @staticmethod
    def _reachability(B):
        """Boolean closure of `B`: R[i, j] iff j is reachable from i in one or more steps.

        This is exactly the support of the SR `sum_{k>=1} gamma^(k-1) B^k` for any gamma in
        (0, 1), independent of gamma. Computed by repeated squaring on booleans, so it is exact.
        """
        R = np.asarray(B, dtype=bool)
        while True:
            R_next = R | ((R.astype(np.int64) @ R.astype(np.int64)) > 0)
            if np.array_equal(R_next, R):
                return R
            R = R_next

    def _local_support(self, B):
        """Structural support of A_local for one base matrix `B`, given this kernel's SR mode."""
        B = np.asarray(B, dtype=bool)
        if not (self.use_sr or self.use_tcmsr):
            return B.copy()
        R = self._reachability(B)
        if self.use_tcmsr:
            # (1 - omega) M + omega M^T M. omega is clipped strictly inside (0, cap), so both
            # terms are always present -- including for the omega = 1 symmetric variant.
            return R | ((R.T.astype(np.int64) @ R.astype(np.int64)) > 0)
        if self.reverse_sr:
            return R.T.copy()
        return R

    def _hard_zero_support(self):
        """K x K boolean support of A_local under hard_zero, diagonal excluded.

        Built from the base structure alone, never from the parameters. It cannot be read off
        the computed SR instead: np.linalg.solve leaves rounding noise (up to ~1.6e-16) at
        structurally unreachable entries while some reachable SR entries are smaller still, so
        no threshold separates the two.
        """
        cached = getattr(self, "_hard_zero_support_cache", None)
        if cached is not None:
            return cached
        if self.base_Ps is not None and self.fixed_base_P is None:
            blocks = [self._local_support(B).astype(float) for B in self.base_Ps]
            S = self._block_diag(blocks) > 0
        else:
            S = self._local_support(np.asarray(self._build_base_P(), dtype=float) > 0)
        S = S & ~np.eye(self.K, dtype=bool)
        self._hard_zero_support_cache = S
        return S

    def hard_zero_absorbing_states(self):
        """Number of states with no structural exit under hard_zero (they can only stay put)."""
        return int(np.sum(~self._hard_zero_support().any(axis=1)))

    def _expected_log_transitions(self, E, A, eps_val=1e-12):
        """sum(E * log A) for the M-step objective.

        Under hard_zero there is no floor: entries with E == 0 contribute exactly nothing, and
        the where() sits INSIDE the log so autograd never differentiates log(0) (placing it
        after the log gives NaN gradients). E is fixed during an M-step, so the mask is a
        constant. The maximum() only guards an entry that underflows mid-step (a shrinking
        Gaussian sigma); it is inactive at the E-step's parameters, where A > 0 wherever E > 0.
        """
        if getattr(self, "hard_zero", False):
            return np.sum(E * np.log(np.where(E > 0, np.maximum(A, 1e-300), 1.0)))
        return np.sum(E * np.log(A + eps_val))

    @property
    def gamma(self):
        # Parameterised as `cap * logistic(u)`, a smooth bijection onto (0, cap). The
        # previous `min((1 - gamma_eps) * logistic(u), cap)` had an identically zero gradient
        # above the cap, so an M-step that overshot froze the parameter there for good.
        return self.gamma_cap * logistic(self.gamma_unconstrained)

    @gamma.setter
    def gamma(self, value):
        # This class floors gamma at gamma_eps rather than 0 (unchanged), so kernels
        # built with gamma=0.0 keep the gamma=1e-4 they have always had.
        cap = float(self.gamma_cap)
        g = np.clip(float(value), self.gamma_eps, cap * (1.0 - 1e-12))
        v = np.clip(g / cap, 1e-12, 1.0 - 1e-12)
        self.gamma_unconstrained = np.log(v) - np.log(1.0 - v)
        self._transition_matrix = None

    @property
    def eps(self):
        # eps is parameterised as `eps_cap * logistic(u)`, a smooth bijection onto
        # (0, eps_cap). The previous form, `min((1 - eps_eps) * logistic(u), eps_cap)`,
        # has an identically zero gradient above the cap, so any M-step that overshot
        # the ceiling froze eps there permanently. This form approaches the cap
        # asymptotically instead, and never has a dead region.
        # Under hard_zero eps is exactly 0, bypassing the logit (which cannot reach 0 without
        # u = -inf, and that NaNs the autograd M-step). The setter below is left alone, so a
        # caller restoring a saved eps onto a hard-zero kernel is a harmless no-op.
        if getattr(self, "hard_zero", False):
            return 0.0
        return self.eps_cap * logistic(self.eps_unconstrained)

    @eps.setter
    def eps(self, value):
        # This class floors eps at eps_eps rather than 0 (unchanged from before, so the
        # stationary kernel keeps the eps=1e-4 it has always been built with).
        cap = float(self.eps_cap)
        e = np.clip(float(value), self.eps_eps, cap * (1.0 - 1e-12))
        v = np.clip(e / cap, 1e-12, 1.0 - 1e-12)
        self.eps_unconstrained = np.log(v) - np.log1p(-v)
        self._transition_matrix = None

    @property
    def omega(self):
        # Parameterised as `cap * logistic(u)`, a smooth bijection onto (0, cap). The
        # previous `min((1 - omega_eps) * logistic(u), cap)` had an identically zero gradient
        # above the cap, so an M-step that overshot froze the parameter there for good.
        return self.omega_cap * logistic(self.omega_unconstrained)

    @omega.setter
    def omega(self, value):
        cap = float(self.omega_cap)
        p = np.clip(float(value), self.omega_eps, cap * (1.0 - 1e-12))
        v = np.clip(p / cap, 1e-12, 1.0 - 1e-12)
        self.omega_unconstrained = np.log(v) - np.log(1.0 - v)
        self._transition_matrix = None

    @property
    def kappa(self):
        km = 1.0 - self.kappa_eps # constrain logistic
        k = km * logistic(self.kappa_unconstrained)
        return k
    
    @kappa.setter
    def kappa(self, value):
        tiny = 1e-12
        km = 1.0 - self.kappa_eps
        kmax = km - tiny
        k = np.clip(float(value), tiny, kmax)
        v = np.clip(k / km, tiny, 1.0 - tiny)
        self.kappa_unconstrained = np.log(v) - np.log(1.0 - v)
        self._transition_matrix = None # if you change kappa, the old transition matrix is stale


    # def sigmoid(self, value):
    #     return 1 / (1 + np.exp(-value))

    def _row_normalize(self, x, eps=1e-12):
        x = np.maximum(x, 0.0)  # keep nonnegative
        s = x.sum(axis=1, keepdims=True) # row sums
        out = x / (s + eps)
        return out

    def _block_diag(self, blocks):
        sizes = [b.shape[0] for b in blocks]
        total = int(np.sum(sizes))

        rows = []
        left = 0
        for b, n in zip(blocks, sizes):
            right = total - left - n
            row = np.concatenate(
                [np.zeros((n, left)), b, np.zeros((n, right))],
                axis=1,
            )
            rows.append(row)
            left += n
        return np.concatenate(rows, axis=0)

    def _build_base_P(self):
        if self.fixed_base_P is not None:
            return np.array(self.fixed_base_P, dtype=float)
        blocks = [np.array(P, dtype=float) for P in self.base_Ps]
        return self._block_diag(blocks)

    # def _successor_representation_geometric(self, P, gamma):
    #     n = P.shape[0]
    #     # M = P + gamma P^2 + gamma^2 P^3 + ...
    #     M = np.zeros((n, n))
    #     term = P
    #     M = M + term
    #     for _ in range(1, self.sr_terms):
    #         term = gamma * np.dot(term, P)
    #         M = M + term
    #     M = M * gamma + np.eye(n) # add back the identity
    #     return M

    def _successor_representation(self, P, gamma):
        n_states = P.shape[0]
        I = np.eye(n_states, dtype=float)
        M = np.linalg.solve(I - gamma * P, P) # one-step look ahead
        return M
    
    def _local_from_blocks(self, gamma, omega):
        local_blocks = []
        for B in self.base_Ps: # iterate over blocks
            M = self._successor_representation(np.asarray(B, float), gamma)
            if self.use_tcmsr:
                M = (1.0 - omega) * M + omega * (M.T @ M)
            elif self.reverse_sr:
                M = self._reverse_sr(M)
            local_blocks.append(M)
        return self._block_diag(local_blocks)

    def _reverse_sr(self, M):
        """The SR of the time-reversed base policy, given the forward SR `M`.

        SR(P.T) == SR(P).T exactly: (I - gamma P)^-1 is a power series in P, so it commutes
        with P, hence (I - gamma P.T)^-1 P.T == ((I - gamma P)^-1 P).T. Transposing is
        therefore not an approximation of "run the SR backwards" -- it is that object, and it
        saves a second linear solve. Plain `.T` is autograd-safe, so the M-step path is
        unaffected.

        Note this identity holds at the A_local level ONLY. The full transition matrix is not
        the transpose of the forward one: the diagonal zeroing and the symmetric Q_off prior
        commute with transposition, but row-normalisation and the
        (1-eps)(kappa I + (1-kappa) A_local) + eps A_jump wrapper do not.
        """
        return M.T


    def _build_transition_matrix(self, P=None, gamma=None, eps=None, kappa=None, omega=None):        
        if gamma is None:
            gamma = self.gamma
        if eps is None:
            eps = self.eps
        if kappa is None:
            kappa = self.kappa
        if omega is None:
            omega = self.omega

        if P is None and self.base_Ps is not None and self.fixed_base_P is None: # if base P has block structure (by linear map), compute SR for each block
            A_local = self._local_from_blocks(gamma, omega) if (self.use_sr or self.use_tcmsr) else self._build_base_P()
        else:
            P_full = self._build_base_P() if P is None else P
            if self.use_sr:
                A_local = self._successor_representation(P_full, gamma)
                if self.reverse_sr:
                    A_local = self._reverse_sr(A_local)
            elif self.use_tcmsr:
                M = self._successor_representation(P_full, gamma)
                A_local = (1.0 - omega) * M + omega * (M.T @ M)
            else:
                A_local = np.array(P_full)


        K = A_local.shape[0] # number of total states across all maps
        # jumps + building full A matrix
        if self.jump_P is None:
            A_jump = np.ones((K, K)) * (1.0 / K)
        else:
            A_jump = np.array(self.jump_P, dtype=float)
        # np.fill_diagonal(A_local, 0) # set diagonal to 0
        A_local *= (1.0 - np.eye(K)) # set diagonal to 0

        if getattr(self, "hard_zero", False):
            # alpha = 0: no Q_off prior. Instead, zero everything outside the structural
            # support, which removes the solve's rounding noise at unreachable SR entries. A
            # constant 0/1 mask, so this is autograd-safe.
            support = self._hard_zero_support()
            A_local = self._row_normalize(A_local * support.astype(float))
            # A state with no structural exit stays put: its A_local row is e_i, so its row
            # of A is exactly e_i whatever kappa is. Without this the row is kappa * e_i and
            # relies on the final normalisation, which breaks as kappa -> 0.
            empty = ~support.any(axis=1)
            if np.any(empty):
                A_local = A_local + np.diag(empty.astype(float))
        else:
            # add a weak uniform prior for the off-diagonal, within segment
            Q_off = np.zeros((K, K))
            bounds = np.cumsum(np.r_[0, self.seg_sizes])
            for s, e in zip(bounds[:-1], bounds[1:]):
                assert e - s > 1, "Expected segment bounds to be increasing when building transition kernel"
                Q_off[s:e, s:e] = 1 / (e - s - 1)
            # np.fill_diagonal(Q_off, 0)
            Q_off *= (1.0 - np.eye(K)) # diagonal to 0

            A_local = A_local + self.alpha * Q_off # add in the prior
            A_local = self._row_normalize(A_local) # row-normalize
        A = (1.0 - eps) * (kappa * np.eye(K) + (1-kappa) * A_local) + eps * A_jump # add sticky self transition, fragmented mixture
        return self._row_normalize(A)

    @property
    def transition_matrix(self):
        if self._transition_matrix is not None:
            return self._transition_matrix
        self._transition_matrix = self._build_transition_matrix()
        s = self._transition_matrix.sum(axis=1, keepdims=True)
        bad = ~np.isfinite(s[:, 0]) | (s[:, 0] <= 1e-12) # if rowsums are bad
        if np.any(bad): # are any row sums unnormalized?
            raise ValueError('bad row after _row_normalize')

        return self._transition_matrix

    def log_transition_matrices(self, data, input, mask, tag):
        T = data.shape[0]
        P = self.transition_matrix
        if getattr(self, "hard_zero", False):
            # No floor: an exact zero stays -inf, so transition_matrices (exp of this) hands
            # the message-passing kernels exact zeros. They handle that correctly -- Ezzp1 is
            # exactly 0 there -- because every row keeps a positive self-transition.
            with np.errstate(divide="ignore"):
                log_P = np.log(P)
        else:
            log_P = np.log(P + 1e-16)
        return np.tile(log_P[None, :, :], (T - 1, 1, 1))

    def _transition_objective(self, gamma_u, eps_u, kappa_u, omega_u, E):
        '''
        scalar score for optimizing gamma_u, eps_u, kappa_u, which are the unconstrained real values of these parameters
        '''
        hard_zero = getattr(self, "hard_zero", False)
        gm = 1.0 - self.gamma_eps
        gamma_eff = self.gamma_cap * logistic(gamma_u) # cap gamma to 0.99

        em = 1.0 - self.eps_eps
        eps_eff = 0.0 if hard_zero else self.eps_cap * logistic(eps_u)

        km = 1.0 - self.kappa_eps
        kappa_eff = km * logistic(kappa_u) # no cap for kappa, can go up to 1

        om = 1.0 - self.omega_eps
        omega_eff = self.omega_cap * logistic(omega_u)

        P = self._build_base_P()
        A = self._build_transition_matrix(P=P, gamma=gamma_eff, eps=eps_eff, kappa=kappa_eff, omega=omega_eff)
        eps_val = 1e-12
        # if self.zero_diag_P:
        #     off = (1.0 - np.eye(self.K))
        #     base = np.sum(E * off * np.log(A + eps_val))
        # else:
        #     base = np.sum(E * np.log(A + eps_val))
        base = self._expected_log_transitions(E, A, eps_val) # compute expected log-likelihood

        # gamma prior on transformed gamma (discount factor)
        v = gamma_eff / gm
        log_prior_gamma = (self.gamma_prior_alpha - 1.0) * np.log(v + eps_val) + (self.gamma_prior_beta - 1.0) * np.log(1.0 - v + eps_val)

        # gamma prior on epsilon (jump factor). Dropped under hard_zero: eps is not a
        # parameter there, so the term would only be a constant.
        log_prior_eps = 0.0
        if not hard_zero:
            w = eps_eff / em
            log_prior_eps = (self.eps_prior_alpha - 1.0) * np.log(w + eps_val) + (self.eps_prior_beta - 1.0) * np.log(1.0 - w + eps_val)

        # beta prior on omega (experimental factor of tcm-sr)
        log_prior_omega = 0.0
        if self.use_tcmsr and self.learn_omega:
            o = omega_eff / om
            log_prior_omega = (
                (self.omega_prior_alpha - 1.0) * np.log(o + eps_val)
                + (self.omega_prior_beta - 1.0) * np.log(1.0 - o + eps_val)
            )

        return base + log_prior_gamma + log_prior_eps + log_prior_omega

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        E = np.zeros((self.K, self.K))
        for exp in expectations:
            Ezzp1 = exp[1] if isinstance(exp, tuple) else exp["Ezzp1"]
            E += np.sum(Ezzp1, axis=0)

        # shorthand gradient functions
        g_gamma = grad(lambda gu: self._transition_objective(gu, self.eps_unconstrained, self.kappa_unconstrained, self.omega_unconstrained, E))
        g_eps = grad(lambda eu: self._transition_objective(self.gamma_unconstrained, eu, self.kappa_unconstrained, self.omega_unconstrained, E))
        g_kappa = grad(lambda ku: self._transition_objective(self.gamma_unconstrained, self.eps_unconstrained, ku, self.omega_unconstrained, E))
        g_omega = grad(lambda ou: self._transition_objective(self.gamma_unconstrained, self.eps_unconstrained, self.kappa_unconstrained, ou, E,))
        for _ in range(self.mstep_steps):
            if self.learn_gamma:
                dg = g_gamma(self.gamma_unconstrained)
                dg = np.clip(dg, -self.mstep_clip, self.mstep_clip)
                self.gamma_unconstrained = self.gamma_unconstrained + self.mstep_lr * dg
            if self.learn_eps and not getattr(self, "hard_zero", False):
                de = g_eps(self.eps_unconstrained)
                de = np.clip(de, -self.mstep_clip, self.mstep_clip)
                self.eps_unconstrained = self.eps_unconstrained + self.mstep_lr * de
            if self.learn_kappa:
                dk = g_kappa(self.kappa_unconstrained)
                dk = np.clip(dk, -self.mstep_clip, self.mstep_clip)
                self.kappa_unconstrained = self.kappa_unconstrained + self.mstep_lr * dk
            if self.learn_omega and self.use_tcmsr:
                do = g_omega(self.omega_unconstrained)
                do = np.clip(do, -self.mstep_clip, self.mstep_clip)
                self.omega_unconstrained = self.omega_unconstrained + self.mstep_lr * do
        self._transition_matrix = None # set cached transition matrix to none so it gets rebuilt with updated parameters

class SevenMapBehaviorStickyTransitions(MultiMapBehaviorStickyTransitions):
    """
    Backward-compatible wrapper for the original 7-map Lustig transition model.

    Segment order is expected to be:
        [R_in, R_out, C_in, C_out, L_in, L_out, Wheel]

    For arbitrary segment counts, use MultiMapBehaviorStickyTransitions.
    """

    def __init__(
        self,
        K,
        D,
        seg_sizes=None,
        base_Ps=None,
        fixed_base_P=None,
        jump_P=None,
        use_sr=True,
        use_tcmsr=False,
        reverse_sr=False,
        eps=0.0,
        learn_eps=False,
        eps_eps=1e-8,
        eps_cap=0.1,
        kappa=0.3,
        kappa_eps=1e-8,
        learn_kappa=True,
        gamma=0.95,
        learn_gamma=True,
        gamma_eps=1e-4,
        gamma_cap=0.99,
        zero_diag_P=False,
        omega=0.5,
        learn_omega=True,
        omega_eps=1e-4,
        omega_cap=0.99,
        omega_prior_alpha=0.5,
        omega_prior_beta=0.5,
        gamma_prior_alpha=0.5,
        gamma_prior_beta=0.5,
        eps_prior_alpha=0.5,
        eps_prior_beta=0.5,
        mstep_steps=25,
        mstep_lr=0.05,
        mstep_clip=5.0,
        alpha=0.001,
    ):
        if seg_sizes is None:
            if base_Ps is None:
                raise ValueError("seg_sizes required when base_Ps is not provided.")
            seg_sizes = [P.shape[0] for P in base_Ps]

        seg_sizes = [int(s) for s in seg_sizes]
        if len(seg_sizes) != 7:
            raise ValueError(
                "SevenMapBehaviorStickyTransitions requires 7 segment sizes "
                "in order [R_in, R_out, C_in, C_out, L_in, L_out, Wheel]. "
                f"Got {len(seg_sizes)}."
            )

        super().__init__(
            K=K,
            D=D,
            seg_sizes=seg_sizes,
            base_Ps=base_Ps,
            fixed_base_P=fixed_base_P,
            jump_P=jump_P,
            use_sr=use_sr,
            use_tcmsr=use_tcmsr,
            reverse_sr=reverse_sr,
            eps=eps,
            learn_eps=learn_eps,
            eps_eps=eps_eps,
            eps_cap=eps_cap,
            kappa=kappa,
            kappa_eps=kappa_eps,
            learn_kappa=learn_kappa,
            gamma=gamma,
            learn_gamma=learn_gamma,
            gamma_eps=gamma_eps,
            gamma_cap=gamma_cap,
            # zero_diag_P=zero_diag_P,
            omega=omega,
            learn_omega=learn_omega,
            omega_eps=omega_eps,
            omega_cap=omega_cap,
            omega_prior_alpha=omega_prior_alpha,
            omega_prior_beta=omega_prior_beta,
            gamma_prior_alpha=gamma_prior_alpha,
            gamma_prior_beta=gamma_prior_beta,
            eps_prior_alpha=eps_prior_alpha,
            eps_prior_beta=eps_prior_beta,
            mstep_steps=mstep_steps,
            mstep_lr=mstep_lr,
            mstep_clip=mstep_clip,
            alpha=alpha,
        )


# class SevenMapContinuousGaussianTransitions(SevenMapBehaviorStickyTransitions):
#     """
#     Continuous 7-map transition model with Gaussian local dynamics and sticky/fragmented mixing:

#         A = (1 - eps) * [kappa * I + (1-kappa) * A_local] + eps * A_jump

#     where A_local is a block-diagonal Gaussian transition over the 7 linearized segments.
#     This class matches the eps/kappa parameterization used by SevenMapBehaviorStickyTransitions
#     and supports EM updates for sigma_wheel and sigma_arm.
#     """

#     def __init__(
#         self,
#         K,
#         D,
#         seg_sizes,
#         sigma_wheel=2.0,
#         sigma_arm=2.0,
#         learn_sigma_wheel=True,
#         learn_sigma_arm=True,
#         sigma_min=1e-3,
#         sigma_cap=20.0,
#         jump_P=None,
#         eps=0.05,
#         learn_eps=True,
#         eps_eps=1e-4,
#         eps_cap=0.1,
#         kappa=0.3,
#         kappa_eps=1e-8,
#         learn_kappa=True,
#         zero_diag_P=False,
#         eps_prior_alpha=0.5,
#         eps_prior_beta=0.5,
#         mstep_steps=25,
#         mstep_lr=0.05,
#         mstep_clip=5.0,
#     ):
#         super(SevenMapContinuousGaussianTransitions, self).__init__(
#             K=K,
#             D=D,
#             seg_sizes=seg_sizes,
#             fixed_base_P=np.eye(K),  # ignored by overridden _build_base_P
#             jump_P=jump_P,
#             use_sr=False,
#             use_tcmsr=False,
#             eps=eps,
#             learn_eps=learn_eps,
#             eps_eps=eps_eps,
#             eps_cap=eps_cap,
#             kappa=kappa,
#             kappa_eps=kappa_eps,
#             learn_kappa=learn_kappa,
#             gamma=0.0,
#             learn_gamma=False,
#             gamma_eps=1e-4,
#             gamma_cap=0.99,
#             zero_diag_P=zero_diag_P,
#             gamma_prior_alpha=0.5,
#             gamma_prior_beta=0.5,
#             eps_prior_alpha=eps_prior_alpha,
#             eps_prior_beta=eps_prior_beta,
#             mstep_steps=mstep_steps,
#             mstep_lr=mstep_lr,
#             mstep_clip=mstep_clip,
#         )

#         self.sigma_min = float(sigma_min)
#         self.sigma_cap = float(sigma_cap)
#         if not (self.sigma_cap > self.sigma_min):
#             raise ValueError("sigma_cap must be > sigma_min.")

#         self.learn_sigma_wheel = bool(learn_sigma_wheel)
#         self.learn_sigma_arm = bool(learn_sigma_arm)
#         self.sigma_wheel_u = self._sigma_to_u(float(sigma_wheel))
#         self.sigma_arm_u = self._sigma_to_u(float(sigma_arm))
#         self._transition_matrix = None

#     def _sigma_to_u(self, sigma):
#         s = np.clip(float(sigma), self.sigma_min + 1e-12, self.sigma_cap - 1e-12)
#         frac = (s - self.sigma_min) / (self.sigma_cap - self.sigma_min)
#         frac = np.clip(frac, 1e-12, 1.0 - 1e-12)
#         return np.log(frac) - np.log(1.0 - frac)

#     def _u_to_sigma(self, u):
#         return self.sigma_min + (self.sigma_cap - self.sigma_min) * logistic(u)

#     @property
#     def params(self):
#         return (
#             self.sigma_wheel_u,
#             self.sigma_arm_u,
#             self.eps_unconstrained,
#             self.kappa_unconstrained,
#         )

#     @params.setter
#     def params(self, value):
#         if len(value) == 3:
#             self.sigma_wheel_u, self.sigma_arm_u, self.eps_unconstrained = value
#         elif len(value) == 4:
#             self.sigma_wheel_u, self.sigma_arm_u, self.eps_unconstrained, self.kappa_unconstrained = value
#         else:
#             raise ValueError("Expected params tuple of length 3 or 4.")
#         self._transition_matrix = None

#     @property
#     def sigma_wheel(self):
#         return self._u_to_sigma(self.sigma_wheel_u)

#     @sigma_wheel.setter
#     def sigma_wheel(self, value):
#         self.sigma_wheel_u = self._sigma_to_u(value)
#         self._transition_matrix = None

#     @property
#     def sigma_arm(self):
#         return self._u_to_sigma(self.sigma_arm_u)

#     @sigma_arm.setter
#     def sigma_arm(self, value):
#         self.sigma_arm_u = self._sigma_to_u(value)
#         self._transition_matrix = None

#     def _gaussian_block(self, n_bins, sigma):
#         xs = np.arange(int(n_bins), dtype=float)
#         d = xs[None, :] - xs[:, None]
#         W = np.exp(-0.5 * (d / sigma) ** 2)
#         return self._row_normalize(W)

#     def _build_base_P(self, sigma_wheel=None, sigma_arm=None):
#         if sigma_wheel is None:
#             sigma_wheel = self.sigma_wheel
#         if sigma_arm is None:
#             sigma_arm = self.sigma_arm

#         # Segment order is [R_in, R_out, C_in, C_out, L_in, L_out, Wheel].
#         blocks = [
#             self._gaussian_block(self.seg_sizes[0], sigma_arm),
#             self._gaussian_block(self.seg_sizes[1], sigma_arm),
#             self._gaussian_block(self.seg_sizes[2], sigma_arm),
#             self._gaussian_block(self.seg_sizes[3], sigma_arm),
#             self._gaussian_block(self.seg_sizes[4], sigma_arm),
#             self._gaussian_block(self.seg_sizes[5], sigma_arm),
#             self._gaussian_block(self.seg_sizes[6], sigma_wheel),
#         ]
#         return self._block_diag(blocks)

#     def _transition_objective(self, sigma_wheel_u, sigma_arm_u, eps_u, kappa_u, E):
#         sigma_wheel_eff = self._u_to_sigma(sigma_wheel_u)
#         sigma_arm_eff = self._u_to_sigma(sigma_arm_u)

#         em = 1.0 - self.eps_eps
#         eps_raw = em * logistic(eps_u)
#         eps_eff = np.minimum(eps_raw, self.eps_cap)

#         km = 1.0 - self.kappa_eps
#         kappa_eff = km * logistic(kappa_u)

#         P = self._build_base_P(sigma_wheel=sigma_wheel_eff, sigma_arm=sigma_arm_eff)
#         A = self._build_transition_matrix(P=P, gamma=0.0, eps=eps_eff, kappa=kappa_eff)

#         eps_val = 1e-12
#         if self.zero_diag_P:
#             off = 1.0 - np.eye(self.K)
#             base = np.sum(E * off * np.log(A + eps_val))
#         else:
#             base = np.sum(E * np.log(A + eps_val))

#         w = eps_eff / em
#         log_prior_eps = (self.eps_prior_alpha - 1.0) * np.log(w + eps_val) + \
#             (self.eps_prior_beta - 1.0) * np.log(1.0 - w + eps_val)

#         return base + log_prior_eps

#     def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
#         E = np.zeros((self.K, self.K))
#         for exp in expectations:
#             Ezzp1 = exp[1] if isinstance(exp, tuple) else exp["Ezzp1"]
#             E += np.sum(Ezzp1, axis=0)

#         g_sigma_wheel = grad(
#             lambda su: self._transition_objective(
#                 su, self.sigma_arm_u, self.eps_unconstrained, self.kappa_unconstrained, E
#             )
#         )
#         g_sigma_arm = grad(
#             lambda au: self._transition_objective(
#                 self.sigma_wheel_u, au, self.eps_unconstrained, self.kappa_unconstrained, E
#             )
#         )
#         g_eps = grad(
#             lambda eu: self._transition_objective(
#                 self.sigma_wheel_u, self.sigma_arm_u, eu, self.kappa_unconstrained, E
#             )
#         )
#         g_kappa = grad(
#             lambda ku: self._transition_objective(
#                 self.sigma_wheel_u, self.sigma_arm_u, self.eps_unconstrained, ku, E
#             )
#         )

#         for _ in range(self.mstep_steps):
#             if self.learn_sigma_wheel:
#                 ds = g_sigma_wheel(self.sigma_wheel_u)
#                 ds = np.clip(ds, -self.mstep_clip, self.mstep_clip)
#                 self.sigma_wheel_u = self.sigma_wheel_u + self.mstep_lr * ds

#             if self.learn_sigma_arm:
#                 da = g_sigma_arm(self.sigma_arm_u)
#                 da = np.clip(da, -self.mstep_clip, self.mstep_clip)
#                 self.sigma_arm_u = self.sigma_arm_u + self.mstep_lr * da

#             if self.learn_eps:
#                 de = g_eps(self.eps_unconstrained)
#                 de = np.clip(de, -self.mstep_clip, self.mstep_clip)
#                 self.eps_unconstrained = self.eps_unconstrained + self.mstep_lr * de

#             if self.learn_kappa:
#                 dk = g_kappa(self.kappa_unconstrained)
#                 dk = np.clip(dk, -self.mstep_clip, self.mstep_clip)
#                 self.kappa_unconstrained = self.kappa_unconstrained + self.mstep_lr * dk

#         self._transition_matrix = None



class MultiMapContinuousGaussianTransitions(MultiMapBehaviorStickyTransitions):
    """
    Continuous multi-map transition model with Gaussian local dynamics and sticky/fragmented mixing:

        A = (1 - eps) * [kappa * I + (1-kappa) * A_local] + eps * A_jump

    where A_local is a block-diagonal Gaussian transition over the 7 linearized segments.
    This class matches the eps/kappa parameterization used by MultiMapBehaviorStickyTransitions
    and supports EM updates for sigma_wheel and sigma_arm.
    """

    def __init__(
        self,
        K,
        D,
        seg_sizes,
        map_is_wheel=None,
        distance_matrix=None,
        support_mask=None,
        sigma_wheel=2.0,
        sigma_arm=2.0,
        learn_sigma_wheel=True,
        learn_sigma_arm=True,
        sigma_min=1e-3,
        sigma_cap=20.0,
        jump_P=None,
        eps=0.05,
        learn_eps=True,
        eps_eps=1e-4,
        eps_cap=0.1,
        kappa=0.3,
        kappa_eps=1e-8,
        learn_kappa=True,
        zero_diag_P=False,
        eps_prior_alpha=0.5,
        eps_prior_beta=0.5,
        mstep_steps=25,
        mstep_lr=0.05,
        mstep_clip=5.0,
    ):
        super(MultiMapContinuousGaussianTransitions, self).__init__(
            K=K,
            D=D,
            seg_sizes=seg_sizes,
            fixed_base_P=np.eye(K),  # ignored by overridden _build_base_P
            jump_P=jump_P,
            use_sr=False,
            use_tcmsr=False,
            eps=eps,
            learn_eps=learn_eps,
            eps_eps=eps_eps,
            eps_cap=eps_cap,
            kappa=kappa,
            kappa_eps=kappa_eps,
            learn_kappa=learn_kappa,
            gamma=0.0,
            learn_gamma=False,
            gamma_eps=1e-4,
            gamma_cap=0.99,
            # zero_diag_P=zero_diag_P,
            gamma_prior_alpha=0.5,
            gamma_prior_beta=0.5,
            eps_prior_alpha=eps_prior_alpha,
            eps_prior_beta=eps_prior_beta,
            mstep_steps=mstep_steps,
            mstep_lr=mstep_lr,
            mstep_clip=mstep_clip,
        )

        self.sigma_min = float(sigma_min)
        self.sigma_cap = float(sigma_cap)
        if not (self.sigma_cap > self.sigma_min):
            raise ValueError("sigma_cap must be > sigma_min.")

        self.learn_sigma_wheel = bool(learn_sigma_wheel)
        self.learn_sigma_arm = bool(learn_sigma_arm)
        self.sigma_wheel_u = self._sigma_to_u(float(sigma_wheel))
        self.sigma_arm_u = self._sigma_to_u(float(sigma_arm))
        self._transition_matrix = None
        if map_is_wheel is None:
            map_is_wheel = np.zeros(len(seg_sizes), dtype=bool)
        self.map_is_wheel = np.asarray(map_is_wheel, dtype=bool).flatten()
        if self.map_is_wheel.shape != (len(self.seg_sizes),):
            raise ValueError(
                "map_is_wheel must have length len(seg_sizes)."
            )

        self.distance_matrix = None
        self.support_mask = None
        if (distance_matrix is None) != (support_mask is None):
            raise ValueError("distance_matrix and support_mask must be provided together.")
        if distance_matrix is not None:
            distance_matrix = np.asarray(distance_matrix, dtype=float)
            support_mask = np.asarray(support_mask, dtype=bool)
            if distance_matrix.shape != (K, K):
                raise ValueError(f"distance_matrix must have shape ({K}, {K}).")
            if support_mask.shape != (K, K):
                raise ValueError(f"support_mask must have shape ({K}, {K}).")
            if np.any(~np.isfinite(distance_matrix[support_mask])):
                raise ValueError("distance_matrix has non-finite values on supported entries.")
            if np.any(distance_matrix[support_mask] < 0.0):
                raise ValueError("distance_matrix must be nonnegative on supported entries.")
            if np.any(np.sum(support_mask, axis=1) <= 0):
                raise ValueError("support_mask must have at least one supported entry per row.")
            self.distance_matrix = distance_matrix
            self.support_mask = support_mask
        self._map_by_state = np.repeat(np.arange(len(self.seg_sizes)), np.asarray(self.seg_sizes, dtype=int))

    def _sigma_to_u(self, sigma):
        s = np.clip(float(sigma), self.sigma_min + 1e-12, self.sigma_cap - 1e-12)
        frac = (s - self.sigma_min) / (self.sigma_cap - self.sigma_min)
        frac = np.clip(frac, 1e-12, 1.0 - 1e-12)
        return np.log(frac) - np.log(1.0 - frac)

    def _u_to_sigma(self, u):
        return self.sigma_min + (self.sigma_cap - self.sigma_min) * logistic(u)

    @property
    def params(self):
        return (
            self.sigma_wheel_u,
            self.sigma_arm_u,
            self.eps_unconstrained,
            self.kappa_unconstrained,
        )

    @params.setter
    def params(self, value):
        if len(value) == 3:
            self.sigma_wheel_u, self.sigma_arm_u, self.eps_unconstrained = value
        elif len(value) == 4:
            self.sigma_wheel_u, self.sigma_arm_u, self.eps_unconstrained, self.kappa_unconstrained = value
        else:
            raise ValueError("Expected params tuple of length 3 or 4.")
        self._transition_matrix = None

    @property
    def sigma_wheel(self):
        return self._u_to_sigma(self.sigma_wheel_u)

    @sigma_wheel.setter
    def sigma_wheel(self, value):
        self.sigma_wheel_u = self._sigma_to_u(value)
        self._transition_matrix = None

    @property
    def sigma_arm(self):
        return self._u_to_sigma(self.sigma_arm_u)

    @sigma_arm.setter
    def sigma_arm(self, value):
        self.sigma_arm_u = self._sigma_to_u(value)
        self._transition_matrix = None

    def _gaussian_block(self, n_bins, sigma):
        xs = np.arange(int(n_bins), dtype=float)
        d = xs[None, :] - xs[:, None]
        W = np.exp(-0.5 * (d / sigma) ** 2)
        return self._row_normalize(W)

    def _gaussian_supported(self, sigma_wheel, sigma_arm):
        row_is_wheel = self.map_is_wheel[self._map_by_state]
        row_sigma = np.where(row_is_wheel, sigma_wheel, sigma_arm)[:, None]
        W = np.exp(-0.5 * (self.distance_matrix / row_sigma) ** 2)
        W = np.where(self.support_mask, W, 0.0)
        return self._row_normalize(W)

    def _build_base_P(self, sigma_wheel=None, sigma_arm=None):
        if sigma_wheel is None:
            sigma_wheel = self.sigma_wheel
        if sigma_arm is None:
            sigma_arm = self.sigma_arm

        if self.distance_matrix is not None:
            return self._gaussian_supported(sigma_wheel, sigma_arm)

        # Segment order is [R_in, R_out, C_in, C_out, L_in, L_out, Wheel].
        blocks = []
        for n_bins, is_wheel in zip(self.seg_sizes, self.map_is_wheel):
            sigma = sigma_wheel if is_wheel else sigma_arm
            blocks.append(self._gaussian_block(n_bins, sigma))
        return self._block_diag(blocks)

    def _hard_zero_support(self):
        """Structural support of the Gaussian kernel: every pair a Gaussian is defined over.

        Overridden because the parent reads the support off `_build_base_P`, which here depends
        on sigma. The Gaussian is positive on its whole support mathematically; an entry that
        underflows to exactly 0 at a small sigma is then forbidden at that sigma.
        """
        cached = getattr(self, "_hard_zero_support_cache", None)
        if cached is not None:
            return cached
        if self.support_mask is not None:
            S = np.array(self.support_mask, dtype=bool)
        else:
            S = self._block_diag([np.ones((n, n)) for n in self.seg_sizes]) > 0
        S = S & ~np.eye(self.K, dtype=bool)
        self._hard_zero_support_cache = S
        return S

    def _transition_objective(self, sigma_wheel_u, sigma_arm_u, eps_u, kappa_u, E):
        hard_zero = getattr(self, "hard_zero", False)
        sigma_wheel_eff = self._u_to_sigma(sigma_wheel_u)
        sigma_arm_eff = self._u_to_sigma(sigma_arm_u)

        em = 1.0 - self.eps_eps
        eps_eff = 0.0 if hard_zero else self.eps_cap * logistic(eps_u)

        km = 1.0 - self.kappa_eps
        kappa_eff = km * logistic(kappa_u)

        P = self._build_base_P(sigma_wheel=sigma_wheel_eff, sigma_arm=sigma_arm_eff)
        A = self._build_transition_matrix(P=P, gamma=0.0, eps=eps_eff, kappa=kappa_eff)

        eps_val = 1e-12
        # if self.zero_diag_P:
        #     off = 1.0 - np.eye(self.K)
        #     base = np.sum(E * off * np.log(A + eps_val))
        # else:
        base = self._expected_log_transitions(E, A, eps_val)

        # Dropped under hard_zero, where eps is not a parameter and the term is a constant.
        log_prior_eps = 0.0
        if not hard_zero:
            w = eps_eff / em
            log_prior_eps = (self.eps_prior_alpha - 1.0) * np.log(w + eps_val) + \
                (self.eps_prior_beta - 1.0) * np.log(1.0 - w + eps_val)

        return base + log_prior_eps

    def m_step(self, expectations, datas, inputs, masks, tags, **kwargs):
        E = np.zeros((self.K, self.K))
        for exp in expectations:
            Ezzp1 = exp[1] if isinstance(exp, tuple) else exp["Ezzp1"]
            E += np.sum(Ezzp1, axis=0)

        g_sigma_wheel = grad(
            lambda su: self._transition_objective(
                su, self.sigma_arm_u, self.eps_unconstrained, self.kappa_unconstrained, E
            )
        )
        g_sigma_arm = grad(
            lambda au: self._transition_objective(
                self.sigma_wheel_u, au, self.eps_unconstrained, self.kappa_unconstrained, E
            )
        )
        g_eps = grad(
            lambda eu: self._transition_objective(
                self.sigma_wheel_u, self.sigma_arm_u, eu, self.kappa_unconstrained, E
            )
        )
        g_kappa = grad(
            lambda ku: self._transition_objective(
                self.sigma_wheel_u, self.sigma_arm_u, self.eps_unconstrained, ku, E
            )
        )

        for _ in range(self.mstep_steps):
            if self.learn_sigma_wheel and np.any(self.map_is_wheel):
                ds = g_sigma_wheel(self.sigma_wheel_u)
                ds = np.clip(ds, -self.mstep_clip, self.mstep_clip)
                self.sigma_wheel_u = self.sigma_wheel_u + self.mstep_lr * ds

            if self.learn_sigma_arm:
                da = g_sigma_arm(self.sigma_arm_u)
                da = np.clip(da, -self.mstep_clip, self.mstep_clip)
                self.sigma_arm_u = self.sigma_arm_u + self.mstep_lr * da

            if self.learn_eps and not getattr(self, "hard_zero", False):
                de = g_eps(self.eps_unconstrained)
                de = np.clip(de, -self.mstep_clip, self.mstep_clip)
                self.eps_unconstrained = self.eps_unconstrained + self.mstep_lr * de

            if self.learn_kappa:
                dk = g_kappa(self.kappa_unconstrained)
                dk = np.clip(dk, -self.mstep_clip, self.mstep_clip)
                self.kappa_unconstrained = self.kappa_unconstrained + self.mstep_lr * dk

        self._transition_matrix = None


class SevenMapContinuousGaussianTransitions(MultiMapContinuousGaussianTransitions):
    """
    Backward-compatible wrapper for the original 7-map continuous Gaussian
    transition model.

    Segment order is expected to be:
        [R_in, R_out, C_in, C_out, L_in, L_out, Wheel]

    The old API used:
        sigma_arm   -> shared sigma for the six non-wheel maps
        sigma_wheel -> shared sigma for the wheel map

    For arbitrary segment counts, use MultiMapContinuousGaussianTransitions.
    """

    def __init__(
        self,
        K,
        D,
        seg_sizes,
        sigma_wheel=2.0,
        sigma_arm=2.0,
        learn_sigma_wheel=True,
        learn_sigma_arm=True,
        sigma_min=1e-3,
        sigma_cap=20.0,
        jump_P=None,
        eps=0.05,
        learn_eps=True,
        eps_eps=1e-4,
        eps_cap=0.1,
        kappa=0.3,
        kappa_eps=1e-8,
        learn_kappa=True,
        zero_diag_P=False,
        eps_prior_alpha=0.5,
        eps_prior_beta=0.5,
        mstep_steps=25,
        mstep_lr=0.05,
        mstep_clip=5.0,
    ):
        seg_sizes = [int(s) for s in seg_sizes]
        if len(seg_sizes) != 7:
            raise ValueError(
                "SevenMapContinuousGaussianTransitions requires 7 segment sizes "
                "in order [R_in, R_out, C_in, C_out, L_in, L_out, Wheel]. "
                f"Got {len(seg_sizes)}."
            )

        super(SevenMapContinuousGaussianTransitions, self).__init__(
            K=K,
            D=D,
            seg_sizes=seg_sizes,
            map_is_wheel=[False, False, False, False, False, False, True],
            sigma_arm=sigma_arm,
            sigma_wheel=sigma_wheel,
            learn_sigma_arm=learn_sigma_arm,
            learn_sigma_wheel=learn_sigma_wheel,
            sigma_min=sigma_min,
            sigma_cap=sigma_cap,
            jump_P=jump_P,
            eps=eps,
            learn_eps=learn_eps,
            eps_eps=eps_eps,
            eps_cap=eps_cap,
            kappa=kappa,
            kappa_eps=kappa_eps,
            learn_kappa=learn_kappa,
            # zero_diag_P=zero_diag_P,
            eps_prior_alpha=eps_prior_alpha,
            eps_prior_beta=eps_prior_beta,
            mstep_steps=mstep_steps,
            mstep_lr=mstep_lr,
            mstep_clip=mstep_clip,
        )

        self._transition_matrix = None
