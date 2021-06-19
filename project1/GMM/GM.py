import torch
import numpy as np
from math import pi

class GMM(torch.nn.Module):
    def __init__(self,n_features):
        super(GMM,self).__init__()
        self.n_components=10
        self.n_features=n_features
        self.eps=1e-6

        self.mu = torch.nn.Parameter(torch.randn(1, self.n_components, self.n_features), requires_grad=False)
        self.var = torch.nn.Parameter(torch.ones(1, self.n_components, self.n_features), requires_grad=False)
        self.pi = torch.nn.Parameter(torch.Tensor(1, self.n_components, 1), requires_grad=False).fill_(1./self.n_components)
        self.var_init=None
        self.log_likelihood=-np.inf

    def bayesian(self,x):
        free_params = self.n_features * self.n_components + self.n_features + self.n_components - 1
        n=x.shape[0]
        bic = -2. * self.__score(x, sum_data=False).mean() * n + free_params * np.log(n)
        return bic

    def fit(self,x,delta=1e-3,n_iter=100):
        i=0
        j=np.inf
        while(i<=n_iter) and(j>=delta):
            print(i)
            log_likelihood_old = self.log_likelihood
            mu_old = self.mu
            var_old = self.var

            self.em(x)
            self.log_likelihood = self.score(x)

            i += 1
            j = self.log_likelihood - log_likelihood_old

            if j <= delta:
                self.update_mu(mu_old)
                self.update_var(var_old)

    def _e_step(self, x):
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=1, keepdim=True)
        log_resp = weighted_log_prob - log_prob_norm
        return torch.mean(log_prob_norm), log_resp

    def _estimate_log_prob(self, x):
        mu = self.mu
        prec = torch.rsqrt(self.var)
        log_p = torch.sum((mu * mu + x * x - 2 * x * mu) * (prec ** 2), dim=2, keepdim=True)
        log_det = torch.sum(torch.log(prec), dim=2, keepdim=True)
        return -.5 * (self.n_features * np.log(2. * pi) + log_p) + log_det

    def _m_step(self, x, log_resp):
        resp = torch.exp(log_resp)
        pi = torch.sum(resp, dim=0, keepdim=True) + self.eps
        mu = torch.sum(resp * x, dim=0, keepdim=True) / pi
        x2 = (resp * x * x).sum(0, keepdim=True) / pi
        mu2 = mu * mu
        xmu = (resp * mu * x).sum(0, keepdim=True) / pi
        var = x2 - 2 * xmu + mu2 + self.eps
        pi = pi / x.shape[0]
        return pi, mu, var


    def em(self, x):
        _, log_resp = self._e_step(x)
        pi, mu, var = self._m_step(x, log_resp)

        self.update_pi(pi)
        self.update_mu(mu)
        self.update_var(var)


    def score(self, x):
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        per_sample_score = torch.logsumexp(weighted_log_prob, dim=1)
        return per_sample_score.sum()


    def update_mu(self, mu):
        if mu.size() == (self.n_components, self.n_features):
            self.mu = mu.unsqueeze(0)
        elif mu.size() == (1, self.n_components, self.n_features):
            self.mu.data = mu


    def update_var(self, var):
        if var.size() == (self.n_components, self.n_features):
            self.var = var.unsqueeze(0)
        elif var.size() == (1, self.n_components, self.n_features):
            self.var.data = var

    def update_pi(self, pi):
        self.pi.data = pi

    def predict(self, x):
        weighted_log_prob = self._estimate_log_prob(x) + torch.log(self.pi)
        p_k = torch.exp(weighted_log_prob)
        return torch.squeeze(torch.max(weighted_log_prob, 1)[1].type(torch.LongTensor))
