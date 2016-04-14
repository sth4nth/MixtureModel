function [label, model, bound] = svebfdigm(X, init, prior)
% sequential variational empirical Bayesian finite Dirichlet isotropic Gaussian mixture
% TODO: remove empty components
if nargin < 3
    prior.eta = 1;        % noninformative setting of Dirichet prior 
    prior.kappa = 1;      % noninformative setting of Gassian prior of Gaussian mean ?
    prior.m = mean(X,2);  % when prior.kappa = 0 it doesnt matter how to set this
    prior.alpha = .5;     % noninformative setting of Gamma prior ?
    prior.beta = .5;      % noninformative setting of Gamma prior ?
end
k = init;
[d,n] = size(X);

idx = randsample(n,k);
m = X(:,idx);
% m = softseeds(X,k);
[~,label] = max(bsxfun(@minus,m'*X,sum(m.^2)'/2));
R = full(sparse(1:n,label,1,n,k,n));

tol = 1e-6;
t = 1;
maxiter = 100;
converged = false;
bound = -inf(1,maxiter*n);


X2 = sum(X.^2,1);

nk = sum(R,1);
xbar = bsxfun(@times,X*R,1./nk);
x2bar = X2*R./nk;
s = (x2bar-sum(xbar.^2,1))/d;

model.nk = nk;
model.xbar = xbar;
model.x2bar = x2bar;
model.s = s;
%%
while  ~converged && t < maxiter
    t = t+1;
    
    order = randperm(n);
    for idx = order
        model = vbm(model, prior);
        x = X(:,idx);
        r = vbe(x, model, prior);
                
        x2 = X2(:,idx);
        r0 = R(idx,:);
        
        model = update(model,x,x2,r0,r);
        
%         model = subsample(model,x,x2,r0);
%         model = addsample(model,x,x2,r);
               
        R(idx,:) = r;
    end
    
    bound(t) = vbound(X,R,model,prior)/n;
    converged = bound(t)-bound(t-1) < tol*abs(bound(t));
end
bound = bound(2:t);
[~,idx] = max(R,[],2);
[~,label(1,:)] = max(R(:,unique(idx)),[],2);
if converged
    fprintf('converged in %d steps.\n',t);
else
    fprintf('not converged in %d steps.\n',maxiter);
end

%% online update statistics
% Done!
function model = update(model,x,x2,r_old,r_new)
% this function is equivalent to:
%   model = subsample(model,x,x2,r_old);
%   model = addsample(model,x,x2,r_new);

nk0 = model.nk;
xbar0 = model.xbar;
x2bar0 = model.x2bar;

d = size(xbar0,1);

r_delta = r_new-r_old;
nk = nk0+r_delta;
x_delta = bsxfun(@plus,-xbar0,x);
xbar = xbar0+bsxfun(@times,x_delta,r_delta./nk);
x2bar = x2bar0+(x2-x2bar0).*r_delta./nk;
s = (x2bar-sum(xbar.^2,1))/d;

model.nk = nk;
model.xbar = xbar;
model.x2bar = x2bar;
model.s = s;


function model = addsample(model,x,x2,r)
nk0 = model.nk;
xbar0 = model.xbar;
x2bar0 = model.x2bar;

d = size(xbar0,1);
nk = nk0+r;
x0 = bsxfun(@plus,-xbar0,x);
xbar = xbar0+bsxfun(@times,x0,r./nk);
x2bar = x2bar0+(x2-x2bar0).*r./nk;
s = (x2bar-sum(xbar.^2,1))/d;

model.nk = nk;
model.xbar = xbar;
model.x2bar = x2bar;
model.s = s;


% Done!
function model = subsample(model,x,x2,r)
nk0 = model.nk;
xbar0 = model.xbar;
x2bar0 = model.x2bar;

d = size(xbar0,1);
nk = nk0-r;

x0 = bsxfun(@plus,-xbar0,x);
xbar = xbar0-bsxfun(@times,x0,r./nk);
x2bar = x2bar0-(x2-x2bar0).*r./nk;
s = (x2bar-sum(xbar.^2,1))/d;

model.nk = nk;
model.xbar = xbar;
model.x2bar = x2bar;
model.s = s;

%% variational Bayesian maximization step for one sample
% Done
function model = vbm(model, prior)
kappa0 = prior.kappa;    % piror of Gaussian mean
m0 = prior.m;            % piror of Gaussian mean
alpha0 = prior.alpha;    % Gamma prior
beta0 = prior.beta;      % Gamma prior

nk = model.nk;
xbar = model.xbar;
s = model.s;

d = size(xbar,1);

kappa = kappa0+nk;
m = bsxfun(@times,bsxfun(@plus,kappa0*m0,bsxfun(@times,xbar,nk)),1./kappa);
alpha = alpha0+0.5*d*nk;
beta = beta0+0.5*d*nk.*s+(0.5*kappa0*nk./(kappa0+nk)).*sqdistance(m0,xbar);

model.kappa = kappa;
model.m = m;
model.alpha = alpha;
model.beta = beta;

%% variational Bayesian expectation step for one sample
% Done: Gaussian with Gaussian-Gamma prior
function r = vbe(x, model, prior)
kappa = model.kappa;
m = model.m;
alpha = model.alpha;
beta = model.beta;

[d,k] = size(m);

lnlambda = psi(0,alpha)-log(beta);

nk = model.nk;
eta0 = prior.eta;

lnr = 0.5*(d*(lnlambda-log(2*pi))-(sqdistance(x,m).*alpha./beta+d./kappa))+log(nk+eta0/k);
lnr = lnr-logsumexp(lnr,2);
r = exp(lnr);


function bound = vbound(X, R, model, prior)
kappa0 = prior.kappa;    % piror of Gaussian mean
m0 = prior.m;            % piror of Gaussian mean
alpha0 = prior.alpha;    % Gamma prior
beta0 = prior.beta;      % Gamma prior
eta0 = prior.eta;

kappa = model.kappa;
m = model.m;
alpha = model.alpha;
beta = model.beta;
lnR = log(R);

[d,n] = size(X);
k = size(m,2);

nk = model.nk;

Nk = bsxfun(@plus,-R,sum(R,1));
W = (Nk+eta0/k)/(n-1+eta0);
Epz = R(:)'*log(W(:));
Eqz = R(:)'*lnR(:);

lnlambda = psi(0,alpha)-log(beta);
aib = alpha./beta;
Epmu = 0.5*(d*(k*log(kappa0/(2*pi))+sum(lnlambda)-sum(kappa0./kappa))-sum(kappa0*aib.*sum(bsxfun(@minus,m,m0).^2,1)));
Eplambda = k*(alpha0*log(beta0)-gammaln(alpha0))+(alpha0-1)*sum(lnlambda)-beta0*sum(aib);
Eptheta = Epmu+Eplambda;

Eqmu = 0.5*d*(sum(lnlambda)+sum(log(kappa))-k*log(2*pi)-k);
Eqlambda = -sum(gammaln(alpha))+sum((alpha-1).*psi(0,alpha))+sum(log(beta))-sum(alpha);
Eqtheta = Eqmu+Eqlambda;

xbar = bsxfun(@times,X*R,1./nk);
s = sum(sqdistance(X,xbar).*R,1)./(d*nk);
EpX = 0.5*(d*(lnlambda-1./kappa-log(2*pi)-aib.*s)-aib.*sum((xbar-m).^2,1))*nk';

bound = Epz-Eqz+Eptheta-Eqtheta+EpX;

