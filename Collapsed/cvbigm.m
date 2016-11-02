function [label, model, bound] = cvbigm(X, init, prior)
% Collapsed variational Bayesian isotropic Gaussian mixture
% mixing coefficients are marginalized out of the model
if nargin < 3
    prior.eta = 1;        % noninformative setting of Dirichet prior 
    prior.kappa = 1;      % noninformative setting of Gassian prior of Gaussian mean ?
    prior.m = mean(X,2);  % when prior.kappa = 0 it doesnt matter how to set this
    prior.alpha = .5;     % noninformative setting of Gamma prior ?
    prior.beta = .5;      % noninformative setting of Gamma prior ?
end
k = init;
[d,n] = size(X);
% label = ceil(k*rand(1,n));
% R = full(sparse(1:n,label,1,n,k,n));

m = softseeds(X,k);
[~,label] = max(bsxfun(@minus,m'*X,sum(m.^2)'/2));
R = full(sparse(1:n,label,1,n,k,n));

tol = 1e-8;
t = 1;
maxiter = 100;
converged = false;
bound = -inf(1,maxiter*n);

nk = sum(R,1);
xbar = bsxfun(@times,X*R,1./nk);
s = sum(sqdistance(X,xbar).*R,1)./(d*nk);

model.nk = nk;
model.xbar = xbar;
model.s = s;

%% update sufficient statistics
% x2bar = sum(X.^2,1)*R./nk;
% model.x2bar = x2bar;
% s = (x2bar-sum(xbar.^2,1))/d;
%%
for idx = 1:n;
    model = vbm(model, prior);
    x = X(:,idx);
    r = vbe(x, model, prior);

    model = subsample(model,x,R(idx,:));
    model = addsample(model,x,r);
    R(idx,:) = r;
end        
bt = 1;
while  ~converged && t < maxiter
    t = t+1;
    
    order = randperm(n);
    for idx = order
        bt = bt+1;
        model = vbm(model, prior);
        x = X(:,idx);
        r = vbe(x, model, prior);
        
        model = subsample(model,x,R(idx,:));
        model = addsample(model,x,r);
        R(idx,:) = r;
        
        bound(bt) = vbound(X,R,model,prior)/n;
        margin =  bound(t)-bound(t-1);
        if abs(margin) < tol*abs(bound(t))
            converged = true;
        elseif margin < 0
            fprintf('margin: %d\n',margin);
        end
    end

%     converged = bound(t)-bound(t-1) < tol*abs(bound(t));
end
bound = bound(2:bt);
[~,idx] = max(R,[],2);
[~,label(1,:)] = max(R(:,unique(idx)),[],2);
if converged
    fprintf('converged in %d steps.\n',t);
else
    fprintf('not converged in %d steps.\n',maxiter);
end

%% online update statistics
% Done!
function model = addsample(model,x,r)
nk0 = model.nk;
xbar0 = model.xbar;
s0 = model.s;

d = size(xbar0,1);
nk = nk0+r;

x0 = bsxfun(@plus,-xbar0,x);
xbar = xbar0+bsxfun(@times,x0,r./nk);
s = (s0+sum((xbar-xbar0).*x0,1)/d).*nk0./nk;

model.nk = nk;
model.xbar = xbar;
model.s = s;

% x2 = sum(x.^2,1);
% x2bar0 = model.x2bar;
% x2bar = x2bar0+(x2-x2bar0).*r./nk;
% model.x2bar = x2bar;
% s = (x2bar-sum(xbar.^2,1))/d;

% Done!
function model = subsample(model,x,r)
nk0 = model.nk;
xbar0 = model.xbar;
s0 = model.s;

d = size(xbar0,1);
nk = nk0-r;

x0 = bsxfun(@plus,-xbar0,x);
xbar = xbar0-bsxfun(@times,x0,r./nk);
s = (s0-sum((xbar0-xbar).*x0,1)/d).*nk0./nk;

model.nk = nk;
model.xbar = xbar;
model.s = s;

% x2 = sum(x.^2,1);
% x2bar0 = model.x2bar;
% x2bar = x2bar0-(x2-x2bar0).*r./nk;
% model.x2bar = x2bar;
% s = (x2bar-sum(xbar.^2,1))/d;

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
% Done: Gaussian pdf
% TODO: Student t pdf (marginalize out mean and variance of Gaussian)
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

% vn = sum(R.*(1-R),1); % 1 by K
% Epz = gammaln(eta0)-gammaln(n+eta0)+ sum(gammaln(eta0/k+nk)-0.5*psi(1,eta0/k + nk).*vn-gammaln(eta0/k));
Epz =  gammaln(eta0)-gammaln(n+eta0)+ sum(gammaln(eta0/k+nk)-gammaln(eta0/k));
Eqz = R(:)'*lnR(:);

lnlambda = psi(0,alpha)-log(beta);
aib = alpha./beta;
% Epmu = 0.5*sum(d*(log(kappa0/(2*pi))+lnlambda-kappa0./kappa)-kappa0*aib.*sum(bsxfun(@minus,m,m0).^2,1));
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

