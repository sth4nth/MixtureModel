function [label, model, bound] = vbacsgm(X, init, prior)
% variational empirical Bayesian approximate collapsed spherical Gaussian mixture 
% TODO: 
% 1) noninformative prior
if nargin < 3
    prior.eta = 1;        % noninformative setting of Dirichet prior 
    prior.kappa = 1;      % noninformative setting of Gassian prior of Gaussian mean ?
    prior.m = mean(X,2);  % when prior.kappa = 0 it doesnt matter how to set this
    prior.alpha = .5;     % noninformative setting of Gamma prior ?
    prior.beta = .5;      % noninformative setting of Gamma prior ?
end
n = size(X,2);

if max(size(init)) == 1  % random initialization
    k = init;
    % idx = randsample(n,k);
    % m = X(:,idx);
    m = softseeds(X,k);
    [~,label] = max(bsxfun(@minus,m'*X,sum(m.^2)'/2));
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == 1 && size(init,2) == n  % initialize with labels
    label = init;
    k = max(label);
    R = full(sparse(1:n,label,1,n,k,n));
elseif size(init,1) == d && size(init,2) > 1  %initialize with only centers
    k = size(init,2);
    m = init;
    [~,label] = max(bsxfun(@minus,m'*X,sum(m.^2)'/2));
    R = full(sparse(1:n,label,1,n,k,n));
else
    error('ERROR: init is not valid.');
end


model.R = R;

tol = 1e-6;
t = 1;
maxiter = 5000;
converged = false;
bound = -inf(1,maxiter);
model = vbmaximization(X,model,prior);  
while  ~converged && t < maxiter
    t = t+1;
    model = vbexpection(X,model,prior);
    model = vbmaximization(X,model,prior);  
    
    bound(t) = vbound(X, model,prior)/n;
    converged = abs(bound(t)-bound(t-1)) < tol*abs(bound(t));
end
bound = bound(2:t);
R = model.R;
[~,idx] = max(R,[],2);
[~,label(1,:)] = max(R(:,unique(idx)),[],2);
if converged
    fprintf('converged in %d steps.\n',t);
else
    fprintf('not converged in %d steps.\n',maxiter);
end
% update latent variables
function model = vbexpection(X, model, prior)
kappa = model.kappa;
m = model.m;
alpha = model.alpha;
beta = model.beta;
R = model.R;

eta0 = prior.eta;

[d,k] = size(m);
n = size(X,2);

EN_i = bsxfun(@plus,-R,sum(R,1));
V = R.*(1-R); % 1 by K
VN_i = bsxfun(@plus,-V,sum(V,1));

lnW = log(EN_i+eta0/k)-0.5*VN_i./((EN_i+eta0/k).^2)-log(n-1+eta0);

lnlambda = psi(0,alpha)-log(beta);

lnR = bsxfun(@times,sqdistance(X,m),alpha./beta);
lnR = bsxfun(@plus,lnR,d./kappa);
lnR = bsxfun(@plus,(-0.5)*lnR,0.5*d*(lnlambda-log(2*pi)));
lnR = lnR+lnW;

% [~,idx] = max(lnR,[],2);
% lnR = lnR(:,unique(idx));   % remove empty components!!!

lnR = bsxfun(@minus,lnR,logsumexp(lnR,2));
R = exp(lnR);

model.lnR = lnR;
model.R = R;

% update the parameters
function model = vbmaximization(X, model, prior)
eta0 = prior.eta;        % Dirichet prior 
kappa0 = prior.kappa;    % piror of Gaussian mean
m0 = prior.m;            % piror of Gaussian mean
alpha0 = prior.alpha;    % Gamma prior
beta0 = prior.beta;      % Gamma prior

R = model.R;
d = size(X,1);



% Dirichlet
nk = sum(R,1);
eta = eta0+nk; 
% Gaussian
kappa = kappa0+nk;
xbar = bsxfun(@times,X*R,1./nk);
m = bsxfun(@times,bsxfun(@plus,kappa0*m0,bsxfun(@times,xbar,nk)),1./kappa);
% Gamma
alpha = alpha0+0.5*d*nk;
beta = beta0+0.5*sum(sqdistance(X,xbar).*R,1)+(0.5*kappa0*nk./(kappa0+nk)).*sqdistance(m0,xbar);

model.eta = eta;
model.kappa = kappa;
model.m = m;
model.alpha = alpha;
model.beta = beta;

function bound = vbound(X, model, prior)

kappa0 = prior.kappa;    % piror of Gaussian mean
m0 = prior.m;            % piror of Gaussian mean
alpha0 = prior.alpha;    % Gamma prior
beta0 = prior.beta;      % Gamma prior
eta0 = prior.eta;

kappa = model.kappa;
m = model.m;
alpha = model.alpha;
beta = model.beta;

lnR = model.lnR;
R = model.R;

[d,k] = size(m);
n = size(X,2); 

nk = sum(R,1);

vn = sum(R.*(1-R),1); % 1 by K
Epz = gammaln(eta0)-gammaln(n+eta0)+ sum(gammaln(eta0/k+nk)-0.5*psi(1,eta0/k + nk).*vn-gammaln(eta0/k));
% Epz = R(:)'*lnW(:);
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
