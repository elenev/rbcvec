clear; clc; close all;

p.beta = 0.99;
p.alpha = 0.33;
p.delta = 0.1;
p.gamma = 2;
p.phi = 3;

mu_a = 0;
sig_a = 0.02;
rho_a = 0.8;

u = @(c) (c.^(1-p.gamma)-1)./(1-p.gamma);
uprime = @(c) c.^(-p.gamma);

% Exog environment
Na = 33;
[P, a_grid] = rouwen(rho_a,mu_a,sig_a,0,Na);
P = P';

% Endog grid
Nk = 101;
kss = fsolve( @(k) -1 + p.beta*(p.alpha*k^(p.alpha-1) + (1-p.delta)), 1);
k_grid = exp( linspace(-0.2, 0.2, Nk) ) * kss;

% Steady state price
qss = p.beta*p.alpha*kss^(p.alpha-1) / (1 - p.beta*(1-p.delta));

% Steady state consumption
css = kss^p.alpha - p.delta*kss;

P3 = reshape(P,[Na,1,Na]);
a_next = reshape(a_grid,[1,1,Na]);
A = repmat(a_grid, [1,Nk,Na]);

% Interpolants
state_space = {a_grid, k_grid};
fK = griddedInterpolant(state_space, repmat(k_grid,Na,1));
fQ = griddedInterpolant(state_space, log(qss)*ones(Na,Nk));
fC = griddedInterpolant(state_space, log(css)*ones(Na,Nk));

%fC = @(a,k,kprime) max(exp(a) .* k.^alpha + (1-delta).*k - kprime, 1e-14);

Y = exp(a_grid) .* k_grid.^p.alpha;

logC = fC(state_space);
logQ = fQ(state_space);

opts = optimoptions('fsolve');
opts.Display = 'none';
opts.JacobPattern = sparse(kron(ones(2),eye(Na*Nk)));
opts.Algorithm='trust-region';

opts2 = opts;
opts2.JacobPattern = ones(2);

beta = p.beta;
phi = p.phi;
alpha = p.alpha;
delta = p.delta;

MAXITER = 50;
iter = 1;
tic;
while iter <= MAXITER
	Kp = fK(state_space);
	KKp = repmat(Kp,[1,1,Na]);
	Qnext = exp(fQ(A, KKp));
	Cnext = exp(fC(A, KKp));

	RHS = sum( P3 .* beta .* uprime( Cnext ) .* (alpha*exp(a_next) .* Kp.^(alpha-1) + (1-delta).*Qnext), 3);
	fEE = @(c,q) uprime(c) .* q - RHS;
	fMC = @(c,q) 1 + phi*(Kp./k_grid-1) - q;

	objfun = @(x)solveEqm(x,fEE,fMC,Na,Nk);
	[x,fx,exfl,out] = fsolve( objfun, [logC(:); logQ(:)], opts);
	if exfl<1
		warning('No solution in iter %d',iter);
	end
	[logC,logQ] = alloc(x,Na,Nk);

	Kupd = Y + (1-delta).*k_grid - exp(logC);

	fK = griddedInterpolant(state_space, Kupd);
	fQ = griddedInterpolant(state_space, logQ);
	fC = griddedInterpolant(state_space, logC);

	dist = log10( max( abs(Kupd - Kp), [], 'all') );
	fprintf('--- Iteration %d: Dist = %0.3f\n', iter, dist);
	if dist < -5
		break;
	else
		iter = iter+1;
	end
end
time = toc;
disp(time/iter);

function err = solveEqm(x,fEE,fMC,Na,Nk)
	vec = @(x)x(:);
	[c,q] = alloc(x,Na,Nk);
	c = exp(c);
	q = exp(q);
	err = [ vec(fEE(c,q)); vec(fMC(c,q)) ];
end

function [c,q] = alloc(x,Na,Nk)
	c = reshape(x(1:Na*Nk),Na,Nk);
	q = reshape(x(Na*Nk+1:end),Na,Nk);
end