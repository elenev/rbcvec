clear; clc; close all;

% Define parameters
p.beta = 0.99;
p.alpha = 0.33;
p.delta = 0.1;
p.gamma = 2;
p.phi = 3;

% Define exog process parameters
mu_a = 0;
sig_a = 0.02;
rho_a = 0.8;

% Define functions
u = @(c) (c.^(1-p.gamma)-1)./(1-p.gamma);
uprime = @(c) c.^(-p.gamma);

% Exog environment
Na = 11;
[P, a_grid] = rouwen(rho_a,mu_a,sig_a,0,Na);
P = P';

% Endog grid
Nk = 200;
kss = fsolve( @(k) -1 + p.beta*(p.alpha*k^(p.alpha-1) + (1-p.delta)), 1);
k_grid = exp( linspace(-0.2, 0.2, Nk) ) * kss;


% Steady state price
qss = p.beta*p.alpha*kss^(p.alpha-1) / (1 - p.beta*(1-p.delta));

% Steady state consumption
css = kss^p.alpha - p.delta*kss;

% Reshape stuff for vectorized solutions
P3 = reshape(P,[Na,1,Na]);
a_next = reshape(a_grid,[1,1,Na]);
A = repmat(a_grid, [1,Nk,Na]);

% Interpolants
state_space = {a_grid, k_grid};
fK = griddedInterpolant(state_space, repmat(k_grid,Na,1));
fQ = griddedInterpolant(state_space, log(qss)*ones(Na,Nk) );
fC = griddedInterpolant(state_space,  log(css)*ones(Na,Nk) );

% Precomputed output (constant at a given grid point)
Y = exp(a_grid) .* k_grid.^p.alpha;

% Initialize guesses for C and Q
logC = fC(state_space);
logQ = fQ(state_space);
Kupd = fK(state_space);

% Define vectorized solve opts
opts = optimoptions('fsolve');
opts.Display = 'none';
opts.JacobPattern = sparse(kron(ones(2),eye(Na*Nk)));
opts.Algorithm='trust-region';
opts.UseParallel = true;

% Define non-vectorized solver opts
opts2 = opts;
opts2.JacobPattern = ones(2);

% Iterate
MAXITER = 100;
iter = 1;
tic;
while iter <= MAXITER
	% Get next period Q and C
	Kp = Kupd;
	KKp = repmat(Kp,[1,1,Na]);
	Qnext = exp(fQ(A, KKp));
	Cnext = exp(fC(A, KKp));

	% Iterate
	[Kupd, logQ, logC] = iterate_vec(p,Kp,Qnext,Cnext,state_space,a_next,P3,Y,logC,logQ,uprime,opts);
	%[Kupd, logQ, logC] =      iterate(p,Kp,Qnext,Cnext,state_space,a_next,P, Y,logC,logQ,uprime,opts2);

	% Update interpolants
	fK = griddedInterpolant(state_space, Kupd);
	fQ = griddedInterpolant(state_space, logQ);
	fC = griddedInterpolant(state_space, logC);

	% Check and report convergence
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

% Wrapper around the equilibrium conditions
function err = solveEqm(x,fEE,fMC,Na,Nk)
	vec = @(x)x(:);
	[c,q] = alloc(x,Na,Nk);
	c = exp(c);
	q = exp(q);
	err = [ vec(fEE(c,q)); vec(fMC(c,q)) ];
end

% Reshape guess into Na x Nk matrices for both C and Q
function [c,q] = alloc(x,Na,Nk)
	c = reshape(x(1:Na*Nk),Na,Nk);
	q = reshape(x(Na*Nk+1:end),Na,Nk);
end

% Iterate using for loops
function [Kupd, logQ, logC] = iterate(p,Kp,Qnext,Cnext,state_space,~,P,Y,logC,logQ,uprime,opts)
	Na = size(Kp,1);
	Nk = size(Kp,2);

	beta = p.beta;
	phi = p.phi;
	alpha = p.alpha;
	delta = p.delta;
	
	a_grid = state_space{1};
	k_grid = state_space{2};

	Kupd = zeros(Na,Nk);

	for i_a = 1:Na
		pvec = P(i_a,:);
		for i_k = 1:Nk
			% Get k, k', c', and q'
			k = k_grid(i_k);
			kp = Kp(i_a,i_k);
			c_next = squeeze(Cnext(i_a,i_k,:));
			q_next = squeeze(Qnext(i_a,i_k,:));

			% RHS of the EE for capital
			RHS = pvec * (beta * uprime( c_next ) .* (alpha*exp(a_grid) .* kp.^(alpha-1) + (1-delta).*q_next) );

			% Eqm conditions
			fEE = @(c,q) uprime(c) * q - RHS;	% EE for capital
			fMC = @(c,q) 1 + phi*(kp/k-1) - q;	% FOC for investment

			% Solve
			objfun = @(x)solveEqm(x,fEE,fMC,1,1);
			[x,~,exfl,~] = fsolve( objfun, [logC(i_a,i_k); logQ(i_a,i_k)], opts);
			if exfl<1
				warning('No solution in iter %d: a = %0.2f, k = %0.2f',iter,a_grid(i_a),k_grid(i_k));
			end
			
			% Store results
			[logC(i_a,i_k),logQ(i_a,i_k)] = alloc(x,1,1);
			Kupd(i_a,i_k) = Y(i_a,i_k) + (1-delta)*k - exp(logC(i_a,i_k));
		end
	end

end

% Iterate with tensors
function [Kupd, logQ, logC] = iterate_vec(p,Kp,Qnext,Cnext,state_space,a_next,P3,Y,logC,logQ,uprime,opts)
	Na = size(Kp,1);
	Nk = size(Kp,2);

	beta = p.beta;
	phi = p.phi;
	alpha = p.alpha;
	delta = p.delta;

	k_grid = state_space{2};

	% RHS of the EE for capital
	% Term inside expectations is N_a x N_k x N_a (3rd dim is possible exog
	% states next period)
	RHS = sum( P3 .* beta .* uprime( Cnext ) .* (alpha*exp(a_next) .* Kp.^(alpha-1) + (1-delta).*Qnext), 3);
	
	% Eqm conditions
	fEE = @(c,q) uprime(c) .* q - RHS;
	fMC = @(c,q) 1 + phi*(Kp./k_grid-1) - q;
	
	% Solve
	objfun = @(x)solveEqm(x,fEE,fMC,Na,Nk);
	[x,~,exfl,~] = fsolve( objfun, [logC(:); logQ(:)], opts);
	if exfl<1
		warning('No solution in iter %d',iter);
	end

	[logC,logQ] = alloc(x,Na,Nk);

	Kupd = Y + (1-delta).*k_grid - exp(logC);
end