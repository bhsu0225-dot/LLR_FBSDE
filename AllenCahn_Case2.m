% High-Dimensional Equation Solver with Forcing term
% Author: Subihao
% Parallelizes over particles (M)
clc
clear
%% Parameters
d = 100;
nu = 1;
T = 1;
N = 400;
M = 100;
theta = 1;
x0 = ones(1,d)*0.1;
kernel_name = 'gaussian'; % 'gaussian' | 'epanechnikov' | 'triweight'

% ===== NEW: nonlinearity parameters a, b > 0 =====
a = 0.4;
b = 0.8;

high_dim_allen_cahn_solver(d,nu,T,N,M,x0,theta,kernel_name,a,b)

%% ====================== MAIN SOLVER ======================
function high_dim_allen_cahn_solver(d,nu,T,N,M,x0,theta,kernel_name,a,b)

    if nargin < 7 || isempty(theta),       theta = 0.5;       end
    if nargin < 8 || isempty(kernel_name), kernel_name = 'gaussian'; end
    
    dt     = T / N;
    eps_k  = sqrt(dt);            
    min_nn = max(d+2, M);        %  min neighbor
    %rng('default');
    %% Forward simulation
    X = zeros(M, d, N+1);
    Y = zeros(M, N+1);
    X(:,:,1) = repmat(x0, M, 1);
    dW = sqrt(dt) * randn(M, d, N);
    sigma = sqrt(2 * nu) * eye(d);

    for k = 1:N
        % drift = 0, pure diffusion with variance 2*nu
        X(:,:,k+1) = X(:,:,k) + dW(:,:,k) * sigma';
    end

    %% Terminal condition
    for m = 1:M
        Y(m,N+1) = exact_solution(T, squeeze(X(m,:,N+1)));
    end

    %% Backward recursion
    fprintf('Solving backward...\n');
    
    t_start = tic;
    last_pct = -1;

    for k = N:-1:1
        pct = (N-k+1)/N*100;
        if pct ~= last_pct
            fprintf('Progress: %6.2f%% | Elapsed: %.2fs\n', pct, toc(t_start));
            last_pct = pct;
        end

        Xk      = squeeze(X(:,:,k));
        Xk_next = squeeze(X(:,:,k+1));
        Y_next  = Y(:,k+1);

        Yk_new = zeros(M,1);

        parfor m = 1:M
            warning('off');
            x_anchor_k  = Xk(m,:);
            x_anchor_k1 = Xk_next(m,:);

            % ----- neighbor selection -----
            dist_k  = sqrt(sum((Xk - x_anchor_k).^2, 2));

            dist_k(m) = inf;                            % 排除自身
            idx_ball = find(dist_k > 0 & dist_k <= eps_k);    
            if numel(idx_ball) < min_nn
                [~,ord] = sort(dist_k);
                idx_ball = ord(2:M);
            end
            dist_k = dist_k(idx_ball);
            X_nb_k  = Xk(idx_ball,:);
            X_nb_k1 = Xk_next(idx_ball,:);
            Y_nb1   = Y_next(idx_ball);

            % bandwidth proxies (keep your original usage)
            h_k  =  pdist_to_anchor(X_nb_k,  x_anchor_k);
            h_k1 =  pdist_to_anchor(X_nb_k1, x_anchor_k1);

            % local linear regression for gradient (at k and k+1 anchors)
            %[~, grad_k]  = local_regression(X_nb_k,  Y_nb1, x_anchor_k,  h_k,  kernel_name);
            %[~, grad_k1] = local_regression(X_nb_k1, Y_nb1, x_anchor_k1, h_k1, kernel_name);

            % local expectations
            w = kernel_eval(dist_k / max(dist_k,1e-12), kernel_name);
            sw = sum(w);
            if sw <= 0, w = ones(numel(w),1) / numel(w); else, w = w / sw; end
            EY = dot(w, Y_nb1);

            t_k  = (k-1)*dt;
            t_k1 =  k   *dt;

            % f at next level (explicit part)
            f_next_vals = arrayfun(@(j) ...
                nonlinear_function(Y_nb1(j),t_k1, X_nb_k1(j,:), d, nu, a, b), ...
                1:numel(Y_nb1));
            Ef_next = mean(f_next_vals);

            % ------ theta scheme + Newton iteration -------
%             f_impl  = @(u) nonlinear_function(u,grad_k,t_k,x_anchor_k, d, nu, a, b);
%             df_impl = @(u) nonlinear_function_deriv(u, a, b);
            f_impl  = @(u) nonlinear_function(u,t_k,x_anchor_k, d, nu, a, b);
            df_impl = @(u) nonlinear_function_deriv(u, a, b);
            % F(u) = u - EY - dt*( theta f_k(u) + (1-theta) E_k[f_{k+1}] )
            F  = @(u) (u - EY) + dt*( theta * f_impl(u) + (1-theta) * Ef_next );
            dF = @(u) 1 + dt * theta * df_impl(u);

            % Newton starting from EY
            Yk_new(m) = newton_solver(F, dF, EY);
        end

        Y(:,k) = Yk_new;
    end

    %% Results
    u0_est = mean(Y(:,1));
    u_exact = exact_solution(0, x0);
    abs_err = abs(u0_est - u_exact);
    rel_err = abs(u0_est - u_exact) / max(1e-16,abs(u_exact)) * 100;
    fprintf('\n============================================\n');
    fprintf('Allen-Cahn Equation (%dD) Solver Result\n', d);
    fprintf('Parameters: T = %.2f, nu = %.2f, a = %.3g, b = %.3g\n', T, nu, a, b);
    fprintf('Particles: %d, Time steps: %d\n', M, N);
    fprintf('\nEstimated u(0,x0): %.6e (exact: %.6e)\n', u0_est, u_exact);
    fprintf('Absolute error:     %.6e\n', abs_err);
    fprintf('Relative error:     %.6f%%\n', rel_err);
    fprintf('Total time:         %.2fs\n', toc(t_start));
    fprintf('============================================\n');
end


function [alpha0, grad] = local_regression(X,y,anchor,h,kernel_name)
    % local linear regression around 'anchor'
    D = X - anchor;
    m = size(D,1);
    A = [ones(m,1), D];

    % NOTE: your original code uses a vector 'h' so that r becomes ~ones;
    % we keep that behavior to minimize structural changes.
    r  = sqrt(sum(D.^2,2)) ./ max(h, 1e-12);
    w  = kernel_eval(r, kernel_name);        
    if all(w==0)
        w = ones(m,1);
    end
    W = spdiags(w, 0, m, m);

    % Ridge regularization (kept)
    p = size(A,2);
    lambda_ridge= 1e-6; 
    R = diag([0; ones(p-1,1)]) * lambda_ridge;

    AtWA = A' * (W * A) + R;
    AtWy = A' * (W * y);

    coeffs = AtWA \ AtWy;    
    alpha0 = coeffs(1);
    grad   = coeffs(2:end)';  % row vector
end
function w = kernel_eval(r, name)
    % r >= 0
    switch lower(name)
        case 'gaussian'
            % e^{-r^2} 
            w = exp(-(r.^2));
        case 'epanechnikov'
            % (1 - r^2)_+ 
            w = max(0, 1 - r.^2);
        case 'triweight'
            % (1 - r^2)^3_+
            z = max(0, 1 - r.^2);
            w = z.^3;
        otherwise
            w = exp(-(r.^2));
    end
end

function val = exact_solution(t, x)
    % careful: prod(sin(x)) may underflow in high d; that's ok for testing
    val = exp(cos(t)) * cos(prod(x)) * exp(-sum(x.^2));
%    val = cos(t) * exp(-sum(x.^2));
end

function u = newton_solver(f, df, u0)
    u = u0;
    for i = 1:20
        fu = f(u);
        d_fu = df(u);
        if abs(d_fu) < 1e-12, break; end
        delta = -fu / d_fu;
        u = u + delta;
        if abs(delta) < 1e-6, break; end
        % optional: keep u in (-1,1) softly if needed
        if ~isfinite(u), u = max(min(u0, 1-1e-8), -1+1e-8); end
    end
end

% d/du[ -g(u) ] = -g'(u)
% g'(u) = (a/2)*log((1+u)/(1-u)) - b*u
function val = nonlinear_function_deriv(u, a, b)
    uu = max(min(u, 1-1e-12), -1+1e-12); % domain protection
    val = -( 0.5*a*log((1+uu)./(1-uu)) - b*uu );  % = -g'(u)
end

% g(u) = a/2 * [ (1+u)ln(1+u) + (1-u)ln(1-u) ] - (b/2) u^2
% Driver uses: f(t,x,u) = -g(u) + source_term(t,x,...) (same pattern as your original)
function val = nonlinear_function(u, t, x, d, nu, a, b)
    g_u = 0.5*a*((1+u).*log(1+u) + (1-u).*log(1-u)) - 0.5*b*(u.^2);
    val = g_u - source_term(t, x, d, nu, a, b);
end

function r = pdist_to_anchor(Xnb, anchor)
    D = Xnb - anchor;
    r = sqrt(sum(D.^2,2));
    if isempty(r), r = 0; end
end

function val = source_term(t, x, d, nu, a, b)
    % SOURCE TERM for u(t,x) = exp(cos t) * cos(prod x_j) * exp(-|x|^2)
    % Driver will be: f(t,x,u) = -g(u) + source_term(t,x,...) 
    % with g(u) = a/2*((1+u)ln(1+u) + (1-u)ln(1-u)) - b/2*u^2.
    % Here we return: u_t - nu * Δu + g(u_exact).
    %
    % Inputs:
    %   t  : scalar time
    %   x  : 1xd (or dx1) vector
    %   d  : dimension (not strictly needed; will infer from x)
    %   nu : diffusion coefficient
    %   a,b: positive constants in g(u)
    %
    % Output:
    %   val: scalar source term

    x = x(:).';                 % make row
    d = numel(x);               % ensure consistency

    % ----- exact u, and factors -----
    Q     = prod(x);            % Π x_j
    cosQ  = cos(Q);
    sinQ  = sin(Q);
    r2    = sum(x.^2);
    A     = exp(cos(t));        % time-only factor
    E     = exp(-r2);           % space Gaussian
    u     = A * cosQ * E;       % exact solution

    % ----- time derivative: u_t = (-sin t) * A * cosQ * E = -sin(t) * u -----
    time_deriv = -sin(t) * u;

    % ----- build dQ_i = Π_{j≠i} x_j in O(d) time (prefix-suffix trick) -----
    pref = ones(1,d);
    for i = 2:d
        pref(i) = pref(i-1) * x(i-1);
    end
    suff = ones(1,d);
    for i = d-1:-1:1
        suff(i) = suff(i+1) * x(i+1);
    end
    dQi = pref .* suff;         % dQ_i = Π_{j≠i} x_j

    % ----- Laplacian of F(x)=cos(Q)*E; Δu = A * ΔF -----
    % For each i:
    % ?_{ii}F = E * [ cos(Q) * ( - dQ_i^2 + 4 x_i^2 - 2 ) + 4 sin(Q) * x_i * dQ_i ].
    % Hence:
    sum1 = sum( - (dQi.^2) + 4*(x.^2) - 2 );     % Σ_i ( -dQ_i^2 + 4 x_i^2 - 2 )
    sum2 = sum( x .* dQi );                      % Σ_i x_i dQ_i
    laplacian = A * E * ( cosQ * sum1 + 4 * sinQ * sum2 );

    % ----- nonlinearity g(u_exact) with light domain protection for logs -----

    g_u = 0.5*a*((1+u).*log(1+u) + (1-u).*log(1-u)) - 0.5*b*(u.^2);

    % ----- final source term -----
    val = time_deriv - nu * laplacian + g_u;
end
