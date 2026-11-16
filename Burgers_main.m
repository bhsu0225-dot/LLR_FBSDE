% High-Dimensional Burgers Equation Solver
% Author: Subihao
% Parallelizes over particles (M)
clc
clear
%% Parameters
d  = 10;
nu = d^2/2;
T  = 0.5;
N  = 800;
M  = 200;
x0 = ones(1,d)*0.2;

theta         = 1;     % theta-scheme
kappa         = 0.5;     % eps_k = kappa * sqrt(dt)
lambda_ridge  = 1e-3;
kernel_name   = 'gaussian'; % 'gaussian' | 'epanechnikov' | 'triweight'
min_nn        = max(d+2, M); % minimal neighbors per anchor

high_dim_burgers_solver(d,nu,T,N,M,x0,theta,kappa,lambda_ridge,kernel_name,min_nn);
 


%% ====================== MAIN SOLVER ======================
function high_dim_burgers_solver(d,nu,T,N,M,x0,theta,kappa,lambda_ridge,kernel_name,min_nn)
    seed= 123;
    rng(seed);
    % ---- time & diffusion ----
    dt    = T / N;
    sigma = sqrt(2*nu) * eye(d);
    eps_k = kappa * sqrt(dt);      % fixed ball radius

    % ---- storage ----
    X = zeros(M, d, N+1);
    u = zeros(M, N+1);

    % ---- init ----
    dW = sqrt(dt) * randn(M, d, N);
    X(:,:,1) = repmat(x0, M, 1);

    % ---- forward: dX = sigma dW ----
    for k = 1:N
        X(:,:,k+1) = X(:,:,k) + dW(:,:,k) * sigma';
    end

    % ---- terminal payoff ----
    for m = 1:M
        u(m,N+1) = exact_solution(T, squeeze(X(m,:,N+1)), d);
    end

    % ---- backward ----
    %fprintf('Backward theta-scheme ($theta$=%.2f)...\n', theta);
    t_start = tic;
    last_pct = -1;

    for k = N:-1:1
        pct = (N-k+1)/N*100;
        if pct ~= last_pct
            fprintf('Progress: %6.2f%% | Elapsed: %.2fs\n', pct, toc(t_start));
            last_pct = pct;
        end

        Xk     = squeeze(X(:,:,k));
        Xk1    = squeeze(X(:,:,k+1));
        u_next = u(:,k+1);

        u_k = zeros(M,1);

        parfor m = 1:M
            xk  = Xk(m,:);
            xk1 = Xk1(m,:);

            % ---------- neighbors at t_k (ball of radius eps_k) ----------
            dist_k = sqrt(sum((Xk - xk).^2, 2));
            idx    = find(dist_k > 0 & dist_k <= eps_k);
            if numel(idx) < min_nn
                [~,ord] = sort(dist_k,'ascend');
                idx = ord(2:min(M, 1+min_nn)); % skip self
            end

            Xnb_k  = Xk(idx,:);
            Xnb_k1 = Xk1(idx,:);
            unb1   = u_next(idx);

            % ---------- tk: WLS (weights from tk geometry) ----------
            [~, grad_k, w_norm_k] = ...
                local_regression(Xnb_k, unb1, xk,  eps_k, kernel_name, lambda_ridge);

            % E_k[u_{k+1}] with tk-weights
            Eu = sum(w_norm_k .* unb1(:)');

            % ---------- tk+1: WLS (weights from tk+1 geometry) ----------
            [~, grad_k1, w_norm_k1] = ...
                local_regression(Xnb_k1, unb1, xk1, eps_k, kernel_name, lambda_ridge);

            % E_k[f_{k+1}] with tk+1-weights
            f_next_vals = arrayfun(@(j) nonlinear_term(unb1(j),grad_k1, d),1:numel(unb1));
            Ef_next = sum(w_norm_k1 .* f_next_vals);

            % ---------- theta-scheme Newton at tk ----------
            f_impl  = @(uv) nonlinear_term(uv, grad_k, d);
            df_impl = @(uv) nonlinear_term_derivative(grad_k);

            F  = @(uv) (uv - Eu) - dt*( theta * f_impl(uv) + (1-theta) * Ef_next );
            dF = @(uv) 1 - dt * theta * df_impl(uv);

            u_k(m) = newton_solver(F, dF, Eu);
        end

        u(:,k) = u_k;
    end

    % ---- results ----
    u0_est  = mean(u(:,1));
    u_exact = exact_solution(0, x0, d);
    abs_err = abs(u0_est - u_exact);
    rel_err = abs_err / max(1e-12, abs(u_exact)) * 100;

    fprintf('\n=========================================\n');
    fprintf('Burgers Equation (%dD) BSDE Solver Result\n', d);
    fprintf('Time steps: %d, particles: %d\n', N, M);
    fprintf('Numerical solution: %.6f\n', u0_est);
    fprintf('Exact solution:     %.6f\n', u_exact);
    fprintf('Absolute error: %.6f\n', abs_err);
    fprintf('Relative Error: %.6f%%\n', rel_err);
    fprintf('Total time: %.2fs\n', toc(t_start));
    fprintf('=========================================\n');
end

%% ===================== WLS local linear regression =====================
% Fit: u ≈ α0 + grad' * (x - anchor)
% Returns: alpha0, grad, and normalized weights w_norm (sum=1)
function [alpha0, grad, w_norm] = local_regression(X, y, anchor, h, kernel_name, lambda_ridge)
    warning('off', 'all'); 
    D = X - anchor;
    m = size(D,1);
    A = [ones(m,1), D];

 
    r  = sqrt(sum(D.^2,2)) / max(h,1e-12);
    w  = kernel_eval(r, kernel_name);        
    if all(w==0)
 
        w = ones(m,1);
    end
    W = spdiags(w, 0, m, m);

 
    % Solve: (A' W A + λ * diag([0,1,...,1])) * coeffs = A' W y
    p = size(A,2);
    R = diag([0; ones(p-1,1)]) * lambda_ridge;

    AtWA = A' * (W * A) + R;
    AtWy = A' * (W * y);

    coeffs = AtWA \ AtWy;    
    alpha0 = coeffs(1);
    grad   = coeffs(2:end)';

    sw = sum(w);
    if sw <= 0, w_norm = (1/m)*ones(1,m);
    else,        w_norm = (w'/sw);
    end
    
end

function w = kernel_eval(r, name)
    switch lower(name)
        case 'gaussian'
            w = exp(-(r.^2));
        case 'epanechnikov'
            w = max(0, 1 - r.^2);
        case 'triweight'
            z = max(0, 1 - r.^2);
            w = z.^3;
        otherwise
            w = exp(-(r.^2));
    end
end


%% ===================== Newton Solver =====================
function u = newton_solver(F, dF, u0)
    u = u0;
    for it = 1:12
        Fu  = F(u);
        dFu = dF(u);
        if abs(dFu) < 1e-12, break; end
        du = -Fu / dFu;
        u  = u + du;
        if abs(du) < 1e-8, break; end
    end
end

% ===================== Nonlinearity derivative =====================
function val = nonlinear_term_derivative(grad_u)

    val = sum(grad_u);
end

% ===================== Nonlinearity  =====================
function val = nonlinear_term(u_val, grad_u, d)
 
    val = (u_val - (2 + d)/(2*d)) * sum(grad_u);
end

function val = exact_solution(t, x, d)
    z = t + sum(x)/d;
    val = exp(z) / (1 + exp(z));
end