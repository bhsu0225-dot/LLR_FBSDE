% High-Dimensional Hamilton-Jacobi Equation Solver
% Author: Subihao
% Parallelizes over particles (M)
clear
clc
%% Parameters
d = 10; 
T = 1; 
N = 400;
M = 1000;
theta = 1;
D = 1;          % diffusion coefficient
kappa = 0.1;    % nonlinearity coefficient
x0 = ones(1,d)*0.1;
kernel_name = 'gaussian'; % 'gaussian' | 'epanechnikov' | 'triweight'
Hamilton_Jacobi_solver(d,T,N,M,D,kappa,x0,theta, kernel_name);

%% ====================== MAIN SOLVER ======================
function Hamilton_Jacobi_solver(d,T,N,M,D,kappa,x0,theta,kernel_name)
    if nargin < 8 || isempty(theta),       theta = 1;       end
    if nargin < 9 || isempty(kernel_name), kernel_name = 'gaussian'; end

    rng('default');
    dt     = T / N;
    eps_k  = sqrt(dt);            
    
    %% Forward paths
    X = zeros(M, d, N+1);
    Y = zeros(M, N+1);
    X(:,:,1) = repmat(x0, M, 1);
    dW = sqrt(dt) * randn(M, d, N);
    for k = 1:N
 
        X(:,:,k+1) = X(:,:,k) + sqrt(2*D) * dW(:,:,k);
    end

    %% Terminal condition
    for m = 1:M
        Y(m,N+1) = exact_solution(T, squeeze(X(m,:,N+1)), d, D);
    end
    
    %% Backward recursion
    fprintf('Solving backward...');
    
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
            x_anchor_k  = Xk(m,:);
            x_anchor_k1 = Xk_next(m,:);

 
            dist_k  = sqrt(sum((Xk - x_anchor_k).^2, 2));
            idx_ball = find(dist_k > 0 & dist_k <= eps_k);    
            if numel(idx_ball) < M
                [~,ord] = sort(dist_k);
                idx_ball = ord(2:M);
            end

            X_nb_k  = Xk(idx_ball,:);
            X_nb_k1 = Xk_next(idx_ball,:);
            Y_nb1   = Y_next(idx_ball);

 
            h_k  =  pdist_to_anchor(X_nb_k, x_anchor_k);
            h_k1 =  pdist_to_anchor(X_nb_k1, x_anchor_k1);

 
            [~, grad_k] =  ...
                local_regression(X_nb_k,Y_nb1,x_anchor_k,h_k,kernel_name);

  
             [~, grad_k1] = ...
                 local_regression(X_nb_k1,Y_nb1,x_anchor_k1,h_k1,kernel_name);

 
            EY = mean(Y_nb1(:));

            t_k  = (k-1)*dt;
            t_k1 =  k   *dt;
 
             f_next_vals = arrayfun(@(j) ...
                 nonlinear_function(Y_nb1(j),grad_k1,t_k1, X_nb_k1(j,:), D, d, kappa), ...
                 1:numel(Y_nb1));
             Ef_next = mean(f_next_vals);

            % ------ theta scheme newton iteration -------
            f_impl  = @(u) nonlinear_function(u,grad_k,t_k,x_anchor_k,D,d,kappa);
            df_impl = @(u) nonlinear_function_deriv(u, grad_k,  kappa);

            % F(u) = u - EY - dt*( theta f_k(u) + (1-theta) E_k[f_{k+1}] )
             F  = @(u) (u - EY) - dt*( theta * f_impl(u) + (1-theta) * Ef_next );
             dF = @(u) 1 - dt * theta * df_impl(u);


 
            Yk_new(m) = newton_solver(F, dF, EY);
        end

        Y(:,k) = Yk_new;
    end

    u0_est  = mean(Y(:,1));
    u_exact = exact_solution(0, x0, d, D);
    abs_err = abs(u0_est - u_exact);
    rel_err = abs_err / max(1e-15, abs(u_exact));
    fprintf('\nEstimated u(0,x0): %.6e (exact: %.6e)\n', u0_est, u_exact);
    fprintf('Absolute error:     %.6e\n', abs_err);
    fprintf('Relative error:     %.6f%%\n', 100*rel_err);
    fprintf('Total time:         %.2fs\n', toc(t_start));
end


function [alpha0, grad] = local_regression(X,y,anchor,h,kernel_name)
 
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
    lambda_ridge= 1e-6; 
    R = diag([0; ones(p-1,1)]) * lambda_ridge;

    AtWA = A' * (W * A) + R;
    AtWy = A' * (W * y);

    coeffs = AtWA \ AtWy;    
    alpha0 = coeffs(1);
    grad   = coeffs(2:end)';

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


function u = newton_solver(F, dF, u0)
    u = u0;
    for it = 1:3
        Fu  = F(u);
        dFu = dF(u);
        if abs(dFu) < 1e-12, break; end
        du = -Fu / dFu;
        u  = u + du;
        if abs(du) < 1e-8, break; end
    end
end

function val = nonlinear_function_deriv(u_val, grad_u, kappa)
    grad_norm_sq = sum(grad_u.^2);
    val = -kappa * grad_norm_sq;
end



function val = nonlinear_function(u_val, grad_u, t, x, D, d, kappa)
    grad_norm_sq = sum(grad_u.^2);
    val = -kappa * u_val * grad_norm_sq + source_term(t, x, d, D, kappa);
end

function r = pdist_to_anchor(Xnb, anchor)
    D = Xnb - anchor;
    r = sqrt(sum(D.^2,2));
    if isempty(r), r = 0; end
end

function val = source_term(t, x, d, D, kappa)
    scale = 1 + 4 * D * t;
    u = exact_solution(t, x, d, D);
    val = (d * 4 * D * u) / scale;
end

function u = exact_solution(t, x, d, D)
    scale = 1 + 4 * D * t;
    u = scale^(-d/2) * exp(-sum(x.^2) / scale);
end