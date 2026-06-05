%% demo_datasnooping_clean.m
% GitHub-friendly MATLAB demo for time-series data snooping.
%
% This script simulates a geodetic/remote-sensing-style time series with:
%   - deterministic terms: offset, linear trend, annual and semi-annual cycles
%   - stochastic noise: white noise + flicker-like power-law noise
%   - artificial blunders/outliers
%
% It then applies Baarda-style data snooping using normalized residuals
% (w-test) to identify suspicious epochs. The test can be run in two modes:
%   1) white-noise assumption
%   2) colored-noise covariance estimated with LS-VCE
%
% All required helper functions are included at the bottom of this file.
%
% Author/original concept: Yusof Ghiasi, 2016
% Cleaned and reorganized for portfolio/GitHub use.

clc; clear; close all;
rng(7);   % fixed random seed for reproducible GitHub demo

%% ------------------------------------------------------------------------
% 1. Simulation settings
% -------------------------------------------------------------------------
% Use 3 years for a fast demo. You can change this to 10*365 to reproduce
% a longer experiment similar to the original script.
m = 3*365;                         % number of daily observations
time = (1:m)'/365.25;              % time in years
freq = [1, 2];                     % annual and semi-annual terms
noiseIndex = [0, 1];               % 0 = white noise, 1 = flicker-like PL noise

% Functional model parameters:
% x = [offset; trend; cos_annual; sin_annual; cos_semiannual; sin_semiannual]
xTrue = [1; 1.5e-3; sqrt(5)/1000; sqrt(4)/1000; sqrt(3)/1000; sqrt(1)/1000];

% Variance components used to simulate the data.
% These are variances, not standard deviations.
sigmaTrue = [0.002, 0.004].^2;

nOffset  = 0;                      % number of simulated step offsets
nBlunder = 2;                      % number of simulated single-epoch blunders
alpha = 0.001;                     % significance level for two-sided test

%% ------------------------------------------------------------------------
% 2. Build power-law cofactor matrices and simulate time series
% -------------------------------------------------------------------------
Q = buildPowerLawCofactors(noiseIndex, time);

[time, y, ySignal, yNoise, yTrend, offsetIndex, offsetValue, ...
    trueBlunderIndex, trueBlunderValue, QyTrue] = ...
    simulateTimeSeriesWithCofactor(m, Q, sigmaTrue, freq, xTrue, ...
    nOffset, nBlunder, true);

A = makeTimeSeriesDesignMatrix(time, freq);

%% ------------------------------------------------------------------------
% 3. Data snooping with colored-noise covariance estimated by LS-VCE
% -------------------------------------------------------------------------
fprintf('\nRunning data snooping with LS-VCE covariance model...\n');
[detectedColored, detailColored] = dataSnooping(y, A, alpha, Q);

%% ------------------------------------------------------------------------
% 4. Data snooping with simple white-noise assumption
% -------------------------------------------------------------------------
fprintf('\nRunning data snooping with white-noise covariance model...\n');
[detectedWhite, detailWhite] = dataSnooping(y, A, alpha);

%% ------------------------------------------------------------------------
% 5. Print summary
% -------------------------------------------------------------------------
fprintf('\n================ DATA SNOOPING SUMMARY ================\n');
fprintf('True blunder indices:                 %s\n', mat2str(sort(trueBlunderIndex(:)')));
fprintf('Detected indices, LS-VCE covariance:  %s\n', mat2str(sort(detectedColored(:)')));
fprintf('Detected indices, white covariance:   %s\n', mat2str(sort(detectedWhite(:)')));
fprintf('Two-sided alpha:                      %.4g\n', alpha);
fprintf('Critical |w| threshold:               %.3f\n', detailColored.threshold);
fprintf('=======================================================\n');

%% ------------------------------------------------------------------------
% 6. Plot results
% -------------------------------------------------------------------------
figure('Name','Data snooping result','Color','w');
plot(time, y, '.', 'DisplayName','Simulated observations'); hold on; grid on;
plot(time, yTrend + ySignal, 'LineWidth', 1.5, 'DisplayName','True deterministic signal');

if ~isempty(trueBlunderIndex)
    plot(time(trueBlunderIndex), y(trueBlunderIndex), 'ko', ...
        'MarkerSize', 8, 'LineWidth', 1.5, 'DisplayName','True blunders');
end
if ~isempty(detectedColored)
    plot(time(detectedColored), y(detectedColored), 'rx', ...
        'MarkerSize', 10, 'LineWidth', 2, 'DisplayName','Detected by LS-VCE model');
end
if ~isempty(detectedWhite)
    plot(time(detectedWhite), y(detectedWhite), 'bs', ...
        'MarkerSize', 7, 'LineWidth', 1.5, 'DisplayName','Detected by white-noise model');
end

xlabel('Time (years)');
ylabel('Simulated observation');
title('Baarda-style data snooping for simulated time series');
legend('Location','best');

figure('Name','Maximum w-test value per iteration','Color','w');
plot(detailColored.maxAbsW, '-o', 'LineWidth', 1.5, 'DisplayName','LS-VCE covariance'); hold on; grid on;
plot(detailWhite.maxAbsW, '-s', 'LineWidth', 1.5, 'DisplayName','White covariance');
yline(detailColored.threshold, '--', 'DisplayName','Critical threshold');
xlabel('Iteration');
ylabel('Maximum |w|');
title('Data snooping iteration history');
legend('Location','best');

%% ========================================================================
% Local functions
% ========================================================================

function Q = buildPowerLawCofactors(noiseIndex, time)
% Build one cofactor matrix for each requested power-law noise exponent.
    Q = cell(1, numel(noiseIndex));
    dt = 1/365.25;
    tRel = time - time(1) + dt;
    for k = 1:numel(noiseIndex)
        alpha = noiseIndex(k);
        Q{k} = makePowerLawCofactor(alpha, tRel);
        Q{k} = Q{k} * dt^(alpha/2);
    end
end

function A = makeTimeSeriesDesignMatrix(t, freq)
% Design matrix for offset, linear trend, and sinusoidal terms.
% t    : time in years, column vector
% freq : vector of frequencies in cycles/year
    t = t(:);
    A = zeros(numel(t), 2 + 2*numel(freq));
    A(:,1) = 1;
    A(:,2) = t;
    for k = 1:numel(freq)
        A(:,2*k+1) = cos(2*pi*freq(k)*t);
        A(:,2*k+2) = sin(2*pi*freq(k)*t);
    end
end

function U = makePowerLawCofactor(alpha, timeYears)
% Generate a simple power-law cofactor matrix.
% alpha = 0 gives a white-noise-like identity cofactor.
% alpha = 1 gives a flicker-like cofactor.
    dayIndex = round(timeYears(:) * 365.25);
    dayIndex(dayIndex < 1) = 1;
    mFull = dayIndex(end);

    H = zeros(mFull,1);
    H(1) = 1;
    for i = 2:mFull
        H(i) = (alpha/2 + i - 2) * H(i-1) / (i - 1);
    end

    keep = ismember(1:mFull, dayIndex);
    Ufull = toeplitz(H);
    Ufull = triu(Ufull);
    U = Ufull(:, keep);
    U = U' * U;
end

function [time, y, ySignal, yNoise, yTrend, offsetIndex, offsetValue, ...
    blunderIndex, blunderValue, Qy] = simulateTimeSeriesWithCofactor(...
    m, Q, sigma, freq, xTrue, nOffset, nBlunder, makePlot)
% Simulate a time series with deterministic signal, colored noise, optional
% step offsets, and optional single-epoch blunders.

    time = (1:m)'/365.25;
    p = numel(Q);
    Qy = zeros(m,m);
    for k = 1:p
        Qy = Qy + sigma(k) * Q{k};
    end

    % Add tiny diagonal jitter for numerical stability in Cholesky factorization.
    Qy = (Qy + Qy')/2 + 1e-14*eye(m);
    R = chol(Qy, 'lower');
    yNoise = R * randn(m,1);

    A = makeTimeSeriesDesignMatrix(time, freq);
    yTrend = A(:,1:2) * xTrue(1:2);
    ySignal = A(:,3:end) * xTrue(3:end);
    y = yTrend + ySignal + yNoise;

    offsetIndex = [];
    offsetValue = [];
    if nOffset > 0
        offsetIndex = sort(randperm(m, nOffset))';
        offsetValue = 10 * std(yNoise) * (2*rand(nOffset,1) - 1);
        for k = 1:nOffset
            y(offsetIndex(k):end) = y(offsetIndex(k):end) + offsetValue(k);
        end
    end

    blunderIndex = [];
    blunderValue = [];
    if nBlunder > 0
        blunderIndex = sort(randperm(m, nBlunder))';
        blunderValue = 10 * std(yNoise) * (2*rand(nBlunder,1) - 1);
        y(blunderIndex) = y(blunderIndex) + blunderValue;
    end

    if makePlot
        figure('Name','Simulated time series','Color','w');
        plot(time, y, '.', 'Color', [0.45 0.45 0.45], 'DisplayName','Observation'); hold on; grid on;
        plot(time, yTrend + ySignal, 'r', 'LineWidth', 1.5, 'DisplayName','Deterministic signal');
        for k = 1:numel(offsetIndex)
            xline(time(offsetIndex(k)), '--b', 'Offset');
        end
        for k = 1:numel(blunderIndex)
            xline(time(blunderIndex(k)), '--g', 'Blunder');
        end
        xlabel('Time (years)');
        ylabel('Simulated observation');
        title('Simulated time series with trend, seasonal terms, noise, and blunders');
        legend('Location','best');
    end
end

function [blunderIndex, details] = dataSnooping(y, A, alpha, Q)
% Iterative Baarda-style data snooping.
%
% INPUTS
% y     : observation vector
% A     : design matrix
% alpha : two-sided significance level
% Q     : optional cell array of cofactor matrices. If omitted, an identity
%         covariance matrix is used.
%
% OUTPUTS
% blunderIndex : indices in the original time series flagged as outliers
% details      : structure containing threshold and iteration history

    yWork = y(:);
    AWork = A;
    originalIndex = (1:numel(yWork))';
    blunderIndex = [];
    maxAbsW = [];

    % Equivalent to norminv(1 - alpha/2), but avoids Statistics Toolbox.
    threshold = sqrt(2) * erfinv(1 - alpha);

    useColoredCovariance = (nargin == 4) && ~isempty(Q);
    if useColoredCovariance
        QWork = Q;
        sigma0 = 1e-6 * ones(1, numel(QWork));
    end

    while true
        mCurrent = numel(yWork);

        if useColoredCovariance
            [~, ~, Qy, QyInv, ~, ~, ehat, PAo] = ...
                lsvce(yWork, AWork, QWork, sigma0, 1e-12, 40, false);
            Qe = PAo * Qy;

            w = zeros(mCurrent,1);
            for i = 1:mCurrent
                ai = zeros(mCurrent,1);
                ai(i) = 1;
                B = ai' * QyInv;
                denom = sqrt(max(B * Qe * B', eps));
                w(i) = (B * ehat) / denom;
            end
        else
            Qy = eye(mCurrent);
            [~, ~, ehat, ~, PAo] = leastSquaresEstimate(yWork, AWork, Qy, false);
            Qe = PAo * Qy;
            w = ehat ./ sqrt(max(diag(Qe), eps));
        end

        [thisMax, localMaxIndex] = max(abs(w));
        maxAbsW(end+1,1) = thisMax; %#ok<AGROW>

        if thisMax <= threshold
            break;
        end

        % Remove the single most suspicious observation and repeat.
        blunderIndex(end+1,1) = originalIndex(localMaxIndex); %#ok<AGROW>
        yWork(localMaxIndex) = [];
        AWork(localMaxIndex,:) = [];
        originalIndex(localMaxIndex) = [];

        if useColoredCovariance
            for k = 1:numel(QWork)
                QWork{k}(localMaxIndex,:) = [];
                QWork{k}(:,localMaxIndex) = [];
            end
        end

        % Safety stop to avoid excessive removal if the model is unsuitable.
        if numel(yWork) <= size(AWork,2) + 1
            warning('Data snooping stopped because too few observations remain.');
            break;
        end
    end

    blunderIndex = sort(blunderIndex);
    details.threshold = threshold;
    details.maxAbsW = maxAbsW;
end

function [SIGMA, QSIGMA, Qy, QyInv, xhat, yhat, ehat, PAo, df, logLikelihood] = ...
    lsvce(y, A, Q, sigma0, threshold, maxIter, computeLogLikelihood)
% Least-Squares Variance Component Estimation with non-negative variance
% components.

    y = y(:);
    p = numel(Q);
    m = numel(y);
    df = size(A,1) - size(A,2);
    sigma0 = sigma0(:)';
    SIGMA = [];
    delta = inf(1,p);
    iter = 1;

    while max(delta) > threshold && iter <= maxIter
        Qy = zeros(m,m);
        for k = 1:p
            Qy = Qy + sigma0(k) * Q{k};
        end
        Qy = (Qy + Qy')/2 + 1e-14*eye(m);
        QyInv = Qy \ eye(m);

        if isempty(A)
            PAo = eye(m);
        else
            N = A' * QyInv * A;
            PAo = eye(m) - A * (N \ (A' * QyInv));
        end

        ehat = PAo * y;
        B = QyInv * PAo;
        C = QyInv * ehat;

        l = zeros(p,1);
        Nvc = zeros(p,p);
        for i = 1:p
            l(i) = 0.5 * C' * Q{i} * C;
            for j = 1:p
                Nvc(i,j) = 0.5 * trace(B * Q{i} * B * Q{j});
            end
        end

        [sigma, QSIGMA] = nonNegativeVarianceLS(Nvc, l);
        sigma = sigma(:)';
        delta = abs(sigma - sigma0);
        sigma0 = sigma;
        SIGMA = [SIGMA; sigma]; %#ok<AGROW>
        iter = iter + 1;
    end

    if isempty(SIGMA)
        sigma = sigma0;
    else
        sigma = SIGMA(end,:);
    end

    Qy = zeros(m,m);
    for k = 1:p
        Qy = Qy + sigma(k) * Q{k};
    end
    Qy = (Qy + Qy')/2 + 1e-14*eye(m);
    QyInv = Qy \ eye(m);

    if isempty(A)
        PAo = eye(m);
        xhat = [];
        yhat = [];
        ehat = PAo * y;
    else
        AtQInv = A' * QyInv;
        Qxhat = (AtQInv * A) \ eye(size(A,2));
        PAo = eye(m) - A * Qxhat * AtQInv;
        xhat = Qxhat * AtQInv * y;
        yhat = A * xhat;
        ehat = PAo * y;
    end

    if exist('Nvc','var')
        QSIGMA = Nvc \ eye(p);
    else
        QSIGMA = NaN(p,p);
    end

    logLikelihood = [];
    if computeLogLikelihood
        eigenValues = eig(Qy);
        logLikelihood = -0.5*m*log(2*pi) - 0.5*sum(log(eigenValues)) - 0.5*ehat'*QyInv*ehat;
    end
end

function [xhat, yhat, ehat, sigma2, PAo, df, QyScaled, logLikelihood] = ...
    leastSquaresEstimate(y, A, Qy, computeLogLikelihood)
% Weighted least-squares estimate for a known covariance matrix Qy.

    y = y(:);
    m = numel(y);
    QyInv = Qy \ eye(m);
    N = A' * QyInv * A;
    xhat = N \ (A' * QyInv * y);
    yhat = A * xhat;
    PAo = eye(m) - A * (N \ (A' * QyInv));
    ehat = PAo * y;
    df = size(A,1) - size(A,2);
    sigma2 = (ehat' * QyInv * ehat) / df;
    QyScaled = sigma2 * Qy;

    logLikelihood = [];
    if computeLogLikelihood
        QyScaledInv = QyScaled \ eye(m);
        eigenValues = eig(QyScaled);
        logLikelihood = -0.5*m*log(2*pi) - 0.5*sum(log(eigenValues)) - 0.5*ehat'*QyScaledInv*ehat;
    end
end

function [s, Qs] = nonNegativeVarianceLS(N, L)
% Simple active-set-like non-negative solution used by the original code.
% Solves approximately: N*s = L, subject to s >= 0.

    p = numel(L);
    mu0 = -L(:);
    s0 = zeros(p,1);
    s = s0;
    previous = s0 + 1;

    while norm(s - previous) > 1e-12
        for k = 1:p
            s(k) = max(0, s0(k) - mu0(k)/N(k,k));
            mu0 = mu0 + (s(k) - s0(k)) * N(:,k);
        end
        previous = s0;
        s0 = s;
    end

    zeroIdx = find(s == 0);
    Ct = zeros(numel(zeroIdx), p);
    for k = 1:numel(zeroIdx)
        Ct(k, zeroIdx(k)) = 1;
    end

    Ni = N \ eye(p);
    if isempty(Ct)
        Qs = Ni;
    else
        C = Ct';
        Qs = Ni * (eye(p) - C * ((Ct * Ni * C) \ Ct) * Ni);
    end
end

%% Optional power-law noise-model w-test utilities -------------------------
% These functions are not required by the main demo above, but they are
% included because they were part of the original data-snooping toolkit.

function [indexPL, wIndexPL, indexGrid, wValues] = powerLawWTest(y, freq, H0)
% Test candidate power-law noise exponents against a null model H0.
% H0 can be [0] for white noise or, for example, [0 1] for white+flicker.

    y = y(:);
    m = numel(y);
    time = (1:m)'/365.25;
    A = makeTimeSeriesDesignMatrix(time, freq);

    if any(H0 ~= 0)
        Q = buildPowerLawCofactors(H0, time);
        sigma0 = 1e-6 * ones(1, numel(Q));
        [~,~,Qy,~,~,~,ehat,PAo,df] = lsvce(y,A,Q,sigma0,1e-12,40,false);
    else
        Qy = eye(m);
        [~,~,ehat,sigma2,PAo,df,Qy] = leastSquaresEstimate(y,A,Qy,false);
    end

    indexGrid = -1:0.1:3;
    indexGrid = setdiff(round(indexGrid, 2), round(H0, 2));
    wValues = zeros(size(indexGrid));

    for k = 1:numel(indexGrid)
        Cy = buildPowerLawCofactors(indexGrid(k), time);
        Cy = Cy{1};
        if any(H0 ~= 0)
            wValues(k) = wTestQ0(ehat, Qy, PAo, Cy, df);
        else
            wValues(k) = wTestWhite(ehat, Cy, PAo, sqrt(sigma2), df);
        end
    end

    [wIndexPL, idx] = max(wValues);
    indexPL = indexGrid(idx);
end

function wValue = wTestWhite(ehat, Cy, PAo, sigma, df)
% w-test under a white-noise null hypothesis.
    a1 = df * ehat' * Cy * ehat;
    a2 = trace(Cy * PAo) * (ehat' * ehat);
    numerator = a1 - a2;
    b1 = 2 * df^2 * trace(Cy * PAo * Cy * PAo);
    b2 = 2 * df * trace(Cy * PAo)^2;
    denominator = sigma^2 * sqrt(max(b1 - b2, eps));
    wValue = numerator / denominator;
end

function wValue = wTestS2Q1(ehat, Qy, PAo, Cy, df)
% w-test from Amiri-Simkooei thesis, Eq. 5.81 style.
    QyInv = Qy \ eye(size(Qy));
    QehatRInv = QyInv * PAo;
    term1 = 0.5 * Cy - (trace(Cy * QehatRInv)/(2*df)) * Qy;
    numerator = ehat' * QyInv * term1 * QyInv * ehat;
    denominator = sqrt(max(0.5*trace(Cy*QehatRInv*Cy*QehatRInv) - ...
        (1/(2*df))*trace(Cy*QehatRInv)^2, eps));
    wValue = numerator / denominator;
end

function wValue = wTestQ0(ehat, Qy, PAo, Cy, df)
% w-test from Amiri-Simkooei thesis, Eq. 5.75 style.
    QyInv = Qy \ eye(size(Qy));
    QehatRInv = QyInv * PAo;
    CQr = Cy * QehatRInv;
    wd = sqrt(max(0.5 * trace(CQr * CQr), eps));
    M = (0.5 * QyInv * Cy * QyInv) / wd;
    m0 = (0.5 * trace(CQr)) / wd;
    wValue = ehat' * M * ehat - m0;
end
