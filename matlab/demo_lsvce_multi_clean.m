%% demo_lsvce_multi_clean.m
% Least-Squares Variance Component Estimation (LS-VCE) demo for multiple
% time series.
%
% This script simulates three correlated-style geodetic/remote-sensing time
% series with a deterministic model and colored noise, then estimates the
% variance components of the noise using LS-VCE.
%
% The code is written as a single GitHub-friendly MATLAB file: the demo is at
% the top and all required helper functions are included below as local
% functions. No external project-specific files are required.
%
% Original concept: LS-VCE for multi-component time-series noise analysis.
% Cleaned version: self-contained demo with comments, safer linear algebra,
% and replacement for missing/custom helper functions.

clc; clear; close all;
rng(7);  % Reproducible simulation

%% User settings
numSeries = 3;                 % Example: three components or three stations
numDays   = 3 * 365;           % Use 10*365 for a larger/full-size demo
freq      = [1, 2];            % Annual and semi-annual frequencies [cycles/year]
noiseExp  = [0, 1];            % Power-law noise exponents: 0=white, 1=flicker-like
plotTimeSeries = true;

% Time vector in decimal years
time = (1:numDays)' / 365.25;
dt = 1 / 365.25;

%% Build cofactor matrices for the selected power-law noise models
numNoiseModels = numel(noiseExp);
Q = cell(1, numNoiseModels);
for k = 1:numNoiseModels
    % Small positive offset avoids zero lag issues at the first epoch.
    Q{k} = makePowerLawCofactor(noiseExp(k), time - time(1) + dt);
    Q{k} = Q{k} * dt^(noiseExp(k) / 2);
end

%% True variance components used for simulation
% Rows are time series; columns are variance components.
% These are variances, not standard deviations.
trueSigma = [0.0015, 0.003; ...
             0.0020, 0.004; ...
             0.0030, 0.006].^2;

%% Deterministic model parameters
% Model columns are: intercept, trend, cos(annual), sin(annual),
% cos(semiannual), sin(semiannual).
xTrue = zeros(2 + 2*numel(freq), numSeries);
xTrue(:,1) = [100; 5; sqrt(2); sqrt(2); 1/sqrt(2); 1/sqrt(2)] / 1000;
xTrue(:,2) = [100; 5; sqrt(2); sqrt(3); 1;         1/sqrt(2)] / 1000;
xTrue(:,3) = [100; 1; sqrt(4); sqrt(5); sqrt(2);   sqrt(2)]   / 1000;

%% Simulate observations
numOffsets  = 0;
numBlunders = 0;
[time, Y, Ysignal, Ynoise, Ytrend, offsetIndex, offsetValue, ...
    blunderIndex, blunderValue, QyTrue] = simulateTimeSeriesWithCofactors( ...
    numSeries, numDays, Q, trueSigma, freq, xTrue, numOffsets, numBlunders, plotTimeSeries);

%% Run LS-VCE
A = makeTimeSeriesDesignMatrix(time, freq);
initialSigma = repmat((0.001)^2, 1, numNoiseModels);
threshold = 1e-6;
maxIter = 40;
computeLogLikelihood = true;

[SIGMA, QSIGMA, QyEst, QyInv, xHat, yHat, eHat, PAo, df, S, Cor, logL] = ...
    lsvceMulti(Y, A, Q, initialSigma, threshold, maxIter, computeLogLikelihood);

%% Print compact results
finalSigma = SIGMA(end,:);
estimatedStd = sqrt(kron(diag(S), finalSigma));
trueStd = sqrt(trueSigma);

fprintf('\nFinal estimated variance components:\n');
disp(finalSigma);

fprintf('Estimated noise standard deviations by series and noise component:\n');
disp(estimatedStd);

fprintf('True noise standard deviations used in the simulation:\n');
disp(trueStd);

fprintf('Estimated inter-series covariance matrix S:\n');
disp(S);

fprintf('Estimated inter-series correlation matrix (diagonal = 1):\n');
disp(Cor);

% The original helper function corr2.m used a display-style matrix:
% off-diagonal values are correlations, while diagonal values are
% 10*standard deviation. This is kept here only for compatibility with
% older outputs.
CorDisplayOriginalStyle = covarianceToOriginalCorr2Display(S);
fprintf('Original corr2-style display matrix: off-diagonal = correlation, diagonal = 10*std:\n');
disp(CorDisplayOriginalStyle);

if ~isempty(logL)
    fprintf('Log-likelihood by time series:\n');
    disp(logL(:)');
end

%% Plot variance-component convergence
figure('Name','LS-VCE convergence');
plot(SIGMA, 'LineWidth', 1.5);
grid on;
xlabel('Iteration');
ylabel('Estimated variance component');
legend(compose('Noise exponent %g', noiseExp), 'Location', 'best');
title('LS-VCE variance-component convergence');

%% Local functions
function A = makeTimeSeriesDesignMatrix(t, freq)
% makeTimeSeriesDesignMatrix builds a design matrix for a time series model.
%
% Model:
%   y(t) = intercept + trend*t + sum_i [a_i cos(2*pi*f_i*t) + b_i sin(2*pi*f_i*t)]

    t = t(:);
    A = [ones(numel(t),1), t];

    for i = 1:numel(freq)
        A(:, 2*i + 1) = cos(2*pi*freq(i)*t);
        A(:, 2*i + 2) = sin(2*pi*freq(i)*t);
    end
end

function U = makePowerLawCofactor(alpha, year)
% makePowerLawCofactor creates a power-law noise cofactor matrix.
%
% alpha = 0 gives a white-noise-like cofactor. Larger alpha values produce
% increasingly time-correlated noise structures.

    year = round(year(:) * 365.25);
    year(year < 1) = 1;
    maxDay = max(year);

    H = zeros(maxDay,1);
    H(1) = 1;
    for i = 2:maxDay
        H(i) = (alpha/2 + i - 2) * H(i-1) / (i - 1);
    end

    selectedDays = ismember(1:maxDay, year);
    Ufull = triu(toeplitz(H));
    U = Ufull(:, selectedDays)' * Ufull(:, selectedDays);
end

function [time, Y, Ysignal, Ynoise, Ytrend, offsetIndex, offsetValue, ...
          blunderIndex, blunderValue, Qy] = simulateTimeSeriesWithCofactors( ...
          numSeries, numDays, Q, sigma, freq, xTrue, numOffsets, numBlunders, makePlot)
% simulateTimeSeriesWithCofactors simulates multiple time series using a
% deterministic seasonal/trend model plus colored noise.

    time = (1:numDays)' / 365.25;
    numNoiseModels = numel(Q);
    Qy = cell(1, numSeries);
    Ynoise = zeros(numDays, numSeries);

    for j = 1:numSeries
        Qy{j} = zeros(numDays);
        for k = 1:numNoiseModels
            Qy{j} = Qy{j} + sigma(j,k) * Q{k};
        end

        % Add a tiny diagonal stabilizer in case the simulated covariance is
        % numerically close to singular.
        R = chol(Qy{j} + 1e-14*eye(numDays), 'upper');
        Ynoise(:,j) = R' * randn(numDays,1);
    end

    A = makeTimeSeriesDesignMatrix(time, freq);
    Ytrend  = A(:,1:2) * xTrue(1:2,:);
    Ysignal = A(:,3:end) * xTrue(3:end,:);
    Y = Ytrend + Ysignal + Ynoise;

    offsetIndex = [];
    offsetValue = [];
    blunderIndex = [];
    blunderValue = [];

    stdY = std(Ynoise, 0, 1);

    if numOffsets > 0
        offsetIndex = randi(numDays, numOffsets, 1);
        offsetValue = zeros(numOffsets, numSeries);
        offsetDesign = tril(ones(numDays));
        for j = 1:numSeries
            offsetValue(:,j) = 3 * stdY(j) * (2*rand(numOffsets,1) - 1);
            for i = 1:numOffsets
                Y(:,j) = Y(:,j) + offsetValue(i,j) * offsetDesign(:, offsetIndex(i));
            end
        end
    end

    if numBlunders > 0
        blunderIndex = randi(numDays, numBlunders, 1);
        blunderValue = zeros(numBlunders, numSeries);
        for j = 1:numSeries
            blunderValue(:,j) = 3 * stdY(j) * (2*rand(numBlunders,1) - 1);
            for i = 1:numBlunders
                Y(blunderIndex(i),j) = Y(blunderIndex(i),j) + blunderValue(i,j);
            end
        end
    end

    if makePlot
        figure('Name','Simulated time series');
        for j = 1:numSeries
            subplot(numSeries,1,j);
            plot(time, Y(:,j), 'Color', [0.5 0.5 0.5]); hold on;
            plot(time, Ytrend(:,j) + Ysignal(:,j), 'r', 'LineWidth', 1.5);
            for i = 1:numel(offsetIndex)
                plot([time(offsetIndex(i)) time(offsetIndex(i))], ylim, '--b');
            end
            for i = 1:numel(blunderIndex)
                plot([time(blunderIndex(i)) time(blunderIndex(i))], ylim, '--g');
            end
            grid on;
            xlabel('Time [year]');
            ylabel(sprintf('Series %d [m]', j));
            legend('Observed simulation', 'True deterministic signal', 'Location', 'best');
        end
    end
end

function [SIGMA, QSIGMA, Qy, QyInv, xHat, yHat, eHat, PAo, df, S, Cor, logL] = ...
    lsvceMulti(Y, A, Q, sigma0, threshold, maxIter, computeLogLikelihood)
% lsvceMulti estimates variance components for multiple time series.
%
% Inputs:
%   Y      : m-by-r observation matrix, where r is the number of time series
%   A      : m-by-n design matrix for the deterministic model
%   Q      : 1-by-p cell array of cofactor matrices
%   sigma0 : 1-by-p initial variance components
%
% Outputs:
%   SIGMA  : variance component estimates at each iteration
%   S      : estimated covariance matrix between the r time series
%   Cor    : correlation matrix derived from S
%   logL   : optional log-likelihood values

    [m, n] = size(A);
    [~, r] = size(Y);
    p = numel(Q);
    df = m - n;

    if df <= 0
        error('The number of observations must be larger than the number of model parameters.');
    end

    sigma0 = sigma0(:)';
    SIGMA = zeros(maxIter, p);
    delta = inf(1,p);
    iter = 1;

    fprintf('Running LS-VCE iterations...\n');

    while max(delta) > threshold && iter <= maxIter
        Qy = buildCovariance(Q, sigma0);
        QyInv = Qy \ eye(m);

        if isempty(A)
            PAo = eye(m);
        else
            normalMat = A' * QyInv * A;
            QxHatTemp = normalMat \ eye(size(normalMat));
            PAo = eye(m) - A * QxHatTemp * A' * QyInv;
        end

        eHatIter = PAo * Y;
        B = QyInv * PAo;
        C = QyInv * eHatIter;

        residualCovInv = (eHatIter' * C / df) \ eye(r);

        l = zeros(p,1);
        N = zeros(p,p);
        for i = 1:p
            l(i) = 0.5 * traceProduct(C' * Q{i} * C, residualCovInv);
            for j = 1:p
                N(i,j) = 0.5 * r * traceProduct(B * Q{i}, B * Q{j});
            end
        end

        [sigmaNew, QSIGMA] = nonnegativeLeastSquaresVCE(N, l);
        sigmaNew = sigmaNew(:)';
        delta = abs(sigmaNew - sigma0);
        sigma0 = sigmaNew;
        SIGMA(iter,:) = sigmaNew;

        fprintf('  Iteration %02d: ', iter);
        fprintf('%12.5e ', sigmaNew);
        fprintf('\n');

        iter = iter + 1;
    end

    SIGMA = SIGMA(1:iter-1,:);
    sigmaFinal = SIGMA(end,:);

    Qy = buildCovariance(Q, sigmaFinal);
    QyInv = Qy \ eye(m);

    if isempty(A)
        PAo = eye(m);
        xHat = [];
        yHat = [];
        eHat = PAo * Y;
    else
        AtQinv = A' * QyInv;
        QxHat = (AtQinv * A) \ eye(n);
        PAo = eye(m) - A * QxHat * AtQinv;
        xHat = QxHat * (AtQinv * Y);
        yHat = A * xHat;
        eHat = PAo * Y;
    end

    Cfinal = QyInv * eHat;
    S = eHat' * Cfinal / df;
    Cor = covarianceToCorrelation(S);

    % This covariance is approximate. It is mainly useful as a quality
    % indicator for the variance-component estimates.
    QSIGMA = pinv(N);

    logL = [];
    if computeLogLikelihood
        eigVals = eig((Qy + Qy')/2);
        eigVals(eigVals <= 0) = eps;
        logL = (-0.5*m*log(2*pi) - 0.5*sum(log(eigVals))) * ones(r,1) ...
               - 0.5 * diag(eHat' * QyInv * eHat);
    end
end

function Qy = buildCovariance(Q, sigma)
% buildCovariance forms Qy = sum_i sigma_i Q_i.
    m = size(Q{1},1);
    Qy = zeros(m);
    for i = 1:numel(Q)
        Qy = Qy + sigma(i) * Q{i};
    end
    Qy = (Qy + Qy') / 2;
    Qy = Qy + 1e-14 * eye(m);
end

function [s, Qs] = nonnegativeLeastSquaresVCE(N, L)
% nonnegativeLeastSquaresVCE solves a small non-negative LS problem for the
% LS-VCE variance components using the iterative scheme from the original
% code, with safer pseudo-inverses for constrained covariance estimation.

    p = length(L);
    mu = -L(:);
    s0 = zeros(p,1);
    sTest = s0 + 1;
    s = s0;

    while norm(s - sTest) > 1e-12
        for k = 1:p
            if abs(N(k,k)) < eps
                s(k) = 0;
            else
                s(k) = max(0, s0(k) - mu(k) / N(k,k));
            end
            mu = mu + (s(k) - s0(k)) * N(:,k);
        end
        sTest = s0;
        s0 = s;
    end

    zeroIdx = find(s == 0);
    if isempty(zeroIdx)
        Qs = pinv(N);
    else
        Ct = zeros(numel(zeroIdx), p);
        for i = 1:numel(zeroIdx)
            Ct(i, zeroIdx(i)) = 1;
        end
        C = Ct';
        Ni = pinv(N);
        middle = pinv(Ct * Ni * C);
        Pco = eye(p) - C * middle * Ct * Ni;
        Qs = Ni * Pco;
    end
end

function z = traceProduct(A, B)
% traceProduct replaces the missing custom trace2(A,B) function.
% It computes trace(A*B), which is the operation used in the LS-VCE normal
% equations.
    z = trace(A * B);
end

function Cor = covarianceToCorrelation(S)
% covarianceToCorrelation converts a covariance matrix to a correlation matrix.
    d = sqrt(abs(diag(S)));
    Cor = S ./ (d * d');
    Cor(~isfinite(Cor)) = 0;
    Cor(1:size(Cor,1)+1:end) = 1;
end

function C = covarianceToOriginalCorr2Display(S)
% covarianceToOriginalCorr2Display reproduces the behavior of the original
% custom corr2.m helper:
%   - off-diagonal entries are correlation coefficients
%   - diagonal entries are 10*standard deviations from the covariance matrix
%
% This is not a standard correlation matrix, so the clean code reports the
% standard correlation matrix separately as Cor.

    m = size(S,1);
    C = zeros(m,m);

    for i = 1:m-1
        for j = i+1:m
            denom = sqrt(S(i,i) * S(j,j));
            if denom > 0
                C(i,j) = S(i,j) / denom;
            else
                C(i,j) = 0;
            end
        end
    end

    C = diag(sqrt(diag(S)) * 10) + C + C';
end
