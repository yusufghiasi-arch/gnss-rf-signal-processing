% delay_doppler_map_synthetic_gnssr.m
%
% Synthetic GNSS-R Delay-Doppler Map demo.
%
% This script simulates a simplified GNSS-R reflected signal and forms a
% Delay-Doppler Map (DDM) by correlating the received signal with a local
% GPS C/A PRN replica over delay and Doppler hypotheses.
%
% The purpose is to demonstrate the signal-processing principle behind
% GNSS-R DDM generation:
%
% reflected GNSS-like signal
% + local PRN replica
% + Doppler search
% + delay search
% = Delay-Doppler Map
%
% Author: Yusof Ghiasi

clear; clc; close all;

rng(11);  % Reproducible noise

%% Simulation settings

prn = 1;
fs = 4.092e6;          % Sampling frequency in Hz
coherent_ms = 1;       % Coherent integration time in milliseconds

% True reflected-signal parameters
true_reflected_delay_samples = 950;
true_reflected_doppler_hz = -1750;

reflected_amplitude = 1.2;
noise_std = 1.0;

%% Generate synthetic reflected signal

[rx_signal, local_code] = simulate_reflected_signal( ...
    prn, ...
    fs, ...
    coherent_ms, ...
    true_reflected_delay_samples, ...
    true_reflected_doppler_hz, ...
    reflected_amplitude, ...
    noise_std);

%% Delay-Doppler search grid

doppler_bins = -5000:250:5000;
delay_bins = 600:5:1300;

%% Form DDM

ddm = form_ddm( ...
    rx_signal, ...
    local_code, ...
    fs, ...
    doppler_bins, ...
    delay_bins);

%% Detect DDM peak

[estimated_delay, estimated_doppler, peak_power] = detect_ddm_peak( ...
    ddm, ...
    doppler_bins, ...
    delay_bins);

%% Print results

fprintf('Synthetic GNSS-R Delay-Doppler Map Demo\n');
fprintf('---------------------------------------\n');
fprintf('PRN: %d\n', prn);
fprintf('Sampling frequency: %.3f MHz\n', fs / 1e6);
fprintf('Coherent integration: %d ms\n\n', coherent_ms);

fprintf('True reflected delay:      %d samples\n', true_reflected_delay_samples);
fprintf('Estimated reflected delay: %d samples\n\n', estimated_delay);

fprintf('True reflected Doppler:      %.1f Hz\n', true_reflected_doppler_hz);
fprintf('Estimated reflected Doppler: %.1f Hz\n\n', estimated_doppler);

fprintf('Peak DDM power: %.2e\n', peak_power);

%% Plot DDM

figure;
imagesc(delay_bins, doppler_bins, 10*log10(ddm + 1e-12));
set(gca, 'YDir', 'normal');
colorbar;
hold on;
plot(estimated_delay, estimated_doppler, 'kx', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('Delay (samples)');
ylabel('Doppler frequency (Hz)');
title('Synthetic GNSS-R Delay-Doppler Map');
legend('Detected DDM peak');
grid on;


%% Local functions

function ca_code = generate_ca_code(prn)
%GENERATE_CA_CODE Generate GPS L1 C/A code for PRNs 1 to 32.

    g2_taps = [
        2  6;
        3  7;
        4  8;
        5  9;
        1  9;
        2 10;
        1  8;
        2  9;
        3 10;
        2  3;
        3  4;
        5  6;
        6  7;
        7  8;
        8  9;
        9 10;
        1  4;
        2  5;
        3  6;
        4  7;
        5  8;
        6  9;
        1  3;
        4  6;
        5  7;
        6  8;
        7  9;
        8 10;
        1  6;
        2  7;
        3  8;
        4  9
    ];

    if prn < 1 || prn > 32
        error('This demo supports GPS PRNs 1 to 32.');
    end

    tap1 = g2_taps(prn, 1);
    tap2 = g2_taps(prn, 2);

    g1 = ones(1, 10);
    g2 = ones(1, 10);

    ca_binary = zeros(1, 1023);

    for i = 1:1023

        g1_output = g1(10);
        g2_output = xor(g2(tap1), g2(tap2));

        ca_binary(i) = xor(g1_output, g2_output);

        g1_feedback = xor(g1(3), g1(10));

        g2_feedback = xor(xor(xor(xor(xor(g2(2), g2(3)), g2(6)), g2(8)), g2(9)), g2(10));

        g1 = [g1_feedback, g1(1:9)];
        g2 = [g2_feedback, g2(1:9)];

    end

    ca_code = 1 - 2 * ca_binary;

end


function sampled_code = resample_ca_code(ca_code, fs, code_rate, n_samples)
%RESAMPLE_CA_CODE Resample GPS C/A code to the sampling frequency.

    sample_index = 0:n_samples-1;
    chip_index = floor(sample_index * code_rate / fs) + 1;

    % Wrap chip index to 1 to 1023
    chip_index = mod(chip_index - 1, 1023) + 1;

    sampled_code = ca_code(chip_index);

end


function [rx_signal, local_code] = simulate_reflected_signal( ...
    prn, ...
    fs, ...
    coherent_ms, ...
    reflected_delay_samples, ...
    reflected_doppler_hz, ...
    reflected_amplitude, ...
    noise_std)
%SIMULATE_REFLECTED_SIGNAL Simulate a simplified GNSS-R reflected signal.

    code_rate = 1.023e6;
    n_samples = round(fs * coherent_ms * 1e-3);

    ca_code = generate_ca_code(prn);
    local_code = resample_ca_code(ca_code, fs, code_rate, n_samples);

    reflected_code = circshift(local_code, reflected_delay_samples);

    t = (0:n_samples-1) / fs;

    reflected_carrier = exp(1j * 2 * pi * reflected_doppler_hz * t);

    clean_reflection = reflected_amplitude * reflected_code .* reflected_carrier;

    noise = noise_std * (randn(1, n_samples) + 1j * randn(1, n_samples)) / sqrt(2);

    rx_signal = clean_reflection + noise;

end


function ddm = form_ddm(rx_signal, local_code, fs, doppler_bins, delay_bins)
%FORM_DDM Form a simplified Delay-Doppler Map.

    n_samples = length(rx_signal);
    t = (0:n_samples-1) / fs;

    ddm = zeros(length(doppler_bins), length(delay_bins));

    for i = 1:length(doppler_bins)

        doppler = doppler_bins(i);

        wipeoff = exp(-1j * 2 * pi * doppler * t);

        mixed_signal = rx_signal .* wipeoff;

        for j = 1:length(delay_bins)

            delay = delay_bins(j);

            shifted_code = circshift(local_code, delay);

            correlation = sum(mixed_signal .* conj(shifted_code));

            ddm(i, j) = abs(correlation).^2;

        end

    end

end


function [estimated_delay, estimated_doppler, peak_power] = detect_ddm_peak( ...
    ddm, doppler_bins, delay_bins)
%DETECT_DDM_PEAK Detect the strongest Delay-Doppler peak.

    [peak_power, linear_index] = max(ddm(:));

    [doppler_index, delay_index] = ind2sub(size(ddm), linear_index);

    estimated_doppler = doppler_bins(doppler_index);
    estimated_delay = delay_bins(delay_index);

end
