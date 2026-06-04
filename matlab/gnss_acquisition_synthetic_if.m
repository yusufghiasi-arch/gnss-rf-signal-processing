% gnss_acquisition_synthetic_if.m
%
% Synthetic GNSS raw IF/IQ acquisition demo.
%
% This script simulates a GPS L1 C/A-like baseband signal with a known code
% delay and Doppler shift, adds noise, and then performs a delay-Doppler
% acquisition search to recover the strongest correlation peak.
%
% This demonstrates the core signal-processing concept behind GNSS acquisition
% and delay-Doppler processing:
%
% received signal + local PRN replica + Doppler wipe-off + correlation
% = delay-Doppler acquisition map
%
% Author: Yusof Ghiasi

clear; clc; close all;

rng(7);  % Reproducible noise

%% Simulation settings

prn = 1;
fs = 4.092e6;              % Sampling frequency in Hz
coherent_ms = 1;           % Coherent integration time in milliseconds

% True signal parameters to recover
true_code_delay_samples = 850;
true_doppler_hz = 1500;

% Increase this value if the peak is not visible enough
cnr_scale = 1.5;

%% Generate synthetic received signal

[rx_signal, local_code] = simulate_gnss_signal( ...
    prn, ...
    fs, ...
    coherent_ms, ...
    true_code_delay_samples, ...
    true_doppler_hz, ...
    cnr_scale);

%% Acquisition search

doppler_bins = -5000:250:5000;

[acquisition_map, estimated_delay, estimated_doppler, peak_value] = ...
    acquire_signal(rx_signal, local_code, fs, doppler_bins);

%% Print results

fprintf('Synthetic GNSS Acquisition Demo\n');
fprintf('--------------------------------\n');
fprintf('PRN: %d\n', prn);
fprintf('Sampling frequency: %.3f MHz\n', fs / 1e6);
fprintf('Coherent integration: %d ms\n\n', coherent_ms);

fprintf('True code delay:      %d samples\n', true_code_delay_samples);
fprintf('Estimated code delay: %d samples\n\n', estimated_delay);

fprintf('True Doppler:         %.1f Hz\n', true_doppler_hz);
fprintf('Estimated Doppler:    %.1f Hz\n\n', estimated_doppler);

fprintf('Peak correlation power: %.2e\n', peak_value);

%% Plot acquisition map

figure;
imagesc(0:size(acquisition_map, 2)-1, doppler_bins, 10*log10(acquisition_map + 1e-12));
set(gca, 'YDir', 'normal');
colorbar;
hold on;
plot(estimated_delay, estimated_doppler, 'kx', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('Code delay (samples)');
ylabel('Doppler frequency (Hz)');
title('Synthetic GNSS Acquisition Delay-Doppler Map');
legend('Detected peak');
grid on;


%% Local functions

function ca_code = generate_ca_code(prn)
%GENERATE_CA_CODE Generate GPS L1 C/A code for PRNs 1 to 32.
%
% Inputs
% ------
% prn : GPS satellite PRN number, from 1 to 32
%
% Output
% ------
% ca_code : 1023-chip GPS C/A code with values -1 and +1

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


function [rx_signal, local_code] = simulate_gnss_signal( ...
    prn, fs, coherent_ms, true_code_delay_samples, true_doppler_hz, cnr_scale)
%SIMULATE_GNSS_SIGNAL Simulate a complex baseband GNSS-like signal.

    code_rate = 1.023e6;
    n_samples = round(fs * coherent_ms * 1e-3);

    ca_code = generate_ca_code(prn);
    local_code = resample_ca_code(ca_code, fs, code_rate, n_samples);

    delayed_code = circshift(local_code, true_code_delay_samples);

    t = (0:n_samples-1) / fs;

    carrier = exp(1j * 2 * pi * true_doppler_hz * t);

    clean_signal = cnr_scale * delayed_code .* carrier;

    noise = (randn(1, n_samples) + 1j * randn(1, n_samples)) / sqrt(2);

    rx_signal = clean_signal + noise;

end


function [acquisition_map, estimated_delay, estimated_doppler, peak_value] = ...
    acquire_signal(rx_signal, local_code, fs, doppler_bins)
%ACQUIRE_SIGNAL Perform delay-Doppler acquisition search.
%
% For each Doppler bin, the received signal is mixed by the negative Doppler
% hypothesis and circularly correlated with the local PRN code using FFTs.

    n_samples = length(rx_signal);
    t = (0:n_samples-1) / fs;

    code_fft = fft(local_code);

    acquisition_map = zeros(length(doppler_bins), n_samples);

    for i = 1:length(doppler_bins)

        doppler = doppler_bins(i);

        wipeoff = exp(-1j * 2 * pi * doppler * t);

        mixed_signal = rx_signal .* wipeoff;

        signal_fft = fft(mixed_signal);

        correlation = ifft(signal_fft .* conj(code_fft));

        acquisition_map(i, :) = abs(correlation).^2;

    end

    [peak_value, linear_index] = max(acquisition_map(:));

    [doppler_index, delay_index] = ind2sub(size(acquisition_map), linear_index);

    estimated_doppler = doppler_bins(doppler_index);

    % Convert MATLAB 1-based index to sample delay starting from 0
    estimated_delay = delay_index - 1;

end
