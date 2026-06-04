% tdoa_fdoa_two_receiver_demo.m
%
% Two-receiver TDOA/FDOA estimation demo.
%
% This script simulates an unknown RF signal received by two receivers. The
% second receiver observes the same signal with a time delay and a frequency
% offset. A joint delay-frequency ambiguity search is then used to estimate
% the relative time delay and frequency offset.
%
% This is a simplified signal-processing demo relevant to RF geolocation:
%
% receiver 1 signal
% receiver 2 signal
% -> delay/frequency search
% -> TDOA estimate
% -> FDOA estimate
%
% Author: Yusof Ghiasi

clear; clc; close all;

rng(21);  % Reproducible noise

%% Signal settings

fs = 2.0e6;              % Sampling frequency in Hz
duration_s = 0.002;      % Signal duration in seconds
bandwidth_hz = 300e3;    % Approximate signal bandwidth in Hz

% True receiver-2 relative observables
true_tdoa_samples = 37;
true_fdoa_hz = -2200;

snr_db = 5.0;

%% Generate unknown RF-like source signal

source_signal = generate_unknown_rf_signal(fs, duration_s, bandwidth_hz);

% Receiver 1 receives the reference version
rx1_clean = source_signal;

% Receiver 2 receives delayed and frequency-shifted version
rx2_clean = apply_delay_and_fdoa( ...
    source_signal, ...
    fs, ...
    true_tdoa_samples, ...
    true_fdoa_hz);

% Add independent receiver noise
rx1 = add_complex_noise(rx1_clean, snr_db);
rx2 = add_complex_noise(rx2_clean, snr_db);

%% Search grid

delay_bins = -80:1:80;
fdoa_bins = -5000:100:5000;

%% Estimate TDOA and FDOA

[ambiguity_surface, estimated_delay, estimated_fdoa, peak_value] = ...
    estimate_tdoa_fdoa(rx1, rx2, fs, delay_bins, fdoa_bins);

%% Print results

fprintf('Two-Receiver TDOA/FDOA Estimation Demo\n');
fprintf('--------------------------------------\n');
fprintf('Sampling frequency: %.3f MHz\n', fs / 1e6);
fprintf('Signal duration: %.2f ms\n', duration_s * 1e3);
fprintf('Signal bandwidth: %.1f kHz\n', bandwidth_hz / 1e3);
fprintf('SNR: %.1f dB\n\n', snr_db);

fprintf('True TDOA:      %d samples\n', true_tdoa_samples);
fprintf('Estimated TDOA: %d samples\n\n', estimated_delay);

fprintf('True FDOA:      %.1f Hz\n', true_fdoa_hz);
fprintf('Estimated FDOA: %.1f Hz\n\n', estimated_fdoa);

fprintf('Peak ambiguity value: %.2e\n', peak_value);

%% Plot ambiguity surface

figure;
imagesc(delay_bins, fdoa_bins, 10*log10(ambiguity_surface + 1e-12));
set(gca, 'YDir', 'normal');
colorbar;
hold on;
plot(estimated_delay, estimated_fdoa, 'kx', 'MarkerSize', 12, 'LineWidth', 2);
xlabel('Relative delay / TDOA (samples)');
ylabel('Relative frequency offset / FDOA (Hz)');
title('Two-Receiver TDOA/FDOA Ambiguity Surface');
legend('Detected TDOA/FDOA peak');
grid on;


%% Local functions

function signal = generate_unknown_rf_signal(fs, duration_s, bandwidth_hz)
%GENERATE_UNKNOWN_RF_SIGNAL Generate synthetic complex RF-like baseband signal.
%
% The signal is generated as complex noise filtered in the frequency domain
% to create a band-limited waveform. This avoids assuming a known PRN code
% and is closer to a generic RF emitter scenario.

    n_samples = round(fs * duration_s);

    white_noise = randn(1, n_samples) + 1j * randn(1, n_samples);

    spectrum = fftshift(fft(white_noise));
    freqs = fftshift((-n_samples/2:n_samples/2-1) * fs / n_samples);

    bandpass_mask = abs(freqs) <= bandwidth_hz / 2;

    filtered_spectrum = spectrum .* bandpass_mask;

    signal = ifft(ifftshift(filtered_spectrum));

    signal = signal ./ sqrt(mean(abs(signal).^2));

end


function shifted_signal = apply_delay_and_fdoa(signal, fs, delay_samples, fdoa_hz)
%APPLY_DELAY_AND_FDOA Apply integer-sample delay and frequency offset.

    n_samples = length(signal);

    delayed_signal = circshift(signal, delay_samples);

    t = (0:n_samples-1) / fs;

    frequency_shift = exp(1j * 2 * pi * fdoa_hz * t);

    shifted_signal = delayed_signal .* frequency_shift;

end


function noisy_signal = add_complex_noise(signal, snr_db)
%ADD_COMPLEX_NOISE Add complex Gaussian noise at selected SNR.

    signal_power = mean(abs(signal).^2);

    snr_linear = 10^(snr_db / 10);

    noise_power = signal_power / snr_linear;

    noise = sqrt(noise_power / 2) * ...
        (randn(size(signal)) + 1j * randn(size(signal)));

    noisy_signal = signal + noise;

end


function [ambiguity_surface, estimated_delay, estimated_fdoa, peak_value] = ...
    estimate_tdoa_fdoa(rx1, rx2, fs, delay_bins, fdoa_bins)
%ESTIMATE_TDOA_FDOA Estimate TDOA/FDOA using joint delay-frequency search.
%
% For each FDOA hypothesis, receiver 2 is frequency-corrected. For each
% delay hypothesis, the corrected receiver 2 signal is shifted and compared
% with receiver 1 using a coherent inner product.

    n_samples = length(rx1);

    t = (0:n_samples-1) / fs;

    ambiguity_surface = zeros(length(fdoa_bins), length(delay_bins));

    for i = 1:length(fdoa_bins)

        fdoa = fdoa_bins(i);

        % Correct receiver 2 by the negative candidate FDOA.
        fdoa_correction = exp(-1j * 2 * pi * fdoa * t);

        rx2_corrected = rx2 .* fdoa_correction;

        for j = 1:length(delay_bins)

            delay = delay_bins(j);

            % Undo candidate delay by shifting receiver 2 backward.
            rx2_aligned = circshift(rx2_corrected, -delay);

            metric = rx1 * rx2_aligned';

            ambiguity_surface(i, j) = abs(metric).^2;

        end

    end

    [peak_value, linear_index] = max(ambiguity_surface(:));

    [fdoa_index, delay_index] = ind2sub(size(ambiguity_surface), linear_index);

    estimated_fdoa = fdoa_bins(fdoa_index);
    estimated_delay = delay_bins(delay_index);

end
