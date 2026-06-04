% raw_bin_signal_inspection_and_peak_search.m
%
% Raw .bin IF/IQ signal inspection and GNSS-style correlation peak search.
%
% This script demonstrates how to work with local raw binary IF/IQ files:
%
% 1. Read binary samples from a .bin file
% 2. Convert samples into complex IQ data
% 3. Plot time-domain I/Q samples
% 4. Plot signal magnitude
% 5. Estimate and plot spectrum
% 6. Detect strong spectral peaks
% 7. Optionally perform GPS C/A PRN delay-Doppler correlation search
% 8. Report strongest delay-Doppler peaks
%
% Important:
% Large .bin files should NOT be uploaded to GitHub. Place them locally in
% the data/ folder and update the parameters below.
%
% Author: Yusof Ghiasi

clear; clc; close all;

%% User settings

% Put your local .bin file in the data folder.
% Example:
% BIN_FILE = 'data/sample_if_data_1.bin';
BIN_FILE = 'data/sample_if_data_1.bin';

% Sampling frequency in Hz.
% Update this based on your file metadata.
FS = 8.184e6;

% Data type in the .bin file.
% Common options: 'int8', 'int16', 'single', 'float32'
DATA_TYPE = 'int8';

% File format options:
% 'real_if'         : file contains real-valued IF samples
% 'interleaved_iq' : file contains I,Q,I,Q,... samples
IQ_FORMAT = 'interleaved_iq';

% Number of complex samples to read.
MAX_COMPLEX_SAMPLES = 200000;

% Preprocessing
REMOVE_DC = true;
NORMALIZE_POWER = true;

% Spectral peak detection
NUMBER_OF_SPECTRAL_PEAKS = 10;

% GNSS PRN search settings
RUN_GNSS_PRN_SEARCH = true;
PRN = 1;

% Coherent integration time in milliseconds.
COHERENT_MS = 1;

% Doppler search bins in Hz.
DOPPLER_BINS = -5000:250:5000;

% Report this many strongest delay-Doppler peaks.
NUMBER_OF_DD_PEAKS = 10;

%% Main processing

fprintf('Raw .bin IF/IQ Signal Inspection and Peak Search\n');
fprintf('================================================\n');
fprintf('File: %s\n', BIN_FILE);
fprintf('Sampling frequency: %.3f MHz\n', FS / 1e6);
fprintf('Data type: %s\n', DATA_TYPE);
fprintf('IQ format: %s\n\n', IQ_FORMAT);

samples = read_raw_bin_file( ...
    BIN_FILE, ...
    DATA_TYPE, ...
    IQ_FORMAT, ...
    MAX_COMPLEX_SAMPLES);

samples = preprocess_samples(samples, REMOVE_DC, NORMALIZE_POWER);

fprintf('Loaded complex samples: %d\n', length(samples));
fprintf('Mean power after preprocessing: %.3f\n\n', mean(abs(samples).^2));

%% Time-domain inspection

plot_time_domain(samples, FS, 2000);

%% Spectrum inspection

[freqs, psd_db] = compute_spectrum(samples, FS);

plot_spectrum(freqs, psd_db);

spectral_peaks = find_spectral_peaks( ...
    freqs, ...
    psd_db, ...
    NUMBER_OF_SPECTRAL_PEAKS, ...
    20);

fprintf('Strongest spectral peaks\n');
fprintf('------------------------\n');

for i = 1:size(spectral_peaks, 1)
    fprintf('%02d: frequency = %+0.6f MHz, magnitude = %.2f dB\n', ...
        i, spectral_peaks(i, 1) / 1e6, spectral_peaks(i, 2));
end

fprintf('\n');

%% Optional GNSS PRN delay-Doppler search

if RUN_GNSS_PRN_SEARCH

    fprintf('Running GNSS PRN delay-Doppler search\n');
    fprintf('-------------------------------------\n');
    fprintf('PRN: %d\n', PRN);
    fprintf('Coherent integration: %d ms\n', COHERENT_MS);
    fprintf('Doppler range: %.1f to %.1f Hz\n', DOPPLER_BINS(1), DOPPLER_BINS(end));
    fprintf('Doppler step: %.1f Hz\n\n', DOPPLER_BINS(2) - DOPPLER_BINS(1));

    [acquisition_map, estimated_delay, estimated_doppler, peak_power] = ...
        gnss_delay_doppler_search( ...
            samples, ...
            FS, ...
            PRN, ...
            COHERENT_MS, ...
            DOPPLER_BINS);

    fprintf('Strongest delay-Doppler result\n');
    fprintf('------------------------------\n');
    fprintf('Estimated code delay: %d samples\n', estimated_delay);
    fprintf('Estimated Doppler:    %.1f Hz\n', estimated_doppler);
    fprintf('Peak power:           %.2e\n\n', peak_power);

    dd_peaks = find_delay_doppler_peaks( ...
        acquisition_map, ...
        DOPPLER_BINS, ...
        NUMBER_OF_DD_PEAKS, ...
        20, ...
        1);

    fprintf('Top delay-Doppler peaks\n');
    fprintf('-----------------------\n');

    for i = 1:size(dd_peaks, 1)
        fprintf('%02d: delay = %5d samples, doppler = %+8.1f Hz, power = %.2e\n', ...
            i, dd_peaks(i, 1), dd_peaks(i, 2), dd_peaks(i, 3));
    end

    plot_acquisition_map( ...
        acquisition_map, ...
        DOPPLER_BINS, ...
        estimated_delay, ...
        estimated_doppler);

else
    fprintf('GNSS PRN delay-Doppler search skipped.\n');
    fprintf('Set RUN_GNSS_PRN_SEARCH = true to enable it.\n');
end


%% Local functions

function samples = read_raw_bin_file(bin_file, data_type, iq_format, max_complex_samples)
%READ_RAW_BIN_FILE Read raw .bin file and return complex samples.

    if ~isfile(bin_file)
        error(['Could not find file:\n%s\n\n' ...
               'Place your .bin file in the data/ folder or update BIN_FILE.'], bin_file);
    end

    fid = fopen(bin_file, 'rb');

    if fid < 0
        error('Could not open file: %s', bin_file);
    end

    cleanupObj = onCleanup(@() fclose(fid));

    switch lower(data_type)
        case 'int8'
            matlab_type = 'int8';
        case 'int16'
            matlab_type = 'int16';
        case {'single', 'float32'}
            matlab_type = 'single';
        case 'double'
            matlab_type = 'double';
        otherwise
            error('Unsupported DATA_TYPE. Use int8, int16, single/float32, or double.');
    end

    if strcmpi(iq_format, 'interleaved_iq')

        raw_count = max_complex_samples * 2;
        raw = fread(fid, raw_count, matlab_type);

        if length(raw) < 2
            error('File does not contain enough samples.');
        end

        if mod(length(raw), 2) ~= 0
            raw = raw(1:end-1);
        end

        i_samples = double(raw(1:2:end));
        q_samples = double(raw(2:2:end));

        samples = i_samples + 1j * q_samples;

    elseif strcmpi(iq_format, 'real_if')

        raw = fread(fid, max_complex_samples, matlab_type);

        if isempty(raw)
            error('File does not contain enough samples.');
        end

        samples = double(raw) + 1j * zeros(size(raw));

    else
        error('IQ_FORMAT must be either real_if or interleaved_iq.');
    end

    samples = samples(:).';

end


function x = preprocess_samples(samples, remove_dc, normalize_power)
%PREPROCESS_SAMPLES Remove DC offset and optionally normalize power.

    x = samples;

    if remove_dc
        x = x - mean(x);
    end

    if normalize_power
        power_value = mean(abs(x).^2);

        if power_value > 0
            x = x ./ sqrt(power_value);
        end
    end

end


function plot_time_domain(samples, fs, number_of_samples)
%PLOT_TIME_DOMAIN Plot I, Q, and magnitude in time domain.

    n = min(number_of_samples, length(samples));
    t_ms = (0:n-1) / fs * 1e3;

    figure;
    plot(t_ms, real(samples(1:n)), 'DisplayName', 'I / real');
    hold on;
    plot(t_ms, imag(samples(1:n)), 'DisplayName', 'Q / imag');
    xlabel('Time (ms)');
    ylabel('Amplitude');
    title('Raw IF/IQ Samples: I and Q');
    legend;
    grid on;

    figure;
    plot(t_ms, abs(samples(1:n)));
    xlabel('Time (ms)');
    ylabel('Magnitude');
    title('Raw IF/IQ Sample Magnitude');
    grid on;

end


function [freqs, psd_db] = compute_spectrum(samples, fs)
%COMPUTE_SPECTRUM Compute simple FFT spectrum.

    n = length(samples);

    window = hann(n).';
    xw = samples .* window;

    spectrum = fftshift(fft(xw));

    if mod(n, 2) == 0
        freqs = (-n/2:n/2-1) * fs / n;
    else
        freqs = (-(n-1)/2:(n-1)/2) * fs / n;
    end

    psd_db = 20 * log10(abs(spectrum) + 1e-12);

end


function plot_spectrum(freqs, psd_db)
%PLOT_SPECTRUM Plot spectrum.

    figure;
    plot(freqs / 1e6, psd_db);
    xlabel('Frequency (MHz)');
    ylabel('Magnitude (dB)');
    title('Raw IF/IQ Spectrum');
    grid on;

end


function peaks = find_spectral_peaks(freqs, psd_db, number_of_peaks, guard_bins)
%FIND_SPECTRAL_PEAKS Find strongest spectral peaks.

    spectrum_copy = psd_db;
    peaks = zeros(number_of_peaks, 2);

    for i = 1:number_of_peaks

        [~, idx] = max(spectrum_copy);

        peaks(i, 1) = freqs(idx);
        peaks(i, 2) = psd_db(idx);

        start_idx = max(1, idx - guard_bins);
        end_idx = min(length(spectrum_copy), idx + guard_bins);

        spectrum_copy(start_idx:end_idx) = -Inf;

    end

end


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
%RESAMPLE_CA_CODE Resample GPS C/A code to sampling frequency.

    sample_index = 0:n_samples-1;
    chip_index = floor(sample_index * code_rate / fs) + 1;

    chip_index = mod(chip_index - 1, 1023) + 1;

    sampled_code = ca_code(chip_index);

end


function [acquisition_map, estimated_delay, estimated_doppler, peak_power] = ...
    gnss_delay_doppler_search(samples, fs, prn, coherent_ms, doppler_bins)
%GNSS_DELAY_DOPPLER_SEARCH Perform GPS C/A PRN delay-Doppler search.

    code_rate = 1.023e6;
    n_samples = round(fs * coherent_ms * 1e-3);

    if length(samples) < n_samples
        error('Not enough samples for %d ms coherent integration.', coherent_ms);
    end

    x = samples(1:n_samples);

    ca_code = generate_ca_code(prn);
    local_code = resample_ca_code(ca_code, fs, code_rate, n_samples);

    t = (0:n_samples-1) / fs;

    code_fft = fft(local_code);

    acquisition_map = zeros(length(doppler_bins), n_samples);

    for i = 1:length(doppler_bins)

        doppler = doppler_bins(i);

        wipeoff = exp(-1j * 2 * pi * doppler * t);

        mixed = x .* wipeoff;

        signal_fft = fft(mixed);

        corr = ifft(signal_fft .* conj(code_fft));

        acquisition_map(i, :) = abs(corr).^2;

    end

    [peak_power, linear_index] = max(acquisition_map(:));

    [doppler_index, delay_index] = ind2sub(size(acquisition_map), linear_index);

    estimated_doppler = doppler_bins(doppler_index);

    % Convert MATLAB 1-based index to delay starting at 0 samples.
    estimated_delay = delay_index - 1;

end


function peaks = find_delay_doppler_peaks( ...
    acquisition_map, doppler_bins, number_of_peaks, delay_guard_bins, doppler_guard_bins)
%FIND_DELAY_DOPPLER_PEAKS Find strongest delay-Doppler peaks.

    surface = acquisition_map;

    peaks = zeros(number_of_peaks, 3);

    for i = 1:number_of_peaks

        [~, linear_index] = max(surface(:));

        [doppler_index, delay_index] = ind2sub(size(surface), linear_index);

        peak_power = acquisition_map(doppler_index, delay_index);
        peak_doppler = doppler_bins(doppler_index);

        delay_samples = delay_index - 1;

        peaks(i, :) = [delay_samples, peak_doppler, peak_power];

        d0 = max(1, doppler_index - doppler_guard_bins);
        d1 = min(size(surface, 1), doppler_index + doppler_guard_bins);

        c0 = max(1, delay_index - delay_guard_bins);
        c1 = min(size(surface, 2), delay_index + delay_guard_bins);

        surface(d0:d1, c0:c1) = -Inf;

    end

end


function plot_acquisition_map(acquisition_map, doppler_bins, estimated_delay, estimated_doppler)
%PLOT_ACQUISITION_MAP Plot delay-Doppler acquisition map.

    delay_axis = 0:size(acquisition_map, 2)-1;

    figure;
    imagesc(delay_axis, doppler_bins, 10*log10(acquisition_map + 1e-12));
    set(gca, 'YDir', 'normal');
    colorbar;
    hold on;
    plot(estimated_delay, estimated_doppler, 'kx', 'MarkerSize', 12, 'LineWidth', 2);
    xlabel('Code delay (samples)');
    ylabel('Doppler frequency (Hz)');
    title('Raw .bin GNSS PRN Delay-Doppler Search');
    legend('Strongest peak');
    grid on;

end
