% generate_ca_code.m
%
% Generate and plot a GPS L1 C/A PRN code.
%
% This script demonstrates the first step in GNSS signal processing:
% creating a local PRN replica that can later be correlated with raw IF/IQ
% samples during acquisition or delay-Doppler processing.
%
% Author: Yusof Ghiasi

clear; clc; close all;

% Select GPS PRN
prn = 1;

% Generate C/A code
ca_code = generate_ca_code(prn);

fprintf('Generated GPS L1 C/A code for PRN %d\n', prn);
fprintf('Code length: %d chips\n', length(ca_code));
fprintf('Unique chip values: ');
disp(unique(ca_code).');

% Plot first 100 chips
figure;
stairs(0:99, ca_code(1:100), 'LineWidth', 1.5);
xlabel('Chip index');
ylabel('Amplitude');
title(sprintf('GPS L1 C/A Code, PRN %d, First 100 Chips', prn));
grid on;


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

    % G2 tap selection for GPS PRNs 1 to 32.
    % Each row corresponds to one PRN.
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

    % Initialize G1 and G2 shift registers with ones.
    g1 = ones(1, 10);
    g2 = ones(1, 10);

    ca_binary = zeros(1, 1023);

    for i = 1:1023

        % G1 output is the last register.
        g1_output = g1(10);

        % G2 output is the modulo-2 sum of the selected taps.
        g2_output = xor(g2(tap1), g2(tap2));

        % C/A code chip in binary form.
        ca_binary(i) = xor(g1_output, g2_output);

        % Feedback definitions for GPS C/A code.
        g1_feedback = xor(g1(3), g1(10));

        g2_feedback = xor(xor(xor(xor(xor(g2(2), g2(3)), g2(6)), g2(8)), g2(9)), g2(10));

        % Shift registers.
        g1 = [g1_feedback, g1(1:9)];
        g2 = [g2_feedback, g2(1:9)];

    end

    % Convert binary 0/1 code to bipolar -1/+1 code.
    ca_code = 1 - 2 * ca_binary;

end
