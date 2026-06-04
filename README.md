# GNSS-RF Signal Processing Portfolio

This repository contains Python and MATLAB examples demonstrating raw IF/IQ signal-processing concepts relevant to GNSS, GNSS-R, delay-Doppler analysis, correlation-based acquisition, and RF geolocation observables.

The goal is to demonstrate practical experience with signal-processing workflows that move from sampled data to observables such as code delay, Doppler frequency, delay-Doppler peaks, TDOA, and FDOA.

## Technical Focus

This repository demonstrates:

- GPS L1 C/A PRN code generation
- local PRN replica construction
- synthetic raw IF/IQ signal simulation
- Doppler wipe-off
- code-delay search
- FFT-based circular correlation
- delay-Doppler acquisition map generation
- correlation peak detection
- MATLAB and Python implementation of the same concepts

## Repository Structure

```text
python/
  01_generate_ca_code.py
  02_gnss_acquisition_synthetic_if.py

matlab/
  generate_ca_code.m
  gnss_acquisition_synthetic_if.m

data/
  README.md

figures/
  README.md

docs/
  signal_processing_notes.md
