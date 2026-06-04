# GNSS-RF Signal Processing Portfolio

This repository contains Python and MATLAB examples demonstrating raw IF/IQ signal processing concepts relevant to GNSS, GNSS-R, delay-Doppler analysis, correlation-based acquisition, and RF geolocation observables.

The goal is to demonstrate practical experience with:

- raw binary IF/IQ data handling
- GPS C/A PRN code generation
- code delay and Doppler search
- delay-Doppler correlation mapping
- peak detection in acquisition/DDM maps
- simplified TDOA/FDOA estimation for two-receiver RF geolocation

## Repository Structure

```text
python/
  01_generate_ca_code.py
  02_gnss_acquisition_raw_if.py
  03_delay_doppler_map_demo.py
  04_tdoa_fdoa_two_receiver_demo.py

matlab/
  generate_ca_code.m
  gnss_acquisition_raw_if.m
  delay_doppler_map_demo.m
  tdoa_fdoa_two_receiver_demo.m

data/
  README.md

figures/
  README.md

docs/
  signal_processing_notes.md
