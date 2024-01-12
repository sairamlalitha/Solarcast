# Solarcast
Using Gaussian Process to Model Solar Calcium HK Indices to Forecast Solar Activity Cycle

Overview

This repository contains Python code for performing Gaussian Process (GP) analysis on solar S-index data using PyAstronomy and celerite libraries. The analysis includes fitting a GP model to the data, handling outliers, and incorporating a sine model for periodic signals. 

In the context of this specific application, each dataset corresponds to a separate solar cycle within the range of cycles 20-25. The input data file "Solar_mtwilson_calcium_hk.txt," covers all five solar cycles. By accommodating multiple input files and addressing datasets that span several solar cycles, the code allows to gain a comprehensive understanding of the temporal variations in solar cycles.

Features

The code uses the celerite library for Gaussian Process regression to model the underlying trends in the time-series data. In addition to the GP model, a sine model is incorporated to capture periodic signals in the data. The sine model is optimised alongside the GP model.

Requirements

Python 3.x
Required libraries: numpy, matplotlib, scipy, celerite, PyAstronomy

Usage

Clone the repository to your local machine.
bash
Copy code
git clone https://github.com/sairamlalitha/Solarcast.git

Steps involved:
  - Place your time-series data files in the 'data' folder.
  - Modify the input_files list in the script to include the filenames you want to analyse.
  - Run the script.
  - Output files are saved in the 'output' folder. 

## Reference
If you use this code in your research or find it helpful, please consider citing:

[Lalitha Sairam & Amaury Triaud, "The need for a public forecast of stellar activity to optimize exoplanet radial velocity detections and transmission spectroscopy", Monthly Notices of the Royal Astronomical Society, Volume 514, Issue 2, April 2021, Pages 2259–2267, https://academic.oup.com/mnras/article/514/2/2259/6595315](https://academic.oup.com/mnras/article/514/2/2259/6595315)

Feel free to contribute to the project by forking and creating pull requests.

Happy solarcasting!
