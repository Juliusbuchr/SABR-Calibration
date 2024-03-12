# SABR Calibration

This Python script is designed to calibrate the SABR model to market data and fit SABR implied volatilities for interest rate swaptions. The script takes market data from an Excel file, calibrates the SABR model parameters, and outputs the calibrated parameters, SABR volatilities, and differences between market volatilities and model-ijmplied volatilities.

## Prerequisites
- Required Python packages: `pandas`, `numpy`, `scipy`

## Getting Started
1. **Install Required Packages**: Ensure that you have installed the required Python packages mentioned above.
2. **Prepare Market Data**: Prepare your market data in an Excel file with the following structure:
   - The first row contains strike spreads.
   - The first column contains expiry tenors.
   - The second column contains tenors.
   - The third column contains forward rates.
   - The remaining cells contain market volatilities for corresponding expiry and tenor combinations.
3. **Update File Paths**: Modify the file paths in the script to point to your input Excel file and specify the output location for the generated results.
4. **Run the Script**: Execute the script, and it will calibrate the SABR model parameters and generate the required outputs.

## File Descriptions
- **`swapsdata.xlsx`**: Input Excel file containing market data.
- **`output.xlsx`**: Output Excel file containing the following sheets:
  - `outvol`: SABR volatilities.
  - `vol_diff`: Differences between market volatilities and model-generated volatilities.
  - `parameters`: Calibrated SABR model parameters for each swaption-set of expiration/tenor.

## Functionality Overview
- **SABR Calibration**: The script calibrates the SABR model parameters (`alpha`, `beta`, `rho`, `nu`) using the market data provided.
- **Volatility Generation**: It generates SABR implied volatilities for each expiry-tenor combination based on the calibrated parameters.
- **Output**: The calibrated parameters, SABR volatilities, and volatility differences are saved in an Excel file.

## Usage Notes
- Ensure that the input market data is correctly formatted as described above.
- Verify that the output file paths are correctly specified to save the generated results.

## Data
- Data is from Bloomberg 8 March 2024 12:05 CET.
- Currency: USD with IBOR rates and OIS discounting. 
