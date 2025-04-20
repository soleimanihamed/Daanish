import pandas as pd
from simulation_utils import LossSimulator

# Load Excel
input_file = "VAR Calculation - CRP.xlsx"
input_sheet = "Lambda Calculation"
output_sheet_poisson = "Simulated Defaults Poisson"
output_sheet_normal = "Simulated Defaults Normal"

df = pd.read_excel(input_file, sheet_name=input_sheet)
df = df[df['Lambda_DCF'] > 0].reset_index(drop=True)

lambdas = df['Lambda_DCF'].values
loss_classes = df['Loss_Class_DCF'].values

simulator = LossSimulator(lambdas=lambdas, loss_classes=loss_classes)

# Run simulations
df_poisson = simulator.simulate_poisson()
df_normal = simulator.simulate_normal()

# Write to Excel
with pd.ExcelWriter(input_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    df_poisson.to_excel(writer, sheet_name=output_sheet_poisson, index=False)
    df_normal.to_excel(writer, sheet_name=output_sheet_normal, index=False)
