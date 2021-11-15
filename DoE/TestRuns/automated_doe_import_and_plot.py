import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime


def convert_time(time_date_value):
    """
    Converts time from special format into seconds (float)
    :param time_date_value: time in dd:mm:yyyy HH:MM:SS,fff
    :return: time in seconds
    """
    time_converted = datetime.datetime.strptime(time_date_value, "%d.%m.%Y %H:%M:%S,%f")
    time_rel = time_converted - datetime.datetime(2000, 1, 1)
    time_seconds = time_rel.total_seconds()
    return time_seconds


# importing

# dirname = os.path.dirname(__file__)
# file_name = os.path.join(dirname,'CC_DAT_DoE_SNAr_CEJ-46_Run-1_211110_V01.csv')

data = pd.read_csv('CC_DAT_DoE_SNAr_CEJ-46_Run-1_211110_V01.csv', sep=";")
data = data.dropna()

# sample_date = datetime.datetime.strptime(data.Date.iloc[23], "%d.%m.%Y %H:%M:%S,%f")
# print(sample_date)
# rel_time = sample_date - datetime.datetime(2000, 1, 1)
# seconds = rel_time.total_seconds()
# print(seconds)

time_date = data["Date"]  # not changed to np.array, because it is transformed in to relative hours later on
total_flowrate_set = np.array(data["Control_DoE_10450.Q_total_1"], dtype=np.float32)
total_flowrate_actual = np.array(data["Control_DoE_10450.Q_total_1_act"], dtype=np.float32)
sty = np.array(data["Control_DoE_10450.STY_1"], dtype=np.float32)
ratio_reagent_set_script = np.array(data["Control_DoE_10450.data_1"], dtype=np.float32)
conc_SM_1_set_script = np.array(data["Control_DoE_10450.data_2"], dtype=np.float32)
residence_time_set_script = np.array(data["Control_DoE_10450.data_3"], dtype=np.float32)
T_set_script = np.array(data["Control_DoE_10450.data_4"], dtype=np.float32)
throughput = np.array(data["Control_DoE_10450.TP_1"], dtype=np.float32)
conc_SM_NMR = np.array(data["Control_DoE_10450.C_SM_2"], dtype=np.float32)
conc_prod_NMR = np.array(data["Control_DoE_10450.C_P1_2"], dtype=np.float32)
product_yield = np.array(data["Control_DoE_10450.Yield_1"], dtype=np.float32)
selectivity = np.array(data["Control_DoE_10450.Selectivity_1"], dtype=np.float32)
e_factor = np.array(data["Control_DoE_10450.E_Factor_1"], dtype=np.float32)
gradient = np.array(data["Control_DoE_10450.gradient_1"], dtype=np.float32)
q_R1_act = np.array(data["Control_DoE_10450.Q_R1_1_act"], dtype=np.float32)
q_solvent1_act = np.array(data["Control_DoE_10450.Q_S1_1_act"], dtype=np.float32)
q_SM1_act = np.array(data["Control_DoE_10450.Q_SM_1_act"], dtype=np.float32)
T_thermostat_act = np.array(data["Control_DoE_10450.thermostat_1_act"], dtype=np.float32)
conc_prod_NMR_raw = np.array(data["Control_DoE_10450.C_NMR_P1_2_raw"], dtype=np.float32)
conc_SM_NMR_raw = np.array(data["Control_DoE_10450.C_NMR_SM_2_raw"], dtype=np.float32)
p_sensor1 = np.array(data["Control_DoE_10450.P_1"], dtype=np.float32)
p_sensor2 = np.array(data["Control_DoE_10450.P_2"], dtype=np.float32)
q_SM1_set = np.array(data["Control_DoE_10450.Q_SM_1"], dtype=np.float32)
q_R1_set = np.array(data["Control_DoE_10450.Q_R1_1"], dtype=np.float32)
q_solvent1_set = np.array(data["Control_DoE_10450.Q_S1_1"], dtype=np.float32)
T_thermostat_set = np.array(data["Control_DoE_10450.T_1"], dtype=np.float32)

time_date_in_seconds = []
rel_time_in_hours = []

# print(time_date)

for time_date_value in time_date:
    time_date_in_seconds.append(convert_time(time_date_value))

for value_in_seconds in time_date_in_seconds:
    rel_time_in_hours.append((value_in_seconds - time_date_in_seconds[0]) / 3600)

rel_time_in_hours = np.array(rel_time_in_hours, dtype=np.float32)

# plotting


fig, axs = plt.subplots(4, sharex=True)

# color scheme
color_SM = "#Af4808"
color_reagent = "#3400ff"
color_solvent = "#867cb5"
color_prod = "#9e9a06"
color_temperature = "#E20a0a"
color_sty = "#000000"
color_yield = "#08af63"

plt.xlabel("Time (h)")

axs[0].plot(rel_time_in_hours, q_R1_act, color=color_reagent)
axs[0].plot(rel_time_in_hours, q_SM1_act, color=color_SM)
axs[0].plot(rel_time_in_hours, q_solvent1_act, color=color_solvent)
axs[0].legend(["Reagent", "Starting Material", "Solvent"], bbox_to_anchor=(1.0, 0.8), loc="upper left")
axs[0].set(ylabel="Flowrate (ml/min)")
axs[0].spines["right"].set_visible(False)
axs[0].spines["top"].set_visible(False)
axs[0].spines["bottom"].set_visible(False)

axs[1].plot(rel_time_in_hours, T_thermostat_act, color=color_temperature)
axs[1].legend(["Thermostat Actual"], bbox_to_anchor=(1.0, 0.8), loc="upper left")
axs[1].set(ylabel="Temperature (°C)")
axs[1].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[1].spines["bottom"].set_visible(False)
axs[1].set_ylim([0, 200])

axs[2].plot(rel_time_in_hours, conc_SM_NMR, color=color_SM)
axs[2].plot(rel_time_in_hours, conc_prod_NMR, color=color_prod)
axs[2].legend(["Starting Material (NMR)", "Product (NMR)"], bbox_to_anchor=(1.0, 0.8), loc="upper left")
axs[2].set(ylabel="Concentration (mol/L)")
axs[2].spines["right"].set_visible(False)
axs[2].spines["top"].set_visible(False)
axs[2].spines["bottom"].set_visible(False)

axs[3].plot(rel_time_in_hours, sty, color=color_sty)
axs[3].legend(["STY"], bbox_to_anchor=(1.0, 1.5), loc="upper left")
axs[3].set(ylabel="STY (kg/L/h)")
axs[3].spines["top"].set_visible(False)

axs3_x2 = axs[3].twinx()
axs3_x2.plot(rel_time_in_hours, product_yield * 100, color=color_yield)
axs3_x2.set(ylabel="Yield (%)")
axs3_x2.legend(["Yield"], bbox_to_anchor=(1.0, 1.30), loc="upper left")
axs3_x2.spines["top"].set_visible(False)


fig.set_size_inches(10, 10)
plt.savefig("CC_DAT_DoE_SNAr_CEJ-46_Run-1_211110_V01.png", dpi=500, bbox_inches="tight")
