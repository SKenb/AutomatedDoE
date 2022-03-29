from audioop import minmax
from re import X
from turtle import xcor
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime

from matplotlib import colors

DRAW_LABELS = False


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


## DoE Data
numberOfExperiments = [10, 19, 21, 23, 25, 27]
r2ScoreHistory = [0.90933, 0.77719, 0.90332, 0.90535, 0.90806, 0.79789]
q2ScoreHistory = [0.53475, 0.66086, 0.74849, 0.77956, 0.78982, 0.71578]

experiments = np.array([ 
    np.array([-1, -1, -1, -1]), np.array([-1,  1, -1, -1]), np.array([-1,  1,  1,  1]),
    np.array([-1, -1,  1,  1]), np.array([ 0, -0,  0,  0]), np.array([ 0, 0, 0 , 0]),
    np.array([ 1, -1, -1, -1]), np.array([ 1,  1, -1, -1]), np.array([ 1,  1,  1,  1]),
    np.array([ 1, -1,  1,  1]), np.array([ 1, -1, -1,  1]), np.array([ 1,  1, -1,  1]),
    np.array([ 1,  1,  1, -1]), np.array([ 1, -1,  1, -1]), np.array([ 0, -0,  0, 0]),
    np.array([-1, -1, -1,  1]), np.array([-1,  1, -1,  1]), np.array([-1,  1,  1, -1]),
    np.array([-1, -1,  1, -1]), np.array([-1, -0,  0,  0]), np.array([ 1, -0,  0,  0]),
    np.array([ 0, -1,  0,  0]), np.array([ 0,  1,  0,  0]), np.array([ 0, -0,  1,  0]),
    np.array([ 0, -0, -1,  0]), np.array([ 0, -0,  0, -1]), np.array([ 0, -0,  0, 1])
])


# importing

# dirname = os.path.dirname(__file__)
# file_name = os.path.join(dirname,'CC_DAT_DoE_SNAr_CEJ-46_Run-1_211110_V01.csv')

data = pd.read_csv('./SNAr_DoE_1Run/CC_DAT_DoE_SNAr_CEJ-46_Run-1_211110_V01.csv', sep=";")
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

plt.rcParams.update({
    'text.usetex': True,
    'font.size': '10',
    'font.weight': 'bold'
})

fig, axs = plt.subplots(6)
plt.rcParams['text.usetex'] = True

# color scheme
color_SM = "tab:blue"
color_reagent = "tab:orange"
color_solvent = "tab:green"
color_prod = "tab:red"
color_temperature = "tab:green"
color_sty = "tab:blue"
color_yield = "tab:orange"

plt.xlabel("Time (h)")

minMax = lambda array: (min(array)-.15, 1.02*max(array))

#### Exp History
expLabels = [chr(65+(index % 26))for index in range(4)]

assert len(expLabels) == experiments.shape[1], "UPS 0.o - we have a different amount of labels compared to experiment variables..."

cmap = colors.ListedColormap(['royalblue', 'lightsteelblue', 'cornflowerblue'])
bounds=[-1.5,-.5, .5, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

axs[0].imshow(experiments.T, cmap=cmap, origin='lower', norm=norm, interpolation='nearest', aspect="auto")

# Show all ticks and label them with the respective list entries
axs[0].set_xticks([])
axs[0].set_yticks(np.arange(len(expLabels)))
axs[0].set_yticklabels(expLabels)

if DRAW_LABELS: axs[0].set_ylabel("Factor")

# Rotate the tick labels and set their alignment.
#axs[0].setp(axs[0].get_xticklabels(), rotation=45, ha="right",
#        rotation_mode="anchor")

def valueToString(val, tol=1e-3):
    if val < -tol: return "-" 
    if val > tol: return "+" 
    return "0"

# Loop over data dimensions and create text annotations.
for i in range(len(expLabels)):
    for j in range(experiments.shape[0]):
        text = valueToString(experiments[j, i])

        if "0" in text:
            axs[0].text(j, i, text, ha="center", va="center", color="w", fontsize=12)
        else:
            axs[0].text(j, i, text, ha="center", va="center", color="w")

axs[0].xaxis.tick_top()
axs[0].set_title("")


#### Clemens stuff
axs[1].plot(rel_time_in_hours, q_R1_act, color=color_reagent)
axs[1].plot(rel_time_in_hours, q_SM1_act, color=color_SM)
axs[1].plot(rel_time_in_hours, q_solvent1_act, color=color_solvent)
if DRAW_LABELS: axs[1].legend(["Reagent", "Starting Material", "Solvent"], loc="upper right", ncol = 3)
if DRAW_LABELS: axs[1].set(ylabel=r"Flow rate ($ml\;min^{-1}$)")
#axs[1].spines["right"].set_visible(False)
#axs[1].spines["top"].set_visible(False)
#axs[1].spines["bottom"].set_visible(False)
axs[1].set_xlim(minMax(rel_time_in_hours))


axs[2].plot(rel_time_in_hours, T_thermostat_act, color=color_temperature)
if DRAW_LABELS: axs[2].legend(["Thermostat Actual"], loc="lower left")
if DRAW_LABELS: axs[2].set(ylabel=r"Temperature ($^\circ C$)")
#axs[2].spines["right"].set_visible(False)
#axs[2].spines["top"].set_visible(False)
#axs[2].spines["bottom"].set_visible(False)
axs[2].set_ylim([0, 200])
axs[2].set_xlim(minMax(rel_time_in_hours))

axs[3].plot(rel_time_in_hours, conc_SM_NMR, color=color_SM)
axs[3].plot(rel_time_in_hours, conc_prod_NMR, color=color_prod)
if DRAW_LABELS: axs[3].legend([r"Starting Material ($NMR$)", "Product (NMR)"], loc="upper left")
if DRAW_LABELS: axs[3].set(ylabel=r"Concentration ($mol\;L^{-1}$)")
#axs[3].spines["right"].set_visible(False)
#axs[3].spines["top"].set_visible(False)
#axs[3].spines["bottom"].set_visible(False)
axs[3].set_xlim(minMax(rel_time_in_hours))
axs[3].set_yticks([.0, .15, .3])

axs[4].plot(rel_time_in_hours, sty, color=color_sty)
if DRAW_LABELS: axs[4].legend(["STY"], bbox_to_anchor=(.001, 1), loc="upper left")
if DRAW_LABELS: axs[4].set(ylabel=r"STY ($kg\;L^{-1}\;h^{-1}$)")
#axs[4].spines["top"].set_visible(False)
axs[4].set_xlim(minMax(rel_time_in_hours))
axs[4].set_yticks([0, .6, 1.2])

axs4_x2 = axs[4].twinx()
axs4_x2.plot(rel_time_in_hours, product_yield * 100, color=color_yield)
if DRAW_LABELS: axs4_x2.set(ylabel=r"Yield ($\%$)")
if DRAW_LABELS: axs4_x2.legend(["Yield"], bbox_to_anchor=(.001, 0.75), loc="upper left")
axs4_x2.set_yticks([0, 35, 70])
#axs4_x2.spines["top"].set_visible(False)


#### Design iteration
x = [a*max(rel_time_in_hours)/max(numberOfExperiments) for a in numberOfExperiments]
axs[5].scatter(x, r2ScoreHistory, 60, c="tab:blue", zorder=200, label=r"$R^2$")
axs[5].scatter(x, q2ScoreHistory, 60, c="tab:orange", zorder=200, label=r"$Q^2$")
axs[5].plot(x, r2ScoreHistory, "--", color="tab:blue")
axs[5].plot(x, q2ScoreHistory, "--", color="tab:orange")
axs[5].set_ylim((0, 1))
axs[5].set_yticks([0, .5, 1])

xTicks = [0]
xTicks.extend(x)
xLabels = [str(0)]
xLabels.extend([str(el+1) for el in range(len(x))])
axs[5].set_xlim(minMax(xTicks))
axs[5].set_xticks(xTicks)
axs[5].set_xticklabels(xLabels)

if DRAW_LABELS: axs[5].set_xlabel("Design iteration")
if DRAW_LABELS: axs[5].set_ylabel("Score")

if DRAW_LABELS: axs[5].legend()
#

fig.set_size_inches(10, 10)
plt.savefig("./SNAr_DoE_1Run/CC_DAT_DoE_SNAr_CEJ-46_Run-1_211110_V01.png", dpi=500, bbox_inches="tight")
