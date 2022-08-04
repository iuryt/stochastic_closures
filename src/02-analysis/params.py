
# Data directory
data_dir = "../../data/"

# Define the parameters for the analysis

task = "ψ" # variable to be analyzed
kmax = 1000 # maximum wavenumber to be used
step_task = 2 # step size for the task variable

# Define noise runs that will be analyzed
noise_runs = ["noise_004", "noise_005", "noise_006", "noise_007", "noise_008", "noise_010", "noise_012"]
# noise_runs = ["noise_010"]

# key args for figures

kw = dict(
    savefig = dict(
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )
)
