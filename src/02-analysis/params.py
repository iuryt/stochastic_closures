

# Define the parameters for the analysis

task = "ψ" # variable to be analyzed
kmax = 1000 # maximum wavenumber to be used

# Define noise runs that will be analyzed

#noise_runs = ["noise00", "noise05", "noise06", "noise07"]
noise_runs = ["noise07"]


# key args for figures

kw = dict(
    savefig = dict(
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
    )
)
