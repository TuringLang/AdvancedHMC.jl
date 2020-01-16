run_time_list_tfp_xla = [
    [3.090347766876220, 2.828183174133301, 3.014273643493652, 4.196374416351318, 10.497739553451538],
    [3.114380121231079, 2.841257333755493, 3.200836181640625, 4.058141469955444, 10.449615716934204],
    [2.979512691497802, 2.800515413284301, 3.025217056274414, 4.022154331207275,  9.217156887054443],
]
run_time_list_ahmc = [
    [2.474283462, 2.775114306, 4.474204685, 17.148073437, 142.031350136],
    [1.968109331, 2.581849350, 4.587503092, 18.799088301, 154.755842153],
    [2.350790260, 2.842742620, 4.750033486, 18.714588798, 142.333408883],
]

using MLToolkit.Plots

fig, ax = plt.subplots()

plot!(ax, LinesWithErrorBar(n_chains_list, run_time_list_tfp_xla); label="TFP (XLA)")
plot!(ax, LinesWithErrorBar(n_chains_list, run_time_list_ahmc); label="AdvancedHMC")

ax.set_xlabel("Number of chains")
ax.set_ylabel("Sampling time (s)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_title("Vectorized mode on CPU")
plt.legend()
savefig(fig, "")

run_time_list_forwarddiff_dist  = [0.109947149, 0.103731421, 0.131234604, 0.190311801, 0.446025397, 1.056527587, 3.213806697, 11.083078182, 38.082333896]
run_time_list_forwarddiff_distX = [0.090850415, 0.764131722, 0.812617863, 1.014190059, 0.648893597, 2.299574609, 5.418227988, 10.502449656, 33.848247298]
run_time_list_zygote_dist       = [4.423139535, 4.847198919, 6.281013862, 9.219753507, 14.85712963, 27.256743158, 50.974715499, 97.031564813, 179.80131999]
run_time_list_zygote_distX      = [1.82309049, 1.793467318, 1.837250077, 1.941552501, 2.091833956, 2.387725306, 2.959626034, 3.983277249, 5.701385978]

using MLToolkit.Plots

fig, ax = plt.subplots()
ax.plot(n_chains_list, run_time_list_forwarddiff,       "-x", label="ForwardDiff")
ax.plot(n_chains_list, run_time_list_forwarddiff_distX, "-x", label="ForwardDiff (DistX)")
ax.plot(n_chains_list, run_time_list_zygote,            "-x", label="Zygote")
ax.plot(n_chains_list, run_time_list_zygote_distX,      "-x", label="Zygote (DistX)")
ax.set_xlabel("Number of chains")
ax.set_ylabel("Sampling time (s)")
ax.set_xscale("log")
ax.set_yscale("log")
plt.legend()
fig
