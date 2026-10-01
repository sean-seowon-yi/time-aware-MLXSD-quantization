import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

with open("benchmark_results/w4a8_adaround_poly_p100_group64/benchmark_checkpoints.json") as f:
    ck = json.load(f)

ns    = [int(k) for k in sorted(ck, key=int)]
fid   = [ck[str(n)]["fid"]  for n in ns]
cmmd  = [ck[str(n)]["cmmd"] for n in ns]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("W4A8 AdaRound poly p100 group64 — metric convergence", fontsize=12)

ax1.plot(ns, fid, marker="o", color="steelblue")
ax1.set_title("FID")
ax1.set_xlabel("# generated images")
ax1.set_ylabel("FID (lower = better)")
ax1.xaxis.set_major_locator(ticker.FixedLocator(ns))
ax1.grid(True, alpha=0.3)

ax2.plot(ns, cmmd, marker="o", color="darkorange")
ax2.set_title("CMMD")
ax2.set_xlabel("# generated images")
ax2.set_ylabel("CMMD (lower = better)")
ax2.xaxis.set_major_locator(ticker.FixedLocator(ns))
ax2.set_ylim(0, 0.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = "benchmark_results/w4a8_adaround_poly_p100_group64/checkpoint_metrics.png"
plt.savefig(out, dpi=150)
print(f"Saved → {out}")
plt.show()
