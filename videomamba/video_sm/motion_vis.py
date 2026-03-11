import numpy as np
import matplotlib.pyplot as plt

# ===================== Labels (x-axis) =====================
cats = [
    "cj_20%", "cj_50%", "cj_80%",
    "fs_20%", "fs_50%", "fs_80%",
    "tn_20%", "tn_50%", "tn_80%",
    "cb_20%", "cb_50%", "cb_80%",
]
x = np.arange(len(cats))

# ===================== Group-1 (Amplified) =====================
clean1 = 0.9450
drop1 = np.array([0.20, 1.60, 5.00,  1.40, 2.60, 1.20,  0.20, 1.20, 3.40,  0.20, 1.80, 7.00]) / 100.0
f1_1 = clean1 * (1 - drop1)

# ===================== Group-2 (Robust) =====================
clean2 = 0.9524
drop2 = np.array([0.05, 0.80, 1.50,  0.50, 0.70, 1.20,  0.05, 0.70, 2.00,  0.15, 0.90, 3.50]) / 100.0
f1_2 = clean2 * (1 - drop2)

# ===================== Convert to percentage =====================
clean1_p = clean1 * 100
clean2_p = clean2 * 100
f1_1_p = f1_1 * 100
f1_2_p = f1_2 * 100

# ===================== Plot =====================
plt.rcParams.update({"font.family": "serif", "font.size": 12})
fig, ax = plt.subplots(figsize=(10.8, 4.4), dpi=160)

# curves
l1, = ax.plot(x, f1_1_p, marker='o', linewidth=2.0, markersize=6.5, label="EndoMamba")
l2, = ax.plot(x, f1_2_p, marker='s', linewidth=2.0, markersize=6.2, color='red', label="FPRL (ours)")

# dashed clean lines with corresponding colors
c1 = l1.get_color()
c2 = l2.get_color()
ax.axhline(clean1_p, linestyle='--', linewidth=1.6, color=c1)
ax.axhline(clean2_p, linestyle='-.', linewidth=1.6, color=c2)

# clean value texts (avoid overlap)
x_text_1 = len(cats) - 0.6
x_text_2 = len(cats) - 0.6
ax.text(x_text_1, clean1_p - 0.45, f"{clean1_p:.2f}", ha="right", va="top", color=c1)
ax.text(x_text_2, clean2_p + 0.45, f"{clean2_p:.2f}", ha="right", va="bottom", color=c2)

# annotate minima (optional)
i1 = int(np.argmin(f1_1_p))
i2 = int(np.argmin(f1_2_p))
ax.text(x[i1], f1_1_p[i1] - 0.55, f"{f1_1_p[i1]:.2f}", ha="center", va="top", color=c1)
ax.text(x[i2], f1_2_p[i2] - 0.55, f"{f1_2_p[i2]:.2f}", ha="center", va="top", color=c2)

# axes
ax.set_ylabel("F1 (%)")
ax.set_xlabel("Perturbation type and level")
ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=30, ha='right')

ymin = min(f1_1_p.min(), f1_2_p.min(), clean1_p, clean2_p) - 2.0
ymax = max(f1_1_p.max(), f1_2_p.max(), clean1_p, clean2_p) + 2.0
ax.set_ylim(ymin, ymax)

# legend (top center)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False,
          handlelength=1.2, columnspacing=0.9, handletextpad=0.4)

# clean look
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
out_name = "f1_two_setting_compare_percent.png"
plt.savefig(out_name, bbox_inches="tight")
plt.close()
print(f"Saved: ./{out_name}")








# import numpy as np
# import matplotlib.pyplot as plt

# # ===================== Labels (x-axis) =====================
# cats = [
#     "cj_20%", "cj_50%", "cj_80%",
#     "fs_20%", "fs_50%", "fs_80%",
#     "tn_20%", "tn_50%", "tn_80%",
#     "cb_20%", "cb_50%", "cb_80%",
# ]
# x = np.arange(len(cats))

# # ===================== Dice values (ONLY Dice) =====================
# # Group-1 (EndoMamba) from the 1st image
# clean1 = 84.5
# dice1 = np.array([
#     84.5, 83.7, 78.3,   # cj_20/50/80
#     83.9, 82.4, 81.9,   # fs_20/50/80
#     83.0, 79.8, 78.0,   # tn_20/50/80
#     82.2, 80.6, 73.3    # cb_20/50/80
# ])

# # Group-2 (FPRL ours) from the 2nd image
# clean2 = 86.1
# dice2 = np.array([
#     86.1, 85.8, 85.0,   # cj_20/50/80
#     85.3, 84.4, 84.3,   # fs_20/50/80
#     84.6, 82.9, 82.1,   # tn_20/50/80
#     83.8, 83.0, 81.9    # cb_20/50/80
# ])

# # ===================== Plot =====================
# plt.rcParams.update({"font.family": "serif", "font.size": 12})
# fig, ax = plt.subplots(figsize=(10.8, 4.4), dpi=160)

# # curves
# l1, = ax.plot(x, dice1, marker='o', linewidth=2.0, markersize=6.5, label="EndoMamba")
# l2, = ax.plot(x, dice2, marker='s', linewidth=2.0, markersize=6.2, color='red', label="FPRL (ours)")

# # dashed clean lines in corresponding colors
# c1 = l1.get_color()
# c2 = l2.get_color()
# ax.axhline(clean1, linestyle='--', linewidth=1.6, color=c1)
# ax.axhline(clean2, linestyle='-.', linewidth=1.6, color=c2)

# # clean value texts (avoid overlap)
# x_text_1 = len(cats) - 0.6
# x_text_2 = len(cats) - 0.6
# ax.text(x_text_1, clean1 - 0.45, f"{clean1:.1f}", ha="right", va="top", color=c1)
# ax.text(x_text_2, clean2 + 0.45, f"{clean2:.1f}", ha="right", va="bottom", color=c2)

# # annotate minima (optional)
# i1 = int(np.argmin(dice1))
# i2 = int(np.argmin(dice2))
# ax.text(x[i1], dice1[i1] - 0.55, f"{dice1[i1]:.1f}", ha="center", va="top", color=c1)
# ax.text(x[i2], dice2[i2] - 0.55, f"{dice2[i2]:.1f}", ha="center", va="top", color=c2)

# # axes
# ax.set_ylabel("Dice (%)")
# ax.set_xlabel("Perturbation type and level")
# ax.set_xticks(x)
# ax.set_xticklabels(cats, rotation=30, ha='right')

# ymin = min(dice1.min(), dice2.min(), clean1, clean2) - 2.0
# ymax = max(dice1.max(), dice2.max(), clean1, clean2) + 2.0
# ax.set_ylim(ymin, ymax)

# # legend (top center)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False,
#           handlelength=1.2, columnspacing=0.9, handletextpad=0.4)

# # clean look
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)

# plt.tight_layout()
# out_name = "dice_two_setting_compare.png"
# plt.savefig(out_name, bbox_inches="tight")
# plt.close()
# print(f"Saved: ./{out_name}")
