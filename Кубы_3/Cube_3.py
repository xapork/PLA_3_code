import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

np.set_printoptions(precision=2, suppress=True)

# Cube with side 6, centered at the origin. Each COLUMN is a vertex (x, y, z, 1).
vertices_cube = np.array([
    [-3, -3, -3, 1],
    [ 3, -3, -3, 1],
    [ 3,  3, -3, 1],
    [-3,  3, -3, 1],
    [-3, -3,  3, 1],
    [ 3, -3,  3, 1],
    [ 3,  3,  3, 1],
    [-3,  3,  3, 1],
], dtype=float).T

faces_cube = [
    [0, 1, 2, 3], [4, 5, 6, 7],
    [0, 1, 5, 4], [2, 3, 7, 6],
    [1, 2, 6, 5], [0, 3, 7, 4],
]


def plot(items, title, fname):
    """items: list of (vertices 4xN, faces, color, label)."""
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    handles = []
    for V, F, color, label in items:
        polys = [V[:3, f].T for f in F]
        ax.add_collection3d(Poly3DCollection(
            polys, facecolors=color, edgecolors="k", alpha=0.45))
        handles.append(Patch(facecolor=color, edgecolor="k", label=label))

    pts = np.hstack([V[:3] for V, _, _, _ in items])
    c = (pts.max(axis=1) + pts.min(axis=1)) / 2
    r = (pts.max(axis=1) - pts.min(axis=1)).max() / 2 * 1.1
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(handles=handles, loc="upper left")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print("saved", fname)


def scale(sx, sy, sz):
    return np.diag([sx, sy, sz, 1.0])


def translate(tx, ty, tz):
    T = np.eye(4)
    T[:3, 3] = [tx, ty, tz]
    return T


def apply(M, V):
    return M @ V          # vertices are columns -> M * V


ORIG = (vertices_cube, faces_cube, "lightgray", "original")

# ---------- Translation matrices ----------
T1 = translate(8, 0, 0)
T2 = translate(0, 6, 4)
T = T2 @ T1
print("T1 =\n", T1, "\nT2 =\n", T2, "\nT = T2*T1 =\n", T)

for M, name, label in [(T1, "fig3_T1.png", "T1"),
                       (T2, "fig3_T2.png", "T2"),
                       (T,  "fig3_T.png",  "T = T2*T1")]:
    V = apply(M, vertices_cube)
    print("\n" + label, "vertices:\n", V)
    plot([ORIG, (V, faces_cube, "tab:green", label)],
         f"Task 3: translation {label}", name)

# ---------- Combination with scaling (S from task 2) ----------
S = scale(2, 1.5, 0.5)

# TS: scale first, then translate.  ST: translate first, then scale.
TS = T @ S
ST = S @ T
print("\nS =\n", S)
print("TS =\n", TS, "\nST =\n", ST)
print("TS == ST ?", np.allclose(TS, ST))

V_TS = apply(TS, vertices_cube)
V_ST = apply(ST, vertices_cube)
for label, V in [("TS", V_TS), ("ST", V_ST)]:
    lo, hi = V[:3].min(axis=1), V[:3].max(axis=1)
    print("\n" + label, "vertices:\n", V)
    print(label, "center:", (lo + hi) / 2, "min:", lo, "max:", hi)

plot([ORIG, (V_TS, faces_cube, "tab:purple", "TS")],
     "Task 3: TS (scale, then translate)", "fig3_TS.png")
plot([ORIG, (V_ST, faces_cube, "tab:cyan", "ST")],
     "Task 3: ST (translate, then scale)", "fig3_ST.png")
plot([ORIG,
      (V_TS, faces_cube, "tab:purple", "TS"),
      (V_ST, faces_cube, "tab:cyan", "ST")],
     "Task 3: TS vs ST", "fig3_TS_vs_ST.png")