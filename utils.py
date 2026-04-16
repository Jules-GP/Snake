import base64, io
import plotly.graph_objects as go
import numpy as np
import cv2
import base64, io
from PIL import Image as PILImage

def showAnim(img, x, y, pct=100, frame_dur=50):
    """
    pct : pourcentage des itérations à afficher (1-100)
    """
    # ── Sous-échantillonnage ──────────────────────────────────────────────────
    Nit = len(x)
    n_frames = max(1, int(Nit * pct / 100))
    indices = np.linspace(0, Nit - 1, n_frames, dtype=int)
    x_sampled = [x[i] for i in indices]
    y_sampled = [y[i] for i in indices]

    # ── Conversion image ──────────────────────────────────────────────────────
    img_uint8 = (img / img.max() * 255).astype(np.uint8) if img.dtype != np.uint8 else img
    if img_uint8.ndim == 2:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

    buf = io.BytesIO()
    PILImage.fromarray(img_uint8).save(buf, format="PNG")
    b64_img = base64.b64encode(buf.getvalue()).decode()

    H, W = img_uint8.shape[:2]
    SNAKE_COLOR = "#00e5ff"

    # ── Helpers ───────────────────────────────────────────────────────────────
    def closed(sx, sy):
        return list(sx) + [sx[0]], list(sy) + [sy[0]]

    # ── Figure de base ────────────────────────────────────────────────────────
    init_xs, init_ys = closed(x_sampled[0], y_sampled[0])
    fig = go.Figure(go.Scatter(
        x=init_xs, y=init_ys,
        mode="lines",
        line=dict(color=SNAKE_COLOR, width=2),
        name="snake",
    ))

    # ── Frames ────────────────────────────────────────────────────────────────
    fig.frames = [
        go.Frame(
            data=[go.Scatter(
                x=xs, y=ys,
                mode="lines",
                line=dict(color=SNAKE_COLOR, width=2),
            )],
            name=str(i),
            layout=go.Layout(title_text=f"Itération {indices[i]} / {Nit}"),  # itération réelle
        )
        for i, (sx, sy) in enumerate(zip(x_sampled, y_sampled))
        for xs, ys in [closed(sx, sy)]
    ]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        title=f"Snake actif — itération 0 / {Nit}  ({pct}% affiché)",
        width=W * 2 + 80, height=H * 2 + 160,
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor="#111", plot_bgcolor="#111",
        font=dict(color="white"),
        xaxis=dict(range=[0, W], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="y"),
        yaxis=dict(range=[H, 0], showgrid=False, zeroline=False,
                   showticklabels=False),
        images=[dict(
            source=f"data:image/png;base64,{b64_img}",
            xref="x", yref="y", x=0, y=0,
            sizex=W, sizey=H, sizing="stretch",
            layer="below", opacity=1.0,
        )],
        updatemenus=[
            dict(type="buttons", showactive=False,
                 x=0.2, xanchor="left", y=0, yanchor="top",
                 buttons=[dict(label="▶  Play", method="animate",
                               args=[None, dict(frame=dict(duration=frame_dur, redraw=True),
                                                fromcurrent=True, transition=dict(duration=0))])]),
            dict(type="buttons", showactive=False,
                 x=0.8, xanchor="right", y=0, yanchor="top",
                 buttons=[dict(label="⏸  Pause", method="animate",
                               args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                  mode="immediate", transition=dict(duration=0))])]),
        ],
        sliders=[dict(
            steps=[dict(method="animate",
                        args=[[str(i)], dict(mode="immediate",
                                             frame=dict(duration=frame_dur, redraw=True),
                                             transition=dict(duration=0))],
                        label=str(indices[i]))  # label = itération réelle
                   for i in range(len(fig.frames))],
            active=0, x=0.05, y=0, len=0.9,
            currentvalue=dict(prefix="Itération : ", font=dict(size=13, color="white"),
                              visible=True, xanchor="center"),
            transition=dict(duration=0),
            pad=dict(b=10, t=45),
            bgcolor="#333", bordercolor="#555", tickcolor="white",
            font=dict(color="white"),
        )],
    )

    fig.show()




def displayGradEnergy(E, window=10, percent=1, reduction="mean"):
    E = np.asarray(E)

    if E.ndim != 3 or E.shape[-1] != 2:
        raise ValueError("E doit avoir la forme (Niter, K, 2)")

    energy_xy = np.linalg.norm(E, axis=2)

    if reduction == "mean":
        energy = energy_xy.mean(axis=1)
    elif reduction == "max":
        energy = energy_xy.max(axis=1)
    elif reduction == "sum":
        energy = energy_xy.sum(axis=1)
    else:
        raise ValueError("reduction doit valoir 'mean', 'max' ou 'sum'")

    n_points = min(window, energy.size)
    if n_points == 0:
        raise ValueError("E est vide")

    # base_energy = energy[0] if energy[0] != 0 else 1.0
    # threshold = (percent / 100.0) * base_energy
    # recent = energy[-n_points:]
    # converged = (recent.max() - recent.min()) < threshold

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(energy.size),
            y=energy,
            mode="lines",
            name=f"Energie globale ({reduction})",
            line=dict(color="#1f77b4", width=2),
        )
    )

    # fig.add_hline(
    #     y=threshold,
    #     line_dash="dash",
    #     line_color="#d62728",
    #     annotation_text=f"Seuil ({percent}% de E0)",
    #     annotation_position="top left",
    # )

    fig.update_layout(
        title="Enegrie du Snake en fonction du temps",
        xaxis_title="Iteration",
        yaxis_title="Norme de l'energie",
        template="plotly_white",
        hovermode="x unified",
        dragmode="pan",
    )

    fig.update_xaxes(rangeslider_visible=True, showspikes=True)
    fig.update_yaxes(showspikes=True)
    fig.show()

    return energy


def plot_energy(e_int_tab, e_ext_tab, e_tot_tab):
    iter_range = list(range(1, len(e_tot_tab) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=iter_range, y=e_int_tab,
        mode="lines", name="Énergie interne",
        line=dict(color="#00e5ff", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=iter_range, y=e_ext_tab,
        mode="lines", name="Énergie externe",
        line=dict(color="#ff4081", width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=iter_range, y=e_tot_tab,
        mode="lines", name="Énergie totale",
        line=dict(color="#ffffff", width=2)
    ))

    fig.update_layout(
        title="Évolution de l'énergie du snake",
        xaxis=dict(title="Itération", color="white", gridcolor="#333"),
        yaxis=dict(title="Énergie", color="white", gridcolor="#333"),
        paper_bgcolor="#111",
        plot_bgcolor="#111",
        font=dict(color="white"),
        legend=dict(bgcolor="#222", bordercolor="#555", borderwidth=1),
        hovermode="x unified",
    )

    fig.show()