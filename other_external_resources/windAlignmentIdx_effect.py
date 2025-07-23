import plotly.graph_objects as go
import math

def interpolate_color(color_start, color_end, t):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(rgb):
        return '#' + ''.join(f'{int(c):02x}' for c in rgb)

    rgb_start = hex_to_rgb(color_start)
    rgb_end = hex_to_rgb(color_end)
    rgb_interp = tuple(
        (1 - t) * s + t * e for s, e in zip(rgb_start, rgb_end)
    )
    return rgb_to_hex(rgb_interp)

def plot_line(fig, start, end, color='blue', name=None, width=2):
    x0, y0 = start
    x1, y1 = end
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode='lines',
        line=dict(color=color, width=width),
        name=name or f"Line {len(fig.data)+1}",
        showlegend=False
    ))

def calculate_all_directions(Cvp, degStep):
    line_list = []
    vector_1 = [1, 0]
    scores = []

    for i in range(round(math.pi*2 / degStep)):
        angle = i * degStep
        vector_2 = [math.cos(angle), math.sin(angle)]
        dot = vector_1[0]*vector_2[0] + vector_1[1]*vector_2[1]
        cross = abs(vector_1[0]*vector_2[1] - vector_1[1]*vector_2[0])
        score = dot - Cvp * cross
        scores.append((vector_2, score))

    neg_scores = [s for _, s in scores if s < 0]
    pos_scores = [s for _, s in scores if s > 0]

    min_neg = min(neg_scores) if neg_scores else -1
    max_neg = max(neg_scores) if neg_scores else -0.01  # close to zero negative
    min_pos = min(pos_scores) if pos_scores else 0.01   # close to zero positive
    max_pos = max(pos_scores) if pos_scores else 1

    for vec, score in scores:
        if score < 0:
            t = (score - max_neg) / (min_neg - max_neg) if (min_neg - max_neg) != 0 else 0  # normalize between max_neg and min_neg
            color = interpolate_color('#ffa500', '#ff0000', t)  # orange to red
        elif score > 0:
            t = (score - min_pos) / (max_pos - min_pos) if (max_pos - min_pos) != 0 else 0  # normalize between min_pos and max_pos
            color = interpolate_color('#ffff00', '#90ee90', t)  # yellow to light green
        else:
            color = '#aaaaaa'  # neutral gray for zero

        line_list.append([vec, color])
    
    line_list.append([vector_1, 'black'])

    return line_list, min_neg, max_neg, min_pos, max_pos

def plot_all(fig, line_list):
    for i in range(len(line_list)-1):
        plot_line(fig, [0,0], line_list[i][0], line_list[i][1], width=2)
    plot_line(fig, [0,0], line_list[-1][0], line_list[-1][1], width=5)

fig = go.Figure()

Cvp = 1

line_list, min_neg, max_neg, min_pos, max_pos = calculate_all_directions(Cvp, math.radians(0.5))

plot_all(fig, line_list)

color_min_neg = interpolate_color('#ffa500', '#ff0000', 1)  # min neg (red)
color_max_neg = interpolate_color('#ffa500', '#ff0000', 0)  # max neg (orange)
color_min_pos = interpolate_color('#ffff00', '#90ee90', 0)  # min pos (yellow)
color_max_pos = interpolate_color('#ffff00', '#90ee90', 1)  # max pos (light green)
color_zero = '#aaaaaa'
color_ref = 'black'

legend_entries = [
    (f"Idx Min Negatiu ({min_neg:.2f})", color_min_neg),
    (f"Idx Max Negatiu ({max_neg:.2f})", color_max_neg),
    (f"Idx Min Positiu ({min_pos:.2f})", color_min_pos),
    (f"Idx Max Positiu ({max_pos:.2f})", color_max_pos),
    ("Vector transferència", color_ref)
]

for label, color in legend_entries:
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='lines',
        line=dict(color=color, width=4),
        name=label,
        showlegend=True
    ))

fig.update_layout(
    title=f"Valors idx per a Cvp: {Cvp}",
    showlegend=True,
    legend=dict(title="Llegenda"),
    xaxis=dict(scaleanchor='y'),
    yaxis=dict(scaleanchor='x'),
)

fig.show()
