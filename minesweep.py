from bokeh.models import CustomJS
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

import streamlit as st
import numpy as np
from streamlit_bokeh_events import streamlit_bokeh_events
import sympy as sp


@st.cache
def gen_map(w, h, mines):
    zeros = np.full((h, w), False)
    i_s, j_s = np.indices((h, w))
    i_s = i_s.flatten()
    j_s = j_s.flatten()
    mines_index = np.random.choice(np.arange(0, len(i_s)), mines)
    zeros[i_s[mines_index], j_s[mines_index]] = True
    return zeros


@st.cache
def collect_adj(array: np.ndarray):
    collected = np.empty_like(array, dtype=object)
    for (i, j), _ in np.ndenumerate(array):
        adj_mask = np.full_like(array, False)
        adj_mask[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2] = True
        adj_mask[i, j] = False
        collected[i, j] = list(array[adj_mask])
    return collected


w, h, mines = [st.sidebar.number_input(name, value=v) for name, v in zip("w,h,mines".split(","), [10, 10, 30])]

has_mine = gen_map(w, h, mines)
numbers = np.vectorize(sum)(collect_adj(has_mine))

fig = figure(plot_width=400, plot_height=400, tools="tap,pan,wheel_zoom,box_zoom,reset")


def render_text(is_mine, number, opening):
    if not opening:
        return "　"
    if is_mine:
        return "🚩"
    elif number == 0:
        return "　"
    return "0１２３４５６７８９"[number]


i_s, j_s = np.indices((h, w))

now : np.ndarray = st.cache(allow_output_mutation=True)(lambda: np.full((h, w), False))()
texts = np.vectorize(render_text)(has_mine, numbers, now)

rect_sources = ColumnDataSource(data={
    "now_alpha": (~now).astype(np.float).flatten().clip(0.2,0.8),
    "x": j_s.flatten(),
    "y": i_s.flatten(),
})

st.write(now)

fig.rect(x="x", y="y", width=0.8, height=0.8, fill_alpha="now_alpha", source=rect_sources)
fig.text(x=j_s.flatten(), y=i_s.flatten() - 0.25, text=texts.flatten(), align="center")

rect_sources.selected.js_on_change(
    "indices",
    CustomJS(
        args=dict(source=texts),
        code="""
        document.dispatchEvent(
            new CustomEvent("TOGGLE", {detail: cb_obj.indices})
        )
        """
    )
)

toggling_index = streamlit_bokeh_events(fig, "TOGGLE", key="toggle", debounce_time=2)
if toggling_index is not None:
    flat_now = now.flatten()
    flat_now[toggling_index["TOGGLE"]] = ~flat_now[toggling_index["TOGGLE"]]
    st.write(flat_now)
    now[:] = flat_now.reshape(now.shape)
    st.write(now)
prob_symbols = np.array([[sp.Symbol(f"P_{{{i},{j}}}") for j in range(w)] for i in range(h)])
