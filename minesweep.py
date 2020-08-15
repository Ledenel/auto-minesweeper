from typing import Iterable, Tuple

import torch
from bokeh.models import CustomJS
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource

import streamlit as st
import numpy as np
from func_timeout import FunctionTimedOut, func_set_timeout
from streamlit_bokeh_events import streamlit_bokeh_events
from torch.distributions import Categorical
from torch.nn import Sigmoid
from lagrange_constrain import LagrangeConstrainedLoss
import sympy as sp


# from models import collect_tensor_adj_sum, MineModel


@st.cache
def gen_map(w, h, mines):
    zeros = np.full((h, w), False)
    i_s, j_s = np.indices((h, w))
    i_s = i_s.flatten()
    j_s = j_s.flatten()
    mines_index = np.random.choice(np.arange(0, len(i_s)), mines, replace=False)
    zeros[i_s[mines_index], j_s[mines_index]] = True
    return zeros


@st.cache
def collect_adj(array: np.ndarray):
    collected = np.empty_like(array, dtype=object)
    padding_array = np.pad(array.astype(object), 1, constant_values=None)
    for (i, j), _ in np.ndenumerate(array):
        pi, pj = i + 1, j + 1
        sliced = padding_array[pi - 1:pi + 2, pj - 1:pj + 2]
        mi, mj = sliced.shape
        si, sj = 1, 1
        collected[i, j] = list(sliced[i, j] for i in range(mi) for j in range(mj) if
                               not (i == si and j == sj) and sliced[i, j] is not None)
    return collected


w, h, mines = [
    st.sidebar.number_input(name, value=v) for name, v in zip(
        "w,h,mines".split(","),
        [3, 3, 4]
    )]

has_mine = gen_map(w, h, mines)
numbers = np.vectorize(sum)(collect_adj(has_mine))

fig = figure(plot_width=300, plot_height=300, tools="tap,pan,wheel_zoom,box_zoom,reset")


def render_text(is_mine, number, opening):
    if not opening:
        return "　"
    if is_mine:
        return "🚩"
    elif number == 0:
        return "　"
    return "0１２３４５６７８９"[number]


i_s, j_s = np.indices((h, w))

now: np.ndarray = st.cache(allow_output_mutation=True)(lambda: np.full((h, w), False))()
texts = np.vectorize(render_text)(has_mine, numbers, now)

i_s_y = h - i_s.flatten() - 1
rect_sources = ColumnDataSource(data={
    "now_alpha": (~now).astype(np.float).flatten().clip(0.2, 0.8),
    "x": j_s.flatten(),
    "y": i_s_y,
})

st.write(has_mine)

probs = st.empty()

fig.rect(x="x", y="y", width=0.8, height=0.8, fill_alpha="now_alpha", source=rect_sources)
fig.text(x=j_s.flatten(), y=i_s_y - 0.25, text=texts.flatten(), align="center")

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
    # st.write(flat_now)
    now[:] = flat_now.reshape(now.shape)
    # st.write(now)


def collect_tensor_adj_sum(tensor: torch.Tensor):
    kernel = torch.ones(3, 3)
    kernel[1, 1] = 0
    conv_adj = torch.nn.functional.conv2d(
        tensor.unsqueeze(0).unsqueeze(0),
        kernel.unsqueeze(0).unsqueeze(0),
        stride=1, padding=1
    )
    return conv_adj.squeeze().squeeze()


def restrict_vars(vars_source: Iterable[sp.Symbol], min_val=None, max_val=None, slack_func=lambda x: x * x) -> Tuple[
    np.ndarray, np.ndarray]:
    source = np.array(vars_source)
    eqs = []
    slack_vars = []
    if min_val is not None:
        slack_min = np.array(sp.symbols(f"zmin_{{:{len(source)}}}", real=True))
        slack_vars += slack_min.tolist()
        slack_min = np.vectorize(slack_func)(slack_min)
        eqs += (source - min_val - slack_min).tolist()
    if max_val is not None:
        slack_max = np.array(sp.symbols(f"zmax_{{:{len(source)}}}", real=True))
        slack_vars += slack_max.tolist()
        slack_max = np.vectorize(slack_func)(slack_max)
        eqs += (max_val - source - slack_max).tolist()
    return np.array(eqs), np.array(slack_vars)


class MineModel(torch.nn.Module):
    def __init__(self, h, w, now_tensor: torch.Tensor):
        super().__init__()
        self.now_tensor = now_tensor
        self.prob_vars_param = torch.nn.Parameter(torch.Tensor(size=(h, w)))
        self.constrain = LagrangeConstrainedLoss(
            eqs_zero_len=int(now_tensor.sum() + (now_tensor & ~has_mine).sum() + 1),
        )

    @property
    def probs(self):
        return Sigmoid()(self.prob_vars_param)

    def forward(self, has_mine_tensor, numbers_tensor, mines):
        prob_vars = self.probs
        opened_is_known = prob_vars - has_mine_tensor.type_as(prob_vars)
        opened_is_known = opened_is_known[self.now_tensor]
        around_numbers = collect_tensor_adj_sum(prob_vars) - numbers_tensor
        around_numbers = around_numbers[self.now_tensor & ~has_mine_tensor]
        all_mines_count = torch.sum(prob_vars) - mines
        all_eqs = torch.cat([opened_is_known, around_numbers, all_mines_count.unsqueeze(0)])
        entropy_energy = Categorical(probs=prob_vars.flatten()).entropy()
        loss = self.constrain(-entropy_energy, all_eqs)
        return loss


def is_symbol_all_probs(solve_dict, sym_set):
    for sym, value in solve_dict.items():
        if sym in sym_set:
            if ((value <= 1) & (value >= 0)) is sp.S.false:
                return False
    return True


solution_timeout = st.sidebar.number_input("solving timeout(seconds):", value=10)
solution_precision = st.sidebar.number_input("solution precision:", value=3)


def entropy(x):
    return -x * sp.log(x) + -(1-x) * sp.log(1-x)

def out_syms():
    prob_symbols = np.array([[sp.Symbol(f"P_{{{i},{j}}}", real=True) for j in range(w)] for i in range(h)])

    opened_is_known = prob_symbols - has_mine.astype(int)
    opened_is_known = opened_is_known[now]

    around_numbers = np.vectorize(sum)(collect_adj(prob_symbols)) - numbers
    around_numbers = around_numbers[now & ~has_mine]

    prob_flatten = prob_symbols.flatten()
    if st.sidebar.checkbox("knowing total mines?"):
        all_bombs = np.array([sum(prob_flatten) - mines])
    else:
        all_bombs = np.array([])


    entropy_energy = sum(np.vectorize(entropy)(prob_flatten))

    lower, upper = None, None
    slack = "x^2"
    if st.sidebar.checkbox("use slacks?"):
        if st.sidebar.checkbox("lower bound?"):
            lower = st.sidebar.number_input("lower bound:", value=0.0)
        if st.sidebar.checkbox("upper bound?"):
            upper = st.sidebar.number_input("upper bound:", value=1.0)
        slack = st.sidebar.text_input("slack variable form:", value=slack)
    slack_expr = sp.sympify(slack)
    r_eqs, slacks = restrict_vars(prob_flatten, lower, upper, sp.lambdify(slack_expr.free_symbols, slack_expr))

    # st.table(r_eqs)
    # st.table(slacks)
    prob_flatten = np.concatenate([prob_flatten, slacks])

    eqs = np.concatenate([opened_is_known, around_numbers, all_bombs, r_eqs])

    lambdas = np.array([sp.Symbol(f"\\lambda_{{{i}}}", real=True) for i in range(len(eqs))])
    ln_lambdas = np.array([sp.Symbol(f"\\mu_{{{i}}}", real=True) for i in range(len(eqs))])
    ln_lambdas_rep = [(l, sp.log(ln_l)) for l, ln_l in zip(lambdas, ln_lambdas)]

    L = entropy_energy + sum(lambdas * eqs)
    L_sym = sp.Symbol("L")
    st.header("target:")
    st.latex(latex_eqs(L_sym, L))

    variables = np.concatenate([prob_flatten, lambdas])
    prob_set = set(prob_symbols.flatten())
    gradL = [{"diff": x, "equation": sp.diff(L, x).subs(ln_lambdas_rep)} for x in variables]
    st.header("before simplification:")
    for x in variables:
        st.latex(latex_eqs(sp.Derivative(L_sym, x, evaluate=False), sp.diff(L, x), 0))

    st.subheader("using new variables:")
    st.latex(" , ".join(latex_eqs(a, b) for a, b in ln_lambdas_rep))
    st.subheader("after substitution:")
    for e in gradL:
        st.latex(latex_eqs(e["equation"], 0))
    # rewrite by poly
    for item in gradL:
        # eq = item["equation"]
        # for var in eq.free_symbols:
        #     if var in prob_set:
        #         eq = var - sp.solve(eq, var)[0]
        #         break
        item["equation"] = poly_rewrite(item["equation"])

    st.subheader("after poly roots rewrite:")
    for e in gradL:
        st.latex(latex_eqs(e["equation"], 0))
    # st.table(gradL)
    vars_with_lnlambda = np.concatenate([prob_flatten, ln_lambdas, slacks])
    simplified_eqs = [e["equation"] for e in gradL]
    try:
        prob_new = solve_eq(prob_symbols, simplified_eqs, vars_with_lnlambda)
        probs.dataframe(prob_new)
    except FunctionTimedOut:
        probs.text("solving time out.")

    # non_lin = sp.nonlinsolve(
    #     simplified_eqs,
    #     vars_with_lnlambda.tolist()
    # )
    #
    # st.write(non_lin)

    # solved = [i for i in solved if is_symbol_all_probs(i, prob_set)]
    # st.write(solved)


def poly_rewrite(eq):
    solved_items = list(poly_rewrite_iter(eq))
    if solved_items:
        return sp.Mul(*[sym - val for sym, val in solved_items])
    else:
        return eq


def poly_rewrite_iter(eq):
    solved_eq = sp.solve(eq, dict=True)
    for solution in solved_eq:
        assert len(solution) == 1, f"it is impossible to get multivariate {solution.keys()} from {eq}"
        yield from solution.items()


@func_set_timeout(solution_timeout)
def solve_eq(prob_symbols, simplified_eqs, vars_with_lnlambda):
    solved = sp.solve(
        simplified_eqs,
        vars_with_lnlambda.tolist(),
        dict=True
    )
    for i1, solution in enumerate(solved):
        st.subheader(f"solution {i1}:")
        for sym, var1 in solution.items():
            st.latex(latex_eqs(sym, var1))
    st.table(solved)
    prob_new = np.vectorize(lambda x: x.subs(solved[0]).evalf(solution_precision))(prob_symbols)
    return prob_new


# @func_set_timeout(solution_timeout)


def latex_eqs(*exprs):
    strs = " = ".join(sp.latex(expr) for expr in exprs)
    return strs


out_syms()


def torch_optim():
    global mine_model
    has_mine_tensor = torch.from_numpy(has_mine)
    now_tensor = torch.from_numpy(now)
    numbers_tensor = torch.from_numpy(numbers)
    mine_model = MineModel(h, w, now_tensor)
    optim = torch.optim.SGD(mine_model.parameters(), lr=0.05, momentum=0.01)
    max_epoch = st.sidebar.number_input("running epoches", value=500)
    prog = st.sidebar.progress(0)
    training = st.sidebar.empty()
    for epoch in range(max_epoch):
        optim.zero_grad()
        loss = mine_model(has_mine_tensor, numbers_tensor, mines)
        loss.backward()
        training.text(f"constrained entropy: {loss}")
        prog.progress(epoch / max_epoch)
        probs.table(mine_model.probs.detach().numpy())
        optim.step()

    probs.table(mine_model.probs.detach().numpy())


# torch_optim()


r = """

prob_symbols = np.array([[sp.Symbol(f"P_{{{i},{j}}}", real=True) for j in range(w)] for i in range(h)])

opened_is_known = prob_symbols - has_mine.astype(int)
opened_is_known = opened_is_known[now]

around_numbers = np.vectorize(sum)(collect_adj(prob_symbols)) - numbers
around_numbers = around_numbers[now]

all_bombs = np.array([sum(prob_symbols.flatten()) - mines])


def entropy(x):
    return -x * sp.log(x)


entropy_energy = sum(np.vectorize(entropy)(prob_symbols.flatten()))

eqs = np.concatenate([opened_is_known, around_numbers, all_bombs])

lambdas = np.array([sp.Symbol(f"\\lambda_{i}", real=True) for i in range(len(eqs))])

L = entropy_energy + sum(lambdas * eqs)
variables = np.concatenate([prob_symbols.flatten(), lambdas])
gradL = [sp.diff(L, x) for x in variables]

slacken_less_than_1_vars = np.array([[sp.Symbol(f"z_{{{i},{j}}}", real=True) for j in range(w)] for i in range(h)])
less_than_1 = 1 - prob_symbols - slacken_less_than_1_vars

# initial_solve = sp.solve(eqs.tolist() + less_than_1.flatten().tolist(), tuple(np.concatenate([prob_symbols.flatten(), slacken_less_than_1_vars.flatten()])), positive=True, dict=True)
initial_solve = sp.solveset(eqs.tolist(), prob_symbols.flatten().tolist(), domain=sp.Interval(0, 1))
initial_solved_dict = {}
for solve_set in initial_solve:
    for sym, value in zip(prob_symbols.flatten(), solve_set):  # solve_set.items():
        st.latex(sp.latex(sp.Eq(sym, value, evaluate=False)))
        initial_solved_dict[sym] = value


# else:
#     st.markdown("**Condition has conflicts.**")

# prob_in_0_1: np.ndarray = np.vectorize(lambda x: np.array(
#     [x >= 0,
#      x <= 1]), signature="()->(k)")(np.array(list(initial_solved_dict.values()))).flatten()
#
# ineq = sp.And(*prob_in_0_1.tolist())
# ineq_solve = sp.solveset(ineq, ineq.free_symbols, domain=sp.S.Reals)
# red = sp.reduce_inequalities(ineq, ineq.free_symbols)
# red


# stationary_points = sp.nsolve(gradL, variables.tolist(), np.ones_like(variables) - 0.5, dict=True, prec=3)
# np.ones_like(variables) - 0.5,
# max_point = max((p for p in stationary_points), key=lambda p: entropy_energy.subs(p))

# st.write(max_point)

@np.vectorize
def get_solved(x):
    return float(max_point[x])

# st.write(get_solved(prob_symbols))
"""
