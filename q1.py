import math
import random
from dataclasses import dataclass
from typing import Callable, Tuple, List
import numpy as np
import pandas as pd
import streamlit as st

# -------------------- Problem Definitions --------------------
@dataclass
class GAProblem:
    name: str
    chromosome_type: str
    dim: int
    bounds: Tuple[float, float] | None
    fitness_fn: Callable[[np.ndarray], float]

# ---- Assignment OneMax with peak at 40 ones ----
def make_peak40(dim: int) -> GAProblem:
    def fitness(x: np.ndarray) -> float:
        ones = np.sum(x)
        return 80 - abs(ones - 40)   # max fitness 80 at ones=40

    return GAProblem(
        name="OneMax Peak@40 (80 bits)",
        chromosome_type="bit",
        dim=dim,
        bounds=None,
        fitness_fn=fitness,
    )

# -------------------- GA Operators (UNCHANGED) --------------------
def init_population(problem: GAProblem, pop_size: int, rng):
    return rng.integers(0, 2, size=(pop_size, problem.dim), dtype=np.int8)

def tournament_selection(fitness, k, rng):
    idxs = rng.integers(0, fitness.size, size=k)
    return int(idxs[np.argmax(fitness[idxs])])

def one_point_crossover(a, b, rng):
    point = int(rng.integers(1, a.size))
    c1 = np.concatenate([a[:point], b[point:]])
    c2 = np.concatenate([b[:point], a[point:]])
    return c1, c2

def bit_mutation(x, mut_rate, rng):
    mask = rng.random(x.shape) < mut_rate
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def evaluate(pop, problem):
    return np.array([problem.fitness_fn(ind) for ind in pop], dtype=float)

# -------------------- GA Engine (UNCHANGED) --------------------
def run_ga(problem, pop_size, generations, crossover_rate, mutation_rate,
           tournament_k, elitism, seed, stream_live=True):

    rng = np.random.default_rng(seed)
    pop = init_population(problem, pop_size, rng)
    fit = evaluate(pop, problem)

    chart_area = st.empty()
    best_area = st.empty()

    history_best, history_avg, history_worst = [], [], []

    for gen in range(generations):
        best = float(np.max(fit))
        avg = float(np.mean(fit))
        worst = float(np.min(fit))

        history_best.append(best)
        history_avg.append(avg)
        history_worst.append(worst)

        if stream_live:
            df = pd.DataFrame({"Best": history_best, "Average": history_avg, "Worst": history_worst})
            chart_area.line_chart(df)
            best_area.markdown(f"Generation {gen+1}/{generations} — Best fitness: **{best:.2f}**")

        # Elitism
        elite_idx = np.argsort(fit)[-elitism:]
        elites = pop[elite_idx]

        next_pop = []

        while len(next_pop) < pop_size - elitism:
            i1 = tournament_selection(fit, tournament_k, rng)
            i2 = tournament_selection(fit, tournament_k, rng)

            p1, p2 = pop[i1], pop[i2]

            if rng.random() < crossover_rate:
                c1, c2 = one_point_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = bit_mutation(c1, mutation_rate, rng)
            c2 = bit_mutation(c2, mutation_rate, rng)

            next_pop.append(c1)
            if len(next_pop) < pop_size - elitism:
                next_pop.append(c2)

        pop = np.vstack([np.array(next_pop), elites])
        fit = evaluate(pop, problem)

    best_idx = np.argmax(fit)
    return pop[best_idx], fit[best_idx], history_best, history_avg, history_worst

# -------------------- Streamlit UI (Same Template) --------------------
st.set_page_config(page_title="GA Peak@40", layout="wide")
st.title("Genetic Algorithm – Peak at 40 Ones")

st.sidebar.header("Fixed Assignment Parameters")
st.sidebar.write("Population = 300")
st.sidebar.write("Chromosome Length = 80")
st.sidebar.write("Generations = 50")
st.sidebar.write("Target Ones = 40")
st.sidebar.write("Max Fitness = 80")

# Fixed problem
problem = make_peak40(80)  # chromosome length = 80

# Fixed GA parameters
pop_size = 300
generations = 50
crossover_rate = 0.9
mutation_rate = 0.01
tournament_k = 3
elitism = 2
seed = 42

if st.button("Run GA"):
    best, best_fit, best_hist, avg_hist, worst_hist = run_ga(
        problem, pop_size, generations, crossover_rate,
        mutation_rate, tournament_k, elitism, seed
    )

    st.subheader("Fitness Curve")
    st.line_chart(pd.DataFrame({
        "Best": best_hist,
        "Average": avg_hist,
        "Worst": worst_hist
    }))

    st.subheader("Best 80-bit Pattern")
    bitstring = "".join(best.astype(str))
    st.code(bitstring)
    st.write("Number of ones:", int(np.sum(best)))
    st.write("Fitness:", best_fit)
