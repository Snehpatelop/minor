import pandapower as pp
import pandapower.networks as nw
import pandapower.topology as top
import pandapower.plotting as pplot
import numpy as np
import pandas as pd
import random
import math
import warnings
from copy import deepcopy
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore")

# Supported IEEE test systems for multi-system benchmarking
IEEE_SYSTEMS = {
    "IEEE 14": nw.case14,
    "IEEE 30": nw.case_ieee30,
    "IEEE 57": nw.case57,
    "IEEE 118": nw.case118,
}

class GridContingencyAnalyzer:
    def __init__(self, system_name="IEEE 118"):
        print(f"--- Initializing {system_name}-Bus System ---")
        self.system_name = system_name
        try:
            loader = IEEE_SYSTEMS.get(system_name, nw.case118)
            self.net = loader()
            print(f"SUCCESS: Loaded {system_name}-Bus System.")
        except Exception as e:
            print(f"WARNING: Could not load {system_name} ({e}). Using simple example.")
            self.net = nw.example_simple()
        self.num_lines = len(self.net.line)
        self.all_line_indices = list(self.net.line.index)
        print(f"System Size: {self.num_lines} Transmission Lines.")

    def run_genetic_algorithm(self, k=2, population_size=30, generations=40, mutation_rate=0.2):
        """
        Advanced: Genetic Algorithm for N-k contingency search.
        """
        print(f"\n--- Running Genetic Algorithm (N-{k}) ---")
        random.seed(42)
        # Each individual is a sorted tuple of k unique line indices
        def create_individual():
            return tuple(sorted(random.sample(self.all_line_indices, k)))
        def mutate(ind):
            ind = list(ind)
            idx = random.randint(0, k-1)
            choices = [x for x in self.all_line_indices if x not in ind]
            ind[idx] = random.choice(choices)
            return tuple(sorted(ind))
        def crossover(parent1, parent2):
            child = set(random.sample(parent1, k//2) + random.sample(parent2, k-k//2))
            while len(child) < k:
                child.add(random.choice(self.all_line_indices))
            return tuple(sorted(child))
        # Initialize population
        population = [create_individual() for _ in range(population_size)]
        best_ind, best_score = None, -float('inf')
        convergence_history = []
        for gen in range(generations):
            scores = [self.calculate_severity_score(list(ind)) for ind in population]
            # Track best
            for ind, score in zip(population, scores):
                if score > best_score:
                    best_score = score
                    best_ind = ind
            convergence_history.append(best_score)
            # Selection (tournament)
            selected = []
            for _ in range(population_size):
                i, j = random.sample(range(population_size), 2)
                selected.append(population[i] if scores[i] > scores[j] else population[j])
            # Crossover & Mutation
            next_pop = []
            for i in range(0, population_size, 2):
                p1, p2 = selected[i], selected[(i+1)%population_size]
                child1 = crossover(p1, p2)
                child2 = crossover(p2, p1)
                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)
                next_pop.extend([child1, child2])
            population = next_pop[:population_size]
        return list(best_ind), best_score, convergence_history

    def calculate_severity_score(self, lines_to_cut):
        """
        ADVANCED COST FUNCTION (Voltage, Topology, Overloads, Cascading)
        1. Checks for Islanding (Grid split) -> Max Penalty
        2. Checks for Voltage Collapse (Divergence) -> Max Penalty
        3. Calculates Voltage Drop -> Severity Score
        4. Adds penalty for overloaded lines
        5. Simulates simple cascading: if >2 lines overloaded, add penalty
        """
        net_copy = deepcopy(self.net)
        try:
            # 1. Disconnect lines
            net_copy.line.loc[lines_to_cut, 'in_service'] = False
            # 2. TOPOLOGY CHECK (Is the grid split?)
            unsupplied = top.unsupplied_buses(net_copy)
            if len(unsupplied) > 0:
                return 10.0 # Catastrophic Failure (Islanding)
            # 3. Run Power Flow
            pp.runpp(net_copy, respect_in_service=True)
            # 4. VOLTAGE METRIC (VSA)
            min_vm_pu = net_copy.res_bus.vm_pu.min()
            score = 1.0 - min_vm_pu
            if min_vm_pu < 0.90:
                score += 0.2
            # 5. LINE OVERLOAD METRIC
            if hasattr(net_copy, 'res_line') and 'loading_percent' in net_copy.res_line:
                overloads = net_copy.res_line[net_copy.res_line.loading_percent > 100]
                n_over = len(overloads)
                if n_over > 0:
                    score += 0.1 * n_over
                # 6. CASCADING: if >2 lines overloaded, add extra penalty
                if n_over > 2:
                    score += 0.3
            return score
        except pp.LoadflowNotConverged:
            return 10.0
        except Exception:
            return 10.0

    def run_n1_analysis(self):
        """
        Brute-force checks EVERY single line failure (N-1).
        """
        print("\n--- Running Full N-1 Contingency Scan ---")
        results = []
        for i in self.all_line_indices:
            score = self.calculate_severity_score([i])
            results.append({"Contingency": [i], "Score": score, "Type": "N-1"})
            
        # Sort by severity (descending)
        results.sort(key=lambda x: x["Score"], reverse=True)
        return results

    def run_simulated_annealing(self, k=2, max_iterations=500):
        """
        Heuristic optimization for N-k (N-2, N-3) failures.
        """
        print(f"\n--- Running Optimization (Simulated Annealing N-{k}) ---")
        
        # Set seed for reproducibility in presentation
        random.seed(42)
        
        current_cut = random.sample(self.all_line_indices, k)
        current_score = self.calculate_severity_score(current_cut)
        
        best_cut = list(current_cut)
        best_score = current_score
        
        T = 100.0
        cooling_rate = 0.96
        convergence_history = []

        for i in range(max_iterations):
            neighbor_cut = list(current_cut)
            neighbor_cut.remove(random.choice(neighbor_cut))
            
            candidates = [x for x in self.all_line_indices if x not in neighbor_cut]
            neighbor_cut.append(random.choice(candidates))
            
            neighbor_score = self.calculate_severity_score(neighbor_cut)
            delta = neighbor_score - current_score
            
            if delta > 0 or random.random() < math.exp(delta / T):
                current_cut = neighbor_cut
                current_score = neighbor_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_cut = list(current_cut)
            
            convergence_history.append(best_score)
            T *= cooling_rate
            if T < 0.01: break
                
        return best_cut, best_score, convergence_history

    def run_multi_seed(self, method="sa", k=2, num_runs=10, **kwargs):
        """
        Run an algorithm multiple times with different seeds for statistical analysis.
        Returns list of (best_cut, best_score, convergence_history) tuples.
        """
        print(f"\n--- Running {num_runs}-seed statistical analysis ({method.upper()}) ---")
        results = []
        for seed in range(num_runs):
            random.seed(seed)
            if method == "sa":
                cut, score, hist = self.run_simulated_annealing(k=k, **kwargs)
            else:
                cut, score, hist = self.run_genetic_algorithm(k=k, **kwargs)
            results.append({"seed": seed, "cut": cut, "score": score, "history": hist})
        return results

    def visualize_contingency(self, lines_to_cut, save_path=None, show=True):
        """
        High-Contrast Visualization for Presentations.
        Highlights broken lines AND the buses they connect.
        """
        print("\n--- Generating Visualization ---")
        try:
            import matplotlib.pyplot as plt
            
            # Generate Coords if missing
            if not hasattr(self.net, 'bus_geodata') or self.net.bus_geodata.empty:
                print("   Generating generic map coordinates...")
                try:
                    pplot.create_generic_coordinates(self.net)
                except:
                    pass

            # 1. Identify components
            all_lines = self.net.line.index
            healthy_lines = all_lines.difference(lines_to_cut)
            
            # Find the specific buses attached to the broken lines
            affected_buses = set()
            for l in lines_to_cut:
                affected_buses.add(self.net.line.at[l, 'from_bus'])
                affected_buses.add(self.net.line.at[l, 'to_bus'])

            # 2. CREATE COLLECTIONS (Layered for visual impact)
            
            # Layer 0: Healthy Grid (Faint Silver) - Fades into background
            lc_healthy = pplot.create_line_collection(self.net, lines=healthy_lines, color="silver", linewidth=1.0, zorder=1)
            bc_healthy = pplot.create_bus_collection(self.net, size=0.03, color="silver", zorder=2)
            
            # Layer 1: The Damage (Bold Red & Orange)
            # Highlight the disconnected lines
            lc_cut = pplot.create_line_collection(self.net, lines=lines_to_cut, color="red", linewidth=4.0, zorder=4)
            # Highlight the endpoints (Buses)
            bc_crit = pplot.create_bus_collection(self.net, buses=list(affected_buses), size=0.15, color="orange", zorder=5)
            
            # 3. DRAW
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(1, 1, 1)
            pplot.draw_collections([lc_healthy, bc_healthy, lc_cut, bc_crit], ax=ax)
            
            # Annotate
            plt.title(f"Critical N-{len(lines_to_cut)} Contingency Identified\nRed Lines = Disconnected | Orange Dots = Affected Buses", fontsize=12)
            plt.legend([lc_cut], [f"Failed Lines: {lines_to_cut}"], loc="upper right")
            
            if save_path:
                fig.tight_layout()
                fig.savefig(save_path, dpi=300, bbox_inches="tight")
            if show:
                print("   Displaying plot...")
                plt.show()
            else:
                plt.close(fig)
            
        except Exception as e:
            print(f"Visualization warning: {e}")
