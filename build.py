import os

# --- 1. REQUIREMENTS ---
requirements_content = """pandapower>=2.10.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.5.0
networkx>=2.5
"""

# --- 2. CORE LOGIC (Analyzer) ---
analyzer_content = '''import pandapower as pp
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

class GridContingencyAnalyzer:
    def __init__(self):
        print("--- Initializing IEEE 118-Bus System ---")
        try:
            # Load the standard IEEE 118 case
            self.net = nw.case118()
            print("SUCCESS: Loaded IEEE 118-Bus System.")
        except Exception as e:
            print(f"WARNING: Could not load IEEE 118 ({e}). Using simple example.")
            self.net = nw.example_simple()
            
        self.num_lines = len(self.net.line)
        self.all_line_indices = list(self.net.line.index)
        print(f"System Size: {self.num_lines} Transmission Lines.")

    def calculate_severity_score(self, lines_to_cut):
        """
        THE COST FUNCTION (Voltage Stability & Topology)
        1. Checks for Islanding (Grid split) -> Max Penalty
        2. Checks for Voltage Collapse (Divergence) -> Max Penalty
        3. Calculates Voltage Drop -> Severity Score
        """
        net_copy = deepcopy(self.net)
        
        try:
            # 1. Disconnect lines
            net_copy.line.loc[lines_to_cut, 'in_service'] = False
            
            # 2. TOPOLOGY CHECK (Is the grid split?)
            # unsupplied_buses returns buses that are disconnected from the slack bus
            unsupplied = top.unsupplied_buses(net_copy)
            if len(unsupplied) > 0:
                return 10.0 # Catastrophic Failure (Islanding)

            # 3. Run Power Flow
            pp.runpp(net_copy, respect_in_service=True)
            
            # 4. VOLTAGE METRIC (VSA)
            min_vm_pu = net_copy.res_bus.vm_pu.min()
            
            # Score formula: 1.0 - Min_Voltage
            score = 1.0 - min_vm_pu
            
            # Critical Penalty (>0.1 means Voltage < 0.90)
            if min_vm_pu < 0.90:
                score += 0.2
                
            return score

        except pp.LoadflowNotConverged:
            # Voltage Collapse / Blackout
            return 10.0
        except Exception:
            # General Failure
            return 10.0

    def run_n1_analysis(self):
        """
        Brute-force checks EVERY single line failure (N-1).
        """
        print("\\n--- Running Full N-1 Contingency Scan ---")
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
        print(f"\\n--- Running Optimization (Simulated Annealing N-{k}) ---")
        
        # Set seed for reproducibility in presentation
        random.seed(42)
        
        current_cut = random.sample(self.all_line_indices, k)
        current_score = self.calculate_severity_score(current_cut)
        
        best_cut = list(current_cut)
        best_score = current_score
        
        T = 100.0
        cooling_rate = 0.96

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
            
            T *= cooling_rate
            if T < 0.01: break
                
        return best_cut, best_score

    def visualize_contingency(self, lines_to_cut):
        """
        High-Contrast Visualization for Presentations.
        Highlights broken lines AND the buses they connect.
        """
        print("\\n--- Generating Visualization ---")
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
            plt.figure(figsize=(10, 8))
            pplot.draw_collections([lc_healthy, bc_healthy, lc_cut, bc_crit])
            
            # Annotate
            plt.title(f"Critical N-{len(lines_to_cut)} Contingency Identified\\nRed Lines = Disconnected | Orange Dots = Affected Buses", fontsize=12)
            plt.legend([lc_cut], [f"Failed Lines: {lines_to_cut}"], loc="upper right")
            
            print("   Displaying plot...")
            plt.show()
            
        except Exception as e:
            print(f"Visualization warning: {e}")
'''

# --- 3. EXECUTION SCRIPT ---
setup_content = '''import subprocess
import sys
import time
import pandas as pd

def install_deps():
    try:
        import pandapower
        import matplotlib
        import networkx
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandapower', 'numpy', 'pandas', 'matplotlib', 'networkx'])

def print_line_details(analyzer, line_indices):
    print("   Failure Details:")
    for l in line_indices:
        try:
            fb = analyzer.net.line.at[l, 'from_bus']
            tb = analyzer.net.line.at[l, 'to_bus']
            print(f"   -> Line {l}: Connects Bus {fb} <-> Bus {tb}")
        except:
            print(f"   -> Line {l}")

def run():
    install_deps()
    import contingency_analyzer
    
    analyzer = contingency_analyzer.GridContingencyAnalyzer()
    
    # --- STEP 1: N-1 SCAN ---
    n1_results = analyzer.run_n1_analysis()
    
    print("\\n--- Top 3 Critical N-1 Contingencies ---")
    for res in n1_results[:3]:
        print(f"Score {res['Score']:.4f} | Line {res['Contingency'][0]}")
    
    # --- STEP 2: N-2 OPTIMIZATION ---
    start = time.time()
    worst_lines, worst_score = analyzer.run_simulated_annealing(k=2, max_iterations=400)
    
    print("\\n" + "="*50)
    print("FINAL ANALYSIS RESULTS")
    print("="*50)
    print(f"Execution Time:         {time.time()-start:.2f}s")
    print(f"Critical N-2 Failure:   Lines {worst_lines}")
    print_line_details(analyzer, worst_lines)
    print(f"Severity Score:         {worst_score:.4f}")
    
    if worst_score >= 10.0:
        print("GRID STATUS:            BLACKOUT (Islanding or Collapse)")
    elif worst_score > 0.15:
        print("GRID STATUS:            CRITICAL VOLTAGE DROP")
    else:
        print("GRID STATUS:            STABLE")
    print("="*50)
    
    # --- STEP 3: SAVE BENCHMARK DATA ---
    print("\\n--- Saving Benchmark Data ---")
    # Combine N-1 and the best N-2 result
    all_data = n1_results
    all_data.append({"Contingency": worst_lines, "Score": worst_score, "Type": "N-2 (Optimized)"})
    
    df = pd.DataFrame(all_data)
    # Convert list of lines to string for CSV readability
    df['Contingency'] = df['Contingency'].astype(str)
    
    filename = "benchmark_results.csv"
    df.to_csv(filename, index=False)
    print(f"Success! Results saved to '{filename}'. You can use this file as your ground truth.")
    
    # --- STEP 4: VISUALIZE ---
    analyzer.visualize_contingency(worst_lines)

if __name__ == "__main__":
    run()
'''

# --- 4. README ---
readme_content = """# Power Grid Contingency Analysis System

## Overview
A complete, self-contained system for identifying critical failures in the IEEE 118-bus power grid. It performs both N-1 (Exhaustive) and N-2 (Heuristic) analysis.

## Features
* **Voltage Stability Cost Function:** $S(x) = 1.0 - V_{min}$.
* **Simulated Annealing:** Finds critical N-2 faults without brute force.
* **Auto-Calibration:** Generates a `benchmark_results.csv` file automatically.
* **High-Contrast Visualization:** Highlights critical failures in Red/Orange against a Silver grid.

## How to Run
1.  Run the build script to generate the system.
2.  `cd Grid_Contingency_Project`
3.  `python setup_and_run.py`
"""

# --- BUILDER ---
def build():
    folder = "Grid_Contingency_Project"
    if not os.path.exists(folder): os.makedirs(folder)
    print(f"Building project in: {folder}")
    
    files = {
        "requirements.txt": requirements_content,
        "contingency_analyzer.py": analyzer_content,
        "setup_and_run.py": setup_content,
        "README.md": readme_content
    }
    
    for name, content in files.items():
        with open(os.path.join(folder, name), "w", encoding='utf-8') as f:
            f.write(content)
        print(f"  - Generated {name}")
    
    print("\\nSuccess. Run 'python setup_and_run.py' inside the folder.")

if __name__ == "__main__":
    build()