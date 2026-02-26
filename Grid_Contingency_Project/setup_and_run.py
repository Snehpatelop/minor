import subprocess
import sys
import time
import os
import logging
import numpy as np
import pandas as pd

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend so script doesn't block

def install_deps():
    try:
        import pandapower
        import matplotlib
        import networkx
    except ImportError:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandapower', 'numpy', 'pandas', 'matplotlib', 'networkx', 'plotly', 'tabulate', 'streamlit'])

def print_line_details(analyzer, line_indices):
    from tabulate import tabulate
    details = []
    for l in line_indices:
        try:
            fb = analyzer.net.line.at[l, 'from_bus']
            tb = analyzer.net.line.at[l, 'to_bus']
            details.append([l, fb, tb])
        except:
            details.append([l, "?", "?"])
    print(tabulate(details, headers=["Line", "From Bus", "To Bus"], tablefmt="fancy_grid"))

def run():
    install_deps()
    import contingency_analyzer

    project_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("Starting contingency analysis run.")
    
    analyzer = contingency_analyzer.GridContingencyAnalyzer()
    
    # --- STEP 1: N-1 SCAN ---
    n1_results = analyzer.run_n1_analysis()
    
    from tabulate import tabulate
    print("\n--- Top 3 Critical N-1 Contingencies ---")
    n1_table = [[res['Contingency'][0], f"{res['Score']:.4f}"] for res in n1_results[:3]]
    print(tabulate(n1_table, headers=["Line", "Severity Score"], tablefmt="fancy_grid"))
    
    # --- STEP 2: N-2 OPTIMIZATION (Simulated Annealing vs Genetic Algorithm) ---
    start = time.time()
    sa_lines, sa_score, sa_history = analyzer.run_simulated_annealing(k=2, max_iterations=400)
    sa_time = time.time()-start
    start = time.time()
    ga_lines, ga_score, ga_history = analyzer.run_genetic_algorithm(k=2, population_size=30, generations=40, mutation_rate=0.2)
    ga_time = time.time()-start
    # Choose best
    if ga_score > sa_score:
        best_lines, best_score, best_method, best_time = ga_lines, ga_score, "Genetic Algorithm", ga_time
    else:
        best_lines, best_score, best_method, best_time = sa_lines, sa_score, "Simulated Annealing", sa_time
    print("\n" + "="*50)
    print("FINAL ANALYSIS RESULTS (Classical Benchmark + Advanced)")
    print("="*50)
    print(f"Best N-2 Method:        {best_method}")
    print(f"Execution Time:         {best_time:.2f}s")
    print(f"Critical N-2 Failure:   Lines {best_lines}")
    print_line_details(analyzer, best_lines)
    print(f"Severity Score:         {best_score:.4f}")
    if best_score >= 10.0:
        status = "BLACKOUT (Islanding or Collapse)"
    elif best_score > 0.15:
        status = "CRITICAL VOLTAGE DROP"
    else:
        status = "STABLE"
    print(f"GRID STATUS:            {status}")
    print("="*50)
    # Benchmark summary for publication
    print("\n[Benchmark Summary] Classical N-1/N-2 Contingency Analysis")
    print(f"Best N-1 Score: {n1_results[0]['Score']:.4f} (Line {n1_results[0]['Contingency'][0]})")
    print(f"Best N-2 Score: {best_score:.4f} (Lines {best_lines}) via {best_method}")
    print(f"Grid Status:    {status}")
    logging.info("Best N-1: score=%.4f line=%s", n1_results[0]['Score'], n1_results[0]['Contingency'][0])
    logging.info("Best N-2: score=%.4f lines=%s method=%s", best_score, best_lines, best_method)
    logging.info("Grid Status: %s", status)
    
    # --- STEP 3: SAVE BENCHMARK DATA ---
    print("\n--- Saving Benchmark Data ---")
    # Save N-2 results from both algorithms in separate CSVs
    import copy
    # Simulated Annealing CSV
    sa_data = copy.deepcopy(n1_results)
    sa_data.append({"Contingency": sa_lines, "Score": sa_score, "Type": "N-2 (Optimized, Simulated Annealing)"})
    df_sa = pd.DataFrame(sa_data)
    df_sa['Contingency'] = df_sa['Contingency'].astype(str)
    sa_filename = os.path.join(project_dir, "benchmark_simulated_annealing.csv")
    df_sa.to_csv(sa_filename, index=False)
    print(f"Success! Simulated Annealing results saved to '{sa_filename}'.")
    logging.info("Saved %s", sa_filename)
    # Genetic Algorithm CSV
    ga_data = copy.deepcopy(n1_results)
    ga_data.append({"Contingency": ga_lines, "Score": ga_score, "Type": "N-2 (Optimized, Genetic Algorithm)"})
    df_ga = pd.DataFrame(ga_data)
    df_ga['Contingency'] = df_ga['Contingency'].astype(str)
    ga_filename = os.path.join(project_dir, "benchmark_genetic.csv")
    df_ga.to_csv(ga_filename, index=False)
    print(f"Success! Genetic Algorithm results saved to '{ga_filename}'.")
    logging.info("Saved %s", ga_filename)

    # --- STEP 3b: CONVERGENCE PLOT ---
    try:
        import matplotlib.pyplot as plt
        fig_conv, ax_conv = plt.subplots(1, 1, figsize=(10, 5))
        ax_conv.plot(sa_history, label="Simulated Annealing", linewidth=2, color="blue")
        ax_conv.plot(ga_history, label="Genetic Algorithm", linewidth=2, color="green")
        ax_conv.set_xlabel("Iteration / Generation")
        ax_conv.set_ylabel("Best Severity Score")
        ax_conv.set_title("Convergence Comparison: SA vs GA")
        ax_conv.legend()
        ax_conv.grid(True, alpha=0.3)
        conv_path = os.path.join(output_dir, "convergence_comparison.png")
        fig_conv.savefig(conv_path, dpi=300, bbox_inches="tight")
        plt.close(fig_conv)
        logging.info("Saved convergence plot to %s", conv_path)
        print(f"Convergence plot saved to '{conv_path}'.")
    except Exception as e:
        logging.warning("Convergence plot failed: %s", e)

    # --- STEP 3c: MULTI-RUN STATISTICAL ANALYSIS ---
    print("\n--- Running Statistical Analysis (10 seeds) ---")
    sa_stats = analyzer.run_multi_seed(method="sa", k=2, num_runs=10, max_iterations=400)
    ga_stats = analyzer.run_multi_seed(method="ga", k=2, num_runs=10, population_size=30, generations=40, mutation_rate=0.2)
    sa_scores = [r["score"] for r in sa_stats]
    ga_scores = [r["score"] for r in ga_stats]
    stats_data = {
        "Algorithm": ["Simulated Annealing", "Genetic Algorithm"],
        "Mean Score": [round(np.mean(sa_scores), 4), round(np.mean(ga_scores), 4)],
        "Std Dev": [round(np.std(sa_scores), 4), round(np.std(ga_scores), 4)],
        "Min Score": [min(sa_scores), min(ga_scores)],
        "Max Score": [max(sa_scores), max(ga_scores)],
    }
    df_stats = pd.DataFrame(stats_data)
    stats_path = os.path.join(project_dir, "statistical_comparison.csv")
    df_stats.to_csv(stats_path, index=False)
    print(tabulate(df_stats.values.tolist(), headers=df_stats.columns.tolist(), tablefmt="fancy_grid"))
    logging.info("Saved statistical comparison to %s", stats_path)

    # --- STEP 3d: MULTI-SYSTEM BENCHMARK ---
    print("\n--- Running Multi-System Benchmark ---")
    multi_sys_results = []
    for sys_name in ["IEEE 14", "IEEE 30", "IEEE 57", "IEEE 118"]:
        try:
            a = contingency_analyzer.GridContingencyAnalyzer(system_name=sys_name)
            n1 = a.run_n1_analysis()
            _, sa_s, _ = a.run_simulated_annealing(k=2, max_iterations=200)
            _, ga_s, _ = a.run_genetic_algorithm(k=2, population_size=20, generations=20, mutation_rate=0.2)
            multi_sys_results.append({
                "System": sys_name,
                "Lines": a.num_lines,
                "Top N-1 Score": round(n1[0]["Score"], 4),
                "SA N-2 Score": round(sa_s, 4),
                "GA N-2 Score": round(ga_s, 4),
            })
        except Exception as e:
            logging.warning("Multi-system %s failed: %s", sys_name, e)
    df_multi = pd.DataFrame(multi_sys_results)
    multi_path = os.path.join(project_dir, "multi_system_benchmark.csv")
    df_multi.to_csv(multi_path, index=False)
    print(tabulate(df_multi.values.tolist(), headers=df_multi.columns.tolist(), tablefmt="fancy_grid"))
    logging.info("Saved multi-system benchmark to %s", multi_path)
    # --- STEP 4: VISUALIZE ---
    try:
        import plotly.graph_objects as go
        # Interactive bar chart for N-1 results
        fig = go.Figure([go.Bar(x=[str(r['Contingency'][0]) for r in n1_results[:10]],
                                y=[r['Score'] for r in n1_results[:10]],
                                marker_color='crimson')])
        fig.update_layout(title="Top 10 N-1 Contingency Severity Scores",
                          xaxis_title="Line Index",
                          yaxis_title="Severity Score")
        # fig.show()  # disabled for non-interactive run
        html_path = os.path.join(output_dir, "top10_n1_contingencies.html")
        fig.write_html(html_path)
        logging.info("Saved Plotly HTML to %s", html_path)
    except Exception as e:
        print(f"Plotly visualization failed: {e}")
        logging.warning("Plotly visualization failed: %s", e)
    # Visualize the power grid with pruned (failed) lines
    grid_path = os.path.join(output_dir, "powergrid_pruned.png")
    analyzer.visualize_contingency(best_lines, save_path=grid_path, show=False)
    logging.info("Saved power grid image to %s", grid_path)
    logging.info("Run complete.")

if __name__ == "__main__":
    run()
