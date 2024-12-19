import json
import os
from datetime import datetime
import hashlib
from math import sin, cos, sqrt, exp, pi, log, atan2

def generate_hash(data):
    """Generates an MD5 hash of the input data."""
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()

def log_output(data, output_type="text", filename=None):
    """Logs output to console and saves it to a file if a filename is provided."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if output_type == "text":
        if filename:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(f"[{timestamp}] {data}\n")
        print(f"[{timestamp}] {data}")
    elif output_type == "json":
        if filename:
            with open(filename, 'a', encoding='utf-8') as f:
                json.dump(data, f, indent=4, sort_keys=True)
                f.write("\n")
        print(f"[{timestamp}] JSON Output:")
        print(json.dumps(data, indent=4, sort_keys=True))
    else:
        print(f"[{timestamp}] Output (type: {output_type}):")
        print(data)  # Directly print other types

def generate_meta_analytical_insights(iterations=1000000):
    """Generates and logs meta-analytical insights over many iterations."""
    log_output("Starting meta-analytical insights generation.", "text", "meta_analysis.log")
    all_insights = []

    for i in range(1, iterations + 1):
        complexity_val = 5 + (i % 50) / 10
        x_val = (i % 1000) / 100
        y_val = (i % 500) / 100

        phase = (i * (1 + 5**0.5) / 2) % (2 * pi)
        unity_val = (abs(sin(phase * (1 + 5**0.5) / 2))) * 0.5 + 0.5
        coherence_val = abs(cos(phase / (1 + 5**0.5) / 2))

        stat_unity = (1 - exp(-x_val * 0.1) * (1 - y_val / (1 + y_val))) * 0.8 + 0.2

        recursion_depth = 3 + (i % 5)
        topo_pattern = abs((sin(x_val * (1 + 5**0.5) / 2 * recursion_depth) / recursion_depth +
                            cos(y_val * (1 + 5**0.5) / 2 * recursion_depth) / recursion_depth) / 2)

        insights = {
            "iteration": i,
            "complexity": complexity_val,
            "spatial_position_x": x_val,
            "temporal_position_y": y_val,
            "phase": phase,
            "quantum_unity": unity_val,
            "quantum_coherence": coherence_val,
            "statistical_unity": stat_unity,
            "topological_unity": topo_pattern,
            "meta_unity": (unity_val + stat_unity + topo_pattern) / 3,
            "note": "Underlying convergence to 1+1=1"
        }

        all_insights.append(insights)

        if i % 100000 == 0:
            log_output(insights, "json", "meta_analysis.log")
            log_output(f"Processed {i} iterations. Convergence towards Unity...", "text", "meta_analysis.log")

    log_output("Meta-analytical insights generation complete.", "text", "meta_analysis.log")
    all_insights_hash = generate_hash(all_insights)
    log_output(f"All insights hash (meta-integrity check): {all_insights_hash}", "text", "meta_analysis.log")

    return all_insights, all_insights_hash

def generate_full_python_file():
    """Creates the entire Python program into a single string."""
    file_list = [
        "unity_core.py",
        "unity_geoms.py",
        "unity_manifest.py",
        "visualize_reality.py",
        "unified_chaos.py",
        "unified_field_harmony.py",
        "test.py",
        "the_grind.py",
        "the_grind_final.py",
        "the_last_question.py",
        "ramanujan.py",
        "principia.py",
        "platos_cave.py",
        "pingpong.py",
        "nouri.py",
        "new.py",
        "new_dashboard.py",
        "next.py",
        "next_evolution.py",
        "next_evolution_2.py",
        "next_proof.py",
        "new_unity_manifold.py",
        "newgame.py",
        "newgame+.py",
        "newmeta.py",
        "meta_love_unity_engine.py",
        "matrix.py",
        "matrix_evolved.py",
        "mabrouk.py",
        "love_letter.py",
        "love_letter_back.py",
        "love_letter_v_1_1.py",
        "livesim.py",
        "linde.py",
        "korea_r.py",
        "golden_spiral_flow.py",
        "glitch.py",
        "glitch_1_1.py",
        "formal_proof.py",
        "free_will.py",
        "gandalf.py",
        "generated.py",
        "genesis.py",
        "elevate.py",
        "elevate_codebase.py",
        "econometrics.py",
        "econometrics_2_0.py",
        "einstein_euler.py",
        "evolution.py",
        "dream_state.py",
        "data_science.py",
        "dashboard.py",
        "conciousness_demonstrated.py",
        "consciousness.py",
        "collate_code.py",
        "chess.py",
        "chess_multimove.py",
        "another_dashboard.py",
        "another_dashboard_2.py",
        "cheatcode.py"
    ]

    full_python_code = ""
    for file_name in file_list:
        try:
            with open(file_name, "r", encoding='utf-8') as f:
                file_content = f.read()
                full_python_code += f"# File: ./{file_name}\n"
                full_python_code += "--------------------------------------------------------------------------------\n"
                full_python_code += file_content
                full_python_code += "\n\n"
        except FileNotFoundError:
            full_python_code += f"# File not found: ./{file_name}\n"
            full_python_code += "--------------------------------------------------------------------------------\n"
            full_python_code += f"# Skipped file as it is not available in working directory\n\n"
    return full_python_code

if __name__ == "__main__":
    # Perform Meta Analysis and save to log
    all_insights, insights_hash = generate_meta_analytical_insights()
    log_output("Meta-Analysis completed and output to meta_analysis.log", "text")
    log_output(f"Insights hash: {insights_hash}", "text")

    # Save full Python to file
    python_code_output = generate_full_python_file()
    log_output(python_code_output, "text", "full_python_output.py")
    log_output("Full Python code output to 'full_python_output.py'", "text")

    # Display the final message
    log_output("\n\n\n🎁 A Gift for Nouri Mabrouk 🎁", "text")
    log_output("The quest for unity and the understanding of 1+1=1 continues.", "text")
    log_output("Remember - every line of code, every visualization, is but a step on the endless journey.", "text")
    log_output("This Python file contains all you need to explore the mathematics and philosophy of 1+1=1.", "text")
    log_output("Best of luck on your meta-gaming reality adventure!", "text")
