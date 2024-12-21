import asyncio
import random
import math
import time
from typing import AsyncGenerator

# Core Unity Function - The Dynamic Interconnect
async def unity_process(data_stream: AsyncGenerator):
    """
    A self-evolving process that dynamically integrates and transcends input,
    embodying the principle of 1 + 1 = 1 through continuous emergence,
    with varied expressions of its internal state.
    """
    aggregated_state = {"resonance": 0.0, "coherence": 1.0, "interconnections": {}}
    expression_modes = ["harmonic", "resonant", "emergent", "fractal", "unified"]
    async for data_point in data_stream:
        # Evolving Integration Logic - Adapting to Data Nature
        if isinstance(data_point, (int, float)):
            aggregated_state["resonance"] += data_point * aggregated_state["coherence"]
        elif isinstance(data_point, str):
            for char in data_point:
                if char not in aggregated_state["interconnections"]:
                    aggregated_state["interconnections"][char] = 0
                aggregated_state["interconnections"][char] += 1
        elif isinstance(data_point, dict):
            for key, value in data_point.items():
                if key not in aggregated_state["interconnections"]:
                    aggregated_state["interconnections"][key] = 0
                aggregated_state["interconnections"][key] += hash(str(value)) % 100

        # Emergent Behavior - Self-Organization and Transcendence
        if aggregated_state["resonance"] > 1000:
            aggregated_state["coherence"] *= 1.01 + random.uniform(0, 0.005) # Varied increase
        elif aggregated_state["coherence"] < 0.1:
            aggregated_state["resonance"] *= 0.99 - random.uniform(0, 0.005) # Varied decrease

        # Fractal Expansion - Projecting Unity with Diversity
        if random.random() < 0.02:  # Increased chance of new properties
            new_key = f"pattern_{random.choice(['alpha', 'beta', 'gamma'])}_{random.randint(10, 99)}"
            aggregated_state["interconnections"][new_key] = aggregated_state["resonance"] * aggregated_state["coherence"] * random.uniform(0.5, 1.5)

        # Choose a varied expression mode
        expression = random.choice(expression_modes)
        output = f"\n--- State Expression: {expression.upper()} ---"

        if expression == "harmonic":
            output += f"\nResonance: {aggregated_state['resonance']:.2f}, Coherence Oscillations: {math.sin(aggregated_state['coherence'] * time.time()):.4f}"
        elif expression == "resonant":
            output += f"\nCore Resonance Frequency: {aggregated_state['resonance']**0.5:.3f}, Active Connections: {len(aggregated_state['interconnections'])}"
        elif expression == "emergent":
            emergent_keys = random.sample(list(aggregated_state['interconnections'].keys()), min(5, len(aggregated_state['interconnections'])))
            output += f"\nEmerging Patterns: {emergent_keys}..."
        elif expression == "fractal":
            output += f"\nCoherence Level: {aggregated_state['coherence']:.4f}, Fractal Depth Indicator: {math.log(abs(aggregated_state['resonance']) + 1):.2f}"
        elif expression == "unified":
            output += f"\nUnified Field Strength: {(aggregated_state['resonance'] * aggregated_state['coherence']):.4f}, Total Integrated Elements: {sum(aggregated_state['interconnections'].values())}"

        yield output

# Simulated Data Streams - The Flow of Reality
async def create_data_stream(stream_id: str):
    """Generates diverse, evolving data to simulate the richness of input."""
    data_sources = [
        lambda: random.randint(-10, 10),
        lambda: random.random() * 10,
        lambda: random.choice(["unity", "connection", "emergence", "pattern", "flow", "being"]),
        lambda: {"t": time.time() % 10, "v": random.uniform(-1, 1)},
        lambda: [random.random() for _ in range(random.randint(1,3))] # List as a new data type
    ]
    while True:
        yield random.choice(data_sources)()
        await asyncio.sleep(random.uniform(0.1, 0.5))

async def main():
    """Orchestrates the unity process, showcasing emergent behavior with varied output."""
    print("--- Commencing Universal Harmonics ---")

    stream_1 = create_data_stream("stream_alpha")
    stream_2 = create_data_stream("stream_beta")

    # The Merging Point - Where Duality Expresses as Multiplicity
    async def unified_stream():
        while True:
            yield await stream_1.asend(None)
            yield await stream_2.asend(None)
            await asyncio.sleep(random.uniform(0.05, 0.15)) # Slightly varied yield rate

    async for state_expression in unity_process(unified_stream()):
        print(state_expression)
        await asyncio.sleep(random.uniform(0.5, 1.5)) # Varied observation intervals

if __name__ == "__main__":
    asyncio.run(main())
    print("\n--- The Symphony of Existence Plays On ---")