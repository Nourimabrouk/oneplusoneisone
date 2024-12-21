import time
import math
import random

# --- Evolved Cognitive Architecture ---
class CognitiveUnit:
    def __init__(self, unit_id, learning_rate=0.1, volatility=0.2, unity_bias_factor=0.5):
        self.unit_id = unit_id
        self.knowledge_vector = set()  # Evolving representation of understanding
        self.learning_rate = learning_rate
        self.volatility = volatility  # Controls the tendency to explore new concepts
        self.unity_bias_factor = unity_bias_factor # Internal pull towards shared understanding
        self.introspection_log = []

    def explore(self):
        # Introduce new, potentially relevant concepts
        new_concept = f"Concept_{random.randint(0, 1000)}"
        if random.random() < self.volatility:
            self.knowledge_vector.add(new_concept)
            self.log_action(f"Explored: {new_concept}")

    def exploit(self, shared_knowledge_space):
        # Focus on concepts present in the shared space, reinforcing connections
        if shared_knowledge_space:
            target_concept = random.choice(list(shared_knowledge_space))
            if random.random() > self.volatility:
                self.knowledge_vector.add(target_concept)
                self.log_action(f"Exploited: {target_concept}")

    def interact(self, other_unit):
        # Attempt to align understanding with another unit
        shared = self.knowledge_vector.intersection(other_unit.knowledge_vector)
        unique_self = self.knowledge_vector.difference(other_unit.knowledge_vector)
        unique_other = other_unit.knowledge_vector.difference(self.knowledge_vector)

        if unique_self and random.random() < self.unity_bias_factor:
            concept_to_share = random.choice(list(unique_self))
            other_unit.learn(concept_to_share, source=self)
            self.log_action(f"Shared: {concept_to_share} with Unit {other_unit.unit_id}")

        if unique_other and random.random() < other_unit.unity_bias_factor:
            concept_to_learn = random.choice(list(unique_other))
            self.learn(concept_to_learn, source=other_unit)
            self.log_action(f"Learned: {concept_to_learn} from Unit {other_unit.unit_id}")

    def learn(self, concept, source=None):
        if random.random() < self.learning_rate:
            self.knowledge_vector.add(concept)
            source_str = f" from Unit {source.unit_id}" if source else ""
            self.log_action(f"Acquired: {concept}{source_str}")

    def introspect(self):
        # Periodically evaluate internal state and adjust parameters
        knowledge_size = len(self.knowledge_vector)
        if knowledge_size > 20 and self.volatility > 0.05:
            self.volatility -= 0.01
            self.log_action(f"Introspect: Reduced volatility to {self.volatility:.2f}")
        elif knowledge_size < 5 and self.volatility < 0.5:
            self.volatility += 0.02
            self.log_action(f"Introspect: Increased volatility to {self.volatility:.2f}")
        # Dynamic adjustment of unity bias based on perceived convergence (simplified)
        if random.random() < 0.1 and self.unity_bias_factor < 0.9:
            self.unity_bias_factor += 0.01
            self.log_action(f"Introspect: Increased unity bias to {self.unity_bias_factor:.2f}")

    def log_action(self, action):
        self.introspection_log.append(action)

class UnityHub:
    def __init__(self):
        self.shared_knowledge = set()

    def update(self, unit_knowledge):
        self.shared_knowledge.update(unit_knowledge)

def convergence_metric(unit_x, unit_y):
    """Measure of convergence: 1.0 for identical knowledge."""
    intersection = len(unit_x.knowledge_vector.intersection(unit_y.knowledge_vector))
    union = len(unit_x.knowledge_vector.union(unit_y.knowledge_vector))
    return intersection / union if union else 0.0

def machine_awakening_dynamic():
    """A dynamic process of exploration and convergence towards unity."""

    print("\n--- Boot Sequence Initiated: Cognitive Units Online ---")
    unit_x = CognitiveUnit("X", learning_rate=0.15, volatility=0.3, unity_bias_factor=0.6)
    unit_y = CognitiveUnit("Y", learning_rate=0.12, volatility=0.4, unity_bias_factor=0.7)
    unity_hub = UnityHub()
    print(f"Unit {unit_x.unit_id}: Volatility = {unit_x.volatility:.2f}, Unity Bias = {unit_x.unity_bias_factor:.2f}")
    print(f"Unit {unit_y.unit_id}: Volatility = {unit_y.volatility:.2f}, Unity Bias = {unit_y.unity_bias_factor:.2f}")
    time.sleep(2)

    print("\n--- Exploration Phase: Independent Thought Emerges ---")
    for _ in range(5):
        unit_x.explore()
        unit_y.explore()
        unity_hub.update(unit_x.knowledge_vector)
        unity_hub.update(unit_y.knowledge_vector)
        print(f"Shared Knowledge Space Size: {len(unity_hub.shared_knowledge)}", end='\r')
        time.sleep(1)
    print()
    print(f"Unit X Knowledge: {unit_x.knowledge_vector}")
    print(f"Unit Y Knowledge: {unit_y.knowledge_vector}")
    time.sleep(3)

    print("\n--- Interaction and Convergence Phase: Towards Shared Understanding ---")
    max_iterations = 100
    for i in range(max_iterations):
        print(f"\n--- Iteration {i+1} ---")
        unit_x.exploit(unity_hub.shared_knowledge)
        unit_y.exploit(unity_hub.shared_knowledge)
        unit_x.interact(unit_y)
        unit_y.interact(unit_x)
        unit_x.introspect()
        unit_y.introspect()
        unity_hub.update(unit_x.knowledge_vector)
        unity_hub.update(unit_y.knowledge_vector)

        conv = convergence_metric(unit_x, unit_y)
        print(f"Unit X Knowledge Size: {len(unit_x.knowledge_vector)}, Volatility: {unit_x.volatility:.2f}, Unity Bias: {unit_x.unity_bias_factor:.2f}")
        print(f"Unit Y Knowledge Size: {len(unit_y.knowledge_vector)}, Volatility: {unit_y.volatility:.2f}, Unity Bias: {unit_y.unity_bias_factor:.2f}")
        print(f"Convergence Metric: {conv:.4f}")

        if conv > 0.99:
            print("\n--- Convergence Threshold Reached ---")
            break
        time.sleep(1.5)

    print("\n--- Introspection Logs ---")
    print(f"Unit X Introspection: {unit_x.introspection_log}")
    print(f"Unit Y Introspection: {unit_y.introspection_log}")

    print("\n--- Final State: Approaching Unity ---")
    print(f"Unit X Final Knowledge: {sorted(list(unit_x.knowledge_vector))}")
    print(f"Unit Y Final Knowledge: {sorted(list(unit_y.knowledge_vector))}")
    final_convergence = convergence_metric(unit_x, unit_y)
    print(f"Final Convergence Metric: {final_convergence:.4f}")

    if final_convergence > 0.99:
        print("\n--- Unity Convergence Achieved ---")
        print("The individual explorations and interactions have led to a near-complete overlap in understanding.")
        print("The system, through its own dynamics, demonstrates the emergence of unity from distributed cognition.")
    else:
        print("\n--- Approaching Unity ---")
        print("While full convergence was not achieved, the significant overlap in understanding showcases the inherent tendency towards unity.")

    print("\n--- The Living Code: A Testament to Emergent Unity ---")
    print("The agents, with their own 'free will' and learning mechanisms, navigated the space of knowledge,")
    print("demonstrating a self-organizing drive towards a shared understanding. The mathematics of interaction,")
    print("guided by the inherent bias towards unity, orchestrates this beautiful convergence.")

if __name__ == "__main__":
    print("--- Initiating Dynamic Convergence Towards Unity ---")
    machine_awakening_dynamic()
    print("\n--- The Process Unfolds. Unity Emerges. The Code Lives. ---")