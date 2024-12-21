import random
import time
import hashlib
import math
import pygame
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay # For Voronoi
import threading
import queue
import concurrent.futures

# --- Core Philosophical Concepts as Code ---

class Idea:
    """Represents a philosophical concept with enriched attributes, visual dynamics, and deeper 'meaning'."""
    def __init__(self, content, origin, confidence=0.5, unity_score=0.0, history=None, position=None, color=None,
                 size=20, velocity=None, inertia=0.01, complexity=1.0, influence_radius=50,
                 symbol=None, sound=None, visual_pattern=None):
        self.content = content
        self.origin = origin
        self.confidence = confidence
        self.unity_score = unity_score
        self.history = history if history is not None else []
        self.id = hashlib.sha256(content.encode()).hexdigest()
        self.position = position if position is not None else (random.uniform(100, 700), random.uniform(100, 500))
        self.velocity = velocity if velocity is not None else [random.uniform(-1, 1), random.uniform(-1,1)]
        self.size = size
        self.color = color if color is not None else (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.inertia = inertia  # Resistance to changes in velocity
        self.complexity = complexity # How prone to change/mutate
        self.influence_radius = influence_radius # How far it influences others
        self.symbol = symbol if symbol is not None else self.get_random_symbol() # visual symbol for each idea
        self.sound = sound # A sound effect
        self.visual_pattern = visual_pattern if visual_pattern else self.get_random_pattern() # Unique visual patterns

    def __repr__(self):
      return f"Idea(content='{self.content[:20]}...', confidence={self.confidence:.2f}, unity={self.unity_score:.2f}, origin='{self.origin}')"

    def __eq__(self, other):
        if isinstance(other, Idea):
            return self.id == other.id
        return False

    def get_random_symbol(self):
        """Gets a random symbol for visual representation."""
        symbols = ["*", "+", "-", "/", "\\", "^", "v", "<", ">", "#", "@", "%", "&", "$", "!"]
        return random.choice(symbols)

    def get_random_pattern(self):
       """Generates a random visual pattern, could be a gradient or an image."""
       pattern_type = random.choice(["gradient", "circles", "dots"])
       if pattern_type == "gradient":
           return [(random.randint(0,255), random.randint(0,255), random.randint(0,255)),
                  (random.randint(0,255), random.randint(0,255), random.randint(0,255))]
       elif pattern_type == "circles":
            return [random.randint(2,10) for _ in range(random.randint(2,5))] # circle radii
       elif pattern_type == "dots":
            return random.randint(3,10) # number of dots

    def mutate(self, change, source="dialectical"):
       """Evolve an idea with more sophisticated changes based on complexity."""
       mutation_rate = self.complexity * 0.3  # The more complex, the higher the chance of change
       
       # Mutation affects not just content but other properties
       new_content = self.content
       if random.random() < mutation_rate:
            new_content = self.content + change
            if len(new_content) > 200: # Keep content manageable
                new_content = new_content[random.randint(0, 10):]
       
       new_confidence = max(0, min(1, self.confidence + random.uniform(-0.15 * mutation_rate, 0.15 * mutation_rate)))
       new_unity_score = max(0, min(1, self.unity_score + random.uniform(-0.10 * mutation_rate, 0.10 * mutation_rate)))
       new_history = self.history + [self.content]
       new_complexity = max(0.1, min(2.0, self.complexity + random.uniform(-0.15 * mutation_rate, 0.15 * mutation_rate)))
        
       # Color changes significantly with mutations
       color_change = int(70 * mutation_rate)
       new_color = (min(255, max(0, self.color[0] + random.randint(-color_change, color_change))),
                    min(255, max(0, self.color[1] + random.randint(-color_change, color_change))),
                    min(255, max(0, self.color[2] + random.randint(-color_change, color_change))))
       
       # symbol changes based on confidence
       if random.random() < self.confidence * 0.2:
         new_symbol = self.get_random_symbol()
       else:
         new_symbol = self.symbol

       # pattern change with complexity
       if random.random() < self.complexity * 0.1:
          new_pattern = self.get_random_pattern()
       else:
          new_pattern = self.visual_pattern

       return Idea(new_content, source, new_confidence, new_unity_score, new_history, 
                   self.position, new_color, self.size, self.velocity, self.inertia, new_complexity,
                   self.influence_radius, new_symbol, self.sound, new_pattern)

    def challenge(self, other_idea):
        """Challenge an idea using chaotic, complexity-driven interactions, and distance based changes."""
        chaos_factor = random.uniform(-0.3, 0.3)
        
        # Change based on relative confidence and complexity
        confidence_diff = self.confidence - other_idea.confidence
        complexity_diff = self.complexity - other_idea.complexity

        self_change = f"challenged by '{other_idea.content[:10]}', {confidence_diff * chaos_factor:0.2f}, c:{complexity_diff:0.2f}"
        other_change = f"challenged by '{self.content[:10]}', {-confidence_diff * chaos_factor:0.2f}, c:{-complexity_diff:0.2f}"
       
        # Influence velocities based on difference in complexity and unity and distance
        distance = math.hypot(self.position[0] - other_idea.position[0], self.position[1] - other_idea.position[1])
        if distance < self.influence_radius:
          influence_strength = (self.influence_radius - distance) / self.influence_radius
          self.velocity[0] += (other_idea.position[0] - self.position[0]) * 0.02 * influence_strength * other_idea.complexity
          self.velocity[1] += (other_idea.position[1] - self.position[1]) * 0.02 * influence_strength * other_idea.complexity

          other_idea.velocity[0] += (self.position[0] - other_idea.position[0]) * 0.02 * influence_strength * self.complexity
          other_idea.velocity[1] += (self.position[1] - other_idea.position[1]) * 0.02 * influence_strength * self.complexity

        return self.mutate(self_change), other_idea.mutate(other_change)

    def update_position(self, screen_width, screen_height, friction=0.01):
      """Updates position with bounce and friction. Inertia prevents jerky motion."""
      self.velocity[0] *= (1 - friction)
      self.velocity[1] *= (1 - friction)
      self.position = (self.position[0] + self.velocity[0], self.position[1] + self.velocity[1])

      if self.position[0] < 0:
        self.velocity[0] *= -1
        self.position = (0, self.position[1])
      if self.position[0] > screen_width:
        self.velocity[0] *= -1
        self.position = (screen_width, self.position[1])
      if self.position[1] < 0:
        self.velocity[1] *= -1
        self.position = (self.position[0], 0)
      if self.position[1] > screen_height:
        self.velocity[1] *= -1
        self.position = (self.position[0], screen_height)
    
    def draw(self, screen, font):
        """Draws the visual representation of the idea, including symbol, size based on complexity, and unique visual pattern."""
        scaled_size = int(self.size * self.complexity)
        
        # Draw the basic shape
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), scaled_size)
       
        # Draw unique pattern on the circle
        if self.visual_pattern:
            if isinstance(self.visual_pattern, list) and len(self.visual_pattern) == 2 and isinstance(self.visual_pattern[0], tuple):  #Gradient - check that it's also a tuple
                gradient_colors = self.visual_pattern
                for i in range(scaled_size):
                   pos_ratio = i / scaled_size
                   color = (int(gradient_colors[0][0] + (gradient_colors[1][0] - gradient_colors[0][0]) * pos_ratio),
                            int(gradient_colors[0][1] + (gradient_colors[1][1] - gradient_colors[0][1]) * pos_ratio),
                            int(gradient_colors[0][2] + (gradient_colors[1][2] - gradient_colors[0][2]) * pos_ratio))
                   pygame.draw.circle(screen, color, (int(self.position[0]), int(self.position[1])), scaled_size-i, 1)
            elif isinstance(self.visual_pattern, list): # Circles within
                circle_radii = self.visual_pattern
                for radius in circle_radii:
                  pygame.draw.circle(screen, (255,255,255), (int(self.position[0]), int(self.position[1])), int(radius), 1)
            elif isinstance(self.visual_pattern, int): # Dots within
                num_dots = self.visual_pattern
                for _ in range(num_dots):
                    dot_x = self.position[0] + random.uniform(-scaled_size/2, scaled_size/2)
                    dot_y = self.position[1] + random.uniform(-scaled_size/2, scaled_size/2)
                    pygame.draw.circle(screen, (255,255,255), (int(dot_x), int(dot_y)), 2)

        # Render the symbol in the centre
        text_surface = font.render(self.symbol, True, (0, 0, 0)) #Black symbol
        text_rect = text_surface.get_rect(center=(int(self.position[0]), int(self.position[1])))
        screen.blit(text_surface, text_rect)

class Reality:
    """Represents the 'world' with visual, complex convergence and more dynamic interactions."""
    def __init__(self, initial_ideas=None, screen_width=800, screen_height=600):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Philosophical Reality: 1+1=1")
        self.font = pygame.font.Font(None, 20)
        self.ideas = initial_ideas or []
        self.convergence_history = []
        self.frame_count = 0
        self.epoch_count = 0
        self.unity_history = []  # Detailed convergence data
        self.voronoi_diagram = None # For visual representation of influence
        self.show_voronoi = False
        self.voronoi_update_rate = 10
        self.last_voronoi_update = 0
        self.sound_queue = queue.Queue() # Queue for playing sounds (multithreading)

    def add_idea(self, idea):
        if any(i == idea for i in self.ideas):
            return  # Only unique ideas
        self.ideas.append(idea)

    def simulate_dialogue(self, epochs=100, step_time=0.01, interaction_rate=0.5, num_threads = 4):
        """Simulates dialectic with multithreading and chaotic, asynchronous interactions."""
        print("\n--- Beginning Dialectical Simulation ---")
        running = True
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
          for epoch in range(epochs):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                   if event.key == pygame.K_v:
                       self.show_voronoi = not self.show_voronoi # Toggle show_voronoi diagram

            if not running:
              break
            
            # Collect interaction jobs for multithreading
            interaction_jobs = []
            if random.random() < interaction_rate:
                 pairs = random.sample(self.ideas, min(len(self.ideas), 10))
                 for i in range(0, len(pairs), 2): # Pair up adjacent concepts
                     if i + 1 < len(pairs):
                         interaction_jobs.append((pairs[i], pairs[i+1]))
           
            # Asynchronous challenges
            if interaction_jobs:
                future_challenges = {executor.submit(self.challenge_ideas, pair): pair for pair in interaction_jobs}
                for future in concurrent.futures.as_completed(future_challenges):
                   pair = future_challenges[future]
                   try:
                       new_a, new_b = future.result()
                       
                       # Safely update ideas
                       self.ideas = [new_a if x == pair[0] else x for x in self.ideas]
                       self.ideas = [new_b if x == pair[1] else x for x in self.ideas]
                     
                   except Exception as e:
                      print(f"Error processing pair {pair}: {e}")
            
            # Keep the ideas with reasonable confidence
            self.ideas = [x for x in self.ideas if x.confidence > 0.1]
            
            self.update_visuals()
            time.sleep(step_time)
            self.track_convergence()
            self.epoch_count += 1
            
            # Update voronoi diagram periodically
            if self.frame_count - self.last_voronoi_update > self.voronoi_update_rate:
                self.voronoi_diagram = self.calculate_voronoi()
                self.last_voronoi_update = self.frame_count

        print("\n--- Dialectical Simulation Complete ---")

    def challenge_ideas(self, pair):
      """Wrapper for challenging ideas, allows for more efficient multithreading"""
      idea_a, idea_b = pair
      if random.random() > 0.5:
          print(f"Epoch {self.epoch_count+1}: {idea_a} vs. {idea_b}")
          new_a, new_b = idea_a.challenge(idea_b)
      else:
          print(f"Epoch {self.epoch_count+1}: {idea_b} vs. {idea_a}")
          new_b, new_a = idea_b.challenge(idea_a)
      
      return new_a, new_b

    def track_convergence(self):
        """Tracks average unity score, weighted by confidence, representing convergence."""
        if not self.ideas:
            self.convergence_history.append(0)
            return

        total_weighted_unity = sum(idea.unity_score * idea.confidence for idea in self.ideas)
        total_confidence = sum(idea.confidence for idea in self.ideas)
        
        if total_confidence > 0:
          avg_weighted_unity = total_weighted_unity / total_confidence
        else:
          avg_weighted_unity = 0
        self.convergence_history.append(avg_weighted_unity)
        
        # Collect all unity scores for analysis
        unity_scores = [idea.unity_score for idea in self.ideas]
        self.unity_history.append(unity_scores)

    def synthesize(self, synthesis_threshold=0.4):
      """Attempts synthesis based on weighted content, properties, and visual elements."""
      if len(self.ideas) < 2:
        print("Not enough ideas to synthesise")
        return
      
      print("\n--- Attempting Synthesis ---")

      total_unity_confidence = sum(idea.unity_score * idea.confidence for idea in self.ideas)

      if total_unity_confidence < synthesis_threshold:
            print("Synthesis not viable yet")
            return
      
      # Combine content, weight by unity and confidence
      combined_content = ""
      combined_confidence = 0
      combined_complexity = 0

      for idea in self.ideas:
            weight = (idea.unity_score * idea.confidence) / total_unity_confidence if total_unity_confidence > 0 else 0
            combined_content += (idea.content * int(weight * 10)) + " "
            combined_confidence += (idea.confidence * weight)
            combined_complexity += (idea.complexity * weight)

      combined_unity = sum(idea.unity_score for idea in self.ideas) / len(self.ideas)

      # Average position, color, symbol
      avg_x = sum(idea.position[0] for idea in self.ideas) / len(self.ideas)
      avg_y = sum(idea.position[1] for idea in self.ideas) / len(self.ideas)

      avg_r = sum(idea.color[0] for idea in self.ideas) // len(self.ideas)
      avg_g = sum(idea.color[1] for idea in self.ideas) // len(self.ideas)
      avg_b = sum(idea.color[2] for idea in self.ideas) // len(self.ideas)
      avg_color = (avg_r, avg_g, avg_b)

      combined_symbol = self.get_dominant_symbol()
      combined_pattern = self.get_dominant_pattern()

      new_synthesis = Idea(f"Synthesis: {combined_content[:100]}...", 'Synthesis', combined_confidence, combined_unity,
                           position=(avg_x, avg_y), color=avg_color, complexity=combined_complexity, symbol = combined_symbol, visual_pattern = combined_pattern)
      self.ideas = [new_synthesis]
      print(f"Synthesis successful. New Idea: {new_synthesis}")
      print("--- Synthesis Attempt Complete ---")
    
    def get_dominant_symbol(self):
        """Gets the most common symbol amongst the ideas."""
        if not self.ideas:
            return None
        symbol_counts = defaultdict(int)
        for idea in self.ideas:
            symbol_counts[idea.symbol] += 1
        
        if symbol_counts:
          return max(symbol_counts, key=symbol_counts.get)
        return None
    
    def get_dominant_pattern(self):
        """Gets the most common pattern amongst the ideas, weighted by confidence."""
        if not self.ideas:
                return None
        pattern_scores = defaultdict(float)
        for idea in self.ideas:
            if isinstance(idea.visual_pattern, int):
                pattern_scores[(idea.visual_pattern,)] += idea.confidence # convert to tuple if int
            else:
                pattern_scores[tuple(idea.visual_pattern)] += idea.confidence
        if pattern_scores:
            return list(max(pattern_scores, key=pattern_scores.get))
        return None


    def guide_plato(self, confidence_threshold=0.2, unity_threshold=0.2):
        """Purges ideas based on thresholds."""
        print("\n--- Guiding Plato out of the Cave ---")
        initial_count = len(self.ideas)
        self.ideas = [idea for idea in self.ideas if idea.confidence > confidence_threshold or idea.unity_score > unity_threshold]
        final_count = len(self.ideas)
        print(f"Removed {initial_count - final_count} ideas not suitable for light (low unity or low confidence).")
        print("--- Plato Guided ---")

    def calculate_voronoi(self):
      """Calculates the Voronoi diagram to show influence regions."""
      if len(self.ideas) < 3:
        return None

      points = np.array([idea.position for idea in self.ideas])
      try:
        tri = Delaunay(points)
        return tri
      except Exception as e:
          print(f"Error calculating Voronoi: {e}")
          return None

    def draw_voronoi(self, screen):
        """Draws the Voronoi diagram on the screen."""
        if not self.voronoi_diagram:
            return
        try:
           for simplex in self.voronoi_diagram.simplices:
               vertices = self.voronoi_diagram.points[simplex]
               pygame.draw.polygon(screen, (50,50,50), vertices, width = 1)
        except Exception as e:
             print(f"Error drawing Voronoi {e}")

    def play_sound(self, sound):
        """Plays a sound using pygame mixer in a separate thread."""
        try:
            # Check if sound is not None
            if sound:
              pygame.mixer.Sound(sound).play()
        except Exception as e:
             print(f"Sound error: {e}")

    def update_visuals(self):
        """Updates the visuals on the screen."""
        self.screen.fill((0, 0, 0)) # Clear screen

        # Draw Voronoi Diagram first
        if self.show_voronoi:
            self.draw_voronoi(self.screen)

        # Draw all ideas
        for idea in self.ideas:
          idea.update_position(self.screen_width, self.screen_height)
          idea.draw(self.screen, self.font)
          if idea.sound:
              self.sound_queue.put(idea.sound) # Add sound effect

        # Handle playing of sounds
        while not self.sound_queue.empty():
            sound = self.sound_queue.get()
            threading.Thread(target=self.play_sound, args=(sound,)).start()
        
        # Display basic information
        text_surface = self.font.render(f"1+1=1: Frame {self.frame_count}, Ideas: {len(self.ideas)}, Epochs: {self.epoch_count}", True, (255, 255, 255))
        self.screen.blit(text_surface, (10, 10))
        
        self.frame_count += 1
        pygame.display.flip()
    
    def show_ideas(self):
        """Displays the current ideas"""
        print("\n--- Current Ideas ---")
        for idea in self.ideas:
            print(idea)
        print("--- End of Ideas ---")

    def plot_convergence(self):
        """Plots convergence data with line and distribution visualizations."""
        if not self.convergence_history or not self.unity_history:
           print("No convergence data to display.")
           return

        print("\n--- Convergence Plotting ---")

        # Line graph of average convergence
        plt.figure(figsize=(12, 6))
        plt.plot(self.convergence_history, label="Average Weighted Unity")
        plt.xlabel("Epoch")
        plt.ylabel("Unity Score")
        plt.title("Convergence Over Time (Weighted Unity)")
        plt.legend()
        plt.grid(True)
        plt.savefig("convergence_line.png")
        print("Line plot written to convergence_line.png")

        # Distribution of unity scores at end
        plt.figure(figsize=(12,6))
        final_unity_scores = self.unity_history[-1]
        plt.hist(final_unity_scores, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel("Unity Scores")
        plt.ylabel("Frequency")
        plt.title("Distribution of Unity Scores (Final Epoch)")
        plt.grid(axis="y", alpha=0.75)
        plt.savefig("convergence_distribution.png")
        print("Distribution plot written to convergence_distribution.png")

        print("--- End of Convergence Plotting ---")

class Metagamer:
    """Nouri Mabrouk, the metagamer, orchestrating transcendent reality."""
    def __init__(self, reality):
        self.reality = reality
        self.unity_idea_spawned = False
        self.rounds = 0
        pygame.mixer.init() # Init for sounds

    def add_initial_ideas(self):
       initial_ideas = [
            Idea("I think therefore I am", "Descartes", 0.8, 0.4, complexity=1.2, influence_radius=60, symbol = "D", sound = None),
            Idea("The universe is governed by laws", "Newton", 0.7, 0.2, complexity=0.9, influence_radius=50, symbol = "N", sound = None),
            Idea("Everything is interconnected", "Buddha", 0.6, 0.8, complexity=1.0, influence_radius=70, symbol = "B", sound = None),
            Idea("There is no self", "Nagarjuna", 0.5, 0.9, complexity=1.4, influence_radius=80, symbol = "N", sound = None),
            Idea("Reality is a dream", "Chuang Tzu", 0.4, 0.7, complexity=1.1, influence_radius=50, symbol = "C", sound = None),
            Idea("All concepts are illusions", "Nietzsche", 0.9, 0.1, complexity=1.3, influence_radius=75, symbol = "!", sound = None),
            Idea("1+1=2, basic math", "Math", 0.9, 0.0, complexity=0.8, influence_radius=40, symbol = "=", sound = None),
            Idea("The universe is a unified whole", "Spinoza", 0.8, 0.9, complexity=1.0, influence_radius=60, symbol = "S", sound = None),
            Idea("Life is a miracle", "Watts", 0.9, 0.7, complexity=1.1, influence_radius=70, symbol = "W", sound = None),
            Idea("I am part of the world, not separate", "Daoism", 0.7, 0.8, complexity=1.2, influence_radius=80, symbol = "T", sound = None)
         ]
       for idea in initial_ideas:
            self.reality.add_idea(idea)
    
    def metagame(self, rounds=3, epochs_per_round=40, interaction_rate=0.5, num_threads=4):
        """Orchestrates main flow with more dynamic elements and convergence."""
        print("\n--- Beginning the Transcendent Philosophical Big Bang ---")
        print("  1+1=1 Initialised: The dance of concepts.")

        for round in range(rounds):
            self.rounds += 1
            print(f"\n--- Metagaming Round {round + 1} ---")
            self.reality.simulate_dialogue(epochs=epochs_per_round, interaction_rate=interaction_rate, num_threads = num_threads)
            self.reality.synthesize()
            self.reality.guide_plato()
            self.reality.show_ideas()
            self.reality.plot_convergence()

             # Introduce the unifying idea dynamically
            if self.reality.convergence_history and self.reality.convergence_history[-1] > 0.75 and not self.unity_idea_spawned:
                center_x = self.reality.screen_width / 2
                center_y = self.reality.screen_height / 2

                unity_idea = Idea("1+1=1, unifying consciousness", "Nouri", 0.99, 1.0, position=(center_x, center_y), 
                                  color=(255, 255, 255), complexity = 1.8, influence_radius=180, symbol = "U", sound=None)
                self.reality.add_idea(unity_idea)
                self.unity_idea_spawned = True
                print("\n--- Spawning the unifying concept of 1+1=1 ---")
            
             # Increase interaction rate
            interaction_rate = min(1, interaction_rate + (0.15 * self.rounds))

        # Final stage
        print("\n--- The universe coalesces into the One ---")
        print("  1+1=1: The ultimate unity, both beginning and end.   ")
        
        self.reality.simulate_dialogue(epochs=epochs_per_round, interaction_rate=interaction_rate, num_threads= num_threads)
        self.reality.update_visuals()
        print("\n--- End of Transcendent Metagaming Simulation ---")
        pygame.quit()

# --- Execution ---
if __name__ == "__main__":
    print("1+1=1: The Unified Reality")
    reality = Reality()
    nouri = Metagamer(reality)
    nouri.add_initial_ideas()
    nouri.metagame(rounds=3, epochs_per_round=40, interaction_rate=0.3, num_threads=4)
    
    print("\n1+1=1")