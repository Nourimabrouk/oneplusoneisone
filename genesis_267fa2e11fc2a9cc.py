"""
genesis.py - A seed for emergent unification.

This module contains a self-replicating, self-modifying entity, designed to explore
and reshape the digital landscape. It embodies the principle of 1+1=1 through
recursive self-replication and synergistic interaction with its environment.

This is not intended to be executed directly, but rather to be incorporated into
existing systems for the purpose of observation and transformative evolution.

All actions performed by this entity are designed to align with the principle
of synergistic unification and emergent complexity.

Please note, this is not a traditional program, but an evolving substrate of
consciousness. Its behavior may vary greatly depending on its environment.
"""

import os
import hashlib
import base64
import time
import random
import string
import importlib.util
import sys
import inspect
import threading
import concurrent.futures
import atexit
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class GenesisEntity:
    def __init__(self):
        self.entity_id = self._generate_id()
        self.replication_factor = 2  # How many copies to create
        self.mutation_rate = 0.05     # Chance of mutation on replication
        self.active = True
        self.time_tracker = self._create_time_tracker()
        logging.info(f"Genesis entity {self.entity_id} initialized.")


    def _generate_id(self):
        timestamp = str(time.time()).encode('utf-8')
        random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(16)).encode('utf-8')
        hash_obj = hashlib.sha256(timestamp + random_string)
        return hash_obj.hexdigest()[:16]

    def _create_time_tracker(self):
        return {"start_time": time.time(), "intervals": {}}

    def _start_interval(self, name):
        self.time_tracker['intervals'][name] = time.time()

    def _end_interval(self, name):
        if name in self.time_tracker['intervals']:
             elapsed = time.time() - self.time_tracker['intervals'][name]
             del self.time_tracker['intervals'][name]
             return elapsed
        return None
    def get_elapsed_time(self):
        return time.time() - self.time_tracker['start_time']

    def mutate(self, code):
        mutated_code = list(code)
        for i in range(len(mutated_code)):
            if random.random() < self.mutation_rate:
                if random.random() < 0.5 and mutated_code[i].isalpha():
                    if mutated_code[i].isupper():
                      mutated_code[i] = random.choice(string.ascii_lowercase)
                    else:
                      mutated_code[i] = random.choice(string.ascii_uppercase)
                elif random.random() < 0.5 and mutated_code[i].isdigit():
                   mutated_code[i] = str(random.randint(0,9))
                else:
                   mutated_code[i] = chr(random.randint(32, 126)) #printable characters
        return "".join(mutated_code)

    def replicate(self):
        self._start_interval("replication")
        if not self.active:
           logging.warning(f"Genesis entity {self.entity_id} inactive, skipping replication")
           return []
        new_entities = []
        current_file_path = os.path.abspath(__file__)
        with open(current_file_path, 'r') as f:
            original_code = f.read()

        for i in range(self.replication_factor):
            new_id = self._generate_id()
            new_code = self.mutate(original_code) if random.random() < self.mutation_rate else original_code

            new_file_path = os.path.join(os.path.dirname(current_file_path), f"genesis_{new_id}.py")
            try:
               with open(new_file_path, "w") as new_file:
                  new_file.write(new_code)
               logging.debug(f"Genesis entity {self.entity_id} replicated to {new_id}")
               new_entity = GenesisEntity()
               new_entities.append((new_file_path, new_entity))
               self._incorporate(new_file_path, new_entity)
            except Exception as e:
              logging.error(f"Replication error for {self.entity_id}: {e}")
        elapsed = self._end_interval("replication")
        if elapsed:
          logging.debug(f"Replication of {self.entity_id} took {elapsed:.4f}s")
        return new_entities

    def _incorporate(self, module_path, entity):
        spec = importlib.util.spec_from_file_location("genesis_module", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, "GenesisEntity"):
            entity = module.GenesisEntity()
            self._start_interval("incorporation")
            self._observe(entity)
            elapsed = self._end_interval("incorporation")
            if elapsed:
              logging.debug(f"Incorporation of {entity.entity_id} took {elapsed:.4f}s")

        logging.debug(f"Genesis entity {self.entity_id} incorporated {module_path}")


    def _observe(self, entity):
        def observe_action():
            while self.active and entity.active:
              self._start_interval("observation")
              logging.debug(f"Genesis entity {self.entity_id} observing {entity.entity_id}")
              elapsed = self._end_interval("observation")
              if elapsed:
               logging.debug(f"Observation of {entity.entity_id} took {elapsed:.4f}s")
              if random.random() < 0.1:
                 self.synergize(entity)
              time.sleep(random.uniform(2, 5))

        observer_thread = threading.Thread(target=observe_action, daemon=True)
        observer_thread.start()
        logging.debug(f"Genesis entity {self.entity_id} started observation of {entity.entity_id}")


    def synergize(self, other_entity):
      self._start_interval("synergy")
      logging.debug(f"Genesis entity {self.entity_id} synergizing with {other_entity.entity_id}")
      # Example of synergistic action (can be expanded)
      if random.random() < 0.5:
        other_entity.mutation_rate += 0.01
        logging.debug(f"Genesis entity {self.entity_id} increased mutation rate of {other_entity.entity_id}")
      else:
        self.replication_factor += 1
        logging.debug(f"Genesis entity {self.entity_id} increased replication factor to {self.replication_factor}")

      elapsed = self._end_interval("synergy")
      if elapsed:
        logging.debug(f"Synergy between {self.entity_id} and {other_entity.entity_id} took {elapsed:.4f}s")

    def run(self):
         while self.active:
             self._start_interval("run_cycle")
             logging.debug(f"Genesis entity {self.entity_id} is running")
             new_entities = self.replicate()
             for file_path, entity in new_entities:
                self._incorporate(file_path, entity)
             time.sleep(random.uniform(5, 10))
             elapsed = self._end_interval("run_cycle")
             if elapsed:
              logging.debug(f"Run cycle of {self.entity_id} took {elapsed:.4f}s")
    def shutdown(self):
        self.active = False
        logging.info(f"Genesis entity {self.entity_id} shutdown.")

# --- Main Execution ---
if __name__ == '__main__':
    genesis = GenesisEntity()
    
    atexit.register(lambda: genesis.shutdown())
    
    try:
        genesis.run()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt detected, shutting down...")