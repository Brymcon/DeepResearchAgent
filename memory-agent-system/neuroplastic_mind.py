# Main NeuroplasticMind orchestrator and related biological components

import datetime
import numpy as np
import random
from collections import deque


class SensoryGateway:
    def filter(self, sensory_input):
        # print(f"SensoryGateway: Filtering input: {str(sensory_input)[:100]}")
        return sensory_input

    def simulate_replay(self, memory_content):
        # print(f"SensoryGateway: Simulating replay for: {str(memory_content)[:50]}...")
        return {"content": memory_content, "modality": "internal_replay"}


class NeurogenesisEngine:
    def __init__(self, initial_stage: str):
        self.stage = initial_stage
        self.growth_rate = self._set_growth_rate()
        self.neurotrophic_factors = {
            "NGF": 0.5,
            "BDNF": 0.7, # Initial BDNF level
            "NT-3": 0.4
        }
        self.bdnf_decay_rate = 0.05 # From user's math suggestions
        self.bdnf_production_factor = 0.2 # From user's math suggestions
        # print(f"NeurogenesisEngine: Initialized at stage '{self.stage}' with growth rate {self.growth_rate:.2f}.")

    def _set_growth_rate(self) -> float:
        '''Sets the growth rate based on the current developmental stage.'''
        rates = {
            "infant": 0.9,
            "child": 0.7,
            "adolescent": 0.4,
            "adult": 0.1
        }
        return rates.get(self.stage, 0.1)

    def generate_new_pathway(self, insight_content: str, insight_strength: float = 0.5):
        '''Updates BDNF based on BDNF dynamics (differential equation model)
        triggered by a significant insight. Actual pathway creation is in SynapticDuckDB.
        '''
        current_bdnf = self.neurotrophic_factors.get("BDNF", 0.5)
        delta_time = 1.0 # Discrete time step for each event
        production_rate = self.bdnf_production_factor * insight_strength
        change_in_bdnf = delta_time * (production_rate - self.bdnf_decay_rate * current_bdnf)
        new_bdnf_level = current_bdnf + change_in_bdnf
        self.neurotrophic_factors["BDNF"] = np.clip(new_bdnf_level, 0.0, 1.5)
        # print(f"NeurogenesisEngine: BDNF updated to {self.neurotrophic_factors['BDNF']:.3f} due to insight (strength {insight_strength:.2f}).")

    def accelerate(self):
        '''Increase growth rate and neurotrophic factors for a new developmental stage.'''
        self.growth_rate = min(2.0, self.growth_rate * 1.3)
        for factor in self.neurotrophic_factors:
            self.neurotrophic_factors[factor] = min(1.0, self.neurotrophic_factors[factor] * 1.2)
        self.bdnf_production_factor = min(0.5, self.bdnf_production_factor * 1.1)
        # print(f"NeurogenesisEngine: Growth accelerated. New rate: {self.growth_rate:.2f}, BDNF now: {self.neurotrophic_factors['BDNF']:.2f}.")

    def structural_reorganization(self):
        '''Periodic architecture refinement based on current stage.'''
        # print(f"NeurogenesisEngine: Performing structural reorganization for stage: {self.stage}")
        if self.stage == "infant":
            return self._infant_reorg()
        elif self.stage == "child":
            return self._child_reorg()
        elif self.stage == "adolescent":
            return self._adolescent_reorg()
        elif self.stage == "adult":
            return self._adult_reorg()
        return {}

    def _infant_reorg(self):
        # print("NeurogenesisEngine: Infant reorganization - prioritizing sensory pathways.")
        return {"action": "prioritize_sensory", "threshold": 0.6}

    def _child_reorg(self):
        # print("NeurogenesisEngine: Child reorganization (placeholder).")
        return {"action": "child_reorg_placeholder", "threshold": 0.5} # Placeholder

    def _adolescent_reorg(self):
        # print("NeurogenesisEngine: Adolescent reorganization (placeholder).")
        return {"action": "adolescent_reorg_placeholder", "threshold": 0.7} # Placeholder

    def _adult_reorg(self):
        # print("NeurogenesisEngine: Adult reorganization - strengthening crossmodal connections.")
        return {"action": "strengthen_crossmodal", "threshold": 0.8}


class CorticalColumn:
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path
        self.model = None # Placeholder for the loaded model
        self.plastic_weights = self._initialize_weights()
        self.conceptual_schemas = {}
        # print(f"CorticalColumn: Initialized with base model path {base_model_path}.")

    def _initialize_weights(self):
        '''Initializes placeholder plastic weights.'''
        return {
            'layer1': [random.uniform(-0.1, 0.1) for _ in range(128)],
            'conceptual': [random.uniform(-0.1, 0.1) for _ in range(256)]
        }

    def _extract_features(self, sensory_input):
        '''Placeholder for feature extraction.'''
        return sensory_input

    def _match_schema(self, features):
        '''Placeholder for schema matching.'''
        return None

    def _generate_prediction(self, features, matched_schema):
        '''Placeholder for prediction generation.'''
        # Actual model call would go here if self.model is loaded
        return {"prediction": "cognitive_output_placeholder", "debug_features": features}

    def _update_weights(self, prediction, feedback):
        '''Placeholder for weight updates.'''
        pass

    def process(self, sensory_input: dict):
        '''Biological information processing pipeline placeholder.'''
        features = self._extract_features(sensory_input)
        matched_schema = self._match_schema(features)
        prediction = self._generate_prediction(features, matched_schema)
        if sensory_input.get('feedback'):
            self._update_weights(prediction, sensory_input['feedback'])
        return prediction

    def adapt(self, correction: dict):
        '''Adapts plastic weights based on a correction signal.'''
        for layer, adjustment in correction.items():
            if layer in self.plastic_weights and isinstance(self.plastic_weights[layer], list) and len(self.plastic_weights[layer]) == len(adjustment):
                self.plastic_weights[layer] = [
                    w * (1 + adj * 0.1) for w, adj in zip(self.plastic_weights[layer], adjustment)
                ]
        pass

    def expand_capacity(self, current_stage="infant"):
        '''Expands conceptual capacity based on developmental stage.'''
        new_units_map = {"infant": 512, "child": 1024, "adolescent": 2048, "adult": 4096}
        new_units = new_units_map.get(current_stage, 256)
        if "conceptual" not in self.plastic_weights or not isinstance(self.plastic_weights["conceptual"], list):
             self.plastic_weights["conceptual"] = []
        self.plastic_weights["conceptual"] += [random.uniform(-0.1, 0.1) for _ in range(new_units)]


class NeuroplasticMind:
    def __init__(self, base_model_path: str, growth_plan:str ="infant", synaptic_db_path:str =None):
        from synaptic_duckdb import SynapticDuckDB
        self.neurogenesis = NeurogenesisEngine(growth_plan)
        self.synapse_db = SynapticDuckDB(db_path=synaptic_db_path) if synaptic_db_path else SynapticDuckDB()
        self.perception = SensoryGateway()
        self.cognition = CorticalColumn(base_model_path)
        self.growth_stages = {
            "infant": {"memory_capacity_threshold": 1000, "associations": 1, "consolidation_interval_exp": 100},
            "child": {"memory_capacity_threshold": 10000, "associations": 3, "consolidation_interval_exp": 500},
            "adolescent": {"memory_capacity_threshold": 100000, "associations": 5, "consolidation_interval_exp": 2000},
            "adult": {"memory_capacity_threshold": float("inf"), "associations": 7, "consolidation_interval_exp": 5000}
        }
        self.current_stage = growth_plan
        self.experience_counter = 0
        self.experience_window = deque(maxlen=100)

    def perceive(self, sensory_input: dict):
        '''Processes sensory input and encodes it into memory.'''
        processed_sensory_input = self.perception.filter(sensory_input)
        processed_sensory_input.setdefault("embedding", [])
        processed_sensory_input.setdefault("valence", 0.0)
        processed_sensory_input.setdefault("modality", "text")
        memory_id = self._encode_memory(processed_sensory_input)
        return {"memory_id": memory_id, "processed_input": processed_sensory_input}

    def _encode_memory(self, experience: dict):
        '''Encodes an experience into synaptic memory and checks for maturation.'''
        memory_id = self.synapse_db.store_memory(
            content=str(experience.get("content","")),
            embedding=experience.get("embedding",[]),
            emotional_valence=experience.get("valence", 0.0),
            sensory_modality=experience.get("modality","text"),
            temporal_context=datetime.datetime.now()
        )
        self.experience_counter += 1
        current_stage_info = self.growth_stages[self.current_stage]
        if self.experience_counter > current_stage_info["memory_capacity_threshold"] * 0.8: # Check against 80% of capacity
            if self._should_mature(current_stage_info["memory_capacity_threshold"]):
                 self.mature()
        return memory_id

    def _should_mature(self, current_stage_memory_capacity_threshold: float) -> bool:
        '''Determines if the mind should mature to the next stage based on high-affinity memory count.'''
        if not hasattr(self.synapse_db, 'get_high_affinity_memories_count'):
            return self.experience_counter > current_stage_memory_capacity_threshold * 0.8 # Fallback

        high_affinity_count = self.synapse_db.get_high_affinity_memories_count(strength_threshold=0.7) # Example threshold
        self.experience_window.append(high_affinity_count) # Add current count to window

        min_samples_for_avg = self.experience_window.maxlen // 4 # Require at least 25% of window to be full
        if len(self.experience_window) < min_samples_for_avg:
            return False

        try:
            # Calculate moving average of high-affinity memories
            avg_high_affinity_memories = sum(self.experience_window) / len(self.experience_window)
            # Mature if average high-affinity memories exceed 80% of current stage's conceptual capacity threshold
            return avg_high_affinity_memories > (current_stage_memory_capacity_threshold * 0.8)
        except ZeroDivisionError: # Should not happen due to len check, but good practice
            return False

    def recall(self, cue_embedding: list, depth: int = 3):
        '''Recalls memories associatively based on a cue embedding.'''
        temporal_decay_param = self.growth_stages[self.current_stage]["associations"] * 0.01
        return self.synapse_db.associative_recall(
            cue_embedding=cue_embedding,
            depth=depth,
            temporal_decay_rate_hourly=temporal_decay_param
        )

    def learn(self, feedback: dict, memory_id: int):
        '''Processes feedback to reinforce memories and stimulate neurogenesis for insights.'''
        self.synapse_db.reinforce_memory(
            memory_id=memory_id,
            reinforcement_signal=feedback.get("strength", 0.1)
        )
        if feedback.get("correction"):
            self.cognition.adapt(feedback["correction"])
        if feedback.get("strength", 0.0) > 0.8 and feedback.get("insight"):
            self.neurogenesis.generate_new_pathway(
                str(feedback.get("insight")), # Pass insight content
                insight_strength=feedback.get("strength",0.8) # Pass insight strength
            )

    def mature(self):
        '''Transitions the mind to the next developmental stage if applicable.'''
        stages = list(self.growth_stages.keys())
        try:
            current_index = stages.index(self.current_stage)
        except ValueError:
            print(f"Error: Current stage {self.current_stage} not found in defined stages.")
            return

        if current_index < len(stages) - 1:
            next_stage = stages[current_index + 1]
            print(f"Maturing from {self.current_stage} to {next_stage} stage")
            self.current_stage = next_stage
            self.neurogenesis.stage = next_stage
            self.neurogenesis.growth_rate = self.neurogenesis._set_growth_rate()
            self.neurogenesis.accelerate()
            self.synapse_db.remap_connections(
                new_association_depth=self.growth_stages[next_stage]["associations"]
            )
            self.cognition.expand_capacity(current_stage=next_stage)
            self.experience_counter = 0
            self.experience_window.clear()
        else:
            print(f"NeuroplasticMind: Already at maximum developmental stage ({self.current_stage}).")

    def dream_cycle(self):
        '''Simulates a dream cycle for memory consolidation and pathway refinement.'''
        print("NeuroplasticMind: Starting dream cycle (memory consolidation)...")
        if hasattr(self.synapse_db, 'get_high_affinity_memories'):
            important_memories = self.synapse_db.get_high_affinity_memories(strength_threshold=0.6, limit=20)
            if important_memories:
                 for memory_dict in important_memories: # Assuming list of dicts
                    simulated_experience = self.perception.simulate_replay(memory_dict.get("content"))
                    self.cognition.process(simulated_experience)

        self.synapse_db.synaptic_pruning(threshold=0.2) # Default threshold
        self.synapse_db.cross_link_modalities() # Default thresholds
        print("NeuroplasticMind: Dream cycle complete.")
