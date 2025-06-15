import duckdb
import os
import numpy as np # Required for logistic_growth, forgetting_curve if used in Python
import time # For strftime in associative_recall updates
import datetime # For datetime objects in associative_recall updates

DB_DIR = "memory_db"
DEFAULT_DB_PATH = os.path.join(DB_DIR, "neuro_memory.duckdb")

# Utility for logistic growth (module level)
def logistic_growth(x, L: float = 1.0, k: float = 0.5, x0: float = 5.0) -> float:
    '''S-shaped growth curve. x is current value (e.g. number of reinforcements)'''
    if x is None: x = 0
    try:
        exp_val = -k * (x - x0)
        if exp_val > 700: return 0.0
        elif exp_val < -700: return L
        return L / (1 + np.exp(exp_val))
    except OverflowError:
        return L if -k * (x - x0) < 0 else 0.0

# Utility for forgetting curve (module level)
def forgetting_curve_factor(t_seconds: float, s: float = 1.0) -> float:
    '''Returns a factor (0-1) representing recall probability after t_seconds, given stability s.'''
    if t_seconds < 0: t_seconds = 0
    if s <= 1e-9: return 0.0
    return 1.0 / (1.0 + (t_seconds / (s * 3600.0)))


class SynapticDuckDB:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(database=self.db_path, read_only=False)
        self._install_and_load_extensions()
        self._create_biomemory_schema()
        self.active_connections = {}
        # print("SynapticDuckDB: Schema initialized and ready.")

    def _install_and_load_extensions(self):
        try:
            self.conn.execute("INSTALL hnsw;")
            self.conn.execute("LOAD hnsw;")
            self.conn.execute("INSTALL json;")
            self.conn.execute("LOAD json;")
        except Exception as e:
            pass # print(f"SynapticDuckDB: Notice - Extensions: {e}")

    def _create_biomemory_schema(self):
        '''Creates the biological memory schema with DuckDB optimizations.'''
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS synaptic_memories (
            neuron_id INTEGER PRIMARY KEY, memory_trace FLOAT[], content TEXT, valence FLOAT,
            modality VARCHAR, timestamp TIMESTAMP, last_accessed TIMESTAMP,
            activation_count INTEGER DEFAULT 0, synaptic_strength FLOAT DEFAULT 0.5
        );
        ''')
        self.conn.execute('''
        CREATE TABLE IF NOT EXISTS neural_pathways (
            pathway_id INTEGER PRIMARY KEY,
            source_neuron INTEGER NOT NULL REFERENCES synaptic_memories(neuron_id) ON DELETE CASCADE,
            target_neuron INTEGER NOT NULL REFERENCES synaptic_memories(neuron_id) ON DELETE CASCADE,
            connection_strength FLOAT DEFAULT 0.1, last_used TIMESTAMP,
            CONSTRAINT unique_pathway UNIQUE(source_neuron, target_neuron)
        );
        ''')
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_trace_generic ON synaptic_memories (memory_trace);")
        except Exception as e:
            pass # print(f"SynapticDuckDB: Note - HNSW index: {e}")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_valence ON synaptic_memories(valence);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_modality ON synaptic_memories(modality);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_strength ON neural_pathways(connection_strength);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_source ON neural_pathways(source_neuron);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_target ON neural_pathways(target_neuron);")

    def store_memory(self, content: str, embedding: list[float], emotional_valence: float,
                     sensory_modality: str, temporal_context: datetime.datetime):
        '''Store with biological context.'''
        if not embedding:
            # print("SynapticDuckDB: Warning - Empty embedding in store_memory. Skipping.")
            return None
        flat_embedding = [float(x) for x in embedding]
        insert_sql = """INSERT INTO synaptic_memories (memory_trace, content, valence, modality, timestamp,
                           last_accessed, activation_count, synaptic_strength) VALUES (?, ?, ?, ?, ?, ?, 1, 0.5) RETURNING neuron_id;"""
        try:
            result = self.conn.execute(insert_sql, [flat_embedding, content, emotional_valence, sensory_modality, temporal_context, temporal_context]).fetchone()
            if result and result[0] is not None:
                new_id = result[0]
                self._form_initial_synapses(new_id)
                return new_id
            return None
        except Exception as e:
            print(f"SynapticDuckDB: Error storing memory: {e}")
            return None

    def _form_initial_synapses(self, new_neuron_id: int,
                             candidate_pool_size: int = 15,
                             connect_top_n: int = 5,
                             base_similarity_threshold: float = 0.5,
                             sigmoid_steepness: float = 10.0):
        '''Connect to related existing memories based on embedding similarity using sigmoid weighting.'''
        new_neuron_embedding_tuple = self.conn.execute(
            "SELECT memory_trace FROM synaptic_memories WHERE neuron_id = ?",
            [new_neuron_id]
        ).fetchone()
        if not new_neuron_embedding_tuple or not new_neuron_embedding_tuple[0]: return
        new_neuron_embedding = new_neuron_embedding_tuple[0]
        find_similar_sql = "SELECT neuron_id, cosine_similarity(memory_trace, CAST(? AS FLOAT[])) AS sim FROM synaptic_memories WHERE neuron_id != ? ORDER BY sim DESC LIMIT ?;"
        candidate_neurons_raw = self.conn.execute(find_similar_sql, [new_neuron_embedding, new_neuron_id, candidate_pool_size]).fetchall()
        if not candidate_neurons_raw: return
        weighted_candidates = []
        for similar_neuron_id, similarity_score in candidate_neurons_raw:
            if similarity_score is None or similarity_score < base_similarity_threshold: continue
            try:
                exp_val = -similarity_score * sigmoid_steepness # Corrected: similarity is positive, so -sim for desired sigmoid behavior
                if exp_val < -700: sigmoid_w = 1.0 # Approaches 1 if sim is high
                elif exp_val > 700: sigmoid_w = 0.0 # Approaches 0 if sim is low (but still > threshold)
                else: sigmoid_w = 1.0 / (1.0 + np.exp(exp_val))
            except OverflowError: sigmoid_w = 1.0 if -similarity_score * sigmoid_steepness < 0 else 0.0
            if sigmoid_w > 0.01: weighted_candidates.append({'id': similar_neuron_id, 'sigmoid_weight': sigmoid_w})

        weighted_candidates.sort(key=lambda x: x['sigmoid_weight'], reverse=True)
        final_connections = weighted_candidates[:connect_top_n]

        insert_pathway_sql = "INSERT INTO neural_pathways (source_neuron, target_neuron, connection_strength, last_used) VALUES (?, ?, ?, CURRENT_TIMESTAMP) ON CONFLICT (source_neuron, target_neuron) DO NOTHING;"
        pathways_created_count = 0
        for conn_candidate in final_connections:
            connection_strength = float(np.clip(conn_candidate['sigmoid_weight'], 0.01, 1.0))
            self.conn.execute(insert_pathway_sql, [new_neuron_id, conn_candidate['id'], connection_strength])
            self.conn.execute(insert_pathway_sql, [conn_candidate['id'], new_neuron_id, connection_strength]) # Bidirectional
            pathways_created_count += 1
        # if pathways_created_count > 0: print(f"SynapticDuckDB: Formed {pathways_created_count} sig-weighted pathways for {new_neuron_id}.")

    def associative_recall(self, cue_embedding: list[float], depth: int = 3,
                         temporal_decay_rate_hourly: float = 0.05,
                         similarity_threshold_initial: float = 0.6,
                         result_limit: int = 20):
        '''Biological pattern completion recall using recursive graph traversal with exponential temporal decay.'''
        if not cue_embedding: import pandas as pd; return pd.DataFrame() # Return empty DataFrame
        flat_cue_embedding = [float(x) for x in cue_embedding]

        decay_constant_per_second = temporal_decay_rate_hourly / 3600.0

        recursive_sql = f"""
        WITH RECURSIVE memory_walk AS (
            -- Anchor member: Direct similarity to the cue
            SELECT
                sm.neuron_id, sm.memory_trace, sm.content, sm.valence, sm.modality,
                sm.timestamp AS creation_timestamp, sm.last_accessed, sm.activation_count, sm.synaptic_strength,
                1 AS depth,
                (cosine_similarity(CAST(? AS FLOAT[]), sm.memory_trace) * exp(-? * (epoch(CURRENT_TIMESTAMP) - epoch(sm.last_accessed)))) AS relevance
            FROM synaptic_memories sm
            WHERE cosine_similarity(CAST(? AS FLOAT[]), sm.memory_trace) > ?

            UNION ALL

            -- Recursive member: Traverse neural pathways
            SELECT
                sm_target.neuron_id, sm_target.memory_trace, sm_target.content, sm_target.valence, sm_target.modality,
                sm_target.timestamp AS creation_timestamp, sm_target.last_accessed, sm_target.activation_count, sm_target.synaptic_strength,
                mw.depth + 1,
                (cosine_similarity(mw.memory_trace, sm_target.memory_trace) * np.connection_strength * exp(-? * (epoch(CURRENT_TIMESTAMP) - epoch(sm_target.last_accessed)))) * mw.relevance AS relevance
            FROM memory_walk mw
            JOIN neural_pathways np ON mw.neuron_id = np.source_neuron
            JOIN synaptic_memories sm_target ON np.target_neuron = sm_target.neuron_id
            WHERE mw.depth < ? AND sm_target.neuron_id != mw.neuron_id -- Avoid self-loops in this step
        )
        SELECT DISTINCT neuron_id, content, valence, modality, creation_timestamp, last_accessed, activation_count, synaptic_strength, depth, relevance
        FROM memory_walk
        WHERE relevance IS NOT NULL AND relevance > 1e-9 -- Filter out negligible or null relevance
        ORDER BY relevance DESC, depth ASC
        LIMIT ?;
        """
        try:
            params = [
                flat_cue_embedding, decay_constant_per_second,
                flat_cue_embedding, similarity_threshold_initial,
                decay_constant_per_second,
                depth, result_limit
            ]
            results_df = self.conn.execute(recursive_sql, params).fetchdf()
            if not results_df.empty:
                recalled_ids = results_df['neuron_id'].tolist()
                current_time_for_update = datetime.datetime.now()
                update_sql = "UPDATE synaptic_memories SET activation_count = activation_count + 1, last_accessed = ? WHERE neuron_id = ?"
                updates = [(current_time_for_update, mem_id) for mem_id in recalled_ids]
                self.conn.executemany(update_sql, updates)
            return results_df
        except Exception as e:
            print(f"SynapticDuckDB: Error during associative recall: {e}")
            import pandas as pd; return pd.DataFrame() # Return empty DataFrame on error

    def reinforce_memory(self, memory_id: int, reinforcement_signal: float,
                         growth_k_neuron: float = 0.2, growth_x0_neuron: float = 5.0,
                         growth_k_pathway: float = 0.1, growth_x0_pathway: float = 7.0,
                         max_strength: float = 1.0, min_strength: float = 0.01):
        '''Strengthens or weakens a memory and its pathways based on a signal.'''
        reinforcement_signal = float(np.clip(reinforcement_signal, -1.0, 1.0))
        if abs(reinforcement_signal) < 1e-6:
            touch_sql = "UPDATE synaptic_memories SET activation_count = activation_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE neuron_id = ?;"
            self.conn.execute(touch_sql, [memory_id])
            return
        try:
            current_memory_details = self.conn.execute("SELECT synaptic_strength FROM synaptic_memories WHERE neuron_id = ?", [memory_id]).fetchone()
            if not current_memory_details: return
            current_strength_neuron = float(current_memory_details[0])

            actual_increment_neuron = 0.0
            if reinforcement_signal > 0: # Strengthening
                room_for_growth = max_strength - current_strength_neuron
                if room_for_growth > 1e-6:
                    effort_neuron = reinforcement_signal * 10.0
                    gain_fraction = logistic_growth(effort_neuron, L=1.0, k=growth_k_neuron, x0=growth_x0_neuron)
                    actual_increment_neuron = room_for_growth * gain_fraction
            else: # Weakening
                room_for_decline = current_strength_neuron - min_strength
                if room_for_decline > 1e-6:
                    effort_neuron = abs(reinforcement_signal) * 10.0
                    decline_fraction = logistic_growth(effort_neuron, L=1.0, k=growth_k_neuron, x0=growth_x0_neuron)
                    actual_increment_neuron = - (room_for_decline * decline_fraction)

            new_synaptic_strength = float(np.clip(current_strength_neuron + actual_increment_neuron, min_strength, max_strength))

            if abs(new_synaptic_strength - current_strength_neuron) > 1e-5:
                self.conn.execute("UPDATE synaptic_memories SET synaptic_strength = ?, activation_count = activation_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE neuron_id = ?;", [new_synaptic_strength, memory_id])

            pathways = self.conn.execute("SELECT pathway_id, connection_strength FROM neural_pathways WHERE source_neuron = ? OR target_neuron = ?;", [memory_id, memory_id]).fetchall()
            pathway_updates = []
            for pathway_id, current_conn_strength_raw in pathways:
                current_conn_strength = float(current_conn_strength_raw)
                actual_increment_pathway = 0.0
                if reinforcement_signal > 0:
                    room_for_growth_pathway = max_strength - current_conn_strength
                    if room_for_growth_pathway > 1e-6:
                        effort_pathway = reinforcement_signal * 10.0
                        gain_fraction_pathway = logistic_growth(effort_pathway, L=1.0, k=growth_k_pathway, x0=growth_x0_pathway)
                        actual_increment_pathway = room_for_growth_pathway * gain_fraction_pathway
                else:
                    room_for_decline_pathway = current_conn_strength - min_strength
                    if room_for_decline_pathway > 1e-6:
                        effort_pathway = abs(reinforcement_signal) * 10.0
                        decline_fraction_pathway = logistic_growth(effort_pathway, L=1.0, k=growth_k_pathway, x0=growth_x0_pathway)
                        actual_increment_pathway = - (room_for_decline_pathway * decline_fraction_pathway)

                if abs(actual_increment_pathway) > 1e-5:
                    new_conn_strength = float(np.clip(current_conn_strength + actual_increment_pathway, min_strength, max_strength))
                    pathway_updates.append((new_conn_strength, datetime.datetime.now(), pathway_id))

            if pathway_updates:
                update_pathways_sql = "UPDATE neural_pathways SET connection_strength = ?, last_used = ? WHERE pathway_id = ?;"
                self.conn.executemany(update_pathways_sql, pathway_updates)
        except Exception as e:
            print(f"SynapticDuckDB: Error during memory reinforcement for ID {memory_id}: {e}")

    def synaptic_pruning(self, absolute_strength_threshold: float = 0.1,
                         enable_entropy_pruning: bool = True,
                         low_entropy_threshold: float = 0.5,
                         min_connections_for_entropy_eval: int = 3,
                         min_strength_for_low_entropy_pathways: float = 0.05):
        '''Prunes weak pathways and optionally those from low-entropy distributions.'''
        pruned_count_total = 0
        try:
            original_count = self.conn.execute("SELECT COUNT(*) FROM neural_pathways").fetchone()[0]
            delete_sql = "DELETE FROM neural_pathways WHERE connection_strength < ?;"
            self.conn.execute(delete_sql, [absolute_strength_threshold])
            current_count = self.conn.execute("SELECT COUNT(*) FROM neural_pathways").fetchone()[0]
            pruned_absolute = original_count - current_count
            pruned_count_total += pruned_absolute

            if enable_entropy_pruning:
                # Placeholder for entropy pruning logic, requires calculate_connection_entropy helper
                pass
        except Exception as e:
            print(f"SynapticDuckDB: Error during synaptic pruning: {e}")

    def cross_link_modalities(self, similarity_threshold: float = 0.65, valence_diff_threshold: float = 0.3, initial_strength: float = 0.4):
        '''Creates pathways between memories of different modalities.'''
        sql = """INSERT INTO neural_pathways (source_neuron, target_neuron, connection_strength, last_used)
               SELECT a.neuron_id, b.neuron_id, ?, CURRENT_TIMESTAMP FROM synaptic_memories a JOIN synaptic_memories b ON a.modality != b.modality AND a.neuron_id != b.neuron_id
               WHERE ABS(a.valence - b.valence) < ? AND cosine_similarity(a.memory_trace, b.memory_trace) > ?
               AND NOT EXISTS (SELECT 1 FROM neural_pathways np WHERE (np.source_neuron = a.neuron_id AND np.target_neuron = b.neuron_id) OR (np.source_neuron = b.neuron_id AND np.target_neuron = a.neuron_id))
               ON CONFLICT (source_neuron, target_neuron) DO NOTHING;"""
        try:
            self.conn.execute(sql, [initial_strength, valence_diff_threshold, similarity_threshold])
        except Exception as e:
            print(f"SynapticDuckDB: Error during cross-modal linking: {e}")

    def remap_connections(self, new_association_depth: int, initial_strength: float = 0.35):
        '''Reorganizes connections for a new developmental stage.'''
        sql = """INSERT INTO neural_pathways (source_neuron, target_neuron, connection_strength, last_used)
               SELECT source, target, ?, CURRENT_TIMESTAMP FROM (
                   SELECT a.neuron_id AS source, b.neuron_id AS target, ROW_NUMBER() OVER (PARTITION BY a.neuron_id ORDER BY cosine_similarity(a.memory_trace, b.memory_trace) DESC) AS rank
                   FROM synaptic_memories a CROSS JOIN synaptic_memories b WHERE a.neuron_id != b.neuron_id
                   AND NOT EXISTS (SELECT 1 FROM neural_pathways np WHERE (np.source_neuron = a.neuron_id AND np.target_neuron = b.neuron_id) OR (np.source_neuron = b.neuron_id AND np.target_neuron = a.neuron_id))
               ) ranked_potential_paths WHERE rank <= ? ON CONFLICT (source_neuron, target_neuron) DO NOTHING;"""
        try:
            self.conn.execute(sql, [initial_strength, new_association_depth])
        except Exception as e:
            print(f"SynapticDuckDB: Error during connection remapping: {e}")

    def get_high_affinity_memories_count(self, strength_threshold: float = 0.7) -> int:
        '''Counts memories with synaptic_strength above a threshold.'''
        if not self.conn: return 0
        try:
            result = self.conn.execute("SELECT COUNT(neuron_id) FROM synaptic_memories WHERE synaptic_strength >= ?;", [strength_threshold]).fetchone()
            return result[0] if result else 0
        except Exception as e:
            return 0

    def get_high_affinity_memories(self, strength_threshold: float = 0.7, limit: int = 100) -> list[dict]:
        '''Fetches high-strength memories, ordered by strength and recency.'''
        if not self.conn: return []
        try:
            query = """SELECT neuron_id, memory_trace, content, valence, modality, timestamp,
                           last_accessed, activation_count, synaptic_strength
                       FROM synaptic_memories
                       WHERE synaptic_strength >= ?
                       ORDER BY synaptic_strength DESC, last_accessed DESC LIMIT ?;"""
            results_raw = self.conn.execute(query, [strength_threshold, limit]).fetchall()
            memories = []
            for row in results_raw:
                memories.append({
                    'neuron_id': row[0], 'memory_trace': row[1], 'content': row[2],
                    'valence': row[3], 'modality': row[4], 'timestamp': row[5],
                    'last_accessed': row[6], 'activation_count': row[7], 'synaptic_strength': row[8]
                })
            return memories
        except Exception as e:
            return []

    def get_memories_for_clustering(self, sample_size: int = None, min_activation_count: int = 0) -> list[dict]:
        '''Fetches neuron_id and memory_trace for memories, optionally sampled or filtered.
        Args:
            sample_size: If not None, randomly samples this many memories.
            min_activation_count: Only include memories with at least this many activations.
        Returns:
            A list of dictionaries, each with 'neuron_id' and 'memory_trace'.
        '''
        if not self.conn:
            return []
        try:
            base_query = "SELECT neuron_id, memory_trace FROM synaptic_memories WHERE activation_count >= ?"
            params = [min_activation_count]

            if sample_size is not None and sample_size > 0:
                query = f"SELECT neuron_id, memory_trace FROM ({base_query} ORDER BY random()) subquery LIMIT ?"
                params.append(sample_size)
            else:
                query = base_query + " ORDER BY neuron_id;"

            results_raw = self.conn.execute(query, params).fetchall()
            memories_for_clustering = []
            for row in results_raw:
                memories_for_clustering.append({'neuron_id': row[0], 'memory_trace': row[1]})
            return memories_for_clustering
        except Exception as e:
            print(f"SynapticDuckDB: Error fetching memories for clustering: {e}")
            return []

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        self.close()
