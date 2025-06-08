import duckdb
import os
import numpy as np # Required for logistic_growth, forgetting_curve if used in Python

DB_DIR = "memory_db" # Re-using the existing directory for organization
DEFAULT_DB_PATH = os.path.join(DB_DIR, "neuro_memory.duckdb")

class SynapticDuckDB:
    def __init__(self, db_path=None):
        self.db_path = db_path if db_path else DEFAULT_DB_PATH
        print(f"SynapticDuckDB: Initializing database at {self.db_path}")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(database=self.db_path, read_only=False)
        self._install_and_load_extensions()
        self._create_biomemory_schema()
        self.active_connections = {} # As per user's code, though usage not yet defined
        print("SynapticDuckDB: Schema initialized and ready.")

    def _install_and_load_extensions(self):
        try:
            self.conn.execute("INSTALL hnsw;")
            self.conn.execute("LOAD hnsw;")
            # JSON extension is usually bundled, but good to have if specific JSON functions are needed
            self.conn.execute("INSTALL json;")
            self.conn.execute("LOAD json;")
            print("SynapticDuckDB: HNSW and JSON extensions loaded/installed.")
        except Exception as e:
            print(f"SynapticDuckDB: Notice - Could not install/load extensions (may be pre-loaded or specific versions might differ): {e}")

    def _create_biomemory_schema(self):
        """Creates the biological memory schema with DuckDB optimizations."""
        # DuckDB uses SERIAL or IDENTITY for auto-incrementing PKs, or INTEGER PRIMARY KEY often defaults to auto-increment behavior.
        # Using INTEGER PRIMARY KEY for simplicity, which implies rowid aliasing or auto-increment.
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS synaptic_memories (
            neuron_id INTEGER PRIMARY KEY,      -- Auto-incrementing unique ID for each memory
            memory_trace FLOAT[],               -- Embedding vector for similarity search
            content TEXT,                       -- Textual content of the memory
            valence FLOAT,                      -- Emotional weight [-1.0, 1.0]
            modality VARCHAR,                   -- Sensory modality (e.g., 'text', 'image', 'audio')
            timestamp TIMESTAMP,                -- When the memory was encoded (renamed from last_accessed in user schema for clarity)
            last_accessed TIMESTAMP,            -- When the memory was last accessed/recalled
            activation_count INTEGER DEFAULT 0, -- How many times this memory has been activated/recalled
            synaptic_strength FLOAT DEFAULT 0.5 -- Current strength of this memory node
        );
        """)

        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS neural_pathways (
            pathway_id INTEGER PRIMARY KEY,     -- Auto-incrementing unique ID for each pathway
            source_neuron INTEGER NOT NULL REFERENCES synaptic_memories(neuron_id) ON DELETE CASCADE,
            target_neuron INTEGER NOT NULL REFERENCES synaptic_memories(neuron_id) ON DELETE CASCADE,
            connection_strength FLOAT DEFAULT 0.1, -- Strength of the connection
            last_used TIMESTAMP,                -- When this pathway was last utilized
            CONSTRAINT unique_pathway UNIQUE(source_neuron, target_neuron) -- Avoid duplicate pathways
        );
        """)

        # Optimized biological indexes
        # HNSW Index for memory_trace for approximate nearest neighbor search
        # DuckDB's HNSW index is created on a column and then used by distance functions in queries.
        # The specific `WITH (distance_function = 'cosine', ...)` syntax might vary or be set by PRAGMA.
        # DuckDB's default for vector similarity with HNSW typically uses Euclidean distance (L2).
        # For cosine similarity, you'd typically use the `cosine_similarity` function in your queries.
        # The user's schema specifies 'cosine' for HNSW. We'll ensure queries use cosine similarity.
        # DuckDB's HNSW creation: CREATE INDEX idx_hnsw_mem_trace ON synaptic_memories USING HNSW (memory_trace (metric 'cosine'));
        # The (metric 'cosine') part depends on specific DuckDB version and HNSW extension capabilities for explicit metric in CREATE INDEX.
        # If not supported directly in CREATE INDEX, queries must use cosine_similarity() and it will leverage HNSW if column is indexed.
        try:
             # Attempt to create HNSW index with cosine if supported, otherwise a generic HNSW for list_distance.
             # The HNSW extension should provide functions like 'cosine_similarity' or 'list_cosine_distance'.
             # We will rely on these functions + standard indexing on the column for now.
             # A simple index on the column helps, and specific HNSW usage is via query functions.
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_trace_generic ON synaptic_memories (memory_trace);")
            # User specific HNSW with options:
            # self.conn.execute("CREATE INDEX IF NOT EXISTS memory_affinity ON synaptic_memories USING HNSW(memory_trace) WITH (metric = 'cosine', M=12, ef_construction=32);")
            # The above WITH syntax is more typical of specialized vector DBs or specific Postgres extensions.
            # For DuckDB HNSW, it's usually about indexing the column and using vector functions in queries.
            # If the HNSW extension supports `metric='cosine'` in `CREATE INDEX`, that would be ideal.
            # For now, we assume the HNSW extension will optimize queries using cosine_similarity() on the memory_trace column.
            print("SynapticDuckDB: Index on memory_trace created/verified.")
        except Exception as e:
            print(f"SynapticDuckDB: Note - Could not create specific HNSW index with options (may rely on query-time functions): {e}")

        # Standard B-tree indexes for filtering and sorting
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_valence ON synaptic_memories(valence);")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_modality ON synaptic_memories(modality);
")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_strength ON neural_pathways(connection_strength);
")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_source ON neural_pathways(source_neuron);
")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pathway_target ON neural_pathways(target_neuron);
")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            print("SynapticDuckDB: Database connection closed.")

    def __del__(self):
        self.close()

    def store_memory(self, content: str, embedding: list[float], emotional_valence: float,
                     sensory_modality: str, temporal_context: "datetime.datetime"):
        """Store with biological context."""
        if len(embedding) == 0: # Basic check for empty embedding
            print("SynapticDuckDB: Warning - Attempted to store memory with empty embedding. Skipping.")
            return None

        # Ensure embedding is a flat list of floats
        flat_embedding = [float(x) for x in embedding]

        insert_sql = """
            INSERT INTO synaptic_memories (
                memory_trace, content, valence, modality, timestamp,
                last_accessed, activation_count, synaptic_strength
            ) VALUES (?, ?, ?, ?, ?, ?, 1, 0.5) RETURNING neuron_id;
        """
        try:
            result = self.conn.execute(insert_sql, [
                flat_embedding, content, emotional_valence, sensory_modality,
                temporal_context, temporal_context # timestamp and last_accessed are same on creation
            ]).fetchone()

            if result and result[0] is not None:
                new_id = result[0]
                print(f"SynapticDuckDB: Stored new memory with neuron_id: {new_id}")
                self._form_initial_synapses(new_id)
                return new_id
            else:
                print("SynapticDuckDB: Failed to retrieve neuron_id after insert.")
                return None
        except Exception as e:
            print(f"SynapticDuckDB: Error storing memory: {e}")
            return None

    def synaptic_pruning(self, threshold: float = 0.2):
        """Remove weak connections (biological pruning) based on a strength threshold."""
        try:
            original_count_query = self.conn.execute("SELECT COUNT(*) FROM neural_pathways").fetchone()
            original_count = original_count_query[0] if original_count_query else 0

            delete_sql = f"DELETE FROM neural_pathways WHERE connection_strength < ?;"
            self.conn.execute(delete_sql, [threshold])

            new_count_query = self.conn.execute("SELECT COUNT(*) FROM neural_pathways").fetchone()
            new_count = new_count_query[0] if new_count_query else 0
            pruned_count = original_count - new_count
            if pruned_count > 0:
                print(f"SynapticDuckDB: Pruned {pruned_count} weak neural pathways (strength < {threshold}).")
            else:
                print(f"SynapticDuckDB: No neural pathways found weaker than {threshold} for pruning.")
            return pruned_count
        except Exception as e:
            print(f"SynapticDuckDB: Error during synaptic pruning: {e}")
            return 0

    def cross_link_modalities(self, similarity_threshold: float = 0.65, valence_diff_threshold: float = 0.3, initial_strength: float = 0.4):
        """Create cross-modal associations during rest/consolidation."""
        # pathway_id is INTEGER PRIMARY KEY (auto-increment), so no need for nextval('pathway_seq') directly in this SQL
        # The user's SQL had nextval, but for DuckDB auto-increment PK this is not needed in the INSERT SELECT.
        # We also need to ensure we don't try to connect a neuron to itself if a and b could be the same row by chance.
        # And ensure the NOT EXISTS subquery correctly references outer a and b.
        sql = """
        INSERT INTO neural_pathways (source_neuron, target_neuron, connection_strength, last_used)
        SELECT
            a.neuron_id AS source_neuron,
            b.neuron_id AS target_neuron,
            ? AS connection_strength, -- initial_strength param
            CURRENT_TIMESTAMP
        FROM synaptic_memories a
        JOIN synaptic_memories b ON a.modality != b.modality AND a.neuron_id != b.neuron_id
        WHERE ABS(a.valence - b.valence) < ?
            AND cosine_similarity(a.memory_trace, b.memory_trace) > ?
            AND NOT EXISTS (
                SELECT 1 FROM neural_pathways np
                WHERE (np.source_neuron = a.neuron_id AND np.target_neuron = b.neuron_id)
                   OR (np.source_neuron = b.neuron_id AND np.target_neuron = a.neuron_id) -- Check for bidirectional existing paths
            )
        ON CONFLICT (source_neuron, target_neuron) DO NOTHING; -- Handles if somehow still a conflict after NOT EXISTS
        """
        try:
            result = self.conn.execute(sql, [initial_strength, valence_diff_threshold, similarity_threshold])
            # The result object for INSERT from SELECT doesn't directly give row count easily in all DBs with Python client.
            # We can infer by checking if an error, or assume it worked if no error.
            # To get number of rows inserted, one might need a different approach or check DB-specific functions.
            # For now, we'll assume success if no error.
            # DuckDB's execute often returns a relation object that can be queried, or rowcount might be available.
            # For INSERT FROM SELECT, rowcount might not be standard. Let's assume it's hard to get affected_rows directly.
            print(f"SynapticDuckDB: Cross-modal linking attempted. (thresholds: sim={similarity_threshold}, valence_diff={valence_diff_threshold})")
        except Exception as e:
            print(f"SynapticDuckDB: Error during cross-modal linking: {e}")

    def remap_connections(self, new_association_depth: int, initial_strength: float = 0.35):
        """Reorganize for new developmental stage, increasing connection density."""
        # This query aims to add new connections up to a certain rank (new_association_depth)
        # for each source neuron, if a connection doesn't already exist.
        # It uses ROW_NUMBER() to rank potential new connections by similarity.
        sql = """
        INSERT INTO neural_pathways (source_neuron, target_neuron, connection_strength, last_used)
        SELECT
            source,
            target,
            ? AS connection_strength, -- initial_strength param
            CURRENT_TIMESTAMP
        FROM (
            SELECT
                a.neuron_id AS source,
                b.neuron_id AS target,
                ROW_NUMBER() OVER (
                    PARTITION BY a.neuron_id
                    ORDER BY cosine_similarity(a.memory_trace, b.memory_trace) DESC
                ) AS rank
            FROM synaptic_memories a
            CROSS JOIN synaptic_memories b -- Potential for large intermediate result, ensure DB can handle
            WHERE a.neuron_id != b.neuron_id
                AND NOT EXISTS (
                    SELECT 1 FROM neural_pathways np
                    WHERE (np.source_neuron = a.neuron_id AND np.target_neuron = b.neuron_id)
                       OR (np.source_neuron = b.neuron_id AND np.target_neuron = a.neuron_id) -- Check bidirectional
                )
        ) ranked_potential_paths
        WHERE rank <= ?
        ON CONFLICT (source_neuron, target_neuron) DO NOTHING;
        """
        try:
            self.conn.execute(sql, [initial_strength, new_association_depth])
            print(f"SynapticDuckDB: Connection remapping attempted for new association depth: {new_association_depth}.")
        except Exception as e:
            print(f"SynapticDuckDB: Error during connection remapping: {e}")

    def reinforce_memory(self, memory_id: int, reinforcement_signal: float, # Renamed from factor for clarity
                         logistic_k: float = 0.5, logistic_x0_memory: float = 5.0,
                         logistic_x0_pathway: float = 10.0, max_strength: float = 1.0):
        """Strengthen synapses based on learning signal, using logistic growth."""
        if reinforcement_signal == 0: # No change if signal is zero
            # Still update last_accessed and activation_count as it was 'touched'
            touch_sql = """UPDATE synaptic_memories
                           SET activation_count = activation_count + 1, last_accessed = CURRENT_TIMESTAMP
                           WHERE neuron_id = ?;"""
            self.conn.execute(touch_sql, [memory_id])
            return

        try:
            # --- 1. Reinforce the memory (neuron) itself ---
            current_memory_details = self.conn.execute(
                "SELECT synaptic_strength, activation_count FROM synaptic_memories WHERE neuron_id = ?",
                [memory_id]
            ).fetchone()

            if not current_memory_details:
                print(f"SynapticDuckDB: Neuron ID {memory_id} not found for reinforcement.")
                return

            current_strength = current_memory_details[0]
            current_activation_count = current_memory_details[1]

            # Apply logistic growth to the change in strength
            # The 'x' in logistic_growth can be considered the number of positive reinforcements or accumulated signal strength.
            # Here, reinforcement_signal is a value (e.g. 0 to 1). Let's assume it's an increment to a conceptual 'effort' or 'evidence' counter.
            # We can use current_activation_count as a proxy for 'x' if reinforcement_signal is a fixed positive value.
            # Or, if reinforcement_signal varies, it's more like a delta to be added to strength, but bounded by logistic growth.

            # Simpler model: reinforcement_signal directly influences the *increment* of strength,
            # and this increment is larger when current_strength is further from max_strength.
            # User plan: reinforcement = logistic_growth(feedback_strength). Assuming feedback_strength is the signal.
            # This implies the signal itself is transformed. Let's try that.
            # The `x` in logistic_growth(x, L, k, x0) is the input variable.
            # If reinforcement_signal is (e.g. 0..1), this might not map well if x0 is 5.
            # Let's assume `x` is the number of times this kind of signal has been received, approximated by activation_count.
            # The amount of strength to ADD can be based on logistic_growth.
            # delta_strength = reinforcement_signal * logistic_growth(current_activation_count, L=0.2, k=logistic_k, x0=logistic_x0_memory) # L is max increment

            # Let's use the user's model: `reinforcement = logistic_growth(feedback_strength)`
            # This means the `reinforcement_signal` (feedback_strength) is the `x`.
            # To make this work with typical x0=5, the feedback_strength needs to be on a scale that can reach x0.
            # Or, we scale reinforcement_signal: e.g., if signal is 0..1, scale it to 0..10 for x.
            # Let's assume reinforcement_signal is the raw 'strength' value from user's 'learn' method.
            # The logistic_growth function as defined returns a value between 0 and L.
            # Let L be the max possible *increment* in one step.
            max_increment_neuron = 0.2 # Max strength to add in one reinforcement for a neuron
            strength_increment_neuron = logistic_growth(reinforcement_signal * 10, L=max_increment_neuron, k=logistic_k, x0=logistic_x0_memory) # Scale signal to ~0-10 for x

            new_synaptic_strength = min(max_strength, current_strength + strength_increment_neuron if reinforcement_signal > 0 else current_strength + strength_increment_neuron) # allow negative signal to reduce
            # Ensure strength doesn't go below a certain floor if reducing (e.g. 0.01)
            new_synaptic_strength = max(0.01, new_synaptic_strength)

            self.conn.execute("""
                UPDATE synaptic_memories
                SET
                    synaptic_strength = ?,
                    activation_count = activation_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE neuron_id = ?;
            """, [new_synaptic_strength, memory_id])
            # print(f"SynapticDuckDB: Reinforced memory {memory_id}. Strength: {current_strength:.2f} -> {new_synaptic_strength:.2f}")

            # --- 2. Strengthen associated neural pathways ---
            # Fetch current strengths of pathways connected to this neuron
            pathways_to_update_sql = """
                SELECT pathway_id, connection_strength, source_neuron, target_neuron
                FROM neural_pathways
                WHERE source_neuron = ? OR target_neuron = ?;
            """
            pathways = self.conn.execute(pathways_to_update_sql, [memory_id, memory_id]).fetchall()

            max_increment_pathway = 0.1 # Max strength to add in one reinforcement for a pathway
            # Use a potentially different x0 for pathways, or reuse.
            # The 'x' for pathway reinforcement could be the new_synaptic_strength of the neuron, or the signal again.
            # Let's use the original reinforcement_signal for pathways too.
            strength_increment_pathway = logistic_growth(reinforcement_signal * 10, L=max_increment_pathway, k=logistic_k, x0=logistic_x0_pathway)

            updates = []
            for pathway_id, current_conn_strength, _, _ in pathways:
                new_conn_strength = min(max_strength, current_conn_strength + strength_increment_pathway if reinforcement_signal > 0 else current_conn_strength + strength_increment_pathway)
                new_conn_strength = max(0.01, new_conn_strength) # Ensure positive strength
                if abs(new_conn_strength - current_conn_strength) > 1e-5: # Update only if changed
                    updates.append((new_conn_strength, pathway_id))

            if updates:
                update_pathways_sql = """UPDATE neural_pathways SET connection_strength = ?, last_used = CURRENT_TIMESTAMP WHERE pathway_id = ?;"""
                self.conn.executemany(update_pathways_sql, updates)
                # print(f"SynapticDuckDB: Reinforced {len(updates)} pathways connected to memory {memory_id}.")

        except Exception as e:
            print(f"SynapticDuckDB: Error during memory reinforcement for ID {memory_id}: {e}")

    def associative_recall(self, cue_embedding: list[float], depth: int = 3, temporal_decay_factor: float = 0.05, similarity_threshold_initial: float = 0.6, result_limit: int = 20):
        """Biological pattern completion recall using recursive graph traversal."""
        if not cue_embedding or len(cue_embedding) == 0:
            print("SynapticDuckDB: Warning - Cue embedding is empty. Cannot perform recall.")
            return self.conn.execute("SELECT NULL LIMIT 0;").fetchdf() # Return empty DataFrame with no columns

        # Ensure cue_embedding is a flat list of floats for the query
        flat_cue_embedding = [float(x) for x in cue_embedding]

        # Note: `np.connection_strength` in user's original SQL was a typo, should be `np_alias.connection_strength`.
        # DuckDB uses epoch(timestamp) instead of EXTRACT(EPOCH FROM timestamp).
        # The relevance calculation includes a temporal decay based on last_accessed.
        # The temporal_decay_factor influences how quickly relevance drops with age since last access.
        # A smaller factor means slower decay of relevance due to time.

        # The recursive query needs to be carefully constructed. Alias for neural_pathways is np.
        # The similarity in the recursive part should be between the *current memory_walk item's trace* (mw.memory_trace)
        # and the *newly joined synaptic_memory's trace* (sm.memory_trace).
        # The connection_strength from neural_pathways (np.connection_strength) acts as a weight for this similarity.

        recursive_sql = f"""
        WITH RECURSIVE memory_walk AS (
            -- Anchor member: Direct similarity to the cue
            SELECT
                sm.neuron_id,
                sm.memory_trace,
                sm.content,
                sm.valence,
                sm.modality,
                sm.timestamp AS creation_timestamp,
                sm.last_accessed,
                sm.activation_count,
                sm.synaptic_strength,
                1 AS depth,
                (cosine_similarity(CAST(? AS FLOAT[]), sm.memory_trace) *
                    (1.0 - ? * (epoch(CURRENT_TIMESTAMP) - epoch(sm.last_accessed)) / 3600.0)
                ) AS relevance
            FROM synaptic_memories sm
            WHERE cosine_similarity(CAST(? AS FLOAT[]), sm.memory_trace) > ? -- Initial similarity threshold for anchor

            UNION ALL

            -- Recursive member: Traverse neural pathways
            SELECT
                sm_target.neuron_id,
                sm_target.memory_trace,
                sm_target.content,
                sm_target.valence,
                sm_target.modality,
                sm_target.timestamp AS creation_timestamp,
                sm_target.last_accessed,
                sm_target.activation_count,
                sm_target.synaptic_strength,
                mw.depth + 1,
                (cosine_similarity(mw.memory_trace, sm_target.memory_trace) * np.connection_strength * mw.relevance *
                    (1.0 - ? * (epoch(CURRENT_TIMESTAMP) - epoch(sm_target.last_accessed)) / 3600.0)
                ) AS relevance
            FROM memory_walk mw
            JOIN neural_pathways np ON mw.neuron_id = np.source_neuron
            JOIN synaptic_memories sm_target ON np.target_neuron = sm_target.neuron_id
            WHERE mw.depth < ? AND sm_target.neuron_id != mw.neuron_id -- Avoid immediate self-loops in this step
        )
        SELECT DISTINCT neuron_id, content, valence, modality, creation_timestamp, last_accessed, activation_count, synaptic_strength, depth, relevance
        FROM memory_walk
        ORDER BY relevance DESC, depth ASC
        LIMIT ?;
        """

        try:
            # Parameters for the SQL query:
            # 1. cue_embedding (for anchor cosine_similarity)
            # 2. temporal_decay_factor (for anchor relevance calc)
            # 3. cue_embedding (for anchor WHERE clause similarity check)
            # 4. similarity_threshold_initial (for anchor WHERE clause)
            # 5. temporal_decay_factor (for recursive member relevance calc)
            # 6. depth (for recursive member WHERE clause)
            # 7. result_limit (for final LIMIT)
            params = [
                flat_cue_embedding,
                temporal_decay_factor,
                flat_cue_embedding,
                similarity_threshold_initial,
                temporal_decay_factor,
                depth,
                result_limit
            ]
            # print(f"SynapticDuckDB: Executing associative_recall with depth={depth}, temporal_decay_factor={temporal_decay_factor}, initial_sim_thresh={similarity_threshold_initial}")
            results_df = self.conn.execute(recursive_sql, params).fetchdf()
            # Update last_accessed and activation_count for recalled memories
            if not results_df.empty:
                recalled_ids = results_df['neuron_id'].tolist()
                current_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                update_sql = "UPDATE synaptic_memories SET activation_count = activation_count + 1, last_accessed = ? WHERE neuron_id = ?"
                updates = [(current_time_str, mem_id) for mem_id in recalled_ids]
                self.conn.executemany(update_sql, updates)
            return results_df
        except Exception as e:
            print(f"SynapticDuckDB: Error during associative recall: {e}")
            # Return an empty DataFrame with expected columns in case of error
            # This is a bit tricky as the actual columns are defined by the CTE.
            # For now, returning a completely empty DF from a null query is simpler.
            return self.conn.execute("SELECT NULL LIMIT 0;").fetchdf()

    def _form_initial_synapses(self, new_neuron_id: int, limit_connections: int = 5, similarity_threshold: float = 0.7, initial_connection_strength: float = 0.3):
        """Connect to related existing memories based on embedding similarity."""
        new_neuron_embedding_tuple = self.conn.execute(
            "SELECT memory_trace FROM synaptic_memories WHERE neuron_id = ?",
            [new_neuron_id]
        ).fetchone()

        if not new_neuron_embedding_tuple or not new_neuron_embedding_tuple[0]:
            print(f"SynapticDuckDB: Could not retrieve embedding for new neuron {new_neuron_id} to form synapses.")
            return

        new_neuron_embedding = new_neuron_embedding_tuple[0]

        find_similar_sql = """
            SELECT neuron_id, cosine_similarity(memory_trace, CAST(? AS FLOAT[])) AS sim
            FROM synaptic_memories
            WHERE neuron_id != ?
            ORDER BY sim DESC
            LIMIT ?;
        """
        similar_neurons = self.conn.execute(find_similar_sql, [new_neuron_embedding, new_neuron_id, limit_connections * 2]).fetchall()

        pathways_created_count = 0
        insert_pathway_sql = """
            INSERT INTO neural_pathways (source_neuron, target_neuron, connection_strength, last_used)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (source_neuron, target_neuron) DO NOTHING;
        """
        for similar_neuron_id, similarity_score in similar_neurons:
            if pathways_created_count >= limit_connections: break
            if similarity_score >= similarity_threshold:
                self.conn.execute(insert_pathway_sql, [new_neuron_id, similar_neuron_id, initial_connection_strength])
                self.conn.execute(insert_pathway_sql, [similar_neuron_id, new_neuron_id, initial_connection_strength])
                pathways_created_count += 1

        if pathways_created_count > 0:
            print(f"SynapticDuckDB: Formed {pathways_created_count} initial bidirectional synaptic pathways for neuron {new_neuron_id}.")
        else:
            print(f"SynapticDuckDB: No sufficiently similar existing neurons found to form initial pathways for neuron {new_neuron_id} (threshold {similarity_threshold}).")

# Utility for logistic growth (can be part of this file or a utils.py)
def logistic_growth(x, L=1.0, k=0.5, x0=5):
    """S-shaped growth curve. x is current value (e.g. number of reinforcements)"""
    if x is None: x = 0 # Handle None case
    try:
        return L / (1 + np.exp(-k * (x - x0)))
    except OverflowError:
        # If -k * (x - x0) is very large, np.exp overflows.
        # If -k * (x - x0) is very small (large negative), np.exp approaches 0.
        val = -k * (x - x0)
        if val > 700: # exp(709) is around max float64
            return 0.0 # Denominator becomes huge, so result is 0
        else: # Should not happen if L is positive, means 1 + very small number
            return L

# Utility for forgetting curve (can be part of this file or a utils.py)
def forgetting_curve_factor(t_seconds, s=0.3):
    """Returns a factor (0-1) representing recall probability after t_seconds, given stability s."""
    if t_seconds < 0: t_seconds = 0
    # s is stability in some unit, let's assume higher s means slower forgetting.
    # The user formula was 1 / (1 + t/(s*3600)). If s is small, forgetting is fast.
    # Let's adjust s to be more intuitive: s is 'half-life' in days for this example.
    # For example, if s_days = 1, then after 1 day, factor is 0.5.
    # (1 / (1 + t_days/s_days)) - this is not exponential, but hyperbolic.
    # A common exponential form: factor = exp(-decay_constant * t_seconds)
    # Let's use the user's provided hyperbolic model: s is a stability parameter.
    # The original formula: 1 / (1 + t/(s*3600)) - t is in seconds, s is stability parameter.
    # If s = 0.3, t = 3600 (1 hour) => 1 / (1 + 1/0.3) = 1 / (1 + 3.33) = 1/4.33 = 0.23
    # If s is larger, the denominator grows slower, so factor is higher (slower forgetting)
    if s <= 0: return 0.0 # Avoid division by zero or nonsensical results
    return 1.0 / (1.0 + (t_seconds / (s * 3600.0)))
