import duckdb
import numpy as np
import time
import os
# import json # No longer needed directly here for tags if using VARCHAR[]

DB_DIR = "memory_db"
DB_PATH = os.path.join(DB_DIR, "memory.duckdb")

class VectorMemory:
    def __init__(self, db_path=DB_PATH, dim=384):
        self.db_path = db_path
        self.dim = dim
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = duckdb.connect(database=self.db_path, read_only=False)
        try:
            self.conn.execute("INSTALL hnsw;")
            self.conn.execute("LOAD hnsw;")
            self.conn.execute("INSTALL json;") # Still useful for other JSON operations if needed
            self.conn.execute("LOAD json;")
        except Exception as e:
            print(f"Notice: Could not install/load extensions (may be pre-loaded or bundled): {e}")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            timestamp TIMESTAMP,
            content TEXT,
            embedding FLOAT[{self.dim}],
            tags VARCHAR[],
            source VARCHAR,
            certainty FLOAT,
            initial_decay_rate FLOAT,
            access_count INTEGER,
            importance FLOAT,
            last_accessed_ts TIMESTAMP,
            bucket_type VARCHAR DEFAULT 'short_term' -- New column for memory buckets
        );
        """
        self.conn.execute(create_table_sql)

        max_id_result = self.conn.execute("SELECT MAX(id) FROM memories").fetchone()
        self.counter = (max_id_result[0] if max_id_result[0] is not None else -1) + 1

    def add_memory(self, embedding: np.ndarray, content: str, tags: list[str] = None,
                   source: str = "unknown", certainty: float = 1.0,
                   initial_decay_rate: float = 0.95, memory_type: str = "fact",
                   bucket_type: str = 'short_term'): # New parameter for bucket type
        current_time = time.time()
        ts_obj = time.localtime(current_time)
        timestamp_str = time.strftime('%Y-%m-%d %H:%M:%S', ts_obj)
        last_accessed_ts_str = timestamp_str

        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        if len(embedding_list) != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {len(embedding_list)}")

        memory_id = self.counter
        initial_importance = 1.0
        final_tags = tags or []
        if memory_type and f"type:{memory_type}" not in final_tags: # Add memory_type to tags if not already
             final_tags.append(f"type:{memory_type}")

        sql = """
        INSERT INTO memories (
            id, timestamp, content, embedding, tags, source, certainty,
            initial_decay_rate, access_count, importance, last_accessed_ts, bucket_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, [
            memory_id, timestamp_str, content, embedding_list, final_tags, source, certainty,
            initial_decay_rate, 0, initial_importance, last_accessed_ts_str, bucket_type
        ])
        self.counter += 1
        return memory_id

    def retrieve(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None, bucket_types: list[str] = None):
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)
        if len(query_embedding_list) != self.dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.dim}, got {len(query_embedding_list)}")

        params = [query_embedding_list]
        bucket_filter_sql = ""
        if bucket_types and isinstance(bucket_types, list) and len(bucket_types) > 0:
            placeholders = ', '.join(['?'] * len(bucket_types))
            bucket_filter_sql = f"WHERE bucket_type IN ({placeholders})"
            params.extend(bucket_types)

        # Ensure LIMIT param is added last
        params.append(k)

        sql = f"""
        SELECT id, content, tags, source, certainty, importance, timestamp, last_accessed_ts, access_count, bucket_type,
               list_distance(embedding, CAST(? AS FLOAT[{self.dim}])) AS distance
        FROM memories
        {bucket_filter_sql} -- This will be empty if no bucket_types are specified
        ORDER BY distance ASC
        LIMIT ?;
        """

        results_raw = self.conn.execute(sql, params).fetchall()
        retrieved_memories = []
        ids_to_update_access = []
        current_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

        for row in results_raw:
            distance = row[10] # distance is now the 11th column (index 10)
            if threshold is not None and distance > threshold:
                continue
            retrieved_memories.append({
                "id": row[0],
                "content": row[1],
                "tags": row[2],
                "source": row[3],
                "certainty": row[4],
                "importance": row[5],
                "timestamp": row[6],
                "last_accessed_ts": row[7],
                "access_count": row[8],
                "bucket_type": row[9],
                "distance": distance
            })
            ids_to_update_access.append(row[0])

        if ids_to_update_access:
            update_sql = "UPDATE memories SET access_count = access_count + 1, last_accessed_ts = ? WHERE id = ?"
            # Consider executemany for batch updates
            updates = [(current_time_str, mem_id) for mem_id in ids_to_update_access]
            self.conn.executemany(update_sql, updates)
        return retrieved_memories

    def get_memory_by_id(self, memory_id: int):
        # Added bucket_type to the select list
        sql = "SELECT id, content, embedding, tags, source, certainty, importance, timestamp, access_count, initial_decay_rate, last_accessed_ts, bucket_type FROM memories WHERE id = ?"
        result = self.conn.execute(sql, [memory_id]).fetchone()
        if result:
            return {
                "id": result[0], "content": result[1], "embedding": np.array(result[2]),
                "tags": result[3], "source": result[4], "certainty": result[5],
                "importance": result[6], "timestamp": result[7], "access_count": result[8],
                "initial_decay_rate": result[9], "last_accessed_ts": result[10],
                "bucket_type": result[11] # Added bucket_type
            }
        return None

    def get_all_memories_for_decay(self):
        # Added bucket_type for potential use in decay/transition logic
        sql = "SELECT id, timestamp, importance, access_count, initial_decay_rate, last_accessed_ts, bucket_type FROM memories"
        return self.conn.execute(sql).fetchall()

    def update_memory_importance(self, memory_id: int, new_importance: float):
        sql = "UPDATE memories SET importance = ? WHERE id = ?"
        self.conn.execute(sql, [new_importance, memory_id])

    def update_memory_bucket(self, memory_id: int, new_bucket_type: str):
        sql = "UPDATE memories SET bucket_type = ? WHERE id = ?"
        self.conn.execute(sql, [new_bucket_type, memory_id])
        print(f"Moved memory ID {memory_id} to bucket '{new_bucket_type}'.")

    def delete_memory(self, memory_id: int):
        sql = "DELETE FROM memories WHERE id = ?"
        self.conn.execute(sql, [memory_id])
        # print(f"Deleted memory with ID: {memory_id}") # Reduced verbosity here, MemoryAgent can log

    def close(self):
        if hasattr(self, 'conn') and self.conn:
             self.conn.close()
             self.conn = None

    def save(self):
        try:
            if self.conn:
                 self.conn.execute("CHECKPOINT;")
                 # print("DuckDB checkpoint successful.") # Reduced verbosity
        except Exception as e:
            print(f"Error during DuckDB checkpoint: {e}")

    def __del__(self):
        self.close()
