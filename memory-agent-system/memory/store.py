import duckdb
import numpy as np
import time
import os
import json # For handling tags list if stored as JSON string

# Define the path for the DuckDB database file
DB_DIR = "memory_db"
DB_PATH = os.path.join(DB_DIR, "memory.duckdb")

class VectorMemory:
    def __init__(self, db_path=DB_PATH, dim=384):
        self.db_path = db_path
        self.dim = dim # Expected dimension of embeddings

        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        self.conn = duckdb.connect(database=self.db_path, read_only=False)
        try:
            self.conn.execute("INSTALL hnsw;")
            self.conn.execute("LOAD hnsw;")
            self.conn.execute("INSTALL json;")
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
            last_accessed_ts TIMESTAMP
        );
        """
        self.conn.execute(create_table_sql)

        max_id_result = self.conn.execute("SELECT MAX(id) FROM memories").fetchone()
        self.counter = (max_id_result[0] if max_id_result[0] is not None else -1) + 1

    def add_memory(self, embedding: np.ndarray, content: str, tags: list[str] = None,
                   source: str = "unknown", certainty: float = 1.0,
                   initial_decay_rate: float = 0.95, memory_type: str = "fact"): # memory_type can go into tags or source
        current_time = time.time()
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
        last_accessed_ts = timestamp
        embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else list(embedding)
        if len(embedding_list) != self.dim:
            raise ValueError(f"Embedding dimension mismatch. Expected {self.dim}, got {len(embedding_list)}")
        memory_id = self.counter
        initial_importance = 1.0
        sql = """
        INSERT INTO memories (
            id, timestamp, content, embedding, tags, source, certainty,
            initial_decay_rate, access_count, importance, last_accessed_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        self.conn.execute(sql, [
            memory_id, timestamp, content, embedding_list, tags or [], source, certainty,
            initial_decay_rate, 0, initial_importance, last_accessed_ts
        ])
        self.counter += 1
        return memory_id

    def retrieve(self, query_embedding: np.ndarray, k: int = 5, threshold: float = None):
        query_embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else list(query_embedding)
        if len(query_embedding_list) != self.dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.dim}, got {len(query_embedding_list)}")
        sql = f"""
        SELECT id, content, tags, source, certainty, importance, timestamp, last_accessed_ts, access_count,
               list_distance(embedding, CAST(? AS FLOAT[{self.dim}])) AS distance
        FROM memories
        ORDER BY distance ASC
        LIMIT ?;
        """
        results_raw = self.conn.execute(sql, [query_embedding_list, k]).fetchall()
        retrieved_memories = []
        ids_to_update_access = []
        current_time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        for row in results_raw:
            distance = row[9]
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
                "distance": distance
            })
            ids_to_update_access.append(row[0])
        if ids_to_update_access:
            update_sql = "UPDATE memories SET access_count = access_count + 1, last_accessed_ts = ? WHERE id = ?"
            for mem_id in ids_to_update_access:
                self.conn.execute(update_sql, [current_time_str, mem_id])
        return retrieved_memories

    def get_memory_by_id(self, memory_id: int):
        sql = "SELECT id, content, embedding, tags, source, certainty, importance, timestamp, access_count, initial_decay_rate, last_accessed_ts FROM memories WHERE id = ?"
        result = self.conn.execute(sql, [memory_id]).fetchone()
        if result:
            return {
                "id": result[0], "content": result[1], "embedding": np.array(result[2]),
                "tags": result[3], "source": result[4], "certainty": result[5],
                "importance": result[6], "timestamp": result[7], "access_count": result[8],
                "initial_decay_rate": result[9], "last_accessed_ts": result[10]
            }
        return None

    def get_all_memories_for_decay(self):
        sql = "SELECT id, timestamp, importance, access_count, initial_decay_rate, last_accessed_ts FROM memories"
        return self.conn.execute(sql).fetchall()

    def update_memory_importance(self, memory_id: int, new_importance: float):
        sql = "UPDATE memories SET importance = ? WHERE id = ?"
        self.conn.execute(sql, [new_importance, memory_id])

    def delete_memory(self, memory_id: int):
        sql = "DELETE FROM memories WHERE id = ?"
        self.conn.execute(sql, [memory_id])
        print(f"Deleted memory with ID: {memory_id}")

    def close(self):
        if hasattr(self, 'conn') and self.conn:
             self.conn.close()
             self.conn = None # Mark as closed

    def save(self):
        try:
            if self.conn:
                 self.conn.execute("CHECKPOINT;")
                 print("DuckDB checkpoint successful.")
        except Exception as e:
            print(f"Error during DuckDB checkpoint: {e}")

    def __del__(self):
        self.close()
