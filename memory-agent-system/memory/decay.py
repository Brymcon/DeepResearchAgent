import time

def apply_memory_decay(memories, decay_rate=0.95, max_age_days=30):
    current_time = time.time()
    decayed_memories = []

    for memory in memories:
        # Calculate age in days
        age_days = (current_time - memory['timestamp']) / (24 * 3600)

        if age_days > max_age_days:
            continue  # Remove old memories

        # Apply exponential decay
        decay_factor = decay_rate ** age_days
        memory['importance'] = memory.get('importance', 1.0) * decay_factor
        decayed_memories.append(memory)

    return decayed_memories
