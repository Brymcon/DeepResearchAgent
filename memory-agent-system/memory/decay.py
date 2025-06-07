import time

def calculate_decayed_importance(memory_data: dict, current_time: float, max_age_days: int = 30, relevance_boost_factor: float = 0.05, importance_floor: float = 0.1):
    """
    Calculates the new importance for a memory item and determines if it should be deleted.

    Args:
        memory_data: A dictionary containing memory attributes like:
            'id': int,
            'timestamp': datetime object (from DuckDB),
            'importance': float (current importance from previous calculation - unused in this model, recalculated),
            'access_count': int,
            'initial_decay_rate': float (e.g., 0.95, for per-day decay),
            'last_accessed_ts': datetime object (from DuckDB)
        current_time: float (current time as time.time())
        max_age_days: int (max age in days before forced deletion)
        relevance_boost_factor: float (small value added to importance per access_count)
        importance_floor: float (if importance drops below this, delete)

    Returns:
        A tuple: (new_importance: float, should_delete: bool)
        If should_delete is True, new_importance can be ignored.
    """
    # DuckDB returns datetime objects for TIMESTAMP columns
    creation_unix_ts = memory_data['timestamp'].timestamp()

    age_seconds = current_time - creation_unix_ts
    age_days = age_seconds / (24 * 3600)

    if age_days > max_age_days:
        return 0.0, True

    # Base importance purely on age and initial decay rate
    base_importance = memory_data['initial_decay_rate'] ** age_days

    # Additive boost for access count
    # This model recalculates total importance each time based on age and total accesses.
    access_boost = memory_data['access_count'] * relevance_boost_factor
    new_importance = base_importance + access_boost

    # Cap total importance to a maximum value (e.g., 1.5 or 2.0)
    # This prevents memories from becoming indefinitely important just due to frequent access.
    new_importance = min(new_importance, 1.5)

    if new_importance < importance_floor:
        return 0.0, True # Delete if importance is too low

    return new_importance, False
