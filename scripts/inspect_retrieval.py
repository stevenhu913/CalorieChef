"""Inspect accepted and rejected user-memory retrieval candidates."""

from long_term_memory import (
    DEFAULT_DISTANCE_THRESHOLD,
    get_user_id,
    retrieve_candidates,
    retrieve_context_memories,
)


def main() -> None:
    user_id = get_user_id()
    queries = [
        "What kind of lunch should I have?",
        "Explain quantum entanglement.",
    ]
    print(f"User scope: {user_id}")
    print(f"Acceptance rule: cosine distance <= {DEFAULT_DISTANCE_THRESHOLD}")
    for query in queries:
        print(f"\nQuery: {query}")
        context = retrieve_context_memories(query, user_id=user_id)
        print("  Final bounded context:")
        if not context:
            print("    No accepted memories.")
        for item in context:
            metadata = item["metadata"]
            print(f"    {item['id']} kind={metadata.get('kind')} value={metadata.get('value')}")
        print("  Raw semantic candidates:")
        candidates = retrieve_candidates(query, user_id=user_id)
        if not candidates:
            print("    No stored memories.")
        for item in candidates:
            metadata = item["metadata"]
            print(
                f"    {item['id']} kind={metadata.get('kind')} "
                f"distance={item['distance']:.4f} accepted={item['accepted']} "
                f"value={metadata.get('value')}"
            )


if __name__ == "__main__":
    main()
