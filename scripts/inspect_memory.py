"""List or delete the current user's long-term memories without showing embeddings."""

import argparse

from long_term_memory import forget_memory, get_user_id, list_memories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forget", metavar="TOPIC", help="Delete one memory by its topic.")
    args = parser.parse_args()
    user_id = get_user_id()
    if args.forget:
        deleted = forget_memory(args.forget, user_id)
        print(f"Deleted topic '{args.forget}'." if deleted else f"Topic '{args.forget}' was not found.")
    records = list_memories(user_id)
    print(f"User scope: {user_id}")
    print(f"Active memories: {len(records)}")
    for record in records:
        print(
            f"- id={record['id']} topic={record['topic']} value={record['value']} "
            f"kind={record['kind']} source={record['source']} version={record['version']} "
            f"updated_at={record['updated_at']} source_turn={record['source_turn']}"
        )


if __name__ == "__main__":
    main()
