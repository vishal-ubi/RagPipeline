import sys
from pathlib import Path

import psycopg

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from app.config import database_connection_settings
from app.database import migrate_document_chunks_schema


def main():
    try:
        settings = database_connection_settings()
    except KeyError as exc:
        print(f"Migration stopped: missing database setting {exc.args[0]}.")
        return 1

    try:
        with psycopg.connect(**settings, autocommit=False) as connection:
            result = migrate_document_chunks_schema(connection)
    except RuntimeError as exc:
        print(f"Migration stopped: {exc}")
        return 1
    except psycopg.Error as exc:
        print(
            "Migration failed; the transaction was rolled back "
            f"({type(exc).__name__})."
        )
        return 1

    if result == "already_migrated":
        print("document_chunks is already migrated; no data was changed.")
    else:
        print("document_chunks migration completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
