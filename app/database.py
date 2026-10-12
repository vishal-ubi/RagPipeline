import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.config import EMBEDDING_DIMENSIONS, database_connection_settings


LEGACY_SOURCE_DOCUMENT = "legacy://document_chunks"
LOCAL_SOURCE_DOCUMENT = LEGACY_SOURCE_DOCUMENT


def _connect_to_database():
    return psycopg.connect(**database_connection_settings())


def _inspect_document_chunks_schema(connection, require_hnsw_index=False):
    columns = connection.execute(
        """
        SELECT column_name, data_type, udt_name, is_nullable
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'document_chunks'
        """
    ).fetchall()
    column_info = {row[0]: (row[1], row[2], row[3]) for row in columns}
    legacy_columns = {"id", "chunk_index", "content", "embedding"}
    migrated_columns = legacy_columns | {"source_document"}
    if set(column_info) not in (legacy_columns, migrated_columns):
        raise RuntimeError(
            "document_chunks has unexpected columns; migration stopped"
        )
    has_source_document = "source_document" in column_info
    expected_columns = {
        "id": ("bigint", "int8", "NO"),
        "chunk_index": ("integer", "int4", "NO"),
        "content": ("text", "text", "NO"),
        "embedding": ("USER-DEFINED", "vector", "NO"),
    }
    if has_source_document:
        expected_columns["source_document"] = ("text", "text", "NO")
    if column_info != expected_columns:
        raise RuntimeError(
            "document_chunks has unexpected column types or nullability; "
            "migration stopped"
        )

    embedding_type = connection.execute(
        """
        SELECT format_type(atttypid, atttypmod)
        FROM pg_attribute
        WHERE attrelid = 'document_chunks'::regclass
          AND attname = 'embedding'
          AND NOT attisdropped
        """
    ).fetchone()
    if embedding_type is None or embedding_type[0] != "vector(384)":
        raise RuntimeError(
            "document_chunks.embedding must use vector(384); migration stopped"
        )

    if require_hnsw_index:
        has_hnsw_index = connection.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_index
                JOIN pg_class AS table_class
                  ON table_class.oid = pg_index.indrelid
                JOIN pg_class AS index_class
                  ON index_class.oid = pg_index.indexrelid
                JOIN pg_am
                  ON pg_am.oid = index_class.relam
                JOIN pg_attribute
                  ON pg_attribute.attrelid = table_class.oid
                 AND pg_attribute.attnum = pg_index.indkey[0]
                JOIN pg_opclass
                  ON pg_opclass.oid = pg_index.indclass[0]
                WHERE table_class.oid = 'document_chunks'::regclass
                  AND index_class.oid =
                      to_regclass('document_chunks_embedding_hnsw_cosine_idx')
                  AND pg_am.amname = 'hnsw'
                  AND NOT pg_index.indisunique
                  AND pg_index.indnkeyatts = 1
                  AND pg_index.indnatts = 1
                  AND pg_index.indpred IS NULL
                  AND NOT pg_index.indisprimary
                  AND pg_index.indisvalid
                  AND pg_index.indisready
                  AND pg_attribute.attname = 'embedding'
                  AND pg_opclass.opcname = 'vector_cosine_ops'
            )
            """
        ).fetchone()[0]
        if not has_hnsw_index:
            raise RuntimeError(
                "document_chunks is missing the expected HNSW index; "
                "migration stopped"
            )

    constraints = connection.execute(
        """
        SELECT conname, pg_get_constraintdef(oid)
        FROM pg_constraint
        WHERE conrelid = 'document_chunks'::regclass
        """
    ).fetchall()
    constraint_definitions = {
        name: " ".join(definition.split()).upper()
        for name, definition in constraints
    }

    if has_source_document:
        expected_constraints = {
            "document_chunks_pkey": "PRIMARY KEY (ID)",
            "document_chunks_source_chunk_key": (
                "UNIQUE (SOURCE_DOCUMENT, CHUNK_INDEX)"
            ),
        }
        schema_state = "migrated"
    else:
        expected_constraints = {
            "document_chunks_pkey": "PRIMARY KEY (ID)",
            "document_chunks_chunk_index_key": "UNIQUE (CHUNK_INDEX)",
        }
        schema_state = "legacy"
    if constraint_definitions != expected_constraints:
        raise RuntimeError(
            "document_chunks has unexpected constraints; migration stopped"
        )
    return schema_state


def _apply_document_chunks_migration(connection):
    connection.execute(
        "ALTER TABLE document_chunks ADD COLUMN source_document TEXT"
    )
    connection.execute(
        """
        UPDATE document_chunks
        SET source_document = %s
        WHERE source_document IS NULL
        """,
        (LEGACY_SOURCE_DOCUMENT,),
    )
    connection.execute(
        """
        ALTER TABLE document_chunks
        ALTER COLUMN source_document SET NOT NULL
        """
    )
    connection.execute(
        """
        ALTER TABLE document_chunks
        ADD CONSTRAINT document_chunks_source_chunk_key
        UNIQUE (source_document, chunk_index)
        """
    )
    connection.execute(
        """
        ALTER TABLE document_chunks
        DROP CONSTRAINT document_chunks_chunk_index_key
        """
    )


def migrate_document_chunks_schema(connection, require_hnsw_index=True):
    if connection.autocommit:
        raise RuntimeError(
            "document_chunks migration requires a non-autocommit transaction"
        )

    if connection.execute(
        "SELECT to_regclass('document_chunks')"
    ).fetchone()[0] is None:
        raise RuntimeError("document_chunks table does not exist; migration stopped")

    connection.execute("LOCK TABLE document_chunks IN ACCESS EXCLUSIVE MODE")
    schema_state = _inspect_document_chunks_schema(
        connection,
        require_hnsw_index=require_hnsw_index,
    )
    if schema_state == "migrated":
        return "already_migrated"

    _apply_document_chunks_migration(connection)
    return "migrated"


def _validate_document_chunks_schema(connection):
    schema_state = _inspect_document_chunks_schema(
        connection,
        require_hnsw_index=False,
    )
    if schema_state == "legacy":
        raise RuntimeError(
            "document_chunks uses the legacy schema; run "
            "python scripts/migrate_document_chunks.py before ingestion"
        )


def store_chunks(
    chunks,
    embeddings,
    source_document=LOCAL_SOURCE_DOCUMENT,
):
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not chunks:
        raise ValueError("cannot replace the active document with no chunks")
    if not source_document:
        raise ValueError("source_document must not be empty")

    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"each embedding must have {EMBEDDING_DIMENSIONS} dimensions"
            )

    # PostgreSQL stores each chunk; pgvector provides the VECTOR column type.
    with _connect_to_database() as connection:
        table_exists = connection.execute(
            "SELECT to_regclass('document_chunks')"
        ).fetchone()[0]
        if not table_exists:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                """
                CREATE TABLE document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    source_document TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding VECTOR(384) NOT NULL,
                    CONSTRAINT document_chunks_source_chunk_key
                        UNIQUE (source_document, chunk_index)
                )
                """
            )
        _validate_document_chunks_schema(connection)

        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(connection)

        # HNSW helps pgvector find nearby vectors using cosine distance.
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_cosine_idx
            ON document_chunks
            USING hnsw (embedding vector_cosine_ops)
            """
        )

        # The connection context commits this delete and all inserts together.
        # If any insert fails, PostgreSQL rolls the whole transaction back.
        connection.execute("DELETE FROM document_chunks")
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            connection.execute(
                """
                INSERT INTO document_chunks (
                    source_document,
                    chunk_index,
                    content,
                    embedding
                )
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_document, chunk_index) DO UPDATE
                SET content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
                """,
                (source_document, index, chunk, Vector(embedding)),
            )
