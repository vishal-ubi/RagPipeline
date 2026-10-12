import unittest
from copy import deepcopy
from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

from app.database import (
    LEGACY_SOURCE_DOCUMENT,
    migrate_document_chunks_schema,
    store_chunks,
)
from app.document import list_s3_documents, read_s3_document
from app.ingestion import ingest_text
import ingest_s3_text


LEGACY_COLUMNS = [
    ("id", "bigint", "int8", "NO"),
    ("chunk_index", "integer", "int4", "NO"),
    ("content", "text", "text", "NO"),
    ("embedding", "USER-DEFINED", "vector", "NO"),
]
LEGACY_CONSTRAINTS = [
    ("document_chunks_pkey", "PRIMARY KEY (id)"),
    ("document_chunks_chunk_index_key", "UNIQUE (chunk_index)"),
]
MIGRATED_COLUMNS = LEGACY_COLUMNS + [
    ("source_document", "text", "text", "NO"),
]
MIGRATED_CONSTRAINTS = [
    ("document_chunks_pkey", "PRIMARY KEY (id)"),
    (
        "document_chunks_source_chunk_key",
        "UNIQUE (source_document, chunk_index)",
    ),
]


class FakeResult:
    def __init__(self, rows=(), row=None):
        self._rows = rows
        self._row = row

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._row


class FakeConnection:
    def __init__(
        self,
        columns=LEGACY_COLUMNS,
        constraints=LEGACY_CONSTRAINTS,
        autocommit=False,
        has_hnsw_index=True,
        embedding_type="vector(384)",
    ):
        self.columns = columns
        self.constraints = constraints
        self.autocommit = autocommit
        self.has_hnsw_index = has_hnsw_index
        self.embedding_type = embedding_type
        self.statements = []
        self.calls = []

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        self.statements.append(normalized)
        self.calls.append((normalized, params))
        if "SELECT to_regclass('document_chunks')" in normalized:
            return FakeResult(row=("public.document_chunks",))
        if "information_schema.columns" in normalized:
            return FakeResult(rows=self.columns)
        if "SELECT format_type(atttypid, atttypmod)" in normalized:
            return FakeResult(row=(self.embedding_type,))
        if "SELECT EXISTS (" in normalized:
            return FakeResult(row=(self.has_hnsw_index,))
        if "FROM pg_constraint" in normalized:
            return FakeResult(rows=self.constraints)
        return FakeResult()


class FakeStorageConnection(FakeConnection):
    def __init__(self, rows, fail_on_insert=None):
        super().__init__(
            columns=MIGRATED_COLUMNS,
            constraints=MIGRATED_CONSTRAINTS,
        )
        self.rows = list(rows)
        self.fail_on_insert = fail_on_insert
        self.insert_count = 0

    def __enter__(self):
        self.saved_rows = deepcopy(self.rows)
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is not None:
            self.rows = self.saved_rows
        return False

    def execute(self, sql, params=None):
        result = super().execute(sql, params)
        normalized = " ".join(sql.split())
        if normalized == "DELETE FROM document_chunks":
            self.rows.clear()
        elif normalized.startswith("INSERT INTO document_chunks"):
            self.insert_count += 1
            if self.insert_count == self.fail_on_insert:
                raise RuntimeError("mock insert failed")
            self.rows.append(params)
        return result


class MigrationSafetyTests(unittest.TestCase):
    def test_autocommit_is_rejected_before_any_sql(self):
        connection = FakeConnection(autocommit=True)

        with self.assertRaisesRegex(RuntimeError, "non-autocommit"):
            migrate_document_chunks_schema(connection)

        self.assertEqual(connection.statements, [])

    def test_legacy_schema_migrates_in_order_without_deleting_rows(self):
        connection = FakeConnection()

        result = migrate_document_chunks_schema(connection)

        self.assertEqual(result, "migrated")
        lock_position = next(
            index
            for index, statement in enumerate(connection.statements)
            if statement.startswith("LOCK TABLE")
        )
        inspection_position = next(
            index
            for index, statement in enumerate(connection.statements)
            if "information_schema.columns" in statement
        )
        self.assertLess(lock_position, inspection_position)
        alterations = [
            statement
            for statement in connection.statements
            if statement.startswith("ALTER TABLE")
        ]
        self.assertEqual(
            alterations,
            [
                "ALTER TABLE document_chunks ADD COLUMN source_document TEXT",
                (
                    "ALTER TABLE document_chunks ALTER COLUMN "
                    "source_document SET NOT NULL"
                ),
                (
                    "ALTER TABLE document_chunks ADD CONSTRAINT "
                    "document_chunks_source_chunk_key UNIQUE "
                    "(source_document, chunk_index)"
                ),
                (
                    "ALTER TABLE document_chunks DROP CONSTRAINT "
                    "document_chunks_chunk_index_key"
                ),
            ],
        )
        self.assertIn(
            "UPDATE document_chunks SET source_document = %s "
            "WHERE source_document IS NULL",
            connection.statements,
        )
        self.assertIn(
            (
                "UPDATE document_chunks SET source_document = %s "
                "WHERE source_document IS NULL",
                (LEGACY_SOURCE_DOCUMENT,),
            ),
            connection.calls,
        )
        self.assertFalse(
            any(
                "DELETE FROM document_chunks" in statement
                for statement in connection.statements
            )
        )

    def test_already_migrated_schema_is_a_no_op(self):
        connection = FakeConnection(
            columns=MIGRATED_COLUMNS,
            constraints=MIGRATED_CONSTRAINTS,
        )

        result = migrate_document_chunks_schema(connection)

        self.assertEqual(result, "already_migrated")
        self.assertFalse(
            any(
                statement.startswith(("ALTER TABLE", "UPDATE "))
                for statement in connection.statements
            )
        )

    def test_unexpected_column_definition_fails_before_alter(self):
        columns = list(LEGACY_COLUMNS)
        columns[1] = ("chunk_index", "integer", "int4", "YES")
        connection = FakeConnection(columns=columns)

        with self.assertRaisesRegex(RuntimeError, "types or nullability"):
            migrate_document_chunks_schema(connection)

        self.assertFalse(
            any(
                statement.startswith("ALTER TABLE")
                for statement in connection.statements
            )
        )

    def test_unexpected_constraint_fails_before_alter(self):
        constraints = list(LEGACY_CONSTRAINTS)
        constraints[1] = ("unexpected_unique_key", "UNIQUE (chunk_index)")
        connection = FakeConnection(constraints=constraints)

        with self.assertRaisesRegex(RuntimeError, "unexpected constraints"):
            migrate_document_chunks_schema(connection)

        self.assertFalse(
            any(
                statement.startswith("ALTER TABLE")
                for statement in connection.statements
            )
        )

    def test_wrong_embedding_dimensions_fail_before_alter(self):
        connection = FakeConnection(embedding_type="vector(768)")

        with self.assertRaisesRegex(RuntimeError, "vector\\(384\\)"):
            migrate_document_chunks_schema(connection)

        self.assertFalse(
            any(
                statement.startswith("ALTER TABLE")
                for statement in connection.statements
            )
        )

    def test_malformed_hnsw_index_fails_before_alter(self):
        connection = FakeConnection(has_hnsw_index=False)

        with self.assertRaisesRegex(RuntimeError, "HNSW index"):
            migrate_document_chunks_schema(connection)

        self.assertFalse(
            any(
                statement.startswith("ALTER TABLE")
                for statement in connection.statements
            )
        )

    def test_s3_listing_returns_only_txt_and_pdf_keys(self):
        s3 = MagicMock()
        s3.list_objects_v2.side_effect = [
            {
                "Contents": [
                    {"Key": "docs/one.txt"},
                    {"Key": "docs/image.png"},
                ],
                "IsTruncated": True,
                "NextContinuationToken": "next-page",
            },
            {
                "Contents": [
                    {"Key": "docs/two.PDF"},
                    {"Key": "notes.md"},
                ],
                "IsTruncated": False,
            },
        ]

        with patch("app.document.boto3.client", return_value=s3):
            keys = list_s3_documents("test-bucket")

        self.assertEqual(keys, ["docs/one.txt", "docs/two.PDF"])
        self.assertEqual(s3.list_objects_v2.call_count, 2)
        self.assertEqual(
            s3.list_objects_v2.call_args_list[1].kwargs["ContinuationToken"],
            "next-page",
        )

    def test_s3_txt_object_is_decoded_as_utf8(self):
        body = BytesIO("Some text".encode("utf-8"))
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": body}
        with patch(
            "app.document.boto3.client",
            return_value=s3,
        ):
            text = read_s3_document("test-bucket", "docs/file.txt")

        self.assertEqual(text, "Some text")
        self.assertTrue(body.closed)

    def test_s3_pdf_object_extracts_page_text(self):
        body = BytesIO(b"mock pdf bytes")
        pdf = MagicMock()
        pdf.__enter__.return_value = [
            MagicMock(get_text=lambda: "first page"),
            MagicMock(get_text=lambda: "second page"),
        ]
        s3 = MagicMock()
        s3.get_object.return_value = {"Body": body}
        with (
            patch(
                "app.document.boto3.client",
                return_value=s3,
            ),
            patch("app.document.pymupdf.open", return_value=pdf) as open_pdf,
        ):
            text = read_s3_document("test-bucket", "docs/file.pdf")

        self.assertEqual(text, "first page\nsecond page")
        open_pdf.assert_called_once_with(
            stream=b"mock pdf bytes",
            filetype="pdf",
        )

    def test_s3_ingestion_uses_exact_selected_key_and_uri(self):
        with (
            patch("ingest_s3_text.S3_BUCKET_NAME", "test-bucket"),
            patch(
                "ingest_s3_text.list_s3_documents",
                return_value=["policies/refunds.pdf"],
            ),
            patch("builtins.input", return_value="policies/refunds.pdf"),
            patch(
                "ingest_s3_text.read_s3_document",
                return_value="refund policy text",
            ) as read_document,
            patch(
                "ingest_s3_text.ingest_text",
                return_value=(["refund policy text"], [[0.0] * 384]),
            ) as ingest,
        ):
            result = ingest_s3_text.main()

        self.assertEqual(result, 0)
        read_document.assert_called_once_with(
            "test-bucket",
            "policies/refunds.pdf",
        )
        ingest.assert_called_once_with(
            "refund policy text",
            source_document="s3://test-bucket/policies/refunds.pdf",
        )

    def test_unlisted_s3_key_is_rejected_before_download_or_storage(self):
        with (
            patch("ingest_s3_text.S3_BUCKET_NAME", "test-bucket"),
            patch(
                "ingest_s3_text.list_s3_documents",
                return_value=["policies/refunds.txt"],
            ),
            patch("builtins.input", return_value="other.txt"),
            patch("ingest_s3_text.read_s3_document") as read_document,
            patch("ingest_s3_text.ingest_text") as ingest,
        ):
            result = ingest_s3_text.main()

        self.assertEqual(result, 1)
        read_document.assert_not_called()
        ingest.assert_not_called()

    def test_empty_s3_document_is_rejected_without_storage(self):
        with (
            patch("ingest_s3_text.S3_BUCKET_NAME", "test-bucket"),
            patch(
                "ingest_s3_text.list_s3_documents",
                return_value=["empty.txt"],
            ),
            patch("builtins.input", return_value="empty.txt"),
            patch("ingest_s3_text.read_s3_document", return_value=" \n "),
            patch("app.ingestion.store_chunks") as save_chunks,
        ):
            result = ingest_s3_text.main()

        self.assertEqual(result, 1)
        save_chunks.assert_not_called()

    def test_empty_store_is_rejected_before_connecting(self):
        with patch("app.database._connect_to_database") as connect:
            with self.assertRaisesRegex(ValueError, "no chunks"):
                store_chunks([], [])

        connect.assert_not_called()

    def test_empty_ingestion_is_rejected_before_embedding_or_storage(self):
        with (
            patch("app.ingestion.generate_embeddings") as generate_embeddings,
            patch("app.ingestion.store_chunks") as save_chunks,
        ):
            with self.assertRaisesRegex(ValueError, "no extractable text"):
                ingest_text("", source_document="s3://bucket/empty.txt")

        generate_embeddings.assert_not_called()
        save_chunks.assert_not_called()

    def test_local_source_identifier_is_rejected(self):
        with (
            patch("app.ingestion.generate_embeddings") as generate_embeddings,
            patch("app.ingestion.store_chunks") as save_chunks,
        ):
            with self.assertRaisesRegex(ValueError, "must be an S3 URI"):
                ingest_text("document text", source_document="local:document.txt")

        generate_embeddings.assert_not_called()
        save_chunks.assert_not_called()

    def test_whitespace_only_document_is_rejected_before_embedding(self):
        with patch("app.ingestion.generate_embeddings") as generate_embeddings:
            with self.assertRaisesRegex(ValueError, "no extractable text"):
                ingest_text(
                    " \n\t ",
                    source_document="s3://bucket/empty.txt",
                )

        generate_embeddings.assert_not_called()

    def test_replacing_document_removes_old_rows_and_uses_new_source(self):
        old_rows = [
            ("s3://bucket/old.txt", 0, "old 0"),
            ("s3://bucket/old.txt", 1, "old 1"),
            ("s3://bucket/old.txt", 2, "old 2"),
        ]
        connection = FakeStorageConnection(old_rows)

        with (
            patch(
                "app.database._connect_to_database",
                return_value=connection,
            ),
            patch("app.database.register_vector"),
        ):
            store_chunks(
                ["new 0", "new 1"],
                [[0.0] * 384, [0.1] * 384],
                source_document="s3://bucket/new.txt",
            )

        self.assertEqual(
            [(row[0], row[1], row[2]) for row in connection.rows],
            [
                ("s3://bucket/new.txt", 0, "new 0"),
                ("s3://bucket/new.txt", 1, "new 1"),
            ],
        )

    def test_renamed_document_replaces_previous_source(self):
        connection = FakeStorageConnection(
            [("s3://bucket/before-rename.txt", 0, "same content")]
        )

        with (
            patch(
                "app.database._connect_to_database",
                return_value=connection,
            ),
            patch("app.database.register_vector"),
        ):
            store_chunks(
                ["same content"],
                [[0.0] * 384],
                source_document="s3://bucket/after-rename.txt",
            )

        self.assertEqual(
            [(row[0], row[1], row[2]) for row in connection.rows],
            [("s3://bucket/after-rename.txt", 0, "same content")],
        )

    def test_insert_failure_rolls_back_to_previous_document(self):
        old_rows = [
            ("s3://bucket/old.txt", 0, "old content"),
        ]
        connection = FakeStorageConnection(old_rows, fail_on_insert=2)

        with (
            patch(
                "app.database._connect_to_database",
                return_value=connection,
            ),
            patch("app.database.register_vector"),
        ):
            with self.assertRaisesRegex(RuntimeError, "mock insert failed"):
                store_chunks(
                    ["replacement 0", "replacement 1"],
                    [[0.0] * 384, [0.1] * 384],
                    source_document="s3://bucket/new.txt",
                )

        self.assertEqual(connection.rows, old_rows)

    def test_embedding_failure_does_not_call_storage(self):
        with (
            patch(
                "app.ingestion.generate_embeddings",
                side_effect=RuntimeError("mock embedding failure"),
            ),
            patch("app.ingestion.store_chunks") as save_chunks,
        ):
            with self.assertRaisesRegex(RuntimeError, "mock embedding failure"):
                ingest_text(
                    "A valid document.",
                    source_document="s3://bucket/document.txt",
                )

        save_chunks.assert_not_called()


if __name__ == "__main__":
    unittest.main()
