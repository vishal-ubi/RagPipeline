import os

from dotenv import load_dotenv


load_dotenv()


MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384
BEDROCK_MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "apac.amazon.nova-lite-v1:0"
)
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME")
EVALUATION_QUESTIONS = [
{
"question": "What is the company's refund policy?",
"expected_chunk_index": 0,
},
{
"question": "How long are refunds normally processed?",
"expected_chunk_index": 1,
},
{
"question": "What must customers provide when requesting a refund?",
"expected_chunk_index": 3,
},
]



def database_connection_settings():
    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


def aws_region():
    return os.environ.get("AWS_REGION")
