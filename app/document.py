import boto3
import pymupdf
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    NoCredentialsError,
    NoRegionError,
    PartialCredentialsError,
)

from app.config import aws_region


def list_s3_documents(bucket):
    try:
        s3 = boto3.client("s3", region_name=aws_region())
        request = {"Bucket": bucket}
        keys = []
        while True:
            response = s3.list_objects_v2(**request)
            keys.extend(
                item["Key"]
                for item in response.get("Contents", [])
                if item["Key"].lower().endswith((".txt", ".pdf"))
            )
            if not response.get("IsTruncated"):
                return keys
            request["ContinuationToken"] = response["NextContinuationToken"]
    except NoCredentialsError as exc:
        raise RuntimeError(
            "AWS credentials were not found in the default credential chain"
        ) from exc
    except PartialCredentialsError as exc:
        raise RuntimeError("AWS credentials are incomplete") from exc
    except NoRegionError as exc:
        raise RuntimeError(
            "AWS region is not configured; set AWS_REGION or a default region"
        ) from exc
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        if code in {"AccessDenied", "AllAccessDisabled", "403"}:
            raise PermissionError(
                f"Access denied listing bucket {bucket}; "
                "the AWS identity needs s3:ListBucket permission"
            ) from exc
        raise RuntimeError(f"S3 object listing failed with AWS error code: {code}") from exc
    except BotoCoreError as exc:
        raise RuntimeError(
            f"AWS request could not be completed ({type(exc).__name__})"
        ) from exc


def read_s3_document(bucket, key):
    suffix = key.lower()
    if not suffix.endswith((".txt", ".pdf")):
        raise ValueError("Only .txt and .pdf S3 objects are supported")

    try:
        response = boto3.client("s3", region_name=aws_region()).get_object(
            Bucket=bucket,
            Key=key,
        )
        body = response["Body"]
        try:
            contents = body.read()
        finally:
            body.close()

        if suffix.endswith(".txt"):
            return contents.decode("utf-8")
        with pymupdf.open(stream=contents, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)
    except NoCredentialsError as exc:
        raise RuntimeError(
            "AWS credentials were not found in the default credential chain"
        ) from exc
    except PartialCredentialsError as exc:
        raise RuntimeError("AWS credentials are incomplete") from exc
    except NoRegionError as exc:
        raise RuntimeError(
            "AWS region is not configured; set AWS_REGION or a default region"
        ) from exc
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "Unknown")
        if code in {"NoSuchKey", "NoSuchBucket", "404"}:
            raise FileNotFoundError(
                f"S3 object s3://{bucket}/{key} was not found"
            ) from exc
        if code in {"AccessDenied", "AllAccessDisabled", "403"}:
            raise PermissionError(
                f"Access denied reading s3://{bucket}/{key}; "
                "the AWS identity needs s3:GetObject permission"
            ) from exc
        raise RuntimeError(f"S3 get_object failed with AWS error code: {code}") from exc
    except BotoCoreError as exc:
        raise RuntimeError(
            f"AWS request could not be completed ({type(exc).__name__})"
        ) from exc
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"S3 object s3://{bucket}/{key} is not valid UTF-8 text"
        ) from exc
    except pymupdf.FileDataError as exc:
        raise ValueError(
            f"S3 object s3://{bucket}/{key} is not a readable PDF"
        ) from exc
