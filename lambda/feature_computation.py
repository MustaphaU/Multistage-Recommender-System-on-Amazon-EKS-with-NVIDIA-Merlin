"""
feature_computation.py — SQS-triggered Lambda for updating behavioral user
features in the Feast online store AND the S3 offline store.

Triggered by: SQS queue fed by the mark_seen Lambda.
Writes to:    Feast online store (Redis) via write_to_online_store.
              S3 offline store   (feast/behavioral_updates/dt=YYYY-MM-DD/)
              Both writes together eliminate training/serving skew for top_category.

SQS message format:
    {"user_id": int}                     # recompute top_category from sorted set
    {"user_id": int, "flush": true}      # reset top_category to OOV on flush

Environment variables:
    REDIS_HOST         — ElastiCache hostname, e.g. master.xxx.cache.amazonaws.com
    DYNAMO_TABLE       — DynamoDB table for item metadata
    FEAST_S3_BUCKET    — S3 bucket holding the Feast registry and behavioral updates
    FEAST_AWS_REGION   — AWS region
"""

import io
import json
import logging
import os
import uuid
from collections import Counter
from datetime import datetime, timedelta

import boto3
import pandas as pd
import redis
from feast import FeatureStore
from feast.repo_config import RepoConfig, RegistryConfig
from feast.infra.online_stores.redis import RedisOnlineStoreConfig

logger = logging.getLogger()
logger.setLevel(logging.INFO)

REDIS_HOST       = os.environ.get("REDIS_HOST") or os.environ["REDIS_URL"]
DYNAMO_TABLE     = os.environ["DYNAMO_TABLE"]
FEAST_S3_BUCKET  = os.environ["FEAST_S3_BUCKET"]
FEAST_AWS_REGION = os.environ["FEAST_AWS_REGION"]
ONE_DAY_SECS  = 24 * 60 * 60

_redis_client = None
_store        = None
_s3_client    = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=1, ssl=True, decode_responses=True)
    return _redis_client


def _get_store():
    global _store
    if _store is None:
        config = RepoConfig(
            project="my_feast_repo",
            provider="aws",
            registry=RegistryConfig(
                path=f"s3://{FEAST_S3_BUCKET}/feast/registry.db",
            ),
            online_store=RedisOnlineStoreConfig(
                connection_string=f"{REDIS_HOST}:6379,db=0,ssl=true"
            ),
            entity_key_serialization_version=2,
        )
        _store = FeatureStore(config=config)
    return _store


def _get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=FEAST_AWS_REGION)
    return _s3_client


def _write_top_category(user_id: int, top_category: int) -> dict:
    existing = _get_store().get_online_features(
        features=["user_features:age", "user_features:gender"],
        entity_rows=[{"user_id": user_id}],
    ).to_dict()
    now = pd.Timestamp.now(tz="UTC")
    row = {
        "user_id":      user_id,
        "age":          existing.get("age", [None])[0],
        "gender":       existing.get("gender", [None])[0],
        "top_category": top_category,
        "datetime":     now,
        "created":      now,
    }
    _get_store().write_to_online_store(feature_view_name="user_features", df=pd.DataFrame([row]))
    logger.info("user_id=%d  top_category=%d", user_id, top_category)
    return row


def _flush_to_s3(rows: list[dict]):
    """Write a batch of behavioral feature rows to the S3 offline store."""
    if not rows:
        return
    buf = io.BytesIO()
    pd.DataFrame(rows).to_parquet(buf, index=False)
    buf.seek(0)
    date_str = datetime.now().strftime("%Y-%m-%d")
    key = f"feast/behavioral_updates/dt={date_str}/{uuid.uuid4()}.parquet"
    _get_s3().put_object(Bucket=FEAST_S3_BUCKET, Key=key, Body=buf.getvalue())
    logger.info("wrote %d row(s) to s3://%s/%s", len(rows), FEAST_S3_BUCKET, key)


def _compute_top_category(user_id: int) -> int:
    cutoff = int((datetime.now() - timedelta(seconds=ONE_DAY_SECS)).timestamp() * 1_000_000)
    recent_ids = [
        int(i) for i in
        _get_redis().zrangebyscore(f"user:{user_id}:recent_items", cutoff, "+inf")
    ]
    if not recent_ids:
        return -1

    feast_result = _get_store().get_online_features(
        features=["item_features:category_l1"],
        entity_rows=[{"item_id": iid} for iid in recent_ids],
    ).to_dict()

    cats = [c for c in feast_result.get("category_l1", []) if c is not None]
    if not cats:
        return -1

    counts = Counter(cats)
    max_count = max(counts.values())
    tied_categories = {category for category, count in counts.items() if count == max_count}

    for category in reversed(cats):
        if category in tied_categories:
            return int(category)

    return -1


def lambda_handler(event, context):
    rows = []
    for record in event["Records"]:
        try:
            body    = json.loads(record["body"])
            user_id = int(body["user_id"])

            if body.get("flush"):
                pass  # Bloom filter and sorted set already cleared; retain existing top_category
            else:
                top_category = _compute_top_category(user_id)
                if top_category != -1:
                    row = _write_top_category(user_id, top_category)
                    rows.append(row)

        except Exception:
            logger.exception("Failed processing record: %s", record.get("body"))
            raise  # let SQS retry

    _flush_to_s3(rows)
