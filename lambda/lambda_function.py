"""
lambda_function.py — AWS Lambda handler for the Multistage Recommender System.

Deploy this Lambda in the same VPC as your ElastiCache cluster.
Triton must be reachable via an internal LoadBalancer or NLB DNS name.

Environment variables (set in Lambda console or Terraform):
    TRITON_HOST   — internal Triton gRPC endpoint, e.g. triton.internal:8001
    REDIS_HOST / REDIS_URL — ElastiCache hostname, e.g. master.xxx.cache.amazonaws.com
    DYNAMO_TABLE  — DynamoDB table name for item metadata, e.g. items

Request body fields:
    user_id      — (int, required) label-encoded user ID
    device_type  — (int, default -1) 0=Mobile, 1=Desktop, 2=Tablet;
                    any other value is normalized to -1
    timestamp    — (int, optional) unix epoch; defaults to current time on server
    top_k        — (int, default 10) number of recommendations to return
    flush        — (bool, optional) clear the seen-items Bloom filter for this user
    mark_seen    — (list[int], optional) item_ids to mark as seen in the Bloom filter
"""

import datetime
import json
import os

import boto3
import numpy as np
import redis
import tritonclient.grpc as grpcclient

TRITON_HOST   = os.environ["TRITON_HOST"]
REDIS_HOST    = os.environ.get("REDIS_HOST") or os.environ["REDIS_URL"]
DYNAMO_TABLE  = os.environ["DYNAMO_TABLE"]
SQS_QUEUE_URL = os.environ["SQS_QUEUE_URL"]

# Reuse connections across warm invocations
_triton_client = None
_redis_client  = None
_dynamo_table  = None
_sqs_client    = None


def _get_triton_client():
    global _triton_client
    if _triton_client is None:
        _triton_client = grpcclient.InferenceServerClient(url=TRITON_HOST)
    return _triton_client


def _get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=6379, db=1, ssl=True, decode_responses=True)
    return _redis_client


def _get_dynamo_table():
    global _dynamo_table
    if _dynamo_table is None:
        _dynamo_table = boto3.resource("dynamodb").Table(DYNAMO_TABLE)
    return _dynamo_table


def _get_sqs_client():
    global _sqs_client
    if _sqs_client is None:
        _sqs_client = boto3.client("sqs")
    return _sqs_client


def _respond(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


def lambda_handler(event, context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _respond(400, {"error": "Invalid JSON body"})

    user_id     = body.get("user_id")
    raw_device_type = body.get("device_type")
    try:
        device_type = int(raw_device_type) if raw_device_type is not None else -1
    except (TypeError, ValueError):
        device_type = -1
    if device_type not in {0, 1, 2}:
        device_type = -1
    timestamp   = body.get("timestamp", None)
    top_k       = int(body.get("top_k", 10))
    flush       = bool(body.get("flush", False))
    mark_seen   = body.get("mark_seen", None)  # list of item_ids to mark as seen

    if user_id is None:
        return _respond(400, {"error": "user_id is required"})

    user_id = int(user_id)

    r = _get_redis_client()

    if flush:
        pipe = r.pipeline()
        pipe.delete(f"bf:seen:{user_id}")
        pipe.delete(f"user:{user_id}:recent_items")
        pipe.execute()
        _get_sqs_client().send_message(
            QueueUrl=SQS_QUEUE_URL,
            MessageBody=json.dumps({"user_id": user_id, "flush": True}),
        )
        return _respond(200, {"message": f"Flushed seen state for user {user_id}"})

    # Mark specific items as seen (called when user interacts with a recommendation)
    if mark_seen is not None:
        if mark_seen:
            now = int(datetime.datetime.now().timestamp())
            pipe = r.pipeline()
            pipe.execute_command("BF.MADD", f"bf:seen:{user_id}", *[int(i) for i in mark_seen])
            pipe.expire(f"bf:seen:{user_id}", 7 * 24 * 60 * 60)
            for item_id in mark_seen:
                pipe.zadd(f"user:{user_id}:recent_items", {str(item_id): now})
            pipe.expire(f"user:{user_id}:recent_items", 7 * 24 * 60 * 60)
            pipe.execute()
            _get_sqs_client().send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps({"user_id": user_id}),
            )
        return _respond(200, {"message": f"Marked {len(mark_seen)} item(s) as seen for user {user_id}"})

    # Build Triton inputs
    client = _get_triton_client()

    user_id_input = grpcclient.InferInput("user_id", [1], "INT32")
    user_id_input.set_data_from_numpy(np.array([user_id], dtype=np.int32))

    device_input = grpcclient.InferInput("device_type", [1], "INT32")
    device_input.set_data_from_numpy(np.array([device_type], dtype=np.int32))

    inputs = [user_id_input, device_input]

    if timestamp is not None:
        ts_input = grpcclient.InferInput("timestamp", [1], "INT32")
        ts_input.set_data_from_numpy(np.array([int(timestamp)], dtype=np.int32))
        inputs.append(ts_input)

    response = client.infer(
        "ensemble_model",
        inputs=inputs,
        outputs=[
            grpcclient.InferRequestedOutput("ordered_ids"),
            grpcclient.InferRequestedOutput("ordered_scores"),
        ],
    )

    ids    = response.as_numpy("ordered_ids")
    scores = response.as_numpy("ordered_scores")

    top_ids    = [int(i) for i in ids[:top_k]]
    top_scores = [float(s) for s in scores[:top_k]]

    # Enrich with item metadata from DynamoDB
    dynamo = boto3.resource("dynamodb", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    dynamo_response = dynamo.batch_get_item(
        RequestItems={
            DYNAMO_TABLE: {
                "Keys": [{"item_id": item_id} for item_id in top_ids]
            }
        }
    )
    metadata_map = {
        int(item["item_id"]): item
        for item in dynamo_response["Responses"].get(DYNAMO_TABLE, [])
    }

    recommendations = []
    for item_id, score in zip(top_ids, top_scores):
        meta = metadata_map.get(item_id, {})
        recommendations.append({
            "item_id":     item_id,
            "score":       score,
            "title":       meta.get("title", ""),
            "category_l1": meta.get("category_l1", ""),
            "category_l2": meta.get("category_l2", ""),
            "gender":      meta.get("gender", ""),
            "price":       str(meta.get("price", "")),
            "image_url":   meta.get("image_url", ""),
        })

    return _respond(200, {
        "user_id": user_id,
        "device_type": device_type,
        "recommendations": recommendations,
    })
