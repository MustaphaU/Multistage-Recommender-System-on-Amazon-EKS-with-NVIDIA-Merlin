"""
NOTE: Please run this from a pod in the same Kubernetes cluster as the Redis/Valkey instance, so it can connect to the Redis service URL.
client_app.py — Multistage Recommender System client.

Gets recommendations for a user, then simulates an interaction by randomly
selecting 2 items from the top 10 and writing them to the Redis Bloom filter
(bf:seen:{user_id}). Run multiple times to see the seen-item filter take effect.

Usage:
    python3 client_app.py \
        --triton-host <host>:8001 \
        --redis-url rediss://<host>:6379/1 \
        --user-id 1008 \
        --device-type 1 \
        --timestamp 1700000000
"""

import argparse
import random
import redis
import numpy as np
import tritonclient.grpc as grpcclient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--triton-host",
                        type=str,
                        required=True,
                        help="Triton gRPC endpoint, e.g. <host>:8001")
    parser.add_argument("--redis-url",
                        type=str,
                        required=True,
                        help="Redis/Valkey URL, e.g. rediss://<host>:6379/1")
    parser.add_argument("--user-id",
                        type=int,
                        required=True)
    parser.add_argument("--device-type",
                        type=int,
                        default=-1)
    parser.add_argument("--timestamp",
                        type=int,
                        default=None,
                        help="Unix epoch timestamp. Defaults to current server time if omitted.")
    parser.add_argument("--top-k",
                        type=int,
                        default=10)
    parser.add_argument("--flush",
                        action="store_true",
                        help="Clear the seen-items Bloom filter for this user and exit")
    args = parser.parse_args()

    if args.flush:
        r = redis.Redis.from_url(args.redis_url, decode_responses=True)
        deleted = r.delete(f"bf:seen:{args.user_id}")
        print(f"Flushed bf:seen:{args.user_id}" if deleted else f"No seen-items filter found for user {args.user_id}")
        return

    # Get recommendations
    client = grpcclient.InferenceServerClient(url=args.triton_host)

    user_id_input = grpcclient.InferInput("user_id", [1], "INT32")
    user_id_input.set_data_from_numpy(np.array([args.user_id], dtype=np.int32))

    device_input = grpcclient.InferInput("device_type", [1], "INT32")
    device_input.set_data_from_numpy(np.array([args.device_type], dtype=np.int32))

    inputs = [user_id_input, device_input]

    if args.timestamp is not None:
        ts_input = grpcclient.InferInput("timestamp", [1], "INT32")
        ts_input.set_data_from_numpy(np.array([args.timestamp], dtype=np.int32))
        inputs.append(ts_input)

    response = client.infer(
        "ensemble_model",
        inputs=inputs,
        outputs=[
            grpcclient.InferRequestedOutput("ordered_ids"),
            grpcclient.InferRequestedOutput("ordered_scores"),
        ],
    )

    ids = response.as_numpy("ordered_ids")
    scores = response.as_numpy("ordered_scores")

    print(f"Recommendations for user_id={args.user_id}  device_type={args.device_type}")
    print("-" * 40)
    for item_id, score in zip(ids[:args.top_k], scores[:args.top_k]):
        print(f"  item {item_id:>5}: {score:.4f}")

    #simulate interaction: randomly pick 2 from the top 10 and mark as seen
    interacted = [int(i) for i in random.sample(list(ids[:args.top_k]), 2)]
    r = redis.Redis.from_url(args.redis_url, decode_responses=True)
    r.execute_command("BF.MADD", f"bf:seen:{args.user_id}", *interacted)
    r.expire(f"bf:seen:{args.user_id}", 7 * 24 * 60 * 60)
    print(f"\nInteracted with items {interacted} — added to bf:seen:{args.user_id}")


if __name__ == "__main__":
    main()
