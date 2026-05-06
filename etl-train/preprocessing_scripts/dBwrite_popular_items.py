import os
import glob
import redis
# import pandas as pd
import cudf
import logging
import argparse

def dbwrite_popular_items(input_path, redis_host, redis_port):
    """
    Writes popular items into Valkey db=3.
    Creates both global and per-category popularity sorted sets.
    """
    train_files = sorted(glob.glob(os.path.join(input_path, "train_day_*.parquet")))
    if not train_files:
        raise FileNotFoundError(f"No train_day_*.parquet files found in {input_path}")

    global_counts = {}
    l1_counts = {}
    l2_counts = {}

    for day_file in train_files:
        day = cudf.read_parquet(day_file, columns=["item_id", "category_l1", "category_l2"])

        day_global = day["item_id"].value_counts(sort=False).to_pandas()
        for item_id, count in day_global.items():
            item_id = int(item_id)
            global_counts[item_id] = global_counts.get(item_id, 0) + int(count)

        day_l1 = day.groupby(["category_l1", "item_id"]).size().to_pandas()
        for (cat, item_id), count in day_l1.items():
            cat = int(cat)
            item_id = int(item_id)
            key = (cat, item_id)
            l1_counts[key] = l1_counts.get(key, 0) + int(count)

        day_l2 = day.groupby(["category_l2", "item_id"]).size().to_pandas()
        for (cat, item_id), count in day_l2.items():
            cat = int(cat)
            item_id = int(item_id)
            key = (cat, item_id)
            l2_counts[key] = l2_counts.get(key, 0) + int(count)

        del day

    client = redis.Redis(host=redis_host, port=redis_port, db=3, ssl=True, decode_responses=True)
    client.ping()
    client.flushdb() 

    pipe = client.pipeline()

    #global popularity
    for item_id, count in global_counts.items():
        pipe.zadd("trending:global", {str(item_id): int(count)})

    #popularity by category_l1
    for (cat, item_id), count in l1_counts.items():
        key = f"trending:cat_l1:{cat}"
        pipe.zadd(key, {str(item_id): int(count)})

    #popularity by category_l2
    for (cat, item_id), count in l2_counts.items():
        key = f"trending:cat_l2:{cat}"
        pipe.zadd(key, {str(item_id): int(count)})

    pipe.execute()


    #verify
    logging.info(f"Wrote {len(global_counts)} items into trending:global")
    n_l1 = len(client.keys("trending:cat_l1:*"))
    n_l2 = len(client.keys("trending:cat_l2:*"))
    logging.info(f"Wrote {n_l1} category_l1 sets, {n_l2} category_l2 sets")

    top_10 = client.zrevrange("trending:global", 0, 9, withscores=True)
    logging.info("Top 10 global:")
    for item_id, score in top_10:
        logging.info(f"  item {item_id}: {int(score)} interactions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", 
                        type=str, 
                        default="/var/lib/data/raw_data", 
                        help="Path to input data directory (e.g. $PV_LOC/raw_data)")
    parser.add_argument("--redis_host", 
                        type=str, 
                        default="master.xxxx-xxxx.xxxx.use1.cache.amazonaws.com", 
                        help="Redis host for Valkey connection")
    parser.add_argument("--redis_port",
                        type=int,
                        default=6379,
                        help="Redis port for Valkey connection")
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%y %H:%M:%S')
    logging.info(f"Args: {args}")

    dbwrite_popular_items(args.input_path, args.redis_host, args.redis_port)