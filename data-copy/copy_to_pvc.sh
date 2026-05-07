s3_bucket=$1
local_data_path=${2:-"/var/lib/data"} 
aws_region=${3:-"us-east-1"}
s3_initial_path=$4
s3_incremental_path=$5


set +e
triton_status=$(helm status triton 2>&1)
set -e 
if [[ "$triton_status" != *"STATUS: deployed"* ]]; then
    echo "FIRST RUN: Triton not deployed"
    if [ -d "$local_data_path" ]; then
        echo "Directory ${local_data_path} exists. Downloading raw data from S3 bucket..."

        mkdir -p $local_data_path/raw_data

        echo "copying from s3://${s3_bucket}/${s3_initial_path} to ${local_data_path}/raw_data"
        aws s3 cp --recursive s3://${s3_bucket}/${s3_initial_path} ${local_data_path}/raw_data --region ${aws_region}
        [ "$(ls -A $local_data_path/raw_data)" ] || { echo "Error: No data downloaded from S3. Exiting." ; exit 1; }
        
        echo "Data copied successfully"
    else
        echo "Error: Directory ${local_data_path} does not exist. Exiting."
        exit 1
    fi
else
    echo "Triton already deployed. WEEKLY FULL RETRAIN RUN"
    #TODO: add the logic for copying all data (old+new) for FULL retrain
    exit 1
fi