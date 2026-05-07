# Deploying a Multistage Recommender System on Amazon EKS
*Autoscaling nodes with Karpenter and Kubernetes Horizontal Pod Autoscaler*

## Follow these steps to set up the infrastructure and deploy the recommender system:

Steps 1 through 5 sets up karpenter, creates the EKS cluster and nodepools. The instructions (1 - 5) were lifted/ adapted from: https://karpenter.sh/docs/getting-started/getting-started-with-karpenter/
### 1. Set environment variables
```bash
export KARPENTER_NAMESPACE="kube-system"
export KARPENTER_VERSION="1.9.0"
export K8S_VERSION="1.34"
export CLUSTER="${USER}-multistage-cluster"
```

AND  

```bash
export AWS_PARTITION="aws" # AWS has multiple partitions for different regions and compliance requirements. aws/aws-cn/aws-us-gov.
export AWS_DEFAULT_REGION="us-east-1"
export AWS_ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
export TEMPOUT="$(mktemp)"
export UBUNTU_AMI_ID="$(aws ssm get-parameter \
--name "/aws/service/canonical/ubuntu/eks/24.04/${K8S_VERSION}/stable/current/amd64/hvm/ebs-gp3/ami-id" \
--query Parameter.Value --output text)"
```

### 2. Create a cluster
The configuration will:
* Use CloudFormation to set up the infrastructure needed by the EKS cluster
* Create a Kubernetes service account and AWS IAM Role, and associate them using IRSA to let Karpenter launch instances.
* Add the Karpenter node role to the aws-auth configmap to allow nodes to connect
* Use AWS EKS managed node groups for the kube-system and karpenter namespaces. 
* Set KARPENTER_IAM_ROLE_ARN variables.
* Create a role to allow spot instances.
* Run Helm to install Karpenter.


* Download Karpenter AWS CloudFormation template for the specified Karpenter version and save it to a temporary file, then deploy the CloudFormation to AWS

```bash
curl -fsSL https://raw.githubusercontent.com/aws/karpenter-provider-aws/v"${KARPENTER_VERSION}"/website/content/en/preview/getting-started/getting-started-with-karpenter/cloudformation.yaml  > "${TEMPOUT}" \
&& aws cloudformation deploy \
--stack-name "Karpenter-${CLUSTER}" \
--template-file "${TEMPOUT}" \
--capabilities CAPABILITY_NAMED_IAM \
--parameter-overrides "ClusterName=${CLUSTER}"
```

* Set up the cluster metadata, configure IAM with OIDC and associate Karpenter service account with the necessary IAM roles and policies, map the Karpenter node role for EC2 instances with Kubernetes node groups, create managed node group for initial cluster nodes, install `eks-pod-identity-agent` addon.

```bash
eksctl create cluster -f - <<EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: ${CLUSTER}
  region: ${AWS_DEFAULT_REGION}
  version: "${K8S_VERSION}"
  tags:
    karpenter.sh/discovery: ${CLUSTER}

iam:
  withOIDC: true
  podIdentityAssociations:
    - namespace: "${KARPENTER_NAMESPACE}"
      serviceAccountName: karpenter
      roleName: ${CLUSTER}-karpenter
      permissionPolicyARNs:
        - arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:policy/KarpenterControllerNodeLifecyclePolicy-${CLUSTER}
        - arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:policy/KarpenterControllerIAMIntegrationPolicy-${CLUSTER}
        - arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:policy/KarpenterControllerEKSIntegrationPolicy-${CLUSTER}
        - arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:policy/KarpenterControllerInterruptionPolicy-${CLUSTER}
        - arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:policy/KarpenterControllerResourceDiscoveryPolicy-${CLUSTER}

iamIdentityMappings:
  - arn: "arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:role/KarpenterNodeRole-${CLUSTER}"
    username: system:node:{{EC2PrivateDNSName}}
    groups:
      - system:bootstrappers
      - system:nodes

managedNodeGroups:
  - instanceType: m5.large
    amiFamily: AmazonLinux2023
    name: ${CLUSTER}-ng
    desiredCapacity: 2
    minSize: 1
    maxSize: 10

addons:
  - name: eks-pod-identity-agent
EOF
```

### 3.  Store the cluster endpoint, ARN of the Karpenter IAM role, and print both values.

```bash
export CLUSTER_ENDPOINT="$(aws eks describe-cluster --name "${CLUSTER}" --query "cluster.endpoint" --output text)"
export KARPENTER_IAM_ROLE_ARN="arn:${AWS_PARTITION}:iam::${AWS_ACCOUNT_ID}:role/${CLUSTER}-karpenter"

echo "${CLUSTER_ENDPOINT} ${KARPENTER_IAM_ROLE_ARN}"
```

### 4. Install Karpenter  

```bash
# Logout of helm registry to perform an unauthenticated pull against the public ECR
helm registry logout public.ecr.aws

helm upgrade --install karpenter oci://public.ecr.aws/karpenter/karpenter --version "${KARPENTER_VERSION}" --namespace "${KARPENTER_NAMESPACE}" --create-namespace \
--set "settings.clusterName=${CLUSTER}" \
--set "settings.interruptionQueue=${CLUSTER}" \
--set controller.resources.requests.cpu=1 \
--set controller.resources.requests.memory=1Gi \
--set controller.resources.limits.cpu=1 \
--set controller.resources.limits.memory=1Gi \
--wait
```

### 5. Create Nodepools   
* CPU nodepool:   
(m5.xlarge, t3.xlarge) -> you can adjust these based on your needs)  
The `consolidationPolicy` set to `WhenEmptyOrUnderutilized` in the `disruption` block configures Karpenter to reduce cluster cost by removing and replacing nodes. As a result, consolidation will terminate any empty nodes on the cluster. This behavior can be disabled by setting `consolidateAfter` to `Never`, telling Karpenter that it should never consolidate nodes.  
**Note:** This NodePool will create capacity as long as the sum of all created capacity is less than the specified limit.  

```bash
kubectl apply -f - <<EOF
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: cpu-node-pool
spec:
  limits:
    cpu: 16
  template:
    metadata:
      labels:
        hardware-type: cpu
    spec:
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: kubernetes.io/os
          operator: In
          values: ["linux"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["x8g.medium", "r6g.large", "x2gd.medium", "t3.xlarge"]
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: ubuntu-cpu
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 1m
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: ubuntu-cpu
spec:
  role: "KarpenterNodeRole-${CLUSTER}"
  amiFamily: Custom
  amiSelectorTerms:
    - id: "${UBUNTU_AMI_ID}"
  blockDeviceMappings:
    - deviceName: /dev/sda1
      ebs:
        volumeSize: 50Gi
        volumeType: gp3
        deleteOnTermination: true
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER}"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER}"
  userData: |
    #!/bin/bash
    /etc/eks/bootstrap.sh '${CLUSTER}' --kubelet-extra-args '--register-with-taints=karpenter.sh/unregistered=true:NoExecute'
EOF
```

* Create GPU node pool  
("g4dn.xlarge","g5.xlarge","g6.xlarge")  

```bash
kubectl apply -f - <<EOF
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-node-pool
spec:
  limits:
    cpu: 16
  template:
    metadata:
      labels:
        hardware-type: gpu
    spec:
      taints:
        - key: nvidia.com/gpu
          value: "present"
          effect: NoSchedule
      requirements:
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
        - key: kubernetes.io/os
          operator: In
          values: ["linux"]
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["on-demand"]
        - key: node.kubernetes.io/instance-type
          operator: In
          values: ["g4dn.xlarge", "g5.xlarge", "g6.xlarge"]
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: ubuntu-gpu
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 1m
---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: ubuntu-gpu
spec:
  role: "KarpenterNodeRole-${CLUSTER}"
  amiFamily: Custom
  amiSelectorTerms:
    - id: "${UBUNTU_AMI_ID}"
  blockDeviceMappings:
    - deviceName: /dev/sda1
      ebs:
        volumeSize: 100Gi
        volumeType: gp3
        deleteOnTermination: true
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER}"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER}"
  userData: |
    #!/bin/bash
    /etc/eks/bootstrap.sh '${CLUSTER}' --kubelet-extra-args '--register-with-taints=karpenter.sh/unregistered=true:NoExecute'
EOF
```

### 6. Install NVIDIA GPU Operator

* Install the helm cli (or SKIP if you alreadly have the helm cli):
    ```bash
    curl -fsSL -o get_helm.sh https://raw.githubusercontent.com/helm/helm/master/scripts/get-helm-3 \
    && chmod 700 get_helm.sh \
    && ./get_helm.sh
    ```

* Create a gpu-operator namespace and set the enforcement policy to previleged
    ```bash
    kubectl create ns gpu-operator
    kubectl label --overwrite ns gpu-operator pod-security.kubernetes.io/enforce=privileged
    ```

* Add the NVIDIA Helm repository
    ```bash
    helm repo add nvidia https://helm.ngc.nvidia.com/nvidia \
        && helm repo update
    ```

* Install the GPU Operator (I have chosen driver 570.XX/ cuda 12.x to avoid potential issue with LocalCUDACluster crashing on 580.XX /cuda 13.XX)
```bash
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --version=v25.10.1 \
  --set driver.version=570.195.03 \
  --set nodeSelector.hardware-type=gpu \
  --set-json "tolerations=[{\"key\":\"nvidia.com/gpu\",\"operator\":\"Exists\",\"effect\":\"NoSchedule\"}]"
```

* To confirm driver installation, create a pod and run `nvidia-smi` inside the pod (a container):  
 
```bash
kubectl apply -f - <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: nvidia-smi-pod
spec:
  restartPolicy: Never
  tolerations:
    - key: "nvidia.com/gpu"
      operator: "Exists"
      effect: "NoSchedule"
  containers:
    - name: nvidia-smi
      image: nvidia/cuda:12.2.0-base-ubuntu22.04
      command: ["nvidia-smi"]
      resources:
        limits:
          nvidia.com/gpu: 1
EOF
```
 
ii. check logs: `kubectl logs nvidia-smi-pod`  
    
```
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 570.195.03             Driver Version: 570.195.03     CUDA Version: 12.8     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA A10G                    On  |   00000000:00:1E.0 Off |                    0 |
    |  0%   22C    P8             23W /  300W |       0MiB /  23028MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                            
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+
```  


### 7. Add the EFS CSI Driver addon
* Find the EFS CSI driver version compatible with your platform version
    ```bash
    aws eks describe-addon-versions --addon-name aws-efs-csi-driver
    ```
    I chose `v2.3.0-eksbuild.1` for my platform version 1.34

* Create IAM Role for EFS CSI Driver and Update Trust Policy:
    This policy `AmazonEFSCSIDriverPolicy` grants the necessary permissions for the EFS CSI (Container Storage Interface) driver to manage Amazon EFS (Elastic File System) resources from within your EKS cluster. You attach this policy to the IAM role used by the EFS CSI driver service account, so the driver can interact with EFS on your behalf.
    ```bash
    export role_name=AmazonEKS_EFS_CSI_DriverRole_${CLUSTER}
    eksctl create iamserviceaccount \
        --name efs-csi-controller-sa \
        --namespace kube-system \
        --cluster $CLUSTER \
        --role-name $role_name \
        --role-only \
        --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy \
        --approve
    TRUST_POLICY=$(aws iam get-role --output json --role-name $role_name --query 'Role.AssumeRolePolicyDocument' | \
        sed -e 's/efs-csi-controller-sa/efs-csi-*/' -e 's/StringEquals/StringLike/')
    aws iam update-assume-role-policy --role-name $role_name --policy-document "$TRUST_POLICY"
    ```

* create EFS CSI addon
    ```bash
    eksctl create addon --cluster $CLUSTER --name aws-efs-csi-driver --version v2.3.0-eksbuild.1 \
    --service-account-role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/${role_name} --force
    ```

### 8. [Install EBS CSI driver](https://docs.aws.amazon.com/eks/latest/userguide/ebs-csi.html)

* Find the driver version compatible with your platform version
    ```bash
    aws eks describe-addon-versions --addon-name aws-ebs-csi-driver
    ```
    v1.55.0-eksbuild.1 works with our platform (kubernetes) version 1.34

* create Amazon EBS CSI driver IAM role for service account and attach `AmazonEBSCSIDriverPolicy`
    ```bash
    eksctl create iamserviceaccount \
    --name ebs-csi-controller-sa \
    --namespace kube-system \
    --cluster $CLUSTER \
    --role-name AmazonEKS_EBS-CSI_DriverRole \
    --role-only \
    --attach-policy-arn arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy \
    --approve
    ```
* create the EBS CSI addon
    ```bash
    eksctl create addon --cluster $CLUSTER --name aws-ebs-csi-driver --version v1.55.0-eksbuild.1 \
    --service-account-role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/AmazonEKS_EBS-CSI_DriverRole --force
    ```

* set default StorageClass: I chose gp2
```bash
kubectl patch storageclass gp2 -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'
```
Why EBS: Some core Kubeflow components need **exclusive** *not* shared storage  (e.g. databases and metadata). EBS is preferred for single pod access.

### 9. [Install Kubeflow Pipelines (Standalone deployment *not* Full)](https://docs.aws.amazon.com/sagemaker/latest/dg/kubernetes-sagemaker-components-install.html#kubeflow-pipelines-standalone)  

I skipped the step *creating a gateway node* because I have a machine that can:  

```
* Call AWS APIs (EKS, IAM, EC2, CloudFormation, S3)

* Talk to the Kubernetes API server

* Authenticate to EKS
```
Also skipped: *Set up an Amazon EKS cluster* (there is an existing cluster)  

i.  [Install the Kubeflow Pipelines.](https://www.kubeflow.org/docs/components/pipelines/operator-guides/installation/)
```bash
export PIPELINE_VERSION=2.16.0
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"                             
```

**TAKES APPROX. 3 minutes to complete**

ii. access the Kubeflow pipelines UI  
- port forward the Kubeflow Pipelines UI  
    ```
    kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80
    ```
- Open http://localhost:8080 on your browser to access the Kubeflow Pipelines UI.

### 10. [Create the Elastic file system (EFS)](https://github.com/kubernetes-sigs/aws-efs-csi-driver/blob/master/docs/efs-create-filesystem.md)
a. Where is the cluster?

* `VPC_ID`: Get the virtual network the cluster lives in.
    ```bash
    VPC_ID=$(aws eks describe-cluster --name $CLUSTER --region $AWS_DEFAULT_REGION \
                --query "cluster.resourcesVpcConfig.vpcId" \
                --output text)
    ```

* `CIDR`: Retrieve the CIDR range for your cluster's VPC: I picked the first entry of Vpcs (ie. Vpcs[0]) since Vpcs is a list. 
    ```bash
    cidr_range=$(aws ec2 describe-vpcs \
    --vpc-ids $VPC_ID \
    --query "Vpcs[0].CidrBlock" \
    --output text \
    --region $AWS_DEFAULT_REGION)
    ```

b. Create a security group with an inbound rule that allows inbound NFS traffic for your Amazon EFS mount points.  
* create a security group
    ```bash
    security_group_id=$(aws ec2 create-security-group \
    --group-name MyEFS_SecurityGroup \
    --description "My EFS security group" \
    --vpc-id $VPC_ID \
    --query 'GroupId' \
    --output text)
    ```

* create an inbound rule that allows inbound NFS traffic to the EFS from the CIDR for your cluster's VPC.    
    ```bash
    aws ec2 authorize-security-group-ingress \
    --group-id $security_group_id \
    --protocol tcp \
    --port 2049 \
    --cidr $cidr_range
    ```

c. create an EFS for the EKS cluster  
* create a file system
    ```bash
    file_system_id=$(aws efs create-file-system \
    --region $AWS_DEFAULT_REGION \
    --performance-mode generalPurpose \
    --query 'FileSystemId' \
    --output text)
    ```
* create mount targets  
    a. Determine the IP (INTERNAL-IP) addresses of your cluster nodes.
    ```
    kubectl get nodes -o wide
    ```

    b. Determine the IDs of the subnets in your (cluster) VPC and which Availability Zone the subnet is in.  
    ```bash
    aws ec2 describe-subnets \
        --filters "Name=vpc-id,Values=$VPC_ID" \
        --query 'Subnets[*].{SubnetId: SubnetId,AvailabilityZone: AvailabilityZone,CidrBlock: CidrBlock}' \
        --output table
    ```

    c. fetch the nodepool subnets: the subnets Karpenter will launch nodes into
    ```bash
    SUBNETS=$(aws ec2 describe-subnets \
    --region "$AWS_DEFAULT_REGION" \
    --filters "Name=vpc-id,Values=$VPC_ID" "Name=tag:karpenter.sh/discovery,Values=$CLUSTER" \
    --query "Subnets[].SubnetId" --output text | tr '\t' '\n' | sort -u)
    echo "$SUBNETS"
    ```
    The script reads the unique subnets (sort -u).

    d. create one mount target per AZ (avoid duplicates)  
    EFS allows one mount target per AZ, so don’t blindly loop every subnet if you have multiple subnets in the same AZ. The loop picks the first subnet it sees per AZ
```bash
declare -A SEEN_AZ
while read -r sn az; do
  [[ -z "$sn" || -z "$az" ]] && continue
  if [[ -z "${SEEN_AZ[$az]}" ]]; then
    aws efs create-mount-target \
      --region "$AWS_DEFAULT_REGION" \
      --file-system-id "$file_system_id" \
      --subnet-id "$sn" \
      --security-groups "$security_group_id" >/dev/null
    SEEN_AZ[$az]=1
    echo "Created mount target in $az using $sn"
  else
    echo "Skipping $sn (already created mount target in $az)"
  fi
done < <(aws ec2 describe-subnets \
  --region "$AWS_DEFAULT_REGION" \
  --subnet-ids $SUBNETS \
  --query 'Subnets[].[SubnetId,AvailabilityZone]' \
  --output text)
```  

e. optional: confirm mount targets created.
```bash
    aws efs describe-mount-targets --file-system-id $file_system_id --region $AWS_DEFAULT_REGION \
    --query 'MountTargets[].{Subnet:SubnetId,AZ:AvailabilityZoneId,State:LifeCycleState,IP:IpAddress}' --output table
```
output like:
```bash
----------------------------------------------------------------------------------------------------
|                                       DescribeMountTargets                                       |
+----------+------------------+--------------------------+------------+----------------------------+
|    AZ    |       IP         |           Id             |   State    |          Subnet            |
+----------+------------------+--------------------------+------------+----------------------------+
|  use1-az6|  192.xxx.yyy.xxx |  fsmt-0hihihihihihihihih |  available |  subnet-hahahahahahahahah  |
|  use1-az1|  192.xxx.yy.cc   |  fsmt-0merrymrrymrymerr  |  available |  subnet-mehdjhdmjdjdjdjej  |
+----------+------------------+--------------------------+------------+----------------------------+
```

### 10 (part b). Tag the shared node security group (SG) so Kubernetes LoadBalancer can identify the SG of the cluster. 
The Cluster nodes currently have multiple SGs attached. Without the Kubernetes cluster tag, the controller can't safely choose the correct SG when creating the LoadBalancer, so it fails with "Multiple untagged security groups found.."
* find the worker-node shared SG
    ```bash
    NODE_SG=$(aws ec2 describe-security-groups \
    --region "$AWS_DEFAULT_REGION" \
    --filters \
        "Name=vpc-id,Values=$VPC_ID" \
        "Name=tag:alpha.eksctl.io/cluster-name,Values=$CLUSTER" \
        "Name=tag:aws:cloudformation:logical-id,Values=ClusterSharedNodeSecurityGroup" \
    --query 'SecurityGroups[0].GroupId' \
    --output text)
    ```
* tag the node 
    ```bash
    aws ec2 create-tags \
    --region "$AWS_DEFAULT_REGION" \
    --resources "$NODE_SG" \
    --tags "Key=kubernetes.io/cluster/$CLUSTER,Value=owned"
    ```

* verify tag
```bash
aws ec2 describe-security-groups \
  --region "$AWS_DEFAULT_REGION" \
  --group-ids "$NODE_SG" \
  --query 'SecurityGroups[0].Tags'
  ```

### 10 (part c). Create ElastiCache (Valkey) for Feast Online Store, Bloom Filters, and Trending Items

a. Create a security group for ElastiCache
```bash
CACHE_SG=$(aws ec2 create-security-group \
  --group-name MyElastiCache_SecurityGroup \
  --description "ElastiCache Valkey security group" \
  --vpc-id $VPC_ID \
  --query 'GroupId' \
  --output text)
```

b. Allow inbound Redis traffic (port 6379) from the cluster VPC CIDR
```bash
aws ec2 authorize-security-group-ingress \
  --group-id $CACHE_SG \
  --protocol tcp \
  --port 6379 \
  --cidr $cidr_range
```
c. Get the private subnets and create a subnet group
```bash
aws elasticache create-cache-subnet-group \
  --cache-subnet-group-name merlin-elasticache-subnets \
  --cache-subnet-group-description "Subnets for Merlin ElastiCache" \
  --subnet-ids $SUBNETS
```

d. Create the Valkey cluster (provisioned, TLS enabled)
```bash
aws elasticache create-replication-group \
  --replication-group-id multistage-valkeycl \
  --replication-group-description "Merlin RecSys Valkey cluster" \
  --engine valkey \
  --engine-version 8.2 \
  --cache-node-type cache.t3.medium \
  --num-cache-clusters 1 \
  --cache-subnet-group-name merlin-elasticache-subnets \
  --security-group-ids $CACHE_SG \
  --transit-encryption-enabled \
  --at-rest-encryption-enabled \
  --region $AWS_DEFAULT_REGION

# Wait for cluster to become available
aws elasticache wait replication-group-available \
  --replication-group-id multistage-valkeycl \
  --region $AWS_DEFAULT_REGION
```

e. Get the endpoint
```bash
export REDIS_HOST=$(aws elasticache describe-replication-groups \
  --replication-group-id multistage-valkeycl \
  --query 'ReplicationGroups[0].NodeGroups[0].PrimaryEndpoint.Address' \
  --output text \
  --region $AWS_DEFAULT_REGION)

echo "Valkey endpoint: $REDIS_HOST"
```
f. Verify connectivity from a test pod
```bash
kubectl run redis-test --image=redis:7 --restart=Never -n kubeflow -- \
  redis-cli --tls --insecure -h $REDIS_HOST -p 6379 PING
kubectl logs redis-test -n kubeflow
kubectl delete pod redis-test -n kubeflow
```

### 11. Create the storage class and persistent volume claim. CSI dynamic provisioning creates PV automatically (reason for efs-ap provisioning mode).  
```bash
kubectl apply -n kubeflow -f - <<EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
parameters:
  provisioningMode: efs-ap
  fileSystemId: ${file_system_id}
  directoryPerms: "777"
mountOptions:
  - tls
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-cluster-pvc
  namespace: kubeflow
spec:
  accessModes: [ReadWriteMany]
  storageClassName: efs-sc
  resources:
    requests:
      storage: 100Gi
EOF
```
    Note: `resources.capacity` is actually ignored by Amazon EFS CSI driver when provisioning the volume claim because Amazon EFS is an elastic file system. Capacity is only specified because it is a required field in Kubernetes. The value doesn't limit the size of your Amazon EFS file system.

* optional: confirm sc and pvc created
    ```bash
    kubectl get sc,pvc -n kubeflow
    ```

* bonus: test the persistent volume claim  
    i. create a pod named *`efs-test`*   
    (PS: YOU can test many pods on the volume)
    ```bash
    kubectl run efs-test --image=busybox --restart=Never --overrides='{"spec":{"containers":[{"name":"efs-test","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"efs-vol","mountPath":"/var/lib/data"}]}],"volumes":[{"name":"efs-vol","persistentVolumeClaim":{"claimName":"my-cluster-pvc"}}]}}' -n kubeflow
    ```
    creates a pod with `busybox` image, with volume mount at `"/var/lib/data/"` 

    ii. exec into the pod
    ```bash
    kubectl exec -it efs-test -n kubeflow -- sh
    ```
    iii. Once inside the shell environment, test the EFS mount like so:
    ```bash
    ls /var/lib/data && mkdir -p /var/lib/data/criteo-data && echo "EFS test successful" > /var/lib/data/criteo-data/test.txt && cat /var/lib/data/criteo-data/test.txt && rm -rf /var/lib/data/criteo-data && exit
    ```
    iv. clean up
    ```bash
    kubectl delete pod efs-test -n kubeflow
    ```
### 12. Create a Service Account and RBAC Role for Pipeline Components to Access Helm and Deploy pods, etc.
The data extraction and preprocess-train components need permission to check Helm release status for Triton (`triton_status=$(helm status triton 2>&1)`).   
**Note:** the RBAC permissions in the Role apply to all pods, services, deployments and replicasets in the `kubeflow` namespace. Any pod or component using the `merlin-kfp-sa` service account will have these permissions for those resource types.
```bash
kubectl apply -f - <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: merlin-kfp-sa
  namespace: kubeflow
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: merlin-kfp-role
  namespace: kubeflow
rules:
# For helm status and install (secrets store helm releases)
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
# For deploying Triton
- apiGroups: [""]
  resources: ["pods", "pods/log", "services"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
# For argo workflows
- apiGroups: ["argoproj.io"]
  resources: ["workflows", "workflowtaskresults"]
  verbs: ["get", "list", "watch", "create", "update", "patch"]
- apiGroups: ["monitoring.coreos.com"]
  resources: ["servicemonitors"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: merlin-kfp-binding
  namespace: kubeflow
subjects:
- kind: ServiceAccount
  name: merlin-kfp-sa
  namespace: kubeflow
roleRef:
  kind: Role
  name: merlin-kfp-role
  apiGroup: rbac.authorization.k8s.io
EOF
```
Verify service account
```bash
kubectl get serviceaccount merlin-kfp-sa -n kubeflow
kubectl get role merlin-kfp-role -n kubeflow
kubectl get rolebinding merlin-kfp-binding -n kubeflow
```

### 13. Setup S3, Glue, and Athena for Feast

a. create the S3 bucket, feast folder structure, and upload the raw training data and embedding models. Please replace `initial_data/` with your desired path.
```bash
export BUCKET="multistage-recsys-bucket"

aws s3 mb s3://$BUCKET --region $AWS_DEFAULT_REGION

aws s3api put-object --bucket $BUCKET --key feast/registry.db
aws s3api put-object --bucket $BUCKET --key feast/athena_staging/
aws s3api put-object --bucket $BUCKET --key feast/data/

aws s3 cp initial_data/ s3://$BUCKET/initial_data/ --recursive

```

b. create the Glue database
```bash
export GLUE_DATABASE="multistage_feast_glue_database"
aws glue create-database \
    --database-input "{\"Name\": \"${GLUE_DATABASE}\"}" \
    --region $AWS_DEFAULT_REGION
```

c. create the Glue tables (these define the schema Athena uses to query Feast feature data in S3)

*`user_features`* table
```bash
aws glue create-table --region $AWS_DEFAULT_REGION \
    --database-name $GLUE_DATABASE \
    --table-input '{
        "Name": "user_features",
        "StorageDescriptor": {
            "Columns": [
                {"Name": "user_id", "Type": "int"},
                {"Name": "age", "Type": "int"},
                {"Name": "gender", "Type": "int"},
                {"Name": "created", "Type": "timestamp"},
                {"Name": "datetime", "Type": "timestamp"}
            ],
            "Location": "s3://'"$BUCKET"'/feast/data/user_features/",
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            }
        },
        "TableType": "EXTERNAL_TABLE"
    }'
```

*`item_features`* table
```bash
aws glue create-table --region $AWS_DEFAULT_REGION \
    --database-name $GLUE_DATABASE \
    --table-input '{
        "Name": "item_features",
        "StorageDescriptor": {
            "Columns": [
                {"Name": "item_id", "Type": "int"},
                {"Name": "price", "Type": "float"},
                {"Name": "category_l1", "Type": "int"},
                {"Name": "category_l2", "Type": "int"},
                {"Name": "item_gender", "Type": "int"},
                {"Name": "created", "Type": "timestamp"},
                {"Name": "datetime", "Type": "timestamp"}
            ],
            "Location": "s3://'"$BUCKET"'/feast/data/item_features/",
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            }
        },
        "TableType": "EXTERNAL_TABLE"
    }'
```

d. create the Athena workgroup for Feast materialization
```bash
export ATHENA_WORKGROUP="feast_materialize_wg"
aws athena create-work-group \
    --name $ATHENA_WORKGROUP \
    --configuration '{"ResultConfiguration":{"OutputLocation":"s3://'"$BUCKET"'/feast/athena_staging/"},"EnforceWorkGroupConfiguration":false}' \
    --description "Workgroup for Feast materialization queries" \
    --region $AWS_DEFAULT_REGION
```

e. optional: verify setup
```bash
aws s3 ls s3://$BUCKET --recursive

aws glue get-tables --database-name $GLUE_DATABASE --region $AWS_DEFAULT_REGION \
    --query 'TableList[].Name' --output text

aws athena get-work-group --work-group $ATHENA_WORKGROUP --region $AWS_DEFAULT_REGION \
    --query 'WorkGroup.Configuration.ResultConfiguration.OutputLocation' --output text
```
**Note:** `feast apply`, `feast materialize`, and `seed_trending` are NOT run here; they are executed automatically by `preprocess.sh` during the ETL pipeline run.

### 14 Create the policy for S3, Athena, and Glue access
```bash
aws iam create-policy \
  --policy-name merlin-s3-${BUCKET}-full \
  --policy-document "$(cat <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "s3:*",
      "Resource": [
        "arn:aws:s3:::${BUCKET}",
        "arn:aws:s3:::${BUCKET}/*"
      ]
    },
  {
    "Effect": "Allow",
    "Action": "glue:*",
    "Resource": [
      "arn:aws:glue:${AWS_DEFAULT_REGION}:${AWS_ACCOUNT_ID}:catalog",
      "arn:aws:glue:${AWS_DEFAULT_REGION}:${AWS_ACCOUNT_ID}:database/${GLUE_DATABASE}",
      "arn:aws:glue:${AWS_DEFAULT_REGION}:${AWS_ACCOUNT_ID}:table/${GLUE_DATABASE}/*"
    ]
  },
  {
    "Effect": "Allow",
    "Action": "athena:*",
    "Resource": [
      "arn:aws:athena:${AWS_DEFAULT_REGION}:${AWS_ACCOUNT_ID}:workgroup/${ATHENA_WORKGROUP}",
      "arn:aws:athena:${AWS_DEFAULT_REGION}:${AWS_ACCOUNT_ID}:datacatalog/AwsDataCatalog"
    ]
  }
  ]
}
EOF
)"
```

### Update the service account with IAM role for S3, Athena, and Glue access
* First, create the IAM role with the S3 access policies attached. Ensure to not override the existing service account (merlin-kfp-sa) by using the --role-only flag.
  ```bash
  export NAMESPACE=kubeflow
  export SERVICE_ACCOUNT=merlin-kfp-sa
  export ROLE_NAME=merlin-kfp_irsa-role
  ```

  ```bash
  eksctl create iamserviceaccount \
  --cluster $CLUSTER \
  --region $AWS_DEFAULT_REGION \
  --namespace $NAMESPACE \
  --name $SERVICE_ACCOUNT \
  --role-name $ROLE_NAME \
  --attach-policy-arn arn:aws:iam::${AWS_ACCOUNT_ID}:policy/merlin-s3-${BUCKET}-full \
  --role-only \
  --approve
  ```

* Next, annotate the existing service account with the newly created role
  ```bash
  kubectl -n $NAMESPACE annotate serviceaccount $SERVICE_ACCOUNT \
  eks.amazonaws.com/role-arn=arn:aws:iam::$AWS_ACCOUNT_ID:role/$ROLE_NAME \
  --overwrite
  ```

### 15. Install Prometheus and Grafana
* Install the kube-prometheus-stack which includes Prometheus Operator, Prometheus, and Grafana.
    ```bash
    export GRAFANA_ADMIN_USERNAME=yourusernameREPLACE # replace
    export GRAFANA_ADMIN_PASSWORD=yourChosenPasswordREPLACE # replace
    ```

    ```bash
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo update
    helm install prometheus prometheus-community/kube-prometheus-stack \
        -n monitoring \
        --create-namespace \
        --set grafana.adminUser=$GRAFANA_ADMIN_USERNAME \
        --set grafana.adminPassword=$GRAFANA_ADMIN_PASSWORD
    ```  

* you can always fetch your username and password:
    ```bash
    kubectl --namespace monitoring get secrets prometheus-grafana -o jsonpath="{.data.admin-user}" | base64 -d ; echo
    kubectl --namespace monitoring get secrets prometheus-grafana -o jsonpath="{.data.admin-password}" | base64 -d ; echo 
    ```
* Update Prometheus to scrape all ServiceMonitors where release is either "triton" or "prometheus":
    ```bash
    helm upgrade prometheus prometheus-community/kube-prometheus-stack -n monitoring \
    --reuse-values \
    --set-json 'prometheus.prometheusSpec.serviceMonitorSelector={"matchExpressions":[{"key":"release","operator":"In","values":["prometheus","triton"]}]}'
    ```
    This ensures it is able to scrape `release: triton` ServiceMonitors; as well as its own internal components with: `release: prometheus`

* Verify installation:
    ```bash
    kubectl get pods -n monitoring
    kubectl get crd | grep monitoring.coreos.com
    ```

* Access Grafana (optional):
    ```bash
    kubectl port-forward svc/prometheus-grafana -n monitoring 3000:80
    # Open http://localhost:3000 (admin/admin)
    ```

### 16. Deploy the Horizontal Pod Autoscaler
The YAML files are located in the [scaling_yamls](../scaling_yamls) directory
* create the `custom-metrics` namespace. Some of the the manifests reference this namespace; the namedspaced resources including ConfigMap, Deployment, ServiceAccount, Service will fail to create without it.

    ```bash
    kubectl create namespace custom-metrics
    ```
* apply [custom-metric-server-config.yaml](../scaling_yamls/custom-metric-server-config.yaml)
    ```bash
    kubectl apply -f custom-metric-server-config.yaml
    ```
    Deploys the ConfigMap which defines the translation rule i.e.,
    
    - Uses the raw triton metrics `nv_inference_queue_duration_us` and `nv_inference_request_success`, to compute **`avg_time_queue_ms`**.  

    - **`avg_time_queue_ms`** = `'avg(delta(nv_inference_queue_duration_us{<<.LabelMatchers>>}[30s])/(1+delta(nv_inference_request_success{<<.LabelMatchers>>}[30s]))/1000) by (<<.GroupBy>>)'` and exposes this `avg_time_queue_ms`to HPA.

* apply [custom-metric-server-rbac.yaml](../scaling_yamls/custom-metric-server-rbac.yaml): role-based access control (RBAC) 
    ```bash
    kubectl apply -f custom-metric-server-rbac.yaml
    ```
    - allows custom metrics adapter to read kubernetes resources and auth config; allows HPA controller to read custom metrics API.

* apply [custom-metric-server.yaml](../scaling_yamls/custom-metric-server.yaml)
    ```bash
    kubectl apply -f custom-metric-server.yaml
    ```
    - runs the Prometheus Adapter pod and points it to Prometheus:  
        `http://<PROMETHEUS_SERVICE_NAME>.<PROMETHEUS_NAMESPACE>.svc.cluster.local:9090`
    - mounts the adapter config deployed earlier.
    - registers APIService custom.metrics.k8s.io so Kubernetes/HPA can query it.

* apply [triton-hpa.yaml](../scaling_yamls/triton-hpa.yaml)
    ```bash
    kubectl apply -f triton-hpa.yaml
    ```
    - creates horizontal pod autoscaler (HPA) resource which adjusts the number of pods in the specified target (the triton-triton-inference-server deployment) based on the custom metric (`avg_time_queue_ms`) that was exposed for the kubeflow namespace by the prometheus custom metrics adapter. If the metric value exceeds 200 milliseconds, the HPA will scale up the number of replicas in the target to 2 and will scale down to 1 if metric falls below 200 miiliseconds.

### 17. Build and Push pipeline containers to Amamzon ECR
#### a. Data extraction (and triton deployment) container:
* navigate to the project root, then run the scripts below:
```bash
chmod +x docker/data-copy/build_and_push_data_copy_container.sh
./docker/data-copy/build_and_push_data_copy_container.sh $AWS_ACCOUNT_ID $AWS_DEFAULT_REGION
```

#### b. Preprocessing and Training Container
* from the project root, run: 
```bash
chmod +x docker/etl-train/build_and_push_etl_container.sh 
./docker/etl-train/build_and_push_etl_container.sh $AWS_ACCOUNT_ID $AWS_DEFAULT_REGION
```

#### c. Triton Inference (server) container
* from the project root, run:
```bash
chmod +x  docker/triton-inference/build_and_push_triton_inference_container.sh
./docker/triton-inference/build_and_push_triton_inference_container.sh $AWS_ACCOUNT_ID $AWS_DEFAULT_REGION
```

### 18. Compile the full training pipeline; Upload in Kubeflow UI, and run
```bash
python kubeflowpipeline_1.py \
-dcoi "$(cat .image_uris/data_copy_image_uri.txt)" \
-ppi "$(cat .image_uris/etl_train_image_uri.txt)" \
-ti "$(cat .image_uris/etl_train_image_uri.txt)" \
-di "$(cat .image_uris/data_copy_image_uri.txt)"
```



# Incremental Run

### 1. Upload new interaction data to a `new_data` path in the same S3 bucket 
```bash
aws s3 cp new_data/ s3://$BUCKET/new_data/ --recursive
```
### 19. Compile the incremental preprocessing pipeline
```bash
python3 kubeflowpipeline_2.py \
  -dcoi "$(cat .image_uris/data_copy_image_uri.txt)" \
  -eti "$(cat .image_uris/etl_train_image_uri.txt)"
  ```