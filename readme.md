### Multistage Recommender System Deployed on Amazon EKS
![Model serving architecture](static/Model_serving.png)  
This Multistage recommender system features:
* Cold-start handling with feature masking and context-aware recommendations.
* Multimodal embeddings – image and text
* Automated finetuning
* Bloom filter for already-seen items
* In-memory feature caching to improve lookup latency
* Triton Server Autoscaling with Kubernetes HPA and Karpenter

### The MLOPs architecture
![](static/MLOps_Arch.png)

### Deployment
For deployment visit [Docs/documentation.md](Docs/documentation.md)

### Medium article: [Deploying a Four-stage Recommender System on Kubernetes featuring Multimodal Embeddings, Cold Start handling, Bloom Filters, and Feature Caching.](https://mustaphaunubi.medium.com/building-a-production-multistage-recommender-system-on-kubernetes-featuring-multimodal-embeddings-5bcd6d7bbf56?postPublishedType=repub)