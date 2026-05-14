### Multistage Recommender System Deployed on Amazon EKS
![Model serving architecture](static/Model_serving.png)  
This Multistage recommender system features:
* Cold-start handling with feature masking and context-aware recommendations.
* Multimodal embeddings – image and text
* Periodic fine-tuning
* Bloom filter for excluding already-seen items
* In-memory caching to improve items feature lookup latency
* Triton Server Autoscaling with Kubernetes HPA and Karpenter

### The MLOps architecture
![The MLOps Architecture](static/MLOps_arch_updated.png)

### Deployment
For deployment visit [Docs/documentation.md](Docs/documentation.md)

### Medium article:  
You can read more about the project on Medium: [Deploying a Four-stage Recommender System on Kubernetes featuring Multimodal Embeddings, Cold Start handling, Bloom Filters, and Feature Caching.](https://mustaphaunubi.medium.com/building-a-production-multistage-recommender-system-on-kubernetes-featuring-multimodal-embeddings-5bcd6d7bbf56?postPublishedType=repub)