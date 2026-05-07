### Multistage Recommender System Deployed on Amazon EKS
![Model serving architecture](static/Model_serving.png)  
This Multistage recommender system features:
* Cold-start handling with feature masking and context-aware recommendations.
* Multimodal embeddings – image and text
* Automated finetuning
* Bloom filter for already-seen items
* In-memory feature caching to improve lookup latency
* Triton Server Autoscaling with Kubernetes HPA and Karpenter

### Deployment
For deployment visit [Docs/documentation.md](Docs/documentation.md)