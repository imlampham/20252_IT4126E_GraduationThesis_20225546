# GRADUATION THESIS: PRIVACY LEAKAGE OF MASQUE PROTOCOL THROUGH WEBSITE FINGERPRINTING
## DEEPMASQUE
DeepMASQUE is a deep learning pipeline for website fingerprinting of MASQUE/QUIC encrypted traffic. Given a captured traffic trace between a client and a MASQUE proxy, the model identifies which website the user is visiting solely from observable packet metadata including direction, inter-arrival time, payload size, and flow-level statistics, without decrypting any content.
The model adapts the Var-CNN architecture with 1D ResNet-18 branches using dilated causal convolutions and extends it with a Supervised Contrastive (SupCon) loss jointly optimized alongside cross-entropy classification during end-to-end training. This joint objective produces a geometrically structured embedding space that enables reliable open-world detection of unmonitored websites as a natural byproduct of training.

## DATASET
The dataset consists of MASQUE encrypted traffic captured through a Cloudflare proxy using the usque client library and a Docker-based automated testbed with Selenium and tcpdump. It covers 300 monitored websites selected from the Tranco top list with up to 500 traces each, along with 6000 unmonitored websites for open-world evaluation and an independently collected Switzerland dataset of 300 websites.

* **[MASQUE 300-Web Dataset](https://www.kaggle.com/datasets/lampdp/masque-300-web)**: Contains up to 500 traces per website for the 300 monitored websites.
* **[MASQUE 6000-Web Dataset](https://www.kaggle.com/datasets/lampdp/masque-open-world)**: Contains up to 10 traces per website for the 6000 unmonitored websites, specifically curated for open-world evaluation scenarios.
* **[Switzerland Dataset](https://www.kaggle.com/datasets/lampdp/masque-switzerland-300-web)**: An independently collected dataset covering 300 websites in Switzerland for cross-dataset evaluation.
