# Who's That Pokémon? A Silhouette-Based Deep Learning Approach to Fine-Grained Character Recognition

## Abstract

Over the decades, numerous contributions have been published towards improvement as well as overall enhancement of the classification task in computer vision. Through this project, our group aims to explore the significance of an object's shape to its identification and, more specifically, based off of its silhouette. Our chosen outlet of exploration in this task is to engage a model in identifying Pokémon primarily on its outline, replicating the popular activity known as Who's That Pokémon?. The model will be trained on a labeled dataset of roughly 2,000 images of various Pokémon (with augmentation should this size not be sufficient). We plan on leveraging/replicating an existing model trained for a similar task, then applying an ensemble of methods fixed on fine-grain segmentation. Output quality and model performance will be evaluated using state-of-the-art metrics relevant in silhouette image classification. We want this project to be both engaging and a valuable contribution to the progress of machine classification.

## Authors
- Illy Hoang (illyhoan@usc.edu)
- Arnav Kamra (akamra@usc.edu) 
- Rahul Puranam (rpuranam@usc.edu)
- Joseph Yue (yuejosep@usc.edu)

## Problem Statement

Pokémon as a franchise has existed for nearly 30 years, with 9 generations of games having been released in this time along with multiple seasons of television. Since its inception, the PokéDex has expanded from the original 151 to over 1000. While early fans of the franchise often pride themselves in being able to recognize Pokémon by their silhouette alone, machines are still lagging in their current capacity. However, the problem statement extends beyond the world of Pokémon and highlights broader challenges in computer vision; To accurately classify over 1,000 distinct classes of Pokémon is a complex task, mirroring real-world classification problems in fields such as ecology where there exists research to distinguish species that appear closely related physically. Within the context of Pokémon, an accurate classifier could serve as a digital Pokédex, helping fans practice in identification. More broadly, a model capable of performing at-least on-par to humans in identifying objects with reduced visual contextual clues could potentially be extended in other subdomains including cataloging biodiversity, monitoring wildlife, or assisting in conservation efforts.

## Current Approaches

### Deep Convolutional Networks and Shape Recognition

Research has shown that Deep Convolutional Neural Networks (DCNNs) tend to diverge from human image recognition by over relying on local features while ignoring the global shape of objects. DCNNs lack shape representations and processing capabilities, and make no special use of the bounding contours of objects, which most reliably define shape in human and biological vision and form the primary bases of human object classification. This poses significant challenges for our project as we are attempting to classify Pokémon on silhouettes alone.

### Existing Pokémon Classification Methods

1. **CNN-based Classification**: A CNN was trained to identify 150 different species of Pokémon with 95.8% accuracy on RGB images. However, there is no indication that the model was relying more on global shape or local textures.

2. **Transfer Learning Approaches**: Using pre-trained models like ResNet101, researchers achieved 95.6% accuracy on Pokémon classification tasks. Transfer learning offers a baseline for building an initial classifier, though these studies were limited to smaller datasets and clean RGB images.

## Our Approach

The project will explore multiple approaches:

1. **Baseline CNN**: Build and test a CNN similar to existing methods on silhouettes alone to establish a baseline and verify findings about shape vs. texture reliance.

2. **Transfer Learning**: Explore pre-trained models and transfer learning techniques to leverage existing knowledge for silhouette classification.

3. **Advanced Methods**: Investigate self-supervised learning and contrastive loss methods for improved generalization.

The model will be tested on labeled images of individual Pokémon with the goal of achieving similar to superior accuracy compared to existing RGB-based approaches.

## Project Timeline

- **Sep 28th - Oct 5: Setup**
  - Extended literature review
  - Setup necessary code repositories
  - Familiarize with dev environment

- **Oct 5 - Oct 12: Initial Results**
  - Setup working model pipeline
  - Model is capable of producing some results

- **Oct 12 - Nov 3: Model Development**
  - Work towards achieving initial goal of identifying silhouettes
  - Document results and write a preliminary report

- **Nov 3 - Nov 24: Model Completion**
  - Continue refining the model and pushing towards stretch goals
  - Work on poster detailing project results

- **Nov 24 - Dec 5: Wrap Up**
  - Wrap up model development and any stretch goals achieved
  - Write up final report for project

## Dataset

Below are some examples of datasets we will use to achieve our basic functionality, which is to ID individual Pokemon.

### Training Datasets:

1. **Complete Pokemon Image Dataset** - [Kaggle](https://www.kaggle.com/datasets/hlrhegemony/pokemon-image-dataset?resource=download)
   - 2,500+ clean labeled images, all official art, for Generations 1 through 8.

2. **Pokedex Large Images** - [GitHub](https://github.com/cristobalmitchell/pokedex/tree/main/images/large_images)
   - PNG images of all pokemon

3. **Pokemon Silhouettes** - [GitHub](https://github.com/poketwo/data/tree/master/silhouettes)
   - 1,577 images
   - PNG pics of pokemon AND their shaded counterparts of their silhouettes

4. **Pokemon Dataset 1000** - [Kaggle](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000)
   - ~40 images per 1,000 Pokémon species
   - Structured in subdirectories for each class
   - Each image is resized to 128x128 pixels and stored as a PNG file

The model will be trained on a combination of these datasets, with augmentation applied if the initial size proves insufficient.

## Quick Start

Download the necessary dependencies:
```
pip install -r requirements.txt
```

Before training, go to this (link)[https://drive.google.com/file/d/1TVcdKHGLcEeOVd2ACCljFEBOPD5AgrBc/view?usp=sharing] and download the augmented data file essential to model training. Move the download to the project director @ `./data/`.

To reproduce the training step, run:
```python3 train.py```

The repository is currently defaulted to train on the simple-CNN structure. To explore training on other models, 

## Evaluation

Output quality and model performance will be evaluated using metrics relevant in silhouette image classification, including accuracy, precision, recall, F1-score, and confusion matrices.
