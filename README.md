# Vision-Transformer-ViT

The overall structure of the vision transformer architecture consists of the following steps:

1. Split an image into patches (fixed sizes)
2. Flatten the image patches
3. Create lower-dimensional linear embeddings from these flattened image patches
4. Include positional embeddings
5. Feed the sequence as an input to a SOTA transformer encoder
6. Pre-train the ViT model with image labels, then fully supervised on a big dataset
7. Fine-tune the downstream dataset for image classification


https://viso.ai/deep-learning/vision-transformer-vit/
