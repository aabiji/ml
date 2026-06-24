You Only Look Once: Detect and classify objects in a single pass of a convolutional network.

- Resize image to 448x448 and divide the image into a 7x7 grid

- Each 64x64 grid cell predicts 2 bounding boxes. Each bounding box has:
  - (x, y) relative to the bounds of the grid cell (normalized)
  - (w, h) relative to the image (normalized, predicting the **square root**)
  - confidence = Pr(Object) * IOU(truth, pred)

- In addition, Pr(Class_i|Object) for each of the 20 labelled classes is outputted if an object is detected.
  At test time, we multiply the conditional class probabilities by the confidence of each bounding box,
  so get confidence scores for each box.

- 7x7x30 model output. At test time after computing confidence scores there would be a 7x7x48 final output
  (x, y, w, h, confidence for each of the 20 classes) * 2

- Architecture: 24 convolutional layers followed by 2 linear layers. Final layer uses a linear **activation function** and the rest of the layers use **leaky ReLU**.

- Pretrain on the ImageNet 1000 class competition dataset (224 x 224 images), using the first 20 convolutional layers followed by an average pooling layer and a linear layer.

- Then train with 4 added convolutional layers and 2 linear layers with randomly initialized weights. The 224 x 224 images are upscaled to 448 x 448.

Intersection over union (IOU) = Overlapping area of 2 bounding boxes / Combined area of 2 bounding boxes
                                How well the predicted bounding box matches the ground truth.

