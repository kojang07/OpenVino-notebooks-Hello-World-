[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkojang07%2520%2F%2520OpenVino-notebooks-Hello-World-&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)

# Hello Image Classification


This basic introduction to OpenVINO™ shows how to do inference with an image classification model.

A pre-trained [MobileNetV3 model](https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/mobilenet-v3-small-1.0-224-tf/README.md) from [Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo/) is used in this tutorial. For more information about how OpenVINO IR models are created, refer to the [TensorFlow to OpenVINO](../tensorflow-classification-to-openvino/tensorflow-classification-to-openvino.ipynb) tutorial.

# mobilenet-v3-small-1.0-224-tf

## Use Case and High-Level Description

`mobilenet-v3-small-1.0-224-tf` is one of MobileNets V3 - next generation of MobileNets,
based on a combination of complementary search techniques as well as a novel architecture design.
`mobilenet-v3-small-1.0-224-tf` is targeted for low resource use cases.
For details see [paper](https://arxiv.org/abs/1905.02244).

## Specification

| Metric                          | Value                                     |
|---------------------------------|-------------------------------------------|
| Type                            | Classification                            |
| GFlops                          | 0.11682                                   |
| MParams                         | 2.537                                     |
| Source framework                | TensorFlow\*                              |

## Accuracy

| Metric | Original model | Converted model |
| ------ | -------------- | --------------- |
| Top 1  | 67.36%         | 67.36%          |
| Top 5  | 87.44%         | 87.44%          |

## Input

### Original Model

Image, name: `input_1`, shape: `1, 224, 224, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `RGB`.

### Converted Model

Image, name: `input_1`, shape: `1, 224, 224, 3`, format: `B, H, W, C`, where:

- `B` - batch size
- `H` - image height
- `W` - image width
- `C` - number of channels

Expected color order: `BGR`.

## Output

### Original Model

Object classifier according to ImageNet classes, name - `StatefulPartitionedCall/MobilenetV3small/Predictions/Softmax`,  shape - `1, 1000`, output data format is `B, C` where:

- `B` - batch size
- `C` - Predicted probabilities for each class in  [0, 1] range

