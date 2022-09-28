# Sequential MNIST LSTM

This repository implements a vanilla LSTMs performing the sequential MNIST task where one pixel of an image of a handwritten digit is inputted per timestep. Please note that the image data is normalized prior to ingesting it into the model. The code to reproduce the results in the image below can be found in the [`main.ipynb`](./main.ipynb) notebook. By default, all performance metrics are printed to the console and logged to Weights and Biases, but these lines can be removed without change of performance if so desired.

## Motivation

This project was put together because there is no (to my knowledge) very simple vanilla LSTM implementation of the sequential MNIST task which is open source and from which the full train/test performance is completely measured and tracable. However, since many papers in the space of recurrent neural networks use sequential MNIST as an 'Hello world' task, the comparison of the performance on this task with an LSTM is often made. Unfortunately, it is hard to replicate their results exactly as they do not provide explicit hyperparameter setup.

All in all, I hope that this mini-project can function as a transparent baseline for future research.

## Baseline model

### Configuration

```json
  "criterion": {"name": "CrossEntropyLoss"},
  "model": {
      "input_size": 1,
      "hidden_size": 128,
      "output_size": 10,
  },
  "optimizer": {
      "grad_clip_value": 1.0,
      "lr": 0.001,
      "name": "RMSProp",
      "weight_decay": 0.0,
  },
  "task": {"T": 784, "name": "sequential_mnist"},
  "test": {"batch_size": 1024},
  "train": {"batch_size": 100, "n_epochs": 100},
```

### Final results

| Name          | Value   |
|--------------:|---------|
| Test accuracy | 96.07%  |
| Test loss     | 0.05891 |

### Training overview

<img width="1252" alt="image" src="https://user-images.githubusercontent.com/24796206/192869588-33f628e9-fe2e-4834-8e60-32ce777daa71.png">

Please see [here](https://wandb.ai/douwe/fptt/reports/LSTM-performance-on-sequential-MNIST--VmlldzoyNzExNTQ3?accessToken=njlxulr3l404ak04huo0fkcju9rb0lapu2mdf2tpasy4zz42tuj5t5zlxex679jq) the full train and test performance results not only for the RMSProp trained model (`sandy-feather-252`), but also for SGD and Adam.

## Collaboration

Feel free to open a PR or issue if there is something that needs fixing or that you would like to see improved. Also, if you have trained differently configured models from the three that were orignally trained, feel free to create a PR with their performance metrics. 

## License

MIT
