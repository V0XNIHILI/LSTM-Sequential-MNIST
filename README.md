# Sequential MNIST LSTM

## Baseline

### Configuration

```json
  "criterion": {"name": "CrossEntropyLoss"},
  "model": {
      "hidden_size": 128,
      "input_size": 1,
      "output_size": 10,
  },
  "optimizer": {
      "grad_clip_value": 1.0,
      "lr": 0.001,
      "momentum": 0.9,
      "name": "RMSProp",
      "weight_decay": 0.0,
  },
  "task": {"T": 784, "name": "sequential_mnist"},
  "test": {"batch_size": 1024},
  "train": {"batch_size": 100, "n_epochs": 100},
```

### Performance

| Name          | Value   |
|--------------:|---------|
| Test accuracy | 96.07%  |
| Test loss     | 0.05891 |

### Alternatives

Please see [here](https://wandb.ai/douwe/fptt/reports/LSTM-performance-on-sequential-MNIST--VmlldzoyNzExNTQ3?accessToken=njlxulr3l404ak04huo0fkcju9rb0lapu2mdf2tpasy4zz42tuj5t5zlxex679jq) the train and test performance results not only for the RMSProp trained model, but also for SGD and Adam.
