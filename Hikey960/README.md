## Usage Instructions

This project is based on [`https://github.com/mofanv/darknetz.git`](https://github.com/mofanv/darknetz.git). The original `darknetz` repository is preserved, but **the `ta` folder is fully replaced with the version from this project**.

### Step 1: Setup

First, follow the original instructions from [darknetz README](./README.md) to set up the environment.

Then, **replace the entire `ta` directory** in `darknetz` with the `ta` directory from this project.

Other files from this project (e.g., `host`) should be deployed and compiled **on the `hikey960` board**, where `darknetp` will be built. The necessary code should already exist on the `hikey960`.

---

### Step 2: On `hikey960`

Assuming you have copied the compiled `darknetp` binary to `/usr/local/bin/`, follow the instructions below.

Examples of usage:

```bash
# Example 1: Basic classification, CPU-only, with optional ReLU obfuscation.
darknetp classifier function datasets_config model_config model_weights figure -nogpu (or not) -tuse_tee_relu 0/1

# Example 2: ReLU obfuscation with GPU, using AlexNet (path may vary; check history on hikey960)
darknetp classifier predict cifar.data cfg/alexnet_224.cfg backup/alexnet_224.start.conv.weights /root/data/cifar/train/1000_truck.png -tuse_tee_relu 1

# Example 3: Layer-by-layer loading (all layers), CPU-only
darknetp classifier predict_per_layer cifar.data cfg/alexnet_224.cfg backup/alexnet_224.start.conv.weights /root/data/cifar/train/1000_truck.png -layer 999 -nogpu

# Example 4: Standard GPU inference without obfuscation
darknetp classifier predict cifar.data cfg/alexnet_224.cfg backup/alexnet_224.start.conv.weights /root/data/cifar/train/1000_truck.png
```

---

### Step 3: Demo Mode

Our proposed approach supports demo runs for various models:

* `-demo 1`: ResNet18
* `-demo 2`: VGG-BN
* `-demo 3`: ResNet50
* `-demo 4`: MobileNetV2

#### To enable or disable linear obfuscation:

Modify line 24 in `/home/user/darknet_tee/host/src/parser.h`:

```cpp
// #define RUN_FUSION  // Comment to disable fusion (i.e., disable obfuscation)
```

After changing the above, **recompile on the `hikey960`**.

Examples:

```bash
# ResNet18 without obfuscation, using GPU
# Make sure RUN_FUSION is commented out
darknetp classifier predict_demo cifar.data cfg/resnet18.cfg backup/resnet18.start.conv.weights /root/data/cifar/train/1000_truck.png -demo 1

# ResNet18 with linear obfuscation, using GPU
# Make sure RUN_FUSION is enabled and recompiled on hikey960
darknetp classifier predict_demo cifar.data cfg/resnet18.cfg backup/resnet18.start.conv.weights /root/data/cifar/train/1000_truck.png -demo 1 -tuse_tee_relu 1
```

> ⚠️ **Note**: Please verify the correct paths to `data`, `cfg`, `weights`, and input images, as they may vary depending on your environment and storage layout.

