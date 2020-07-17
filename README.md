# LSQuantization
The PyTorch implementation of Learned Step size Quantization (LSQ) in ICLR2020 (unofficial)

**The project is working in progress, and experimental results on ImageNet are not as good as shown in the paper.**

<img src="alpha_curve.png" width="50%" height="50%">

## Experimental Results
====VGGsmall + Cifar10=======

|      | VGGsmall |
|------|----------|
| fp32 | 93.34    |
| w4a4 | **94.26**    |
| w3a3 | 93.89    |
| w2a2 | 93.42    |

# LSQ
| **LSQ**  | fp32         | w4a4 | w3a3 | w2a2 |
|----------|--------------|------|------|------|
| AlexNet  | 56.55, 79.09 | 56.96, 79.46 [√](https://tensorboard.dev/experiment/MNSkwpg9SJySk201OqJLhw/) | 55.31, 78.59 |  51.18, 75.38 |
