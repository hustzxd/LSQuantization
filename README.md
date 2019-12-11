# LSQuantization
The PyTorch implementation of Learned Step size Quantization (LSQ) in ICLR2020 open review (unofficial)

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
