# GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

**Google**

Neural network scaling has been critical for improving the model quality in many
 real-world machine learning applications with vast amounts of training data and
 compute. Although this trend of scaling is affirmed to be a sure-fire approach for
 better model quality, there are challenges on the path such as the computation cost,
 ease of programming, and efficient implementation on parallel devices. GShard
 is a module composed of a set of lightweight annotation APIs and an extension
 to the XLA compiler. It provides an elegant way to express a wide range of
 parallel computation patterns with minimal changes to the existing model code.
 GShard enabled us to scale up multilingual neural machine translation Transformer
 model with Sparsely-Gated Mixture-of-Experts beyond 600 billion parameters
 using automatic sharding. We demonstrate that such a giant model can efficienctly
 be trained on 2048 TPU v3 accelerators in 4 days to achieve far superior quality
 for translation from 100 languages to English compared to the prior art.

利用自动分片技术，GShard 使我们能够将带有SGMoE（Spararsely-Gated Mixture-of-Experts ）的多语言神经机器翻译 Transformer 模型扩展到超过 6000 亿个参数。我们证明，这样一个巨大的模型可以在 2048 个 TPU v3 加速器上进行高效训练，只需 4 天，就能实现从 100 种语言到英语的翻译，质量远远优于现有技术。


## 什么是MoE

https://www.zhihu.com/tardis/zm/art/677638939

