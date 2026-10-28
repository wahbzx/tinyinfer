## Tinyinfer - A small inference engine for local inference of open weight models

This is just a small project to practically apply what I'm learning with CUDA.
This will not be anything close to something like vllm or llama.cpp, but is more of a learning project to:
- Understand the intricacies of different transformer architectures from first principles
- Apply parallel programming with concepts with cuda for a real use case
- Practice optimisation techniques for faster inference on my RTX 5070 Ti

For this project I will be attempting to get inference working on the [TinyLlama 1.1b model](https://github.com/jzhang38/tinyllama)
No quanitization (yet), for now focusing on FP16 suporting the SafeTensors format
