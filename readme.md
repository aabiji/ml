A repository of jupyter notebook exercises and notes for Understanding Deep Learning, as well as some small projects to apply what I've learnt so far.

Plan:
- Review/rewrite all notes
- Projects:
  - Implement [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) from scratch using
    numpy to make sure I really understand CNNs.
      1. Implement the paper verbatim using the AlexNet dataset.
      2. Might be cool to visualize the convolutional kernels after training is done
      3. Change the loss function in order to classify multilabel images of facial expressions using the EMOTIC dataset.
  - Implement object detection and segmentation: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640)
  - Implement [YamNet](https://arxiv.org/pdf/2002.00476) to classify audio. Apply this model to label the bird species from clips of their songs.
- Read chapter 11
- Play around with different cellular automata to play around with the emergent complexity of life

# Ideas
- Implement papers:
  - Gaussian splatting
    - [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/pdf/2308.04079)

  - Word2Vec
    - [Can we turn music audio into embeddings?](https://arxiv.org/html/2501.01108v1)
    - [Can we turn images into embeddings?](https://openai.com/index/clip/)
    - Implement Skip-Gram and Negative Sampling. Learn embeddings, contrastive objectives and representational learning.

  - Attention Is All You Need
    - Implement multi-head attention, positional encoding, encoder, decoder.

  - CLIP
    - Learn multimodel learning, contrastive learning, embeddings, zero-shot classification.

  - SegFormer
    - Learn segmentation and dense prediction.

  - Speech to text and text to speech
    - [Tacotron](https://arxiv.org/pdf/1703.10135)
    - [Whisper](https://arxiv.org/pdf/2212.04356)

- Small projects:
  - Real time pose estimation with YoloV8

  - Add an env to PufferLib

  - Build a model to predict next period date from past period dates

Setup:
```bash
sudo apt install okular latexmk
```
