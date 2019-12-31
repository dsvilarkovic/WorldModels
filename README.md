# WorldModels
An implementation of the ideas from this paper https://arxiv.org/pdf/1803.10122.pdf

Code base adapted from https://github.com/hardmaru/estool

For full installation and run instructions see this blog post:

https://applied-data.science/blog/hallucinogenic-deep-reinforcement-learning-using-python-and-keras


## Leonhard/Euler cluster settings
For downloading Google Drive files and folders to cluster:

Follow installation instructions:

https://github.com/gdrive-org/gdrive

We also need some Google Drive API adjustments so follow instructions on this issue:
https://github.com/gdrive-org/gdrive/issues/506#issuecomment-567253689

For getting needed packages:
<code>module load StdEnv openmpi/4.0.1 openblas/0.2.19 libpng/1.6.27 gtkplus/3.20.10 mesa/12.0.3 sdl2/2.0.5 gcc/4.8.5 cmake/3.4.3 jpeg/9b python_gpu/3.6.1 mesa-glu/9.0.0 libgme/0.6.2 boost/1.62.0 eth_proxy mesa/12.0.6 mesa-glu/9.0.0 opencv/3.4.6 ffmpeg/3.2.14 hdf5/1.10.1</code>

<code>python -m pip install --user keras </code>

For running RNN train:

<code>python 04_train_rnn.py --new_model --batch_size 100</code>
