# Deep Learning Malaysia Meetup

Deep Learning Malaysia Meetup</br>
15th Feb 2017</br>
ADAX</br>
Presenter: Poo Kuan Hoong, Ph.D

## Handwritten Recognition using Deep Learning with R

<img src="https://kuanhoong.files.wordpress.com/2016/01/mnistdigits.gif?w=450&h=299">

The [MNIST](http://yann.lecun.com/exdb/mnist/) database consists of handwritten digits. The training set has 60,000 examples, and the test set has 10,000 examples. The MNIST database is a subset of a larger set available from [NIST](http://www.nist.gov/srd/nistsd19.cfm). The digits have been size-normalized and centered in a fixed-size image. The original NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students. For MNIST dataset, half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset.

For the MNIST dataset, the original black and white (bilevel) images from NIST were size normalized to fit in a 20X20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28X28 image (for a total of 784 pixels in total) by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28X28 field.

Download the training and testing dataset from [Kaggle](https://www.kaggle.com/c/digit-recognizer/data).

* [Training Dataset](https://www.kaggle.com/c/digit-recognizer/download/train.csv) 
* [Testing Dataset](https://www.kaggle.com/c/digit-recognizer/download/test.csv)
