Sharpkit.Learn - Machine Learning library for .Net (C#, F#)
==============

Sharpkit.Learn is machine learning library for .Net and mono.
Currently the library is a port of the very nice scikit-learn python library and is very
close to it's code base (hence the name Sharpkit.Learn).
Eventually the codebase may become different.
It is based on Math.Net library and different state-of-the art
machine learning libraries. In particular it uses [liblinear](http://www.csie.ntu.edu.tw/~cjlin/liblinear/).

It supports both sparse and dense matrices. If you need extra
performance, you can use various BLAS providers (like MKL and cublas)
which can be plugged in into Math.Net.

What it contains now
==============
   * Linear regression
   * Logistic regression
   * Ridge regression
   * Ridge classifier
      

What it will contain soon
==============
   * SVM
   * Decision Trees
   * Random Forests
   * Nearest neighbors
   * Naive Bayes
   * Cross validation utilities
   * PCA
   * Ensemble methods

Documentation
===============
   Currently the design is very close to the Scikit-learn python library.
   You can start by reading [Scikit Learn](http://scikit-learn.org/stable/documentation.html) documentation.
   Examples in C# and F# are available [here](EXAMPLES.md)


How you can help
===============
   * Use library in your projects and report issues
   * Port some scikit.learn functionality which you need
      The list is here: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm
   * Implement some missing feature you need     
   * Perform benchmarking
   * Blob about the library.

License
===============

*Math.NET Numerics is covered under the terms of the [MIT/X11](http://mathnetnumerics.codeplex.com/license)
*Sharpkit.Learn is covered under the terms of the BSD Clause 3 license.
*Liblinear is covered under the terms of the Modified BSD License.
