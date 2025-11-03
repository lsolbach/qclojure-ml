# QClojure ML
QClojure ML is the Quantum Machine Learning (QML) extension library for QClojure.

QClojure ML contains QML algorithms and depends on scicloj/noj and scicloj/clay
for the Clojure data science stack and notebook support.
QClojure ML uses QClojure algorithms like Variational Quantum Eigensolver (VQE)
for the quantum part and the data science libraries like noj, tech.ml.dataset and
dtype.next for the classical parts of QML algorithms.

[![Clojars Project](https://img.shields.io/clojars/v/org.soulspace/qclojure-ml.svg)](https://clojars.org/org.soulspace/qclojure-ml)
[![cljdoc badge](https://cljdoc.org/badge/org.soulspace/qclojure-ml)](https://cljdoc.org/d/org.soulspace/qclojure-ml)
![GitHub](https://img.shields.io/github/license/lsolbach/qclojure-ml)
[![DOI](https://zenodo.org/badge/1027283360.svg)](https://doi.org/10.5281/zenodo.17138247)

## Features
* Feature encoding
* Training
* Variational Quantum Classifier algorithm
* Quantum Kernels
* Quantum Neural Networks

## Prerequsites
As Clojure runs on the Java Virtual Machine, you need a JVM installed.
While QClojure and QClojure ML will run with Java 11, a recent JVM >= 21 is preferred.

## Usage
QClojure ML is a Clojure library to be used in programs or interactive within the REPL.

To use QClojure ML, add a dependency to your project definition.

See [![Clojars Project](https://img.shields.io/clojars/v/org.soulspace/qclojure-ml.svg)](https://clojars.org/org.soulspace/qclojure-ml)


## Build
QClojure ML is currently build with [Leiningen](https://leiningen.org/).

Compile the code with:

```
lein compile
```

Run the test suite with:

```
lein test
```

## Copyright
© 2025 Ludger Solbach

## License
Eclipse Public License 1.0 (EPL1.0)

