# QClojure ML
QClojure ML is the Quantum Machine Learning (QML) extension library for QClojure.

QClojure ML contains QML algorithms and depends on scicloj/noj and scicloj/clay
for the Clojure data science stack and notebook support.
QClojure ML uses QClojure algorithms like Variational Quantum Eigensolver (VQE)
for the quantum part and the data science libraries like noj, tech.ml.dataset and
dtype.next for the classical parts of QML algorithms.

## Features
* Feature encoding
* Training
* Variational Quantum Classifier algorithm

## Usage

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

