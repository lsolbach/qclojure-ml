# Scicloj Integration for QClojure-ML

A comprehensive guide for integrating QClojure-ML with the Scicloj ecosystem for enhanced data science and machine learning workflows.

## Overview

The Scicloj ecosystem provides a rich set of tools for data science in Clojure. QClojure-ML is designed to integrate seamlessly with these tools to provide a complete quantum machine learning pipeline. This guide explores how to leverage Scicloj infrastructure for:

- **Data preprocessing** with tablecloth and tech.ml.dataset
- **Statistical analysis** with fastmath and kixi.stats  
- **Visualization** with hanami and plotly-clj
- **ML metrics** with scicloj.ml
- **Numerical computing** with dtype-next and neanderthal

## Current Integration Status

QClojure-ML currently includes foundation components for Scicloj integration:

### ✅ Implemented
- Dataset creation and management utilities (`create-qml-dataset`, `split-dataset`)
- ML metrics calculation (accuracy, precision, recall, F1, MSE, RMSE, R²) via `calculate-ml-metrics`
- ASCII visualization for training progress (`visualize-training-progress`)
- Normalization with fallback to built-in methods (`normalize-features-with-noj`)
- Tablecloth integration in notebooks (see `notebooks/vqc_iris_example.clj`)
- Noj ecosystem integration via single dependency
- Training infrastructure via `train-qml-model`
- Quantum kernel methods (`quantum-kernel-matrix`, `quantum-kernel-overlap`)
- Multiple data encoding strategies (angle, amplitude, basis, IQP)

## Dependencies

QClojure-ML uses the Scicloj **Noj** library, which bundles a comprehensive, pre-tested collection of data science libraries:

```clojure
:dependencies [
  ;; Core QClojure-ML with Noj ecosystem
  [org.soulspace/qclojure-ml "0.2.0-SNAPSHOT"]
  [org.scicloj/noj "2-beta18"]
]
```

### What's Included in Noj

Noj already includes these essential libraries (no need to add separately):

- **tablecloth** `7.042` - Data manipulation
- **tech.ml.dataset** - Dataset abstraction
- **dtype-next** - High-performance numerical arrays
- **fastmath** `3.0.0-alpha3` - Mathematical functions (v3)
- **hanami** `0.20.1` - Grammar of graphics visualization
- **kindly** - Data visualization specification
- **clay** - REPL-friendly notebooks
- **metamorph.ml** - Machine learning pipelines
- **libpython-clj** - Python interoperability
- **clojisr** - R language interoperability

### Optional Additional Libraries

For specialized use cases, you may want to add:

```clojure
:dependencies [
  ;; ... noj dependencies above ...
  
  ;; Statistical analysis (not in noj core)
  [kixi/stats "0.5.5"]                  ; Statistical functions with transducers
  
  ;; High-performance linear algebra (not in noj core)
  [uncomplicate/neanderthal "0.50.0"]   ; GPU-accelerated numerical computing
  
  ;; Clustering (recommended by noj, not included)
  [generateme/fastmath-clustering "3.0.0"] ; Smile clustering algorithms
]
```

## Data Pipeline Integration

### Enhanced Dataset Management with Tablecloth

Since Noj includes tablecloth, you can directly use it for data manipulation:

```clojure
(require '[tablecloth.api :as tc]
         '[org.soulspace.qclojure.ml.application.training :as training])

;; Create dataset from tablecloth (this pattern is used in notebooks/vqc_iris_example.clj)
(defn create-qml-dataset-from-tablecloth
  "Create QML dataset from tablecloth dataset"
  [tc-dataset feature-columns label-column task-type]
  (let [;; Extract features as vectors
        features (mapv (fn [row]
                        (mapv #(get row %) feature-columns))
                      (tc/rows tc-dataset :as-maps))
        ;; Extract labels
        labels (vec (tc/column tc-dataset label-column))]
    (training/create-qml-dataset features labels task-type)))

;; Example usage with remote dataset
(def iris-url "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
(def iris-tc (tc/dataset iris-url))

(def iris-qml (create-qml-dataset-from-tablecloth 
                iris-tc 
                [:sepal_length :sepal_width :petal_length :petal_width]
                :species
                :classification))
```

### Advanced Data Preprocessing

Using fastmath (v3, included in Noj) for preprocessing:

```clojure
(require '[fastmath.stats :as fstats]
         '[fastmath.core :as fm])

(defn preprocess-features-with-scicloj
  "Advanced feature preprocessing using Scicloj tools"
  [tc-dataset feature-columns]
  (-> tc-dataset
      ;; Standardize features using fastmath
      (tc/map-columns (fn [col]
                        (let [col-vec (vec col)
                              mean (fstats/mean col-vec)
                              std (fstats/stddev col-vec)]
                          (map #(/ (- % mean) std) col)))
                      feature-columns)
      ;; Handle missing values
      (tc/replace-missing :down)
      ;; Apply transformations
      (tc/map-columns fm/log1p [:feature1 :feature2])))

(defn detect-outliers
  "Detect outliers using IQR method with fastmath"
  [tc-dataset column]
  (let [values (vec (tc/column tc-dataset column))
        q1 (fstats/quantile values 0.25)
        q3 (fstats/quantile values 0.75)
        iqr (- q3 q1)
        lower-bound (- q1 (* 1.5 iqr))
        upper-bound (+ q3 (* 1.5 iqr))]
    (tc/select-rows tc-dataset 
                    (fn [row] 
                      (let [val (get row column)]
                        (or (< val lower-bound) (> val upper-bound)))))))
```

**Note:** For additional statistical functions like hypothesis testing, add `kixi/stats` as an optional dependency.

## Visualization Integration

QClojure-ML includes built-in ASCII visualization for quick feedback during REPL development:

```clojure
(require '[org.soulspace.qclojure.ml.application.training :as training])

;; ASCII plot of training progress
(defn show-training-progress
  [training-history]
  (println (training/visualize-training-progress training-history 60)))
```

The ASCII visualization displays training cost over iterations as a simple text-based plot, useful for monitoring convergence without external dependencies.

## Statistical Analysis Integration

Fastmath v3 (included in Noj) provides comprehensive statistical functions:

```clojure
(require '[fastmath.stats :as fstats])

(defn analyze-parameter-distribution
  "Analyze distribution of optimized parameters using fastmath"
  [optimal-parameters]
  (let [params-vec (vec optimal-parameters)]
    {:mean (fstats/mean params-vec)
     :variance (fstats/variance params-vec)
     :stddev (fstats/stddev params-vec)
     :median (fstats/median params-vec)
     :kurtosis (fstats/kurtosis params-vec)
     :skewness (fstats/skewness params-vec)}))

(defn compare-quantum-classical
  "Compare quantum ML results with classical baseline"
  [quantum-metrics classical-metrics]
  {:quantum-advantage (- (:accuracy quantum-metrics) (:accuracy classical-metrics))
   :precision-improvement (- (:precision quantum-metrics) (:precision classical-metrics))
   :f1-improvement (- (:f1-score quantum-metrics) (:f1-score classical-metrics))})
```

### Cross-Validation with Tablecloth

```clojure
(require '[fastmath.stats :as fstats])

(defn k-fold-cross-validation
  "K-fold cross-validation for quantum ML models"
  [dataset ansatz k-folds training-config feature-cols label-col]
  (let [;; Shuffle and create fold indices
        n-samples (tc/row-count dataset)
        fold-size (quot n-samples k-folds)
        shuffled (tc/shuffle dataset)
        
        ;; Create k folds
        results (for [test-fold (range k-folds)]
                   (let [test-start (* test-fold fold-size)
                         test-end (if (= test-fold (dec k-folds))
                                   n-samples
                                   (* (inc test-fold) fold-size))
                         
                         ;; Split data
                         test-data (tc/select-rows shuffled (range test-start test-end))
                         train-data (tc/concat 
                                      (tc/select-rows shuffled (range 0 test-start))
                                      (tc/select-rows shuffled (range test-end n-samples)))
                         
                         ;; Convert to QML format
                         train-qml (create-qml-dataset-from-tablecloth 
                                     train-data feature-cols label-col :classification)
                         test-qml (create-qml-dataset-from-tablecloth
                                    test-data feature-cols label-col :classification)
                         
                         ;; Train model
                         result (training/train-qml-model ansatz train-qml training-config)
                         
                         ;; Evaluate on test fold
                         ;; Note: Implement prediction function for actual evaluation
                         test-metrics {:accuracy (rand) :precision (rand)}]
                     
                     test-metrics))]
    
    {:mean-accuracy (fstats/mean (map :accuracy results))
     :std-accuracy (fstats/stddev (map :accuracy results))
     :mean-precision (fstats/mean (map :precision results))
     :fold-results results}))
```

## Quantum Kernel Methods

QClojure-ML provides quantum kernel methods for computing similarity measures between data points:

```clojure
(require '[org.soulspace.qclojure.ml.application.quantum-kernel :as qk]
         '[org.soulspace.qclojure.adapter.backend.ideal-simulator :as sim])

;; Initialize backend
(def backend (sim/create-simulator))

;; Sample data
(def sample-data [[0.1 0.2] [0.8 0.9] [0.2 0.1]])

;; Configure quantum kernel
(def kernel-config
  {:encoding-type :angle
   :num-qubits 2
   :shots 1024
   :encoding-options {:gate-type :ry}})

;; Compute kernel matrix (pairwise similarities)
(def kernel-matrix 
  (qk/quantum-kernel-matrix backend sample-data kernel-config))

;; Compute overlap between two specific points
(def overlap-result 
  (qk/quantum-kernel-overlap backend 
                            (first sample-data) 
                            (second sample-data) 
                            kernel-config))

;; Analyze kernel matrix properties
(def analysis (qk/analyze-kernel-matrix kernel-matrix))

;; Create kernel function for ML pipelines
(def kernel-fn (qk/create-quantum-kernel backend kernel-config))
(kernel-fn [0.1 0.2] [0.8 0.9])  ; Returns kernel value
```

The quantum kernel implementation uses SWAP test circuits for hardware-compatible overlap estimation and supports multiple encoding strategies (angle, amplitude, basis, IQP).

## Data Encoding Strategies

QClojure-ML supports multiple strategies for encoding classical data into quantum states:

```clojure
(require '[org.soulspace.qclojure.ml.application.encoding :as encoding])

;; Angle encoding - encodes features as rotation angles
(def angle-encoder (encoding/angle-encoding [0.5 0.3 0.1] 2 :ry))

;; Amplitude encoding - encodes features in quantum amplitudes
(def amp-encoder (encoding/amplitude-encoding [0.5 0.5 0.5 0.5] 2))

;; Basis encoding - encodes binary features in computational basis
(def basis-encoder (encoding/basis-encoding [0 1 1 0] 2))

;; IQP encoding - Instantaneous Quantum Polynomial encoding
(def iqp-encoder (encoding/iqp-encoding [0.1 0.2 0.3 0.4] 2))

;; Feature normalization
(def norm-result (encoding/normalize-features [1.0 2.0 3.0 4.0]))
(when (:success norm-result)
  (println "Normalized:" (:result norm-result)))
```

Each encoding strategy has different properties suitable for various machine learning tasks.

## Complete Workflow Example

This example demonstrates a real-world workflow using QClojure-ML with Noj. See `notebooks/vqc_iris_example.clj` for the complete working implementation.

```clojure
(require '[tablecloth.api :as tc]
         '[fastmath.stats :as fstats]
         '[org.soulspace.qclojure.ml.application.training :as training]
         '[org.soulspace.qclojure.domain.ansatz :as ansatz]
         '[org.soulspace.qclojure.adapter.backend.ideal-simulator :as sim])

(defn quantum-ml-workflow
  "Complete quantum ML workflow with Noj/Scicloj integration"
  [data-url feature-cols label-col]
  
  ;; 1. Data loading and preprocessing with tablecloth
  (let [raw-data (tc/dataset data-url)
        
        ;; 2. Normalize features
        processed-data (-> raw-data
                           (tc/map-columns (fn [col]
                                            (let [col-vec (vec col)
                                                  min-val (apply min col-vec)
                                                  max-val (apply max col-vec)
                                                  range-val (- max-val min-val)]
                                              (if (> range-val 0)
                                                (map #(/ (- % min-val) range-val) col-vec)
                                                col-vec)))
                                          feature-cols))
        
        ;; 3. Convert to QML format
        qml-dataset (create-qml-dataset-from-tablecloth 
                      processed-data feature-cols label-col :classification)
        
        ;; 4. Split data for training/testing
        split-data (training/split-dataset qml-dataset 0.8 :random-seed 42)
        
        ;; 5. Create quantum circuit ansatz
        circuit-ansatz (ansatz/hardware-efficient-ansatz 2 3)
        
        ;; 6. Configure and train model
        training-result (training/train-qml-model 
                          circuit-ansatz 
                          (:train split-data) 
                          {:max-iterations 100
                           :optimizer :nelder-mead
                           :loss-function :cross-entropy
                           :backend (sim/create-backend)})
        
        ;; 7. Calculate metrics on test set
        metrics (training/calculate-ml-metrics 
                  (:labels (:test split-data))
                  predicted-labels  ; from model predictions
                  :classification)
        
        ;; 8. Visualize training progress
        training-viz (training/visualize-training-progress 
                       (:training-history training-result) 60)
        
        ;; 9. Analyze parameter distribution
        param-stats {:mean (fstats/mean (:optimal-parameters training-result))
                     :stddev (fstats/stddev (:optimal-parameters training-result))}]
    
    {:model training-result
     :metrics metrics
     :training-viz training-viz
     :param-analysis param-stats}))

;; Example usage with Iris dataset
(def result 
  (quantum-ml-workflow
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    [:sepal_length :sepal_width :petal_length :petal_width]
    :species))
```

## Implementation Summary

### ✅ Available Now
1. **Noj integration** - Single dependency provides tablecloth, hanami, fastmath v3, tech.ml.dataset, clay
2. **Tablecloth support** - Working data manipulation example in `notebooks/vqc_iris_example.clj`
3. **Dataset utilities** - `create-qml-dataset`, `split-dataset` for ML workflows
4. **Training infrastructure** - `train-qml-model` with multiple optimization strategies
5. **ML metrics** - Comprehensive metrics via `calculate-ml-metrics` (classification and regression)
6. **Quantum kernels** - Production-ready kernel methods with SWAP test implementation
7. **Data encoding** - Multiple strategies (angle, amplitude, basis, IQP encoding)
8. **ASCII visualization** - Built-in training progress monitoring
9. **Fastmath statistics** - Statistical analysis using fastmath v3

### � Integration Patterns
- Load data with tablecloth from local files or URLs
- Preprocess with fastmath statistical functions
- Convert to QML dataset format
- Train with quantum circuits and ansatzes
- Evaluate with standard ML metrics
- Visualize training progress with ASCII plots

## Resources

- **Noj Documentation**: https://scicloj.github.io/noj/
- **Tablecloth Guide**: https://scicloj.github.io/tablecloth/
- **Fastmath**: https://github.com/generateme/fastmath
- **QClojure-ML Examples**: `notebooks/vqc_iris_example.clj`
- **QClojure**: https://github.com/lsolbach/qclojure

This integration makes QClojure-ML a functional member of the Scicloj ecosystem, providing quantum machine learning capabilities that work alongside standard Clojure data science tools.
