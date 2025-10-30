(ns qnn-example
  "Comprehensive example demonstrating Quantum Neural Network (QNN) functionality.
  
  This example shows how to:
  1. Create QNN models with different architectures
  2. Set up training data for classification and regression
  3. Train QNNs using quantum backends
  4. Evaluate model performance
  5. Visualize network structures
  
  The examples use the QClojure simulator backend for demonstration."
  (:require
   [org.soulspace.qclojure.ml.application.qnn :as qnn]
   [org.soulspace.qclojure.adapter.backend.ideal-simulator :as sim]
   [clojure.pprint :as pprint]))

;;;
;;; Example 1: Binary Classification QNN
;;;

(comment
  "This example demonstrates binary classification using a QNN.
  We create a simple dataset where features close to [0,0] are class 0
  and features close to [1,1] are class 1."

  ;; Create training data for binary classification
  (def binary-classification-data
    [{:features [0.1 0.2] :label 0}
     {:features [0.2 0.1] :label 0}
     {:features [0.0 0.3] :label 0}
     {:features [0.1 0.0] :label 0}
     {:features [0.8 0.9] :label 1}
     {:features [0.9 0.8] :label 1}
     {:features [1.0 0.7] :label 1}
     {:features [0.7 1.0] :label 1}])

  ;; Create a QNN model for binary classification
  (def binary-qnn-model
    (qnn/create-qnn-model {:num-qubits 2
                           :hidden-layers 2
                           :feature-map-type :angle
                           :activation-type :quantum-tanh}))

  ;; Analyze the model structure
  (println "=== Binary Classification QNN Analysis ===")
  (def binary-analysis ((:analyze binary-qnn-model)))
  (pprint/pprint (:network-structure binary-analysis))

  ;; Display network visualization
  (println "\n" ((:visualize binary-qnn-model)))

  ;; Create simulator backend for training
  (def simulator-backend (sim/create-simulator))

  ;; Training options
  (def training-options
    {:optimizer :nelder-mead
     :loss-type :hinge
     :max-iterations 20
     :learning-rate 0.1
     :shots 1024})

  ;; Train the model
  (def training-result
    ((:train binary-qnn-model) binary-classification-data
                               simulator-backend
                               :options training-options))

  ;; Display training results
  (when (= (:status training-result) :success)
      (println "=== Training Results ===")
      (println "Final cost:" (:final-cost training-result))
      (println "Iterations:" (:iterations training-result))
      (println "Final accuracy:" (:final-accuracy training-result)))

  ;; Test prediction on new data
  (def test-features [0.15 0.25])
  (def prediction ((:predict binary-qnn-model) test-features
                                                 (:trained-parameters training-result)
                                                 simulator-backend))
  (println "Prediction for" test-features ":" prediction)
  )

;;;
;;; Example 2: Multi-Class Classification QNN
;;;

(comment
  "This example demonstrates multi-class classification using a larger QNN.
  We create a dataset with 3 classes representing different regions of feature space."

  ;; Create training data for 3-class classification
  (def multiclass-data
    [{:features [0.1 0.1 0.2] :label 0}  ; Class 0: low values
     {:features [0.0 0.2 0.1] :label 0}
     {:features [0.2 0.0 0.1] :label 0}
     {:features [0.5 0.5 0.6] :label 1}  ; Class 1: medium values
     {:features [0.4 0.6 0.5] :label 1}
     {:features [0.6 0.4 0.5] :label 1}
     {:features [0.9 0.8 0.9] :label 2}  ; Class 2: high values
     {:features [0.8 0.9 0.8] :label 2}
     {:features [1.0 0.7 0.9] :label 2}])

  ;; Create a larger QNN for multi-class classification
  (def multiclass-qnn-model
    (qnn/create-qnn-model {:num-qubits 3
                           :hidden-layers 3
                           :feature-map-type :angle
                           :activation-type :quantum-tanh}))

  ;; Analyze the larger model
  (println "=== Multi-Class QNN Analysis ===")
  (def multiclass-analysis ((:analyze multiclass-qnn-model)))
  (println "Network depth:" (get-in multiclass-analysis [:network-structure :depth]))
  (println "Total parameters:" (get-in multiclass-analysis [:network-structure :total-parameters]))
  (println "Complexity metrics:")
  (pprint/pprint (:complexity-metrics multiclass-analysis))

  ;; Training with cross-entropy loss for multi-class
  (def multiclass-training-options
    {:optimizer :nelder-mead
     :loss-type :cross-entropy
     :max-iterations 30
     :learning-rate 0.05
     :shots 2048}))

;;;
;;; Example 3: Regression QNN
;;;

(comment
  "This example demonstrates quantum regression using a QNN.
  We create a simple function approximation task."

  ;; Create training data for regression (approximating x^2 + y^2)
  (def regression-data
    (for [x (range 0 1.1 0.2)
          y (range 0 1.1 0.2)]
      {:features [x y] 
       :label (+ (* x x) (* y y))}))  ; Target function: x² + y²

  (println "=== Regression Training Data ===")
  (doseq [example (take 5 regression-data)]
    (println "Features:" (:features example) "→ Label:" (:label example)))

  ;; Create QNN for regression
  (def regression-qnn-model
    (qnn/create-qnn-model {:num-qubits 2
                           :hidden-layers 2
                           :feature-map-type :angle
                           :activation-type :quantum-tanh}))

  ;; Training with MSE loss for regression
  (def regression-training-options
    {:optimizer :nelder-mead
     :loss-type :mse
     :max-iterations 50
     :learning-rate 0.01
     :shots 1024}))

;;;
;;; Example 4: Custom QNN Architecture
;;;

(comment
  "This example shows how to create custom QNN architectures manually."

  ;; Create custom layers manually
  (def custom-input-layer
    {:layer-type :input
     :num-qubits 3
     :feature-map-type :amplitude
     :layer-name "Custom Input Encoding"
     :parameter-count 0})

  (def custom-dense-layer
    {:layer-type :dense
     :num-qubits 3
     :layer-name "Custom Dense Layer"
     :parameter-count (qnn/count-layer-parameters {:layer-type :dense :num-qubits 3})})

  (def custom-entangling-layer
    {:layer-type :entangling
     :num-qubits 3
     :entangling-pattern :circular
     :layer-name "Custom Circular Entangling"
     :parameter-count (qnn/count-layer-parameters {:layer-type :entangling :num-qubits 3})})

  (def custom-measurement-layer
    {:layer-type :measurement
     :num-qubits 3
     :measurement-basis :computational
     :layer-name "Custom Measurement"
     :parameter-count 0})

  ;; Assemble custom network
  (def custom-network-layers
    [custom-input-layer
     custom-dense-layer
     custom-entangling-layer
     custom-measurement-layer])

  ;; Allocate parameters for the custom network
  (def custom-network
    (:network (qnn/allocate-parameters custom-network-layers)))

  (println "=== Custom QNN Network ===")
  (println "Custom network valid?" (clojure.spec.alpha/valid? 
                                    :org.soulspace.qclojure.ml.application.qnn/qnn-network 
                                    custom-network))

  ;; Analyze custom network
  (def custom-analysis (qnn/analyze-qnn-network custom-network))
  (println "Custom network structure:")
  (pprint/pprint (:network-structure custom-analysis)))

;;;
;;; Example 5: QNN Performance Analysis
;;;

(comment
  "This example demonstrates how to analyze and compare different QNN architectures."

  ;; Create several QNN models with different configurations
  (def small-qnn (qnn/create-qnn-model {:num-qubits 2 :hidden-layers 1}))
  (def medium-qnn (qnn/create-qnn-model {:num-qubits 3 :hidden-layers 2}))
  (def large-qnn (qnn/create-qnn-model {:num-qubits 4 :hidden-layers 3}))

  ;; Analyze each model
  (def models {:small small-qnn :medium medium-qnn :large large-qnn})

  (println "=== QNN Architecture Comparison ===")
  (doseq [[size model] models]
    (let [analysis ((:analyze model))
          structure (:network-structure analysis)
          complexity (:complexity-metrics analysis)]
      (println (format "\n%s QNN:" (name size)))
      (println "  Depth:" (:depth structure))
      (println "  Parameters:" (:total-parameters structure))
      (println "  Parameters/Qubit:" (:parameters-per-qubit complexity))
      (println "  Expressivity Score:" (:expressivity-score complexity))))

  ;; Performance recommendations
  (println "\n=== Performance Recommendations ===")
  (doseq [[size model] models]
    (let [analysis ((:analyze model))
          recommendations (:recommendations analysis)]
      (when (seq recommendations)
        (println (format "\n%s QNN recommendations:" (name size)))
        (doseq [rec recommendations]
          (println "  -" rec))))))

;;;
;;; Example 6: Quantum Feature Maps
;;;

(comment
  "This example explores different quantum feature map encodings."

  ;; Test different feature map types
  (def angle-encoding-qnn 
    (qnn/create-qnn-model {:num-qubits 2 
                           :hidden-layers 1 
                           :feature-map-type :angle}))

  (def amplitude-encoding-qnn 
    (qnn/create-qnn-model {:num-qubits 2 
                           :hidden-layers 1 
                           :feature-map-type :amplitude}))

  ;; Compare encoding strategies
  (println "=== Feature Map Comparison ===")
  (println "Angle encoding network:")
  (println ((:visualize angle-encoding-qnn)))
  
  (println "\nAmplitude encoding network:")
  (println ((:visualize amplitude-encoding-qnn))))

;;;
;;; Example 7: Training Monitoring and Debugging
;;;

(comment
  "This example shows how to monitor training progress and debug QNN issues."

  ;; Create a QNN model
  (def debug-qnn-model
    (qnn/create-qnn-model {:num-qubits 2
                           :hidden-layers 1}))

  ;; Simple training data
  (def debug-data
    [{:features [0.0 0.0] :label 0}
     {:features [1.0 1.0] :label 1}])

  ;; Test cost function behavior
  (def debug-cost-fn
    (qnn/create-qnn-cost-function (:network debug-qnn-model)
                                  debug-data
                                  simulator-backend
                                  :mse))

  ;; Test with different parameter values
  (println "=== Cost Function Debugging ===")
  (let [zero-params (vec (repeat 9 0.0))
        random-params (vec (repeatedly 9 #(rand 2.0)))
        large-params (vec (repeat 9 10.0))]
    
    (println "Cost with zero parameters:" (debug-cost-fn zero-params))
    (println "Cost with random parameters:" (debug-cost-fn random-params))
    (println "Cost with large parameters:" (debug-cost-fn large-params)))

  ;; Test individual loss functions
  (println "\n=== Loss Function Testing ===")
  (println "MSE(0.5, 1):" (qnn/compute-loss 0.5 1 :mse))
  (println "Cross-entropy([0.3 0.7], 1):" (qnn/compute-loss [0.3 0.7] 1 :cross-entropy))
  (println "Hinge(0.2, 1):" (qnn/compute-loss 0.2 1 :hinge))
  (println "Accuracy('1', 1):" (qnn/compute-loss "1" 1 :accuracy)))

;;;
;;; Example 8: Circuit Analysis
;;;

(comment
  "This example demonstrates how to analyze the quantum circuits generated by QNNs."

  ;; Create a simple QNN
  (def circuit-analysis-qnn
    (qnn/create-qnn-model {:num-qubits 2 :hidden-layers 1}))

  ;; Create sample input
  (def sample-features [0.5 0.7])
  (def sample-parameters (vec (repeatedly 9 #(rand 2.0))))

  ;; Generate the quantum circuit
  (def qnn-circuit
    (qnn/apply-qnn-network (:network circuit-analysis-qnn)
                           sample-features
                           sample-parameters))

  (println "=== QNN Circuit Analysis ===")
  (println "Circuit name:" (:name qnn-circuit))
  (println "Number of qubits:" (:num-qubits qnn-circuit))
  (println "Number of operations:" (count (:operations qnn-circuit)))
  
  ;; Analyze circuit depth and gate types
  (let [operations (:operations qnn-circuit)
        gate-types (map :gate-type operations)
        gate-counts (frequencies gate-types)]
    (println "Gate type distribution:")
    (pprint/pprint gate-counts)))

;;;
;;; Utility Functions for Examples
;;;

(defn run-qnn-experiment
  "Run a complete QNN experiment with the given configuration.
  
  Parameters:
  - config: QNN model configuration
  - training-data: Training dataset
  - test-data: Test dataset (optional)
  - training-options: Training configuration
  
  Returns:
  Experiment results map"
  [config training-data & {:keys [test-data training-options]
                           :or {training-options {}}}]
  
  (let [model (qnn/create-qnn-model config)
        analysis ((:analyze model))
        
        ;; Default training options
        default-options {:optimizer :nelder-mead
                         :loss-type :mse
                         :max-iterations 20
                         :shots 1024}
        options (merge default-options training-options)]
    
    {:model-config config
     :model-analysis analysis
     :training-data-size (count training-data)
     :test-data-size (when test-data (count test-data))
     :training-options options
     :model model
     
     ;; Results placeholders (would be filled during actual training)
     :training-result nil
     :test-accuracy nil
     :final-parameters nil}))

(defn compare-qnn-architectures
  "Compare multiple QNN architectures on the same dataset.
  
  Parameters:
  - configs: Vector of QNN configuration maps
  - training-data: Common training dataset
  
  Returns:
  Comparison results map"
  [configs training-data]
  
  (let [experiments (mapv #(run-qnn-experiment % training-data) configs)]
    
    {:num-architectures (count configs)
     :training-data-size (count training-data)
     :experiments experiments
     :comparison-table
     (mapv (fn [i exp]
             (let [analysis (:model-analysis exp)
                   structure (:network-structure analysis)]
               {:architecture-id i
                :num-qubits (get-in exp [:model-config :num-qubits])
                :hidden-layers (get-in exp [:model-config :hidden-layers])
                :depth (:depth structure)
                :total-parameters (:total-parameters structure)
                :complexity-score (:expressivity-score (:complexity-metrics analysis))}))
           (range)
           experiments)}))

(println "QNN examples loaded successfully!")
(println "Use (comment ...) blocks to run individual examples.")
(println "Examples cover:")
(println "  1. Binary classification")
(println "  2. Multi-class classification") 
(println "  3. Regression")
(println "  4. Custom architectures")
(println "  5. Performance analysis")
(println "  6. Feature map comparison")
(println "  7. Training debugging")
(println "  8. Circuit analysis")
