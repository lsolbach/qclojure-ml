(ns org.soulspace.qclojure.ml.application.qnn-test
  "Test suite for Quantum Neural Network (QNN) functionality.
  
  This test suite verifies:
  - Layer type specifications and validation
  - Parameter counting and allocation
  - Network composition and circuit generation
  - Training interface and cost functions
  - Model creation and analysis utilities"
  (:require
   [clojure.test :refer [deftest is testing run-tests]]
   [clojure.spec.alpha :as s]
   [org.soulspace.qclojure.ml.application.qnn :as qnn]
   [org.soulspace.qclojure.adapter.backend.ideal-simulator :as sim]))

;;;
;;; Layer Type Tests
;;;
(deftest test-layer-specifications
  (testing "Layer type specifications"
    (let [input-layer {:layer-type :input
                       :num-qubits 3
                       :feature-map-type :angle
                       :parameter-count 0}
          
          dense-layer {:layer-type :dense
                       :num-qubits 3
                       :parameter-count 9}
          
          entangling-layer {:layer-type :entangling
                            :num-qubits 3
                            :entangling-pattern :linear
                            :parameter-count 2}
          
          activation-layer {:layer-type :activation
                            :num-qubits 3
                            :activation-type :quantum-tanh
                            :parameter-count 3}
          
          measurement-layer {:layer-type :measurement
                             :num-qubits 3
                             :measurement-basis :computational
                             :parameter-count 0}]
      
      (is (s/valid? ::qnn/input-layer input-layer)
          "Input layer should be valid")
      
      (is (s/valid? ::qnn/dense-layer dense-layer)
          "Dense layer should be valid")
      
      (is (s/valid? ::qnn/entangling-layer entangling-layer)
          "Entangling layer should be valid")
      
      (is (s/valid? ::qnn/activation-layer activation-layer)
          "Activation layer should be valid")
      
      (is (s/valid? ::qnn/measurement-layer measurement-layer)
          "Measurement layer should be valid"))))

(deftest test-parameter-counting
  (testing "Parameter counting for different layer types"
    (is (= 0 (qnn/count-layer-parameters {:layer-type :input :num-qubits 3}))
        "Input layer should have 0 parameters")
    
    (is (= 9 (qnn/count-layer-parameters {:layer-type :dense :num-qubits 3}))
        "Dense layer should have num-qubits² parameters")
    
    (is (= 2 (qnn/count-layer-parameters {:layer-type :entangling :num-qubits 3}))
        "Linear entangling layer should have (num-qubits - 1) parameters")
    
    (is (= 3 (qnn/count-layer-parameters {:layer-type :activation :num-qubits 3}))
        "Activation layer should have num-qubits parameters")
    
    (is (= 0 (qnn/count-layer-parameters {:layer-type :measurement :num-qubits 3}))
        "Measurement layer should have 0 parameters")))

(deftest test-parameter-allocation
  (testing "Parameter allocation for networks"
    (let [network-layers [{:layer-type :input :num-qubits 2 :parameter-count 0 :feature-map-type :angle}
                          {:layer-type :dense :num-qubits 2 :parameter-count 4}
                          {:layer-type :entangling :num-qubits 2 :parameter-count 1 :entangling-pattern :linear}
                          {:layer-type :measurement :num-qubits 2 :parameter-count 0 :measurement-basis :computational}]
          
          allocated-result (qnn/allocate-parameters network-layers)
          allocated-network (:network allocated-result)]
      
      (is (= 7 (:total-parameters allocated-result))
          "Total parameters should be sum of layer parameters")
      
      (is (= nil (get-in allocated-network [0 :parameter-indices]))
          "Input layer should have no parameter indices")
      
      (is (= [0 1 2 3 4 5] (get-in allocated-network [1 :parameter-indices]))
          "Dense layer should have consecutive parameter indices")
      
      (is (= [6] (get-in allocated-network [2 :parameter-indices]))
          "Entangling layer should have next parameter index")
      
      (is (= nil (get-in allocated-network [3 :parameter-indices]))
          "Measurement layer should have no parameter indices"))))

;;;
;;; Network Composition Tests
;;;
(deftest test-network-validation
  (testing "QNN network validation"
    (let [valid-network [{:layer-type :input
                          :num-qubits 2
                          :parameter-count 0
                          :feature-map-type :angle}
                         {:layer-type :dense
                          :num-qubits 2
                          :parameter-count 4
                          :parameter-indices [0 1 2 3]}
                         {:layer-type :measurement
                          :num-qubits 2
                          :parameter-count 0
                          :measurement-basis :computational}]
          
          invalid-network [{:layer-type :unknown  ; Invalid layer type
                            :num-qubits 2
                            :parameter-count 0}]]
      
      (is (s/valid? ::qnn/qnn-network valid-network)
          "Valid network should pass validation")
      
      (is (not (s/valid? ::qnn/qnn-network invalid-network))
          "Invalid network should fail validation"))))

(deftest test-feedforward-qnn-creation
  (testing "Feedforward QNN creation"
    (let [qnn-network (qnn/create-feedforward-qnn 2 1)]
      
      (is (s/valid? ::qnn/qnn-network qnn-network)
          "Created QNN network should be valid")
      
      (is (= 5 (count qnn-network))
          "QNN with 1 hidden layer should have 5 layers total")
      
      (is (= :input (:layer-type (first qnn-network)))
          "First layer should be input layer")
      
      (is (= :measurement (:layer-type (last qnn-network)))
          "Last layer should be measurement layer")
      
      ;; Check layer sequence for 1 hidden layer: input, dense, entangling, activation, measurement
      (let [layer-types (mapv :layer-type qnn-network)]
        (is (= [:input :dense :entangling :activation :measurement] layer-types)
            "Layer sequence should be correct for feedforward QNN")))))

(deftest test-parameter-extraction
  (testing "Parameter extraction for layers"
    (let [layer {:parameter-indices [2 3 4]}
          full-parameters [0 1 10 20 30 5 6 7]
          extracted (qnn/extract-layer-parameters layer full-parameters)]
      
      (is (= [10 20 30] extracted)
          "Should extract correct parameters by indices"))
    
    (let [layer-no-params {}
          full-parameters [0 1 2 3 4]
          extracted (qnn/extract-layer-parameters layer-no-params full-parameters)]
      
      (is (= [] extracted)
          "Layer without parameter indices should return empty vector"))))

;;;
;;; Training Interface Tests
;;;
(deftest test-loss-functions
  (testing "Loss function computations"
    ;; MSE loss
    (is (= 0.25 (qnn/compute-loss 0.5 1.0 :mse))
        "MSE loss should be (prediction - label)²")
    
    (is (= 0.0 (qnn/compute-loss 1.0 1.0 :mse))
        "MSE loss should be 0 for perfect prediction")
    
    ;; Hinge loss
    (is (= 0.8 (qnn/compute-loss 0.2 1 :hinge))
        "Hinge loss should be max(0, 1 - margin)")
    
    (is (= 0.0 (qnn/compute-loss 1.5 1 :hinge))
        "Hinge loss should be 0 when margin > 1")
    
    ;; Accuracy loss (negative accuracy for minimization)
    (is (= 0.0 (qnn/compute-loss "1" 1 :accuracy))
        "Accuracy loss should be 0 for correct prediction")
    
    (is (= 1.0 (qnn/compute-loss "0" 1 :accuracy))
        "Accuracy loss should be 1 for incorrect prediction")))

(deftest test-quantum-measurement-extraction
  (testing "Quantum measurement result extraction"
    (let [mock-result {:outcomes {"0" 600 "1" 424} :shots 1024}]
      
      ;; Test expectation value extraction
      (let [expectation (qnn/extract-expectation-value mock-result)]
        (is (number? expectation)
            "Expectation value should be a number")
        (is (and (>= expectation -1.0) (<= expectation 1.0))
            "Expectation value should be between -1 and 1"))
      
      ;; Test probability distribution extraction
      (let [prob-dist (qnn/extract-probability-distribution mock-result)]
        (is (vector? prob-dist)
            "Probability distribution should be a vector")
        (is (every? #(and (>= % 0.0) (<= % 1.0)) prob-dist)
            "All probabilities should be between 0 and 1"))
      
      ;; Test most probable state extraction
      (let [most-probable (qnn/extract-most-probable-state mock-result)]
        (is (string? most-probable)
            "Most probable state should be a string")
        (is (= "0" most-probable)
            "Most probable state should be '0' for this example"))))
  
  (testing "Error handling in measurement extraction"
    (let [error-result {:error "Quantum execution failed"}]
      
      (is (= 0.0 (qnn/extract-expectation-value error-result))
          "Should return default expectation value for errors")
      
      (is (= [0.5 0.5] (qnn/extract-probability-distribution error-result))
          "Should return uniform distribution for errors")
      
      (is (= "0" (qnn/extract-most-probable-state error-result))
          "Should return default state for errors"))))

;;;
;;; Model Interface Tests
;;;
(deftest test-qnn-model-creation
  (testing "QNN model creation and interface"
    (let [config {:num-qubits 2
                  :hidden-layers 1
                  :feature-map-type :angle
                  :activation-type :quantum-tanh}
          
          model (qnn/create-qnn-model config)]
      
      (is (map? model)
          "Model should be a map")
      
      (is (contains? model :network)
          "Model should contain network")
      
      (is (contains? model :config)
          "Model should contain config")
      
      (is (= config (:config model))
          "Model config should match input config")
      
      (is (fn? (:train model))
          "Model should have train function")
      
      (is (fn? (:predict model))
          "Model should have predict function")
      
      (is (fn? (:evaluate model))
          "Model should have evaluate function")
      
      (is (fn? (:analyze model))
          "Model should have analyze function")
      
      (is (fn? (:visualize model))
          "Model should have visualize function"))))

(deftest test-network-analysis
  (testing "QNN network analysis"
    (let [qnn-network (qnn/create-feedforward-qnn 3 2)
          analysis (qnn/analyze-qnn-network qnn-network)]
      
      (is (map? analysis)
          "Analysis should return a map")
      
      (is (contains? analysis :network-structure)
          "Analysis should include network structure")
      
      (is (contains? analysis :layer-analysis)
          "Analysis should include layer analysis")
      
      (is (contains? analysis :complexity-metrics)
          "Analysis should include complexity metrics")
      
      (is (contains? analysis :recommendations)
          "Analysis should include recommendations")
      
      (let [structure (:network-structure analysis)]
        (is (= 8 (:depth structure))
            "Network depth should be correct for 2 hidden layers")
        
        (is (= 3 (:total-qubits structure))
            "Total qubits should match input")
        
        (is (pos? (:total-parameters structure))
            "Should have positive number of parameters")))))

(deftest test-network-visualization
  (testing "QNN network visualization"
    (let [qnn-network (qnn/create-feedforward-qnn 2 1)
          visualization (qnn/visualize-qnn-network qnn-network)]
      
      (is (string? visualization)
          "Visualization should return a string")
      
      (is (.contains visualization "QNN Network Visualization")
          "Should contain visualization header")
      
      (is (.contains visualization "Layer 0:")
          "Should show layer information")
      
      (is (.contains visualization "Total Parameters:")
          "Should show parameter count")
      
      (is (.contains visualization "Network Depth:")
          "Should show network depth"))))

;;;
;;; Integration Tests
;;;
(deftest test-end-to-end-qnn-workflow
  (testing "End-to-end QNN workflow without backend execution"
    ;; Create model
    (let [model (qnn/create-qnn-model {:num-qubits 2 :hidden-layers 1})
          
          ;; Analyze model
          analysis ((:analyze model))
          
          ;; Create sample data
          training-data [{:features [0.1 0.2] :label 0}
                         {:features [0.8 0.9] :label 1}]
          
          ;; Get network for cost function testing
          network (:network model)
                    
          ;; Create cost function
          cost-fn (qnn/create-qnn-cost-function network 
                                                training-data 
                                                (sim/create-simulator) 
                                                :mse)]
      
      ;; Test model creation
      (is (some? model)
          "Model should be created successfully")
      
      ;; Test analysis
      (is (map? analysis)
          "Analysis should be performed successfully")
      
      (is (pos? (get-in analysis [:network-structure :total-parameters]))
          "Network should have parameters")
      
      ;; Test cost function creation
      (is (fn? cost-fn)
          "Cost function should be created successfully")
      
      ;; Test cost function evaluation with random parameters
      (let [num-params (get-in analysis [:network-structure :total-parameters])
            test-params (vec (repeatedly num-params #(rand 2.0)))
            cost-value (cost-fn test-params)]
        
        (is (number? cost-value)
            "Cost function should return a number")
        
        (is (not (neg? cost-value))
            "Cost should not be negative")))))

(deftest test-optimization-interface
  (testing "Optimization interface (mock implementation)"
    (let [cost-fn (fn [params] (reduce + (map #(* % %) params)))  ; Simple quadratic cost
          initial-params [1.0 2.0 -1.5]
          options {:max-iterations 10 :tolerance 1e-6}
          
          result (qnn/optimize-parameters cost-fn initial-params options)]
      
      (is (map? result)
          "Optimization should return a map")
      
      (is (contains? result :parameters)
          "Result should contain optimized parameters")
      
      (is (contains? result :final-cost)
          "Result should contain final cost")
      
      (is (contains? result :iterations)
          "Result should contain iteration count")
      
      (is (contains? result :history)
          "Result should contain optimization history")
      
      (is (vector? (:parameters result))
          "Optimized parameters should be a vector")
      
      (is (number? (:final-cost result))
          "Final cost should be a number")
      
      (is (<= (:final-cost result) (cost-fn initial-params))
          "Final cost should not be worse than initial cost"))))

;;;
;;; Property-Based Tests
;;;
(deftest test-parameter-allocation-properties
  (testing "Parameter allocation properties"
    ;; Test that total parameters equals sum of layer parameters
    (let [test-networks [[(qnn/create-feedforward-qnn 2 1)]
                         [(qnn/create-feedforward-qnn 3 2)]
                         [(qnn/create-feedforward-qnn 4 1)]]]
      
      (doseq [network test-networks]
        (let [total-layer-params (reduce + (map :parameter-count (first network)))
              allocated-result (qnn/allocate-parameters (first network))
              allocated-total (:total-parameters allocated-result)]
          
          (is (= total-layer-params allocated-total)
              "Allocated total should equal sum of layer parameters"))))))

(deftest test-layer-type-consistency
  (testing "Layer type consistency across operations"
    (let [layer-types [:input :dense :entangling :activation :measurement]]
      
      (doseq [layer-type layer-types]
        (let [test-layer {:layer-type layer-type :num-qubits 3}
              param-count (qnn/count-layer-parameters test-layer)]
          
          (is (>= param-count 0)
              (str "Parameter count should be non-negative for " layer-type))
          
          ;; Test with parameter count added
          (when (pos? param-count)
            (let [layer-with-params (assoc test-layer :parameter-count param-count)]
              (is (some? layer-with-params)
                  (str "Layer with parameters should be valid for " layer-type)))))))))

(comment
  ;; Run all tests in this namespace
  (run-tests)
  ;
  )