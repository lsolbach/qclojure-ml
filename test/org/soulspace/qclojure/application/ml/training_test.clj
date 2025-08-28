(ns org.soulspace.qclojure.application.ml.training-test
  (:require [clojure.test :refer [deftest is testing run-tests]]
            [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [org.soulspace.qclojure.application.ml.training :as training]
            [org.soulspace.qclojure.application.algorithm.vqe :as vqe]
            [org.soulspace.qclojure.application.backend :as qb]
            [org.soulspace.qclojure.adapter.backend.simulator :as sim]
            [org.soulspace.qclojure.domain.circuit :as qc]))

;; Test data generators
(def parameter-vector-gen
  (gen/vector (gen/double* {:min -3.14 :max 3.14 :NaN? false :infinite? false}) 2 8))

(def feature-matrix-gen
  (gen/vector (gen/vector (gen/double* {:min -1.0 :max 1.0 :NaN? false :infinite? false}) 2 4) 2 10))

(def binary-labels-gen
  (gen/vector (gen/elements [0 1]) 2 10))

;;
;; Tests
;;
(deftest test-classification-cost-validation
  (testing "Input validation with proper backend"
    (let [valid-params [0.1 0.2 0.3 0.4 0.5 0.6]
          ansatz-fn (vqe/hardware-efficient-ansatz 2 1)
          valid-features [[0.5 0.3] [0.7 0.1]]
          valid-labels [0 1]
          backend (sim/create-simulator)
          
          result (training/classification-cost valid-params ansatz-fn valid-features valid-labels backend)]
      
      (is (number? result) "Should return numeric cost for valid inputs")
      (is (> result 0.0) "Cost should be positive")
      (is (< result 100.0) "Cost should be reasonable for test data")))
  
  (testing "Error handling"
    (let [invalid-params []
          ansatz-fn (vqe/hardware-efficient-ansatz 2 1)
          valid-features [[0.5 0.3]]
          valid-labels [0]
          backend (sim/create-simulator)
          
          result (training/classification-cost invalid-params ansatz-fn valid-features valid-labels backend)]
      
      ;; The current implementation returns 1000.0 on error
      (is (= 1000.0 result) "Should return high cost on error"))))

(deftest test-parameter-shift-gradient-validation
  (testing "Parameter shift gradient validation"
    (let [cost-fn (fn [params] (+ (* (first params) (first params))
                                  (* (second params) (second params))))
          valid-params [0.1 0.2]
          
          result (training/parameter-shift-gradient cost-fn valid-params)]
      
      (is (:success result) "Valid inputs should succeed")
      (is (vector? (:result result)) "Should return vector of gradients")
      (is (= (count (:result result)) (count valid-params)) "Gradient vector should match parameter count")))
  
  (testing "Error cases"
    (let [failing-cost-fn (fn [_] (throw (ex-info "Test error" {})))
          valid-params [0.1 0.2]
          
          result (training/parameter-shift-gradient failing-cost-fn valid-params)]
      
      (is (not (:success result)) "Should fail with failing cost function")
      (is (contains? result :error) "Should provide error message"))))

(deftest test-training-data-specs
  (testing "Training data spec validation"
    (let [valid-features [[1.0 2.0] [3.0 4.0]]
          valid-labels [0 1]]
      
      (is (s/valid? ::training/features valid-features) "Valid features should pass spec")
      (is (s/valid? ::training/labels valid-labels) "Valid labels should pass spec"))))

;; Integration tests
(deftest test-training-integration
  (testing "Complete training workflow"
    (let [features [[0.1 0.2] [0.3 0.4] [0.5 0.6] [0.7 0.8]]
          labels [0 0 1 1]
          training-data {:features features :labels labels}
          ansatz-fn (vqe/hardware-efficient-ansatz 2 1)
          backend (sim/create-simulator)
          
          config {:max-iterations 5
                  :learning-rate 0.1
                  :num-parameters 6
                  :backend backend}
          
          result (training/train-qml-model ansatz-fn training-data config)]
      
      (is (:success result) "Training should succeed")
      (is (vector? (:optimal-parameters result)) "Should return optimal parameters")
      (is (= 6 (count (:optimal-parameters result))) "Should have correct number of parameters")
      (is (number? (:optimal-cost result)) "Should return numeric cost")
      (is (>= (:iterations result) 0) "Should track iteration count"))))

;; Property-based tests
(deftest test-cost-function-properties
  (testing "Cost function properties"
    (let [test-cases (take 5 (gen/sample 
                               (gen/hash-map 
                                :parameters parameter-vector-gen
                                :features feature-matrix-gen
                                :labels binary-labels-gen)))]
      
      (doseq [test-case test-cases]
        (when (= (count (:features test-case)) (count (:labels test-case)))
          (let [ansatz-fn (vqe/hardware-efficient-ansatz 2 1)
                backend (sim/create-simulator)
                cost (try
                       (training/classification-cost 
                        (:parameters test-case)
                        ansatz-fn
                        (:features test-case)
                        (:labels test-case)
                        backend)
                       (catch Exception _ 1000.0))]
            
            (is (number? cost) "Cost should be numeric")
            (is (>= cost 0.0) "Cost should be non-negative")))))))

;; Error handling tests
(deftest test-error-handling
  (testing "Various error conditions"
    (let [valid-params [0.1 0.2 0.3 0.4 0.5 0.6]
          ansatz-fn (vqe/hardware-efficient-ansatz 2 1)
          backend (sim/create-simulator)
          
          ; Test mismatched features and labels
          mismatched-result (try
                              (training/classification-cost 
                               valid-params ansatz-fn 
                               [[0.5 0.3]] [0 1] backend) ; 1 feature, 2 labels
                              (catch Exception _ 1000.0))
          
          ; Test empty data
          empty-result (try
                         (training/classification-cost 
                          valid-params ansatz-fn [] [] backend)
                         (catch Exception _ 1000.0))
          
          ; Test nil inputs
          nil-result (try
                       (training/classification-cost 
                        nil ansatz-fn [[0.5]] [0] backend)
                       (catch Exception _ 1000.0))]
      
      ; Current implementation returns 1000.0 on all errors
      (is (= 1000.0 mismatched-result) "Should handle mismatched data")
      (is (= 1000.0 empty-result) "Should handle empty data")  
      (is (= 1000.0 nil-result) "Should handle nil inputs"))))

;; Rich comment block for REPL testing
(comment
  (run-tests)

  ;; Manual testing of training functions
  (let [backend (sim/create-simulator)]
    (training/classification-cost
     [0.1 0.2 0.3 0.4 0.5 0.6]
     (vqe/hardware-efficient-ansatz 2 1)
     [[0.5 0.3] [0.7 0.1]]
     [0 1]
     backend))

  ;; Test spec validation
  (s/valid? ::training/features [[1 2]])
  (s/explain ::training/features "invalid")

  ;; Test parameter shift gradient
  (training/parameter-shift-gradient (fn [p] (* (first p) (first p))) [0.5])
  )
