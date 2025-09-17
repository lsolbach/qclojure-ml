(ns org.soulspace.qclojure.ml.application.quantum-kernel-test
  "Tests for quantum kernel methods."
  (:require [clojure.test :refer [deftest is testing run-tests]]
            [clojure.spec.alpha :as s]
            [org.soulspace.qclojure.domain.circuit :as circuit]
            [org.soulspace.qclojure.domain.state :as state]
            [org.soulspace.qclojure.adapter.backend.ideal-simulator :as sim]
            [org.soulspace.qclojure.ml.application.quantum-kernel :as qk]))

;; Test helper function to convert frequency maps to QClojure measurement result format
(defn frequency-map-to-measurement-data
  "Convert a frequency map to QClojure measurement result format for testing.
  
  Parameters:
  - freq-map: Map of bit strings to counts, e.g. {'00' 400, '01' 300}
  
  Returns:
  - QClojure measurement result format"
  [freq-map]
  (let [shots (reduce + (vals freq-map))
        ;; Convert bit strings to measurement outcome indices
        outcomes (reduce-kv (fn [acc bit-string count]
                             (let [outcome-index (state/bits-to-index 
                                                  (map #(Integer/parseInt (str %)) bit-string))]
                               (concat acc (repeat count outcome-index))))
                           []
                           freq-map)]
    {:measurement-outcomes outcomes
     :shot-count shots
     :frequencies freq-map}))

(deftest test-kernel-config-validation
  (testing "Valid kernel configurations"
    (is (s/valid? ::qk/kernel-config 
                  {:encoding-type :angle
                   :num-qubits 2
                   :shots 1024
                   :encoding-options {:gate-type :ry}}))
    
    (is (s/valid? ::qk/kernel-config 
                  {:encoding-type :basis
                   :shots 512})))
  
  (testing "Invalid kernel configurations"
    (is (not (s/valid? ::qk/kernel-config 
                       {:encoding-type :invalid})))
    
    (is (not (s/valid? ::qk/kernel-config 
                       {:shots -100})))))

(deftest test-swap-test-measurement-analysis
  (testing "SWAP test overlap estimation"
    ;; Test case where ancilla is measured in |0⟩ 70% of the time
    ;; Using 5-qubit format: ancilla + 2 registers of 2 qubits each
    (let [measurements {"00000" 200 "00001" 150 "00010" 150 "00011" 200 
                        "10000" 100 "10001" 100 "10010" 100}
          measurement-data (frequency-map-to-measurement-data measurements)
          ancilla-qubit 0
          result (qk/estimate-overlap-from-swap-measurements measurement-data ancilla-qubit)]
      
      (is (= (:prob-ancilla-0 result) 7/10))
      (is (< (Math/abs (- (:overlap-squared result) 0.4)) 1e-10))
      (is (> (:overlap-value result) 0.6))
      (is (< (:overlap-value result) 0.64))))
  
  (testing "Perfect overlap case"
    (let [measurements {"00000" 1000}  ; All measurements with ancilla=0
          measurement-data (frequency-map-to-measurement-data measurements)
          result (qk/estimate-overlap-from-swap-measurements measurement-data 0)]
      
      (is (= (:overlap-value result) 1.0))
      (is (= (:overlap-squared result) 1.0))))
  
  (testing "Zero overlap case"
    (let [measurements {"10000" 1000}  ; All measurements with ancilla=1
          measurement-data (frequency-map-to-measurement-data measurements)
          result (qk/estimate-overlap-from-swap-measurements measurement-data 0)]
      
      (is (= (:overlap-value result) 0.0))
      (is (= (:overlap-squared result) 0.0)))))

(deftest test-data-encoding-integration
  (testing "Angle encoding"
    (let [encoder (qk/encode-data-for-kernel [0.5 0.3] :angle 2 {:gate-type :ry})]
      (is (fn? encoder))))
  
  (testing "Basis encoding"
    (let [encoder (qk/encode-data-for-kernel [0.2 0.8] :basis 2 {})]
      (is (fn? encoder))
      
      ;; Test that it correctly applies gates
      (let [base-circuit (circuit/create-circuit 2)
            encoded-circuit (encoder base-circuit)]
        (is (= (count (:operations encoded-circuit)) 1)))))  ; One X gate for second qubit
  
  (testing "IQP encoding"
    (let [encoder (qk/encode-data-for-kernel [0.1 0.2] :iqp 2 {})]
      (is (fn? encoder))))
  
  (testing "Invalid encoding type"
    (is (thrown? Exception 
                 (qk/encode-data-for-kernel [0.5] :invalid 1 {})))))

(deftest test-swap-test-circuit-creation
  (testing "SWAP test circuit structure"
    (let [base-circuit (circuit/create-circuit 5)
          register1 [0 1]
          register2 [2 3]
          ancilla 4
          swap-circuit (qk/swap-test-circuit base-circuit register1 register2 ancilla)]
      
      ;; Should have more operations than the base circuit
      (is (> (count (:operations swap-circuit)) 
             (count (:operations base-circuit))))
      
      ;; Should preserve circuit metadata
      (is (= (:num-qubits swap-circuit) 
             (:num-qubits base-circuit)))))
  
  (testing "Register size validation"
    (is (thrown? AssertionError
                 (qk/swap-test-circuit 
                  (circuit/create-circuit 4)
                  [0 1]    ; 2 qubits
                  [2]      ; 1 qubit - mismatch!
                  3)))))

(deftest test-kernel-matrix-analysis
  (testing "Symmetric matrix analysis"
    (let [matrix [[1.0 0.8 0.6]
                  [0.8 1.0 0.7]
                  [0.6 0.7 1.0]]
          analysis (qk/analyze-kernel-matrix matrix)]
      
      (is (= (:dimensions analysis) [3 3]))
      (is (:symmetric? (:properties analysis)))
      (is (:diagonal-correct? (:properties analysis)))
      (is (= (:diagonal-values (:statistics analysis)) [1.0 1.0 1.0]))))
  
  (testing "Asymmetric matrix analysis"
    (let [matrix [[1.0 0.8]
                  [0.1 1.0]]  ; Clearly not symmetric
          analysis (qk/analyze-kernel-matrix matrix)]
      
      (is (not (:symmetric? (:properties analysis))))))
  
  (testing "Incorrect diagonal analysis"
    (let [matrix [[0.5 0.8]
                  [0.8 0.5]]  ; Diagonal not 1.0
          analysis (qk/analyze-kernel-matrix matrix)]
      
      (is (not (:diagonal-correct? (:properties analysis)))))))

(deftest test-kernel-function-creation
  (testing "Kernel function creation"
    (let [config {:encoding-type :angle
                  :num-qubits 2
                  :shots 1024
                  :encoding-options {:gate-type :ry}}
          kernel-fn (qk/create-quantum-kernel (sim/create-simulator) config)]  ; No backend for unit test
      
      (is (fn? kernel-fn))
      
      ;; The function should be callable (though will return 0.0 without real backend)
      (is (number? (kernel-fn [0.1 0.2] [0.8 0.9])))))
  
  (testing "Invalid configuration"
    (is (thrown? AssertionError
                 (qk/create-quantum-kernel (sim/create-simulator) {:encoding-type :invalid})))))

(deftest test-precompute-encodings
  (testing "Encoding precomputation"
    (let [data-matrix [[0.1 0.2] [0.8 0.9] [0.3 0.4]]
          encodings (qk/precompute-encodings data-matrix :angle 2 {:gate-type :ry})]
      
      (is (= (count encodings) 3))
      (is (every? fn? encodings)))))

(deftest test-batch-computation-setup
  (testing "Batch computation configuration"
    (let [data-matrix [[0.1 0.2] [0.8 0.9]]
          config {:encoding-type :angle :num-qubits 2 :shots 1024}]
      
        
        ;; This should set up the computation without error
        (is (not (nil? (qk/batch-kernel-computation (sim/create-simulator) data-matrix config 10)))))))

(deftest test-svm-matrix-regularization
  (testing "SVM kernel matrix with regularization"
      (let [data-matrix [[0.1 0.2] [0.8 0.9]]
            config {:encoding-type :angle :num-qubits 2}
            regularization 0.1
            svm-matrix (qk/quantum-kernel-svm-matrix (sim/create-simulator) data-matrix config regularization)]
        
        ;; Diagonal should be regularized (1.0 + 0.1 = 1.1)
        (is (= (get-in svm-matrix [0 0]) 1.1))
        (is (= (get-in svm-matrix [1 1]) 1.1))
        
        ;; Off-diagonal elements depend on actual quantum kernel computation
        ;; With ideal simulator and these data points, they compute to 0.0
        (is (= (get-in svm-matrix [0 1]) 0.0))
        (is (= (get-in svm-matrix [1 0]) 0.0)))))

(comment
  ;; Run tests in this namespace
  (run-tests)
  
  ;; Individual test examples
  (test-kernel-config-validation)
  (test-swap-test-measurement-analysis)
  (test-data-encoding-integration)
  )


(comment
  ;; Run tests in this namespace
  (run-tests)

  ;
  )