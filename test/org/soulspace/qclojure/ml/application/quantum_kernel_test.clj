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
        (is (= (get-in svm-matrix [0 0]) 1.1)
            "Diagonal [0,0] should be 1.0 + regularization")
        (is (= (get-in svm-matrix [1 1]) 1.1)
            "Diagonal [1,1] should be 1.0 + regularization")
        
        ;; Off-diagonal elements depend on actual quantum kernel computation
        ;; For these data points with angle encoding, there is non-zero overlap
        ;; The matrix should be symmetric
        (is (= (get-in svm-matrix [0 1]) (get-in svm-matrix [1 0]))
            "Matrix should be symmetric")
        
        ;; Off-diagonal should be between 0 and 1 (valid overlap range)
        (is (and (>= (get-in svm-matrix [0 1]) 0.0)
                 (<= (get-in svm-matrix [0 1]) 1.0))
            "Off-diagonal elements should be valid kernel values")
        
        ;; Off-diagonal should NOT be regularized (only diagonal gets regularization)
        (is (< (get-in svm-matrix [0 1]) 1.0)
            "Off-diagonal elements should not include regularization"))))

(deftest test-quantum-kernel-overlap
  (testing "Overlap computation with angle encoding"
    (let [backend (sim/create-simulator)
          config {:encoding-type :angle
                  :num-qubits 2
                  :shots 1024
                  :encoding-options {:gate-type :ry}}
          data-point1 [0.1 0.2]
          data-point2 [0.8 0.9]
          result (qk/quantum-kernel-overlap backend data-point1 data-point2 config)]
      
      ;; Verify result structure
      (is (map? result))
      (is (contains? result :overlap-value))
      (is (contains? result :measurement-data))
      
      ;; Verify overlap value is in valid range [0, 1]
      (is (number? (:overlap-value result)))
      (is (>= (:overlap-value result) 0.0))
      (is (<= (:overlap-value result) 1.0))))
  
  (testing "Self-overlap should be 1.0"
    (let [backend (sim/create-simulator)
          config {:encoding-type :angle
                  :num-qubits 2
                  :shots 1024
                  :encoding-options {:gate-type :ry}}
          data-point [0.5 0.3]
          result (qk/quantum-kernel-overlap backend data-point data-point config)]
      
      ;; Self-overlap should be very close to 1.0 (allowing for numerical precision)
      (is (> (:overlap-value result) 0.99))
      (is (<= (:overlap-value result) 1.0))))
  
  (testing "Overlap with basis encoding"
    (let [backend (sim/create-simulator)
          config {:encoding-type :basis
                  :num-qubits 2
                  :shots 1024}
          data-point1 [0.0 1.0]
          data-point2 [1.0 0.0]
          result (qk/quantum-kernel-overlap backend data-point1 data-point2 config)]
      
      (is (number? (:overlap-value result)))
      (is (>= (:overlap-value result) 0.0))
      (is (<= (:overlap-value result) 1.0))))
  
  (testing "Overlap with IQP encoding"
    (let [backend (sim/create-simulator)
          config {:encoding-type :iqp
                  :num-qubits 2
                  :shots 1024}
          data-point1 [0.2 0.3]
          data-point2 [0.7 0.8]
          result (qk/quantum-kernel-overlap backend data-point1 data-point2 config)]
      
      (is (number? (:overlap-value result)))
      (is (>= (:overlap-value result) 0.0))
      (is (<= (:overlap-value result) 1.0)))))


(comment
  ;; Run tests in this namespace
  (run-tests)

  ;
  )

(deftest test-parametrized-feature-map
  (testing "Feature map creation with trainable parameters"
    (let [data-point [0.1 0.2]
          trainable-params [0.5 0.3 0.7 0.4 0.6 0.8 0.2 0.9]
          num-qubits 2
          num-layers 2
          options {:gate-type :ry :entangling true}
          feature-map (qk/parametrized-feature-map data-point trainable-params num-qubits num-layers options)]
      
      ;; Should return a function
      (is (fn? feature-map))
      
      ;; Function should be applicable to a circuit
      (let [base-circuit (circuit/create-circuit num-qubits)
            encoded-circuit (feature-map base-circuit)]
        (is (map? encoded-circuit))
        (is (= (:num-qubits encoded-circuit) num-qubits))
        ;; Should have operations from both encoding and trainable layers
        (is (> (count (:operations encoded-circuit)) 0)))))
  
  (testing "Parameter count calculation"
    (let [num-qubits 3
          num-layers 2
          expected-params (* num-qubits 2 num-layers)]
      (is (= (qk/calculate-trainable-parameter-count num-qubits num-layers) expected-params))
      (is (= (qk/calculate-trainable-parameter-count num-qubits num-layers) 12))))
  
  (testing "Feature map with different layer counts"
    (let [data-point [0.1 0.2 0.3]
          num-qubits 3
          num-layers 3
          num-params (qk/calculate-trainable-parameter-count num-qubits num-layers)
          trainable-params (vec (repeat num-params 0.5))
          feature-map (qk/parametrized-feature-map data-point trainable-params num-qubits num-layers {})]
      
      (is (fn? feature-map)))))

(deftest test-trainable-quantum-kernel-overlap
  (testing "Trainable kernel overlap computation"
    (let [backend (sim/create-simulator)
          config {:encoding-type :trainable
                  :num-qubits 2
                  :num-trainable-layers 2
                  :shots 1024
                  :encoding-options {:gate-type :ry}}
          data-point1 [0.1 0.2]
          data-point2 [0.8 0.9]
          num-params (qk/calculate-trainable-parameter-count 2 2)
          trainable-params (vec (repeat num-params 0.5))
          result (qk/trainable-quantum-kernel-overlap backend data-point1 data-point2 trainable-params config)]
      
      ;; Verify result structure
      (is (map? result))
      (is (contains? result :overlap-value))
      
      ;; Verify overlap is in valid range
      (is (number? (:overlap-value result)))
      (is (>= (:overlap-value result) 0.0))
      (is (<= (:overlap-value result) 1.0))))
  
  (testing "Trainable kernel self-overlap"
    (let [backend (sim/create-simulator)
          config {:encoding-type :trainable
                  :num-qubits 2
                  :num-trainable-layers 1
                  :shots 1024}
          data-point [0.3 0.4]
          num-params (qk/calculate-trainable-parameter-count 2 1)
          trainable-params (vec (repeat num-params 0.5))
          result (qk/trainable-quantum-kernel-overlap backend data-point data-point trainable-params config)]
      
      ;; Self-overlap should be close to 1.0
      (is (> (:overlap-value result) 0.99)))))

(deftest test-compute-trainable-kernel-matrix
  (testing "Trainable kernel matrix computation"
    (let [backend (sim/create-simulator)
          data-matrix [[0.1 0.2] [0.8 0.9] [0.3 0.4]]
          config {:encoding-type :trainable
                  :num-qubits 2
                  :num-trainable-layers 1
                  :shots 1024}
          num-params (qk/calculate-trainable-parameter-count 2 1)
          trainable-params (vec (repeat num-params 0.5))
          kernel-matrix (qk/compute-trainable-kernel-matrix backend data-matrix trainable-params config)]
      
      ;; Verify matrix dimensions
      (is (= (count kernel-matrix) 3))
      (is (every? #(= (count %) 3) kernel-matrix))
      
      ;; Verify diagonal elements are close to 1.0
      (is (> (get-in kernel-matrix [0 0]) 0.99))
      (is (> (get-in kernel-matrix [1 1]) 0.99))
      (is (> (get-in kernel-matrix [2 2]) 0.99))
      
      ;; Verify symmetry
      (is (= (get-in kernel-matrix [0 1]) (get-in kernel-matrix [1 0])))
      (is (= (get-in kernel-matrix [0 2]) (get-in kernel-matrix [2 0])))
      (is (= (get-in kernel-matrix [1 2]) (get-in kernel-matrix [2 1])))
      
      ;; Verify all values are in valid range
      (is (every? #(and (>= % 0.0) (<= % 1.0))
                  (flatten kernel-matrix))))))

(deftest test-train-quantum-kernel
  (testing "Quantum kernel training with supervised alignment"
    (let [backend (sim/create-simulator)
          training-data [[0.1 0.2] [0.15 0.25] [0.8 0.9] [0.85 0.95]]
          training-labels [0 0 1 1]
          config {:num-qubits 2
                  :num-trainable-layers 1
                  :alignment-objective :supervised
                  :optimization-method :adam
                  :max-iterations 5  ; Keep low for testing
                  :learning-rate 0.1
                  :shots 512  ; Lower shots for faster testing
                  :parameter-strategy :random
                  :regularization :none}
          result (qk/train-quantum-kernel backend training-data training-labels config)]
      
      ;; Verify result structure
      (is (map? result))
      (is (contains? result :optimal-parameters))
      (is (contains? result :optimal-alignment))
      (is (contains? result :initial-alignment))
      (is (contains? result :iterations))
      
      ;; Verify optimal parameters
      (is (vector? (:optimal-parameters result)))
      (let [expected-params (qk/calculate-trainable-parameter-count 2 1)]
        (is (= (count (:optimal-parameters result)) expected-params)))
      
      ;; Verify alignment values are numbers
      (is (number? (:optimal-alignment result)))
      (is (number? (:initial-alignment result)))
      
      ;; Verify iterations completed
      (is (pos-int? (:iterations result)))
      (is (<= (:iterations result) 5))))
  
  (testing "Quantum kernel training with L2 regularization"
    (let [backend (sim/create-simulator)
          training-data [[0.1 0.2] [0.8 0.9]]
          training-labels [0 1]
          config {:num-qubits 2
                  :num-trainable-layers 1
                  :alignment-objective :supervised
                  :optimization-method :gradient-descent
                  :max-iterations 3
                  :learning-rate 0.05
                  :shots 512
                  :parameter-strategy :zero
                  :regularization :l2
                  :reg-lambda 0.01}
          result (qk/train-quantum-kernel backend training-data training-labels config)]
      
      ;; Should complete without error
      (is (map? result))
      (is (vector? (:optimal-parameters result)))))
  
  (testing "Quantum kernel training with different parameter strategies"
    (let [backend (sim/create-simulator)
          training-data [[0.1 0.2] [0.8 0.9]]
          training-labels [0 1]
          base-config {:num-qubits 2
                       :num-trainable-layers 1
                       :alignment-objective :supervised
                       :optimization-method :gradient-descent
                       :max-iterations 2
                       :shots 512}]
      
      ;; Test :random strategy
      (let [config (assoc base-config :parameter-strategy :random)
            result (qk/train-quantum-kernel backend training-data training-labels config)]
        (is (vector? (:optimal-parameters result))))
      
      ;; Test :zero strategy
      (let [config (assoc base-config :parameter-strategy :zero)
            result (qk/train-quantum-kernel backend training-data training-labels config)]
        (is (vector? (:optimal-parameters result))))
      
      ;; Test :custom strategy
      (let [num-params (qk/calculate-trainable-parameter-count 2 1)
            custom-params (vec (repeat num-params 0.3))
            config (assoc base-config 
                          :parameter-strategy :custom
                          :initial-parameters custom-params)
            result (qk/train-quantum-kernel backend training-data training-labels config)]
        (is (vector? (:optimal-parameters result)))))))

(deftest test-quantum-kernel-matrix
  (testing "Full kernel matrix computation"
    (let [backend (sim/create-simulator)
          data-matrix [[0.1 0.2] [0.8 0.9] [0.3 0.4]]
          config {:encoding-type :angle
                  :num-qubits 2
                  :shots 1024}
          kernel-matrix (qk/quantum-kernel-matrix backend data-matrix config)]
      
      ;; Verify matrix dimensions
      (is (= (count kernel-matrix) 3))
      (is (every? #(= (count %) 3) kernel-matrix))
      
      ;; Verify diagonal elements are close to 1.0
      (is (> (get-in kernel-matrix [0 0]) 0.99))
      (is (> (get-in kernel-matrix [1 1]) 0.99))
      (is (> (get-in kernel-matrix [2 2]) 0.99))
      
      ;; Verify symmetry
      (is (= (get-in kernel-matrix [0 1]) (get-in kernel-matrix [1 0])))
      (is (= (get-in kernel-matrix [0 2]) (get-in kernel-matrix [2 0])))
      (is (= (get-in kernel-matrix [1 2]) (get-in kernel-matrix [2 1])))
      
      ;; Verify all values are in valid range
      (is (every? #(and (>= % 0.0) (<= % 1.0))
                  (flatten kernel-matrix)))))
  
  (testing "Kernel matrix with asymmetric computation"
    (let [backend (sim/create-simulator)
          data-matrix [[0.1 0.2] [0.8 0.9]]
          config {:encoding-type :angle
                  :num-qubits 2
                  :shots 1024}
          kernel-matrix (qk/quantum-kernel-matrix backend data-matrix config false)]
      
      ;; Should still work with asymmetric flag
      (is (= (count kernel-matrix) 2))
      (is (every? #(= (count %) 2) kernel-matrix)))))

(deftest test-kernel-integration-workflow
  (testing "End-to-end kernel workflow"
    (let [backend (sim/create-simulator)
          ;; 1. Prepare data
          data-matrix [[0.1 0.2] [0.15 0.25] [0.8 0.9] [0.85 0.95]]
          labels [0 0 1 1]
          
          ;; 2. Configure and compute initial kernel
          kernel-config {:encoding-type :angle
                         :num-qubits 2
                         :shots 512}
          initial-kernel (qk/quantum-kernel-matrix backend data-matrix kernel-config)
          
          ;; 3. Analyze initial kernel
          initial-analysis (qk/analyze-kernel-matrix initial-kernel)
          
          ;; 4. Train quantum kernel
          training-config {:num-qubits 2
                           :num-trainable-layers 1
                           :alignment-objective :supervised
                           :optimization-method :adam
                           :max-iterations 3
                           :learning-rate 0.1
                           :shots 512
                           :parameter-strategy :random}
          training-result (qk/train-quantum-kernel backend data-matrix labels training-config)
          
          ;; 5. Compute trained kernel matrix
          trained-params (:optimal-parameters training-result)
          trained-config (assoc training-config :encoding-type :trainable)
          trained-kernel (qk/compute-trainable-kernel-matrix backend data-matrix trained-params trained-config)
          
          ;; 6. Analyze trained kernel
          trained-analysis (qk/analyze-kernel-matrix trained-kernel)]
      
      ;; Verify initial kernel properties
      (is (= (:dimensions initial-analysis) [4 4]))
      (is (:symmetric? (:properties initial-analysis)))
      (is (:diagonal-correct? (:properties initial-analysis)))
      
      ;; Verify training completed
      (is (vector? trained-params))
      (is (number? (:optimal-alignment training-result)))
      
      ;; Verify trained kernel properties
      (is (= (:dimensions trained-analysis) [4 4]))
      (is (:symmetric? (:properties trained-analysis)))
      
      ;; Both kernels should have valid values
      (is (every? #(and (>= % 0.0) (<= % 1.0)) (flatten initial-kernel)))
      (is (every? #(and (>= % 0.0) (<= % 1.0)) (flatten trained-kernel)))))
  
  (testing "Kernel function creation and usage"
    (let [backend (sim/create-simulator)
          config {:encoding-type :angle
                  :num-qubits 2
                  :shots 512}
          kernel-fn (qk/create-quantum-kernel backend config)
          
          ;; Test kernel function on various data points
          k11 (kernel-fn [0.1 0.2] [0.1 0.2])  ; Self-overlap
          k12 (kernel-fn [0.1 0.2] [0.8 0.9])  ; Different points
          k21 (kernel-fn [0.8 0.9] [0.1 0.2])] ; Symmetric
      
      ;; Self-overlap should be close to 1.0
      (is (> k11 0.99))
      
      ;; Kernel should be symmetric
      (is (= k12 k21))
      
      ;; All kernel values in valid range
      (is (and (>= k11 0.0) (<= k11 1.0)))
      (is (and (>= k12 0.0) (<= k12 1.0))))))
