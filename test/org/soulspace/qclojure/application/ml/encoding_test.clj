(ns org.soulspace.qclojure.application.ml.encoding-test
  (:require [clojure.test :refer [deftest is testing run-tests]]
            [clojure.spec.alpha :as s]
            [clojure.spec.gen.alpha :as gen]
            [org.soulspace.qclojure.application.ml.encoding :as enc]))

;; Test data generators
(def feature-vector-gen
  (gen/vector (gen/double* {:min -10.0 :max 10.0 :NaN? false :infinite? false}) 1 10))

(def small-positive-int-gen
  (gen/choose 1 5))

;; Unit Tests

(deftest test-normalize-features
  (testing "Basic normalization functionality"
    (let [result (enc/normalize-features [1.0 2.0 3.0 4.0])]
      (is (:success result) "Normalization should succeed")
      (is (< (Math/abs (- (nth (:result result) 1) (double 1/3))) 1e-10) "Should normalize correctly")))
  
  (testing "Edge cases"
    (let [same-values (enc/normalize-features [2.0 2.0 2.0])
          negative-values (enc/normalize-features [-2.0 -1.0 0.0 1.0])]
      (is (:success same-values) "Should handle constant values")
      (is (= [2.0 2.0 2.0] (:result same-values)) "Constant values stay unchanged")
      
      (is (:success negative-values) "Should handle negative values")
      (let [normalized (:result negative-values)]
        (is (< (Math/abs (first normalized)) 1e-10) "First element should be ~0")
        (is (< (Math/abs (- (last normalized) 1.0)) 1e-10) "Last element should be ~1"))))
  
  (testing "Error handling"
    (let [empty-result (enc/normalize-features [])
          invalid-result (enc/normalize-features "not-a-vector")]
      (is (not (:success empty-result)) "Empty vector should fail")
      (is (not (:success invalid-result)) "Invalid input should fail")
      (is (contains? empty-result :error) "Should provide error message")
      (is (contains? invalid-result :error) "Should provide error message"))))

(deftest test-amplitude-encoding
  (testing "Basic amplitude encoding"
    (let [result (enc/amplitude-encoding [0.5 0.5 0.5 0.5] 2)]
      (is (:success result) "Amplitude encoding should succeed")
      (is (map? (:result result)) "Should return quantum state map")))
  
  (testing "Normalization in amplitude encoding"
    (let [result (enc/amplitude-encoding [1.0 1.0] 1)]
      (is (:success result) "Should normalize amplitudes automatically")))
  
  (testing "Error cases"
    (let [too-many-features (enc/amplitude-encoding [1 2 3 4 5] 2)  ; 5 features > 2^2 amplitudes
          invalid-qubits (enc/amplitude-encoding [1 2] -1)
          invalid-features (enc/amplitude-encoding "invalid" 2)]
      
      (is (not (:success too-many-features)) "Should fail when features exceed capacity")
      (is (contains? too-many-features :error) "Should provide error for too many features")
      
      (is (not (:success invalid-qubits)) "Should fail with invalid qubit count")
      (is (contains? invalid-qubits :error) "Should provide error for invalid qubits")
      
      (is (not (:success invalid-features)) "Should fail with invalid features")
      (is (contains? invalid-features :error) "Should provide error for invalid features"))))

(deftest test-angle-encoding
  (testing "Basic angle encoding"
    (let [result (enc/angle-encoding [0.5 0.3 0.7] 3 :ry)]
      (is (:success result) "Angle encoding should succeed")
      (is (fn? (:result result)) "Should return circuit encoder function")))
  
  (testing "Default gate type"
    (let [result-default (enc/angle-encoding [0.5 0.3] 2)
          result-explicit (enc/angle-encoding [0.5 0.3] 2 :ry)]
      (is (:success result-default) "Default gate type should work")
      (is (:success result-explicit) "Explicit gate type should work")))
  
  (testing "Different gate types"
    (doseq [gate-type [:rx :ry :rz]]
      (let [result (enc/angle-encoding [0.5] 1 gate-type)]
        (is (:success result) (str "Gate type " gate-type " should work")))))
  
  (testing "Error cases"
    (let [too-many-features (enc/angle-encoding [1 2 3] 2)  ; 3 features > 2 qubits
          invalid-gate (enc/angle-encoding [1] 1 :invalid)
          invalid-qubits (enc/angle-encoding [1] 0)]
      
      (is (not (:success too-many-features)) "Should fail when features exceed qubits")
      (is (contains? too-many-features :error) "Should provide error for too many features")
      
      (is (not (:success invalid-gate)) "Should fail with invalid gate type")
      (is (contains? invalid-gate :error) "Should provide error for invalid gate")
      
      (is (not (:success invalid-qubits)) "Should fail with invalid qubit count")
      (is (contains? invalid-qubits :error) "Should provide error for invalid qubits"))))

(deftest test-basis-encoding
  (testing "String bit encoding"
    (let [result (enc/basis-encoding "101" 3)]
      (is (:success result) "String bit encoding should succeed")
      (is (map? (:result result)) "Should return quantum state")))
  
  (testing "Vector bit encoding"
    (let [result (enc/basis-encoding [1 0 1] 3)]
      (is (:success result) "Vector bit encoding should succeed")
      (is (map? (:result result)) "Should return quantum state")))
  
  (testing "Error cases"
    (let [length-mismatch (enc/basis-encoding "10" 3)  ; 2 bits for 3 qubits
          invalid-bits (enc/basis-encoding [1 2 3] 3)  ; invalid bit values
          invalid-qubits (enc/basis-encoding "101" -1)]
      
      (is (not (:success length-mismatch)) "Should fail with length mismatch")
      (is (contains? length-mismatch :error) "Should provide error for length mismatch")
      
      (is (not (:success invalid-bits)) "Should fail with invalid bit values")
      (is (contains? invalid-bits :error) "Should provide error for invalid bits")
      
      (is (not (:success invalid-qubits)) "Should fail with invalid qubit count")
      (is (contains? invalid-qubits :error) "Should provide error for invalid qubits"))))

;; Property-based tests using spec generators
(deftest test-normalize-features-properties
  (testing "Normalization properties"
    (let [test-cases (gen/sample feature-vector-gen 50)]
      (doseq [test-data test-cases]
        (when (seq test-data)  ; Skip empty vectors
          (let [result (enc/normalize-features test-data)]
            (when (:success result)
              (let [normalized (:result result)]
                (is (<= (apply min normalized) (apply max normalized))
                    "Max should be >= min after normalization")
                (when (> (count (distinct test-data)) 1)  ; Skip constant vectors
                  (is (or (zero? (apply min normalized))
                          (= normalized test-data))  ; constant case
                      "Min should be 0 for non-constant vectors")
                  (is (or (= 1.0 (apply max normalized))
                          (= normalized test-data))  ; constant case
                      "Max should be 1 for non-constant vectors"))))))))))

;; Integration tests
(deftest test-encoding-pipeline
  (testing "Complete encoding workflow"
    (let [raw-data [0.2 0.8 0.1 0.9]
          norm-result (enc/normalize-features raw-data)]
      (is (:success norm-result) "Normalization step should succeed")
      
      (when (:success norm-result)
        (let [normalized (:result norm-result)
              amp-result (enc/amplitude-encoding normalized 2)
              angle-result (enc/angle-encoding normalized 4)]
          
          (is (:success amp-result) "Amplitude encoding should succeed")
          (is (:success angle-result) "Angle encoding should succeed")
          
          (is (map? (:result amp-result)) "Amplitude encoding should return state")
          (is (fn? (:result angle-result)) "Angle encoding should return function"))))))

;; Performance and edge case tests
(deftest test-encoding-performance
  (testing "Large feature vectors"
    (let [large-vector (vec (range 100))
          result (enc/normalize-features large-vector)]
      (is (:success result) "Should handle large vectors")
      (is (= 100 (count (:result result))) "Should preserve vector length")))
  
  (testing "Extreme values"
    (let [extreme-values [Double/MIN_VALUE Double/MAX_VALUE]
          result (enc/normalize-features extreme-values)]
      (is (:success result) "Should handle extreme values")
      (when (:success result)
        (is (= [0.0 1.0] (:result result)) "Should normalize extreme values correctly")))))

;; Helper function to run all tests
(defn run-encoding-tests []
  (println "Running encoding tests...")
  (run-tests 'org.soulspace.qclojure.application.ml.encoding-test))

;; Rich comment block for REPL testing
(comment
  ;; Run individual tests in REPL
  (test-normalize-features)
  (test-amplitude-encoding)
  (test-angle-encoding)
  (test-basis-encoding)
  
  ;; Run all tests
  (run-encoding-tests)
  
  ;; Manual testing
  (enc/normalize-features [1 2 3 4])
  (enc/amplitude-encoding [0.5 0.5] 1)
  (enc/angle-encoding [0.1 0.2] 2 :ry)
  (enc/basis-encoding "10" 2)
  
  ;; Test error cases
  (enc/normalize-features [])
  (enc/amplitude-encoding [1 2 3 4 5] 2)
  (enc/angle-encoding [1 2 3] 2)
  (enc/basis-encoding "10" 3)
  )