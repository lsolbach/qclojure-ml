(ns org.soulspace.qclojure.application.ml.encoding
 "Quantum data encoding strategies for quantum machine learning"
 (:require [org.soulspace.qclojure.domain.state :as qs]
           [org.soulspace.qclojure.domain.circuit :as qc]
           [fastmath.core :as m]
           [fastmath.complex :as fc]
           [clojure.spec.alpha :as s]))

(defn normalize-features
  "Normalize feature vector using min-max scaling to [0, 1].

   Parameters:
   - feature-vector: Vector of feature values
   
   Returns:
   - Normalized vector with values scaled to [0, 1]
   
   Example:
   (normalize-features [1.0 2.0 3.0 4.0])"
  [feature-vector]
  (let [min-val (apply min feature-vector)
        max-val (apply max feature-vector)
        range-val (- max-val min-val)]
    (if (zero? range-val)
      feature-vector
      (mapv #(/ (- % min-val) range-val) feature-vector))))

(defn amplitude-encoding
  "Encode classical data into quantum amplitude encoding.
   Maps data values to quantum state amplitudes.
   
   Parameters:
   - feature-vector: Vector of feature values (should be normalized)
   - num-qubits: Number of qubits to use (determines 2^n amplitudes)
   
   Returns:
   - Quantum state with amplitudes corresponding to feature values
   
   Example:
   (amplitude-encoding [0.5 0.5 0.5 0.5] 2)"
  [feature-vector num-qubits]
  (let [num-amplitudes (int (m/pow 2 num-qubits))
        padded-features (take num-amplitudes (concat feature-vector (repeat 0.0)))
        sum-squares (m/sqrt (reduce + (map #(* % %) padded-features)))
        normalized-amplitudes (if (> sum-squares 0.0)
                                (mapv #(/ % sum-squares) padded-features)
                                padded-features)
        complex-amplitudes (mapv #(fc/complex % 0) normalized-amplitudes)]
    (qs/multi-qubit-state complex-amplitudes)))

(defn angle-encoding
  "Encode classical data as rotation angles in quantum circuit gates.
   
   This encoding maps classical data values to rotation angles, creating
   a parameterized quantum circuit that encodes the data in qubit rotations.
   
   Parameters:
   - data-row: Vector of classical data values
   - num-qubits: Number of qubits to use for encoding
   - gate-type: Type of rotation gate (:rx, :ry, :rz), default :ry
   
   Returns:
   - Function that takes a circuit and applies the encoding gates
   
   Example:
   (def encoder (angle-encoding [0.5 0.3 0.7] 3 :ry))
   (encoder (qc/create-circuit 3))"
  ([data-row num-qubits] (angle-encoding data-row num-qubits :ry))
  ([data-row num-qubits gate-type]
   {:pre [(<= (count data-row) num-qubits)
          (#{:rx :ry :rz} gate-type)]}
   (fn [circuit]
     (reduce-kv (fn [c idx angle]
                  (if (< idx num-qubits)
                    (case gate-type
                      :rx (qc/rx-gate c idx (* angle m/PI))
                      :ry (qc/ry-gate c idx (* angle m/PI))
                      :rz (qc/rz-gate c idx (* angle m/PI)))
                    c))
                circuit
                (vec data-row)))))

(defn basis-encoding
  "Encode classical data using computational basis encoding.
   
   Maps classical bit strings to computational basis states directly.
   Useful for encoding discrete/categorical data.
   
   Parameters:
   - bit-string: String of bits like '101' or vector of 0/1 values
   - num-qubits: Number of qubits (should match bit string length)
   
   Returns:
   - Quantum state in computational basis
   
   Example:
   (basis-encoding '101' 3)  ; Creates |101> state
   (basis-encoding [1 0 1] 3)"
  [bit-string num-qubits]
  {:pre [(= (count bit-string) num-qubits)]}
  (let [bit-vec (if (string? bit-string)
                  (mapv #(Character/digit % 10) bit-string)
                  (vec bit-string))
        state-index (reduce + (map-indexed (fn [i bit]
                                             (* bit (int (m/pow 2 (- num-qubits i 1)))))
                                           bit-vec))]
    (qs/computational-basis-state state-index num-qubits)))

(defn iqp-encoding
  "Instantaneous Quantum Polynomial (IQP) encoding circuit.
   
   Creates a quantum circuit that implements IQP-style encoding where
   classical data is encoded through Hadamard gates followed by 
   diagonal unitaries with data-dependent phases.
   
   Parameters:
   - data-row: Vector of classical data values
   - num-qubits: Number of qubits to use
   
   Returns:
   - Function that creates the IQP encoding circuit
   
   Example:
   (def iqp-encoder (iqp-encoding [0.1 0.2 0.3 0.4] 2)))"
  [data-row num-qubits]
  {:pre [(<= (count data-row) (* num-qubits num-qubits))]}
  (fn [circuit]
    (let [;; First apply Hadamard to all qubits
          h-circuit (reduce #(qc/h-gate %1 %2) circuit (range num-qubits))
          ;; Then apply data-dependent phase gates
          data-phases (take (* num-qubits num-qubits) (concat data-row (repeat 0.0)))]
      (reduce-kv (fn [c idx phase]
                   (let [qubit1 (quot idx num-qubits)
                         qubit2 (mod idx num-qubits)]
                     (if (= qubit1 qubit2)
                       ;; Single qubit phase
                       (qc/rz-gate c qubit1 phase)
                       ;; Two qubit phase (if qubits are different)
                       (if (< qubit1 qubit2)
                         (-> c
                             (qc/cnot-gate qubit1 qubit2)
                             (qc/rz-gate qubit2 phase)
                             (qc/cnot-gate qubit1 qubit2))
                         c))))
                 h-circuit
                 (vec data-phases)))))

;; Specs for encoding validation
(s/def ::data-vector (s/coll-of number? :kind vector?))
(s/def ::num-qubits pos-int?)
(s/def ::gate-type #{:rx :ry :rz})
(s/def ::bit-string (s/or :string string? :vector (s/coll-of #{0 1} :kind vector?)))
