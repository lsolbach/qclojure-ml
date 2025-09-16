(ns org.soulspace.qclojure.ml.application.encoding
  "Quantum data encoding strategies for quantum machine learning"
  (:require [clojure.spec.alpha :as s]
            [fastmath.core :as fm]
            [fastmath.complex :as fc]
            [org.soulspace.qclojure.domain.state :as state]
            [org.soulspace.qclojure.domain.circuit :as circuit]))

;; Specs for input validation (using QClojure specs where available)
(s/def ::feature-value number?)
(s/def ::feature-vector (s/coll-of ::feature-value :min-count 1 :kind vector?))
(s/def ::normalized-vector (s/and ::feature-vector 
                                  #(every? (fn [x] (<= 0.0 x 1.0)) %)))
(s/def ::gate-type #{:rx :ry :rz})
(s/def ::bit-vector (s/coll-of #{0 1} :kind vector?))  ; Renamed for clarity
(s/def ::bit-string (s/or :string string? :vector ::bit-vector))  ; Accept both string and vector

;; Specs for encoding validation
(s/def ::data-vector (s/coll-of number? :kind vector?))
(s/def ::gate-type #{:rx :ry :rz})
(s/def ::bit-string (s/or :string string? :vector (s/coll-of #{0 1} :kind vector?)))

;; Error handling utilities
(defn- safe-operation
  "Safely execute an operation with error handling."
  [operation error-msg & args]
  (try
    {:success true :result (apply operation args)}
    (catch Exception e
      {:success false 
       :error error-msg 
       :details (.getMessage e)
       :suggestion "Check input parameters and try again"})))

(defn- validate-input
  "Validate input against spec, returning error info if invalid."
  [spec input desc]
  (if (s/valid? spec input)
    {:valid true}
    {:valid false 
     :error (str "Invalid " desc)
     :details (s/explain-str spec input)}))

(defn normalize-features
  "Normalize feature vector using min-max scaling to [0, 1].

   Parameters:
   - feature-vector: Vector of feature values
   
   Returns:
   - Result map with :success boolean and :result/:error
   
   Example:
   (normalize-features [1.0 2.0 3.0 4.0])"
  [feature-vector]
  (let [validation (validate-input ::feature-vector feature-vector "feature vector")]
    (if (:valid validation)
      (safe-operation
       (fn [fv]
         (let [min-val (apply min fv)
               max-val (apply max fv)
               range-val (- max-val min-val)]
           (if (zero? range-val)
             fv
             (mapv #(/ (- % min-val) range-val) fv))))
       "Failed to normalize features"
       feature-vector)
      {:success false :error (:error validation) :details (:details validation)})))

(defn amplitude-encoding
  "Encode classical data into quantum amplitude encoding.
   Maps data values to quantum state amplitudes.
   
   Parameters:
   - feature-vector: Vector of feature values (should be normalized)
   - num-qubits: Number of qubits to use (determines 2^n amplitudes)
   
   Returns:
   - Result map with :success boolean and :result/:error
   
   Example:
   (amplitude-encoding [0.5 0.5 0.5 0.5] 2)"
  [feature-vector num-qubits]
  (let [fv-validation (validate-input ::feature-vector feature-vector "feature vector")
        qb-validation (validate-input ::num-qubits num-qubits "qubit count")]
    (cond
      (not (:valid fv-validation))
      {:success false :error (:error fv-validation) :details (:details fv-validation)}
      
      (not (:valid qb-validation))
      {:success false :error (:error qb-validation) :details (:details qb-validation)}
      
      :else
      (safe-operation
       (fn [fv n-qubits]
         (let [num-amplitudes (int (fm/pow 2 n-qubits))
               _ (when (> (count fv) num-amplitudes)
                   (throw (ex-info "Feature vector too large for qubit count"
                                   {:feature-count (count fv)
                                    :max-amplitudes num-amplitudes
                                    :num-qubits n-qubits})))
               padded-features (take num-amplitudes (concat fv (repeat 0.0)))
               sum-squares (fm/sqrt (reduce + (map #(* % %) padded-features)))
               normalized-amplitudes (if (> sum-squares 0.0)
                                       (mapv #(/ % sum-squares) padded-features)
                                       padded-features)
               complex-amplitudes (mapv #(fc/complex % 0) normalized-amplitudes)]
           (state/multi-qubit-state complex-amplitudes)))
       "Failed to perform amplitude encoding"
       feature-vector
       num-qubits))))

(defn angle-encoding
  "Encode classical data as rotation angles in quantum circuit gates.
   
   This encoding maps classical data values to rotation angles, creating
   a parameterized quantum circuit that encodes the data in qubit rotations.
   
   Parameters:
   - data-row: Vector of classical data values
   - num-qubits: Number of qubits to use for encoding
   - gate-type: Type of rotation gate (:rx, :ry, :rz), default :ry
   
   Returns:
   - Result map with :success boolean and circuit encoder function or error
   
   Example:
   (def result (angle-encoding [0.5 0.3 0.7] 3 :ry))
   (when (:success result)
     ((:result result) (qc/create-circuit 3)))"
  ([data-row num-qubits] (angle-encoding data-row num-qubits :ry))
  ([data-row num-qubits gate-type]
   (let [data-validation (validate-input ::feature-vector data-row "data row")
         qubits-validation (validate-input ::num-qubits num-qubits "qubit count")
         gate-validation (validate-input ::gate-type gate-type "gate type")]
     (cond
       (not (:valid data-validation))
       {:success false :error (:error data-validation) :details (:details data-validation)}
       
       (not (:valid qubits-validation))
       {:success false :error (:error qubits-validation) :details (:details qubits-validation)}
       
       (not (:valid gate-validation))
       {:success false :error (:error gate-validation) :details (:details gate-validation)}
       
       (> (count data-row) num-qubits)
       {:success false 
        :error "Data dimension exceeds qubit count"
        :details (str "Data has " (count data-row) " features but only " num-qubits " qubits available")}
       
       :else
       {:success true
        :result (fn [circuit]
                  (try
                    (reduce-kv (fn [c idx angle]
                                 (if (< idx num-qubits)
                                   (case gate-type
                                     :rx (circuit/rx-gate c idx (* angle fm/PI))
                                     :ry (circuit/ry-gate c idx (* angle fm/PI))
                                     :rz (circuit/rz-gate c idx (* angle fm/PI)))
                                   c))
                               circuit
                               data-row)
                    (catch Exception e
                      (throw (ex-info "Failed to apply angle encoding to circuit"
                                      {:gate-type gate-type
                                       :data-size (count data-row)
                                       :num-qubits num-qubits
                                       :original-error (.getMessage e)}
                                      e)))))}))))

(defn basis-encoding
  "Encode classical data using computational basis encoding.
   
   Maps classical bit strings to computational basis states directly.
   Useful for encoding discrete/categorical data.
   
   Parameters:
   - bit-string: String of bits like '101' or vector of 0/1 values
   - num-qubits: Number of qubits (should match bit string length)
   
   Returns:
   - Result map with :success boolean and :result/:error
   
   Example:
   (basis-encoding \"101\" 3)  ; Creates |101> state
   (basis-encoding [1 0 1] 3)"
  [bit-string num-qubits]
  (let [qubits-validation (validate-input ::num-qubits num-qubits "qubit count")
        bit-string-validation (validate-input ::bit-string bit-string "bit string")]
    (cond
      (not (:valid qubits-validation))
      {:success false :error (:error qubits-validation) :details (:details qubits-validation)}
      
      (not (:valid bit-string-validation))
      {:success false :error (:error bit-string-validation) :details (:details bit-string-validation)}
      
      (not= (count bit-string) num-qubits)
      {:success false 
       :error "Bit string length mismatch"
       :details (str "Bit string has length " (count bit-string) " but expected " num-qubits)}
      
      :else
      (safe-operation
       (fn [bs n-qubits]
         ;; Convert string to bit vector if needed
         (let [bit-vec (cond
                         (string? bs) (mapv #(Character/digit % 10) bs)
                         (vector? bs) bs
                         :else (throw (ex-info "Invalid bit string format" {:input bs})))]
           ;; Validate that all elements are 0 or 1
           (when-not (every? #(or (= % 0) (= % 1)) bit-vec)
             (throw (ex-info "Bit string contains invalid values" 
                             {:input bs :converted bit-vec})))
           ;; Use QClojure's computational-basis-state function
           (state/computational-basis-state n-qubits bit-vec)))
       "Failed to perform basis encoding"
       bit-string
       num-qubits))))

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
          h-circuit (reduce #(circuit/h-gate %1 %2) circuit (range num-qubits))
          ;; Then apply data-dependent phase gates
          data-phases (take (* num-qubits num-qubits) (concat data-row (repeat 0.0)))]
      (reduce-kv (fn [c idx phase]
                   (let [qubit1 (quot idx num-qubits)
                         qubit2 (mod idx num-qubits)]
                     (if (= qubit1 qubit2)
                       ;; Single qubit phase
                       (circuit/rz-gate c qubit1 phase)
                       ;; Two qubit phase (if qubits are different)
                       (if (< qubit1 qubit2)
                         (-> c
                             (circuit/cnot-gate qubit1 qubit2)
                             (circuit/rz-gate qubit2 phase)
                             (circuit/cnot-gate qubit1 qubit2))
                         c))))
                 h-circuit
                 (vec data-phases)))))

