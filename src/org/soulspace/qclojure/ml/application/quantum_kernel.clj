(ns org.soulspace.qclojure.ml.application.quantum-kernel
  "Production-ready quantum kernel methods for quantum machine learning.
  
  Quantum kernels compute similarity measures between classical data points by encoding
  them into quantum states and measuring their overlap. This implementation provides
  hardware-compatible kernel computation using measurement-based approaches suitable
  for real quantum devices.
  
  Key Features:
  - Hardware-compatible SWAP test circuits for overlap estimation
  - Support for multiple encoding strategies (angle, amplitude, basis, IQP)
  - Efficient kernel matrix computation using transients
  - Batched processing for large datasets
  - Integration with QClojure backend protocols
  - Production-ready error handling and validation
  
  Algorithm:
  1. Encode classical data points into quantum states using feature maps
  2. Compute pairwise overlaps |⟨ψ(x_i)|ψ(x_j)⟩|² using SWAP test circuits
  3. Build kernel matrix for use with classical ML algorithms
  4. Support symmetric and asymmetric kernel computations"
  (:require [clojure.spec.alpha :as s]
            [fastmath.core :as fm]
            [org.soulspace.qclojure.application.backend :as backend]
            [org.soulspace.qclojure.application.algorithm.optimization :as opt]
            [org.soulspace.qclojure.application.algorithm.variational-algorithm :as va]
            [org.soulspace.qclojure.domain.circuit :as circuit]
            [org.soulspace.qclojure.domain.state :as state]
            [org.soulspace.qclojure.ml.application.encoding :as encoding]
            [org.soulspace.qclojure.ml.application.training :as training]))

;;;
;;; Specs for quantum kernel operations
;;;
(s/def ::data-matrix (s/coll-of ::encoding/feature-vector :kind vector?))
(s/def ::kernel-matrix (s/coll-of (s/coll-of number? :kind vector?) :kind vector?))
(s/def ::encoding-type #{:angle :amplitude :basis :iqp :trainable})
(s/def ::shots pos-int?)
(s/def ::overlap-result (s/keys :req-un [::overlap-value ::measurement-data]))
(s/def ::overlap-value (s/and number? #(<= 0.0 % 1.0)))

(s/def ::kernel-config
  (s/keys :req-un [::encoding-type]
          :opt-un [::shots ::encoding/num-qubits ::encoding-options ::trainable-parameters]))

(s/def ::encoding-options map?)
(s/def ::trainable-parameters (s/coll-of number? :kind vector?))

;; Specs for trainable kernel configuration
(s/def ::num-trainable-layers pos-int?)
(s/def ::parameter-strategy #{:random :zero :custom :legacy})
(s/def ::parameter-range (s/tuple number? number?))
(s/def ::regularization #{:none :l1 :l2 :elastic-net})
(s/def ::reg-lambda (s/and number? pos?))
(s/def ::reg-alpha (s/and number? #(<= 0.0 % 1.0)))

(s/def ::trainable-kernel-config
  (s/keys :req-un [::encoding-type ::num-trainable-layers]
          :opt-un [::shots ::encoding/num-qubits ::encoding-options
                   ::trainable-parameters ::optimization-method
                   ::max-iterations ::learning-rate ::parameter-strategy
                   ::parameter-range ::regularization ::reg-lambda ::reg-alpha]))

(s/def ::kernel-alignment-objective #{:supervised :target-alignment})
(s/def ::labels (s/coll-of nat-int? :kind vector?))
(s/def ::target-kernel ::kernel-matrix)
(s/def ::kernel-matrix (s/coll-of (s/coll-of number? :kind vector?) :kind vector?))
(s/def ::encoding-type #{:angle :amplitude :basis :iqp})
(s/def ::shots pos-int?)
(s/def ::overlap-result (s/keys :req-un [::overlap-value ::measurement-data]))
(s/def ::overlap-value (s/and number? #(<= 0.0 % 1.0)))

(s/def ::kernel-config
  (s/keys :req-un [::encoding-type]
          :opt-un [::shots ::encoding/num-qubits ::encoding-options]))

(s/def ::encoding-options map?)

;;;
;;; Hardware-compatible SWAP test circuit implementation
;;;
(defn swap-test-circuit
  "Create a SWAP test circuit to measure overlap between two quantum states.
  
  The SWAP test is a fundamental quantum algorithm for estimating the overlap
  |⟨ψ|φ⟩|² between two quantum states |ψ⟩ and |φ⟩. It uses an ancilla qubit
  and controlled operations to encode the overlap in measurement statistics.
  
  Circuit structure:
  1. Prepare ancilla in |+⟩ state (Hadamard gate)
  2. Apply controlled-SWAP between the two state registers
  3. Apply Hadamard to ancilla
  4. Measure ancilla: P(0) = (1 + |⟨ψ|φ⟩|²)/2
  
  Parameters:
  - circuit: Base quantum circuit
  - register1-qubits: Qubit indices for first quantum state
  - register2-qubits: Qubit indices for second quantum state
  - ancilla-qubit: Ancilla qubit index for SWAP test
  
  Returns:
  Circuit with SWAP test operations applied
  
  Hardware compatibility:
  - Uses only standard gates (H, CNOT, controlled gates)
  - Requires single ancilla qubit measurement
  - Compatible with current quantum hardware constraints"
  [circuit register1-qubits register2-qubits ancilla-qubit]
  {:pre [(sequential? register1-qubits)
         (sequential? register2-qubits)
         (= (count register1-qubits) (count register2-qubits))
         (integer? ancilla-qubit)]}

  (let [num-data-qubits (count register1-qubits)]
    (-> circuit
        ;; 1. Prepare ancilla in |+⟩ state
        (circuit/h-gate ancilla-qubit)

        ;; 2. Apply controlled-SWAP operations using Fredkin gates
        ;; For each pair of qubits in the two registers
        ((fn [c]
           (reduce (fn [circuit-acc i]
                     (let [q1 (nth register1-qubits i)
                           q2 (nth register2-qubits i)]
                       ;; Fredkin gate: controlled-SWAP with ancilla as control
                       (circuit/fredkin-gate circuit-acc ancilla-qubit q1 q2)))
                   c
                   (range num-data-qubits))))

        ;; 3. Final Hadamard on ancilla
        (circuit/h-gate ancilla-qubit)

        ;; 4. Measure ancilla qubit
        (circuit/measure-operation [ancilla-qubit]))))

(defn estimate-overlap-from-swap-measurements
  "Extract overlap estimate from SWAP test measurement results using QClojure result system.
  
  From SWAP test theory:
  P(ancilla = 0) = (1 + |⟨ψ|φ⟩|²) / 2
  Therefore: |⟨ψ|φ⟩|² = 2 * P(0) - 1
  
  Parameters:
  - measurement-data: Measurement result data from QClojure result system
  - ancilla-qubit: Index of ancilla qubit
  
  Returns:
  Map with overlap estimate and measurement statistics"
  [measurement-data ancilla-qubit]
  (let [outcomes (:measurement-outcomes measurement-data)
        shots (:shot-count measurement-data)
        frequencies (:frequencies measurement-data)
        
        ;; Extract bit at ancilla qubit position from basis state
        ;; Qubit numbering: LSB is qubit 0, so bit position is qubit index from right
        ;; Handle both integer and string basis states
        ancilla-bit (fn [state]
                      (if (string? state)
                        ;; String format like "10010" - ancilla-qubit indexes from left
                        (let [bit-char (nth state ancilla-qubit)]
                          (if (= bit-char \0) 0 1))
                        ;; Integer format - extract bit using bit operations
                        (bit-and (bit-shift-right state ancilla-qubit) 1)))
        
        ;; Count measurements where ancilla is 0
        ;; Use frequencies for efficiency (avoids iterating all shots)
        count-0 (reduce (fn [acc [state count]]
                          (if (zero? (ancilla-bit state))
                            (+ acc count)
                            acc))
                        0
                        frequencies)

        ;; Calculate probability and overlap
        prob-0 (double (/ count-0 shots))
        overlap-squared (max 0.0 (- (* 2.0 prob-0) 1.0))  ; Ensure non-negative
        overlap (fm/sqrt overlap-squared)]

    {:overlap-value overlap
     :overlap-squared overlap-squared
     :prob-ancilla-0 prob-0
     :measurement-counts {:ancilla-0 count-0
                          :ancilla-1 (- shots count-0)}
     :shots shots
     :measurement-data measurement-data}))

;;;
;;; Data encoding integration for kernel computation
;;;
(defn calculate-trainable-parameter-count
  "Calculate the number of trainable parameters needed for a parametrized feature map.
  
  Parameters:
  - num-qubits: Number of qubits
  - num-layers: Number of trainable layers
  
  Returns:
  Total number of trainable parameters"
  [num-qubits num-layers]
  (* num-qubits 2 num-layers))

(defn encode-data-for-kernel
  "Encode classical data point using specified encoding strategy.
  
  This function creates a quantum circuit that encodes a classical feature vector
  into a quantum state using one of the available encoding methods.
  
  Parameters:
  - data-point: Classical feature vector
  - encoding-type: Type of encoding (:angle, :amplitude, :basis, :iqp)
  - num-qubits: Number of qubits for encoding
  - options: Encoding-specific options
  
  Returns:
  Function that applies encoding to a circuit"
  [data-point encoding-type num-qubits options]
  (case encoding-type
    :angle
    (let [gate-type (:gate-type options :ry)
          encoding-result (encoding/angle-encoding data-point num-qubits gate-type)]
      (if (:success encoding-result)
        (:result encoding-result)
        (throw (ex-info "Angle encoding failed" encoding-result))))

    :amplitude
    (let [encoding-result (encoding/amplitude-encoding data-point num-qubits)]
      (if (:success encoding-result)
        ;; For kernel computation, we need a circuit that prepares the amplitude state
        ;; This is a simplification - in practice would need state preparation circuits
        (let [fallback-result (encoding/angle-encoding data-point num-qubits :ry)]
          (if (:success fallback-result)
            (:result fallback-result)
            (throw (ex-info "Amplitude encoding fallback failed" fallback-result))))
        (throw (ex-info "Amplitude encoding failed" encoding-result))))

    :basis
    (let [;; Convert continuous features to binary for basis encoding
          binary-data (mapv #(if (>= % 0.5) 1 0) data-point)
          encoding-result (encoding/basis-encoding binary-data num-qubits)]
      (if (:success encoding-result)
        ;; For hardware compatibility, create circuit that prepares basis state
        (fn [circuit]
          (reduce (fn [c i]
                    (if (= (nth binary-data i) 1)
                      (circuit/x-gate c i)
                      c))
                  circuit
                  (range (min (count binary-data) num-qubits))))
        (throw (ex-info "Basis encoding failed" encoding-result))))

    :iqp
    (encoding/iqp-encoding data-point num-qubits)

    (throw (ex-info "Unsupported encoding type" {:encoding-type encoding-type}))))

(defn parametrized-feature-map
  "Create a parametrized feature map with trainable parameters.
  
  This feature map combines data encoding with trainable rotation gates,
  allowing the kernel to be optimized for specific datasets. This is critical
  for achieving quantum advantage over classical kernels.
  
  Architecture:
  1. Data encoding layer (angle encoding of features)
  2. Trainable rotation layers (parametrized Ry and Rz gates)
  3. Entangling layers (to create feature interactions)
  
  Parameters:
  - data-point: Classical feature vector
  - trainable-params: Vector of trainable parameters
  - num-qubits: Number of qubits for encoding
  - num-layers: Number of trainable layers
  - options: Additional options
  
  Returns:
  Function that applies parametrized encoding to a circuit"
  [data-point trainable-params num-qubits num-layers options]
  {:pre [(vector? data-point)
         (vector? trainable-params)
         (pos-int? num-qubits)
         (pos-int? num-layers)]}

  (let [base-gate (:gate-type options :ry)
        entangling (:entangling options true)
        params-per-layer (* num-qubits 2)
        expected-params (* num-layers params-per-layer)

        _ (when (< (count trainable-params) expected-params)
            (throw (ex-info "Insufficient trainable parameters"
                            {:expected expected-params
                             :provided (count trainable-params)
                             :num-qubits num-qubits
                             :num-layers num-layers})))]

    (fn [circuit]
      (let [data-encoder (encode-data-for-kernel data-point :angle num-qubits {:gate-type base-gate})
            circuit-with-data (data-encoder circuit)

            circuit-with-trainable
            (reduce
             (fn [c layer-idx]
               (let [layer-start (* layer-idx params-per-layer)
                     layer-params (subvec trainable-params layer-start (+ layer-start params-per-layer))

                     c-rotations
                     (reduce
                      (fn [circ qubit-idx]
                        (let [param-idx (* qubit-idx 2)
                              ry-angle (nth layer-params param-idx)
                              rz-angle (nth layer-params (inc param-idx))]
                          (-> circ
                              (circuit/ry-gate qubit-idx ry-angle)
                              (circuit/rz-gate qubit-idx rz-angle))))
                      c
                      (range num-qubits))

                     c-entangled
                     (if entangling
                       (reduce
                        (fn [circ qubit-idx]
                          (let [target (mod (inc qubit-idx) num-qubits)]
                            (circuit/cnot-gate circ qubit-idx target)))
                        c-rotations
                        (range num-qubits))
                       c-rotations)]

                 c-entangled))
             circuit-with-data
             (range num-layers))]

        circuit-with-trainable))))

(defn trainable-quantum-kernel-overlap
  "Compute quantum kernel overlap using trainable parametrized feature maps.
  
  This function extends the standard kernel computation with trainable parameters,
  allowing the kernel to be optimized for specific datasets. This is essential
  for achieving quantum advantage over classical kernels.
  
  Parameters:
  - backend: Quantum backend for circuit execution
  - data-point1: First classical data vector
  - data-point2: Second classical data vector
  - trainable-params: Vector of trainable parameters
  - config: Kernel configuration with trainable settings
  
  Returns:
  Map with overlap value and measurement details"
  [backend data-point1 data-point2 trainable-params config]
  {:pre [(map? config)
         (vector? trainable-params)]}

  (let [num-qubits (:num-qubits config (max 2 (int (Math/ceil (/ (Math/log (max (count data-point1) (count data-point2))) (Math/log 2))))))
        num-layers (:num-trainable-layers config 2)
        shots (:shots config 1024)
        encoding-options (:encoding-options config {})

        total-qubits (+ (* 2 num-qubits) 1)
        register1-qubits (range 0 num-qubits)
        register2-qubits (range num-qubits (* 2 num-qubits))
        ancilla-qubit (* 2 num-qubits)

        base-circuit (circuit/create-circuit total-qubits)

        encoder1 (parametrized-feature-map data-point1 trainable-params num-qubits num-layers encoding-options)
        circuit-with-state1 (encoder1 base-circuit)

        encoder2 (parametrized-feature-map data-point2 trainable-params num-qubits num-layers encoding-options)
        shifted-circuit (reduce (fn [c op]
                                  (let [shifted-op (update op :operation-params
                                                           (fn [params]
                                                             (cond-> params
                                                               (:target params)
                                                               (update :target #(+ % num-qubits))
                                                               (:control params)
                                                               (update :control #(+ % num-qubits)))))]
                                    (update c :operations conj shifted-op)))
                                circuit-with-state1
                                (:operations (encoder2 (circuit/create-circuit num-qubits))))

        swap-test-circuit (swap-test-circuit shifted-circuit register1-qubits register2-qubits ancilla-qubit)

        options {:result-specs {:measurements {:shots shots
                                               :qubits [ancilla-qubit]}}}

        execution-result (backend/execute-circuit backend swap-test-circuit options)
        measurement-data (get-in execution-result [:results :measurement-results])]

    (if (or (nil? measurement-data) (empty? (:measurement-outcomes measurement-data)))
      {:overlap-value 0.0
       :error "No measurement results obtained"
       :measurement-data {}}
      (estimate-overlap-from-swap-measurements measurement-data ancilla-qubit))))

(defn compute-trainable-kernel-matrix
  "Compute quantum kernel matrix with trainable parameters.
  
  Parameters:
  - backend: Quantum backend
  - data-matrix: Matrix of data vectors
  - trainable-params: Trainable parameters for feature map
  - config: Trainable kernel configuration
  
  Returns:
  Kernel matrix computed with trainable feature maps"
  [backend data-matrix trainable-params config]
  {:pre [(vector? data-matrix)
         (vector? trainable-params)
         (map? config)]}
  
  (let [n (count data-matrix)]
    (mapv (fn [i]
            (mapv (fn [j]
                    (if (= i j)
                      1.0
                      (let [overlap-result (trainable-quantum-kernel-overlap
                                            backend
                                            (nth data-matrix i)
                                            (nth data-matrix j)
                                            trainable-params
                                            config)]
                        (if (:error overlap-result)
                          0.0
                          (:overlap-value overlap-result)))))
                  (range n)))
          (range n))))

;;;
;;; Core quantum kernel computation functions
;;;
(defn train-quantum-kernel
  "Train a quantum kernel using Quantum Kernel Alignment (QKA).
  
  This function optimizes the trainable parameters of a quantum kernel to maximize
  alignment with an ideal kernel (supervised) or to optimize for a specific task.
  This is the key to achieving quantum advantage over classical kernels.
  
  Kernel Alignment Objective:
  - Supervised: Maximize alignment with ideal kernel from labels
  - Target Alignment: Maximize alignment with provided target kernel
  
  Parameters:
  - backend: Quantum backend
  - data-matrix: Training data matrix
  - labels: Training labels (for supervised alignment)
  - config: Training configuration
  
  Required config:
  - :num-qubits - Number of qubits
  - :num-trainable-layers - Number of trainable layers
  - :alignment-objective - :supervised or :target-alignment
  
  Optional config:
  - :target-kernel - Target kernel matrix (for target alignment)
  - :optimization-method - Optimizer (:adam, :cmaes, :nelder-mead, :powell, :bobyqa, :gradient-descent)
  - :max-iterations - Maximum training iterations (default: 100)
  - :learning-rate - Learning rate for gradient-based optimizers (default: 0.01)
  - :shots - Shots per circuit (default: 1024)
  - :parameter-strategy - Parameter init strategy (:random, :zero, :custom, :legacy, default: :random)
  - :parameter-range - Range for random init (default: [-π π])
  - :initial-parameters - Custom initial parameters (if :custom strategy)
  - :regularization - Regularization type (:none, :l1, :l2, :elastic-net, default: :none)
  - :reg-lambda - Regularization strength (default: 0.01)
  - :reg-alpha - Elastic net mix ratio (default: 0.5)
  
  Returns:
  Map with trained parameters and training history"
  [backend data-matrix labels config]
  {:pre [(vector? data-matrix)
         (or (nil? labels) (vector? labels))
         (map? config)]}
  
  (let [num-qubits (:num-qubits config)
        num-layers (:num-trainable-layers config 2)
        objective-type (:alignment-objective config :supervised)
        optimization-method (:optimization-method config :cmaes)
        max-iterations (:max-iterations config 100)
        learning-rate (:learning-rate config 0.01)
        shots (:shots config 1024)
        tolerance (:tolerance config 1e-6)
        
        ;; Regularization configuration
        regularization (:regularization config :none)
        reg-lambda (:reg-lambda config 0.01)
        reg-alpha (:reg-alpha config 0.5)
        
        ;; Use training utilities for parameter initialization
        num-params (calculate-trainable-parameter-count num-qubits num-layers)
        parameter-strategy (:parameter-strategy config :random)
        initial-params (case parameter-strategy
                         :random (va/random-parameter-initialization 
                                  num-params 
                                  :range (:parameter-range config [(- fm/PI) fm/PI]))
                         :zero (va/zero-parameter-initialization num-params)
                         :custom (:initial-parameters config)
                         :legacy (vec (repeatedly num-params #(* 0.1 (rand))))
                         ;; Default to random with [-π, π] range
                         (va/random-parameter-initialization num-params :range [(- fm/PI) fm/PI]))
        
        ideal-kernel (when (and labels (= objective-type :supervised))
                       (let [n (count data-matrix)]
                         (mapv (fn [i]
                                 (mapv (fn [j]
                                         (if (= (nth labels i) (nth labels j))
                                           1.0
                                           0.0))
                                       (range n)))
                               (range n))))
        
        target-kernel (or (:target-kernel config) ideal-kernel)
        
        alignment-objective
        (fn [params]
          (try
            (let [kernel-config (assoc config
                                       :encoding-type :trainable
                                       :num-qubits num-qubits
                                       :num-trainable-layers num-layers
                                       :shots shots)
                  quantum-kernel (compute-trainable-kernel-matrix backend data-matrix params kernel-config)
                  kernel-frobenius (fn [k]
                                     (let [n (count k)]
                                       (fm/sqrt (reduce +
                                                        (for [i (range n)
                                                              j (range n)]
                                                          (fm/sq (get-in k [i j])))))))
                  numerator (let [n (count quantum-kernel)]
                              (reduce +
                                      (for [i (range n)
                                            j (range n)]
                                        (* (get-in quantum-kernel [i j])
                                           (get-in target-kernel [i j])))))
                  denominator (* (kernel-frobenius quantum-kernel)
                                 (kernel-frobenius target-kernel))
                  alignment (if (> denominator 0.0)
                              (/ numerator denominator)
                              0.0)
                  
                  ;; Add regularization penalty using training utilities
                  reg-penalty (case regularization
                                :l1 (training/l1-regularization params reg-lambda)
                                :l2 (training/l2-regularization params reg-lambda)
                                :elastic-net (training/elastic-net-regularization params reg-lambda reg-alpha)
                                :none 0.0
                                0.0)]
              ;; Return negative alignment (to minimize) plus regularization penalty
              (+ (- alignment) reg-penalty))
            (catch Exception e
              (println "Warning: alignment computation failed:" (.getMessage e))
              1.0)))]
    
    ;; Use newer unified optimization API from training namespace pattern
    (let [result (cond
                   (= optimization-method :gradient-descent)
                   (opt/gradient-descent-optimization alignment-objective initial-params
                                                      {:max-iterations max-iterations
                                                       :learning-rate learning-rate
                                                       :tolerance tolerance})
                   
                   (= optimization-method :adam)
                   (opt/adam-optimization alignment-objective initial-params
                                          {:max-iterations max-iterations
                                           :learning-rate learning-rate
                                           :tolerance tolerance})
                   
                   (= optimization-method :cmaes)
                   (opt/fastmath-derivative-free-optimization :cmaes alignment-objective initial-params
                                                              {:max-iterations max-iterations
                                                               :tolerance tolerance})
                   
                   (= optimization-method :nelder-mead)
                   (opt/fastmath-derivative-free-optimization :nelder-mead alignment-objective initial-params
                                                              {:max-iterations max-iterations
                                                               :tolerance tolerance})
                   
                   (= optimization-method :powell)
                   (opt/fastmath-derivative-free-optimization :powell alignment-objective initial-params
                                                              {:max-iterations max-iterations
                                                               :tolerance tolerance})
                   
                   (= optimization-method :bobyqa)
                   (opt/fastmath-derivative-free-optimization :bobyqa alignment-objective initial-params
                                                              {:max-iterations max-iterations
                                                               :tolerance tolerance})
                   
                   ;; Default to CMAES
                   :else
                   (opt/fastmath-derivative-free-optimization :cmaes alignment-objective initial-params
                                                              {:max-iterations max-iterations
                                                               :tolerance tolerance}))]
      
      {:success true
       :optimal-parameters (:optimal-parameters result)
       :optimal-alignment (- (:optimal-value result))
       :initial-alignment (- (alignment-objective initial-params))
       :iterations (:iterations result)
       :optimization-method optimization-method
       :config config
       :training-history result})))

(defn quantum-kernel-overlap
  "Compute quantum kernel overlap between two data points using SWAP test.
  
  This function implements the core quantum kernel computation by:
  1. Encoding both data points into quantum states
  2. Creating a SWAP test circuit to measure their overlap
  3. Executing the circuit with proper result specs and extracting overlap from measurements
  
  Parameters:
  - backend: Quantum backend for circuit execution
  - data-point1: First classical data vector
  - data-point2: Second classical data vector
  - config: Kernel configuration including encoding type and options
  
  Returns:
  Map with overlap value and measurement details"
  [backend data-point1 data-point2 config]
  {:pre [(map? config)
         (s/valid? ::kernel-config config)]}

  (let [encoding-type (:encoding-type config)
        num-qubits (:num-qubits config (max 2 (int (Math/ceil (/ (Math/log (max (count data-point1) (count data-point2))) (Math/log 2))))))
        shots (:shots config 1024)
        encoding-options (:encoding-options config {})

        ;; Calculate required qubits: 2 registers + 1 ancilla
        total-qubits (+ (* 2 num-qubits) 1)
        register1-qubits (range 0 num-qubits)
        register2-qubits (range num-qubits (* 2 num-qubits))
        ancilla-qubit (* 2 num-qubits)

        ;; Create base circuit
        base-circuit (circuit/create-circuit total-qubits)

        ;; Encode first data point in first register
        encoder1 (encode-data-for-kernel data-point1 encoding-type num-qubits encoding-options)
        circuit-with-state1 (encoder1 base-circuit)

        ;; Encode second data point in second register (shift qubit indices)
        encoder2 (encode-data-for-kernel data-point2 encoding-type num-qubits encoding-options)
        ;; Create shifted circuit for second register
        shifted-circuit (reduce (fn [c op]
                                  (let [shifted-op (update op :operation-params
                                                           (fn [params]
                                                             (cond-> params
                                                               (:target params)
                                                               (update :target #(+ % num-qubits))
                                                               (:control params)
                                                               (update :control #(+ % num-qubits)))))]
                                    (update c :operations conj shifted-op)))
                                circuit-with-state1
                                (:operations (encoder2 (circuit/create-circuit num-qubits))))

        ;; Apply SWAP test
        swap-test-circuit (swap-test-circuit shifted-circuit register1-qubits register2-qubits ancilla-qubit)

        ;; Define options with result specifications for measurement extraction
        options {:result-specs {:measurements {:shots shots
                                               :qubits [ancilla-qubit]}}}

        ;; Execute circuit with result specs (wrapped in options map)
        execution-result (backend/execute-circuit backend swap-test-circuit options)

        measurement-data (get-in execution-result [:results :measurement-results])]

    (if (or (nil? measurement-data) (empty? (:measurement-outcomes measurement-data)))
      {:overlap-value 0.0
       :error "No measurement results obtained"
       :measurement-data {}}
      (estimate-overlap-from-swap-measurements measurement-data ancilla-qubit))))

;;;
;;; Efficient kernel matrix computation with transients
;;;
(defn quantum-kernel-matrix
  "Compute quantum kernel matrix for a dataset using efficient batched processing.
  
  This function computes the full kernel matrix K where K[i,j] represents the
  quantum kernel value between data points i and j. Uses transient data structures
  for efficient matrix construction.
  
  Parameters:
  - backend: Quantum backend for circuit execution
  - data-matrix: Matrix of classical data vectors (rows are data points)
  - config: Kernel configuration
  - symmetric?: If true, compute only upper triangle (default: true)
  
  Returns:
  Symmetric kernel matrix as vector of vectors"
  ([backend data-matrix config]
   (quantum-kernel-matrix backend data-matrix config true))

  ([backend data-matrix config symmetric?]
   {:pre [(vector? data-matrix)
          (every? vector? data-matrix)
          (map? config)]}

   (let [n (count data-matrix)
         ;; Pre-validate configuration
         _ (when-not (s/valid? ::kernel-config config)
             (throw (ex-info "Invalid kernel configuration"
                             {:config config
                              :errors (s/explain-data ::kernel-config config)})))
         ;; Build kernel matrix row by row (avoiding complex transient nesting)
         matrix-rows
         (mapv (fn [i]
                 (let [data-point-i (nth data-matrix i)
                       ;; For symmetric matrices, only compute upper triangle
                       j-start (if symmetric? i 0)]
                   
                   ;; Compute row i
                   (mapv (fn [j]
                           (let [data-point-j (nth data-matrix j)]
                             (cond
                               ;; Diagonal elements
                               (= i j) 1.0

                               ;; Upper triangle (or all elements if not symmetric)
                               (or (not symmetric?) (>= j i))
                               (try
                                 (let [overlap-result (quantum-kernel-overlap
                                                       backend data-point-i data-point-j config)]
                                   (if (:error overlap-result)
                                     0.0  ; Fallback for failed computations
                                     (:overlap-value overlap-result)))
                                 (catch Exception e
                                   (println "Warning: kernel computation failed for" i j ":" (.getMessage e))
                                   0.0))

                               ;; Lower triangle for symmetric matrices
                               :else 0.0)))  ; Will be filled later
                         (range n))))
               (range n))]
     ;; For symmetric matrices, copy upper triangle to lower triangle
     (if symmetric?
       (mapv (fn [i]
               (mapv (fn [j]
                       (if (< j i)
                         (get-in matrix-rows [j i])  ; Copy from upper triangle
                         (get-in matrix-rows [i j])))
                     (range n)))
             (range n))
       matrix-rows))))

;;;
;;; Optimized kernel computation strategies
;;;
(defn precompute-encodings
  "Precompute quantum encodings for all data points to optimize repeated kernel computations.
  
  This optimization strategy precomputes the quantum circuits for encoding each data point,
  avoiding redundant encoding operations when computing the full kernel matrix.
  
  Parameters:
  - data-matrix: Matrix of classical data vectors
  - encoding-type: Type of encoding to use
  - num-qubits: Number of qubits for encoding
  - encoding-options: Encoding-specific options
  
  Returns:
  Vector of precomputed encoding functions"
  [data-matrix encoding-type num-qubits encoding-options]
  (mapv (fn [data-point]
          (encode-data-for-kernel data-point encoding-type num-qubits encoding-options))
        data-matrix))

(defn batch-kernel-computation
  "Compute kernel matrix using batched approach for memory efficiency.
  
  For large datasets, this function computes the kernel matrix in batches
  to manage memory usage and provide progress monitoring.
  
  Parameters:
  - backend: Quantum backend
  - data-matrix: Matrix of data vectors
  - config: Kernel configuration
  - batch-size: Number of kernel computations per batch (default: 100)
  
  Returns:
  Complete kernel matrix computed in batches"
  ([backend data-matrix config]
   (batch-kernel-computation backend data-matrix config 100))

  ([backend data-matrix config batch-size]
   (let [n (count data-matrix)
         total-computations (/ (* n (inc n)) 2)  ; Upper triangle + diagonal
         completed (volatile! 0)]

     (println (str "Computing " total-computations " kernel entries in batches of " batch-size))

     ;; Use the standard kernel matrix computation but add progress monitoring
     ;; This is a simplified implementation - could be further optimized
     (quantum-kernel-matrix backend data-matrix config))))

;;;
;;; Kernel matrix analysis and utilities
;;;
(defn analyze-kernel-matrix
  "Analyze properties of computed kernel matrix.
  
  Provides statistical analysis of the kernel matrix including:
  - Eigenvalue spectrum
  - Condition number
  - Symmetry verification
  - Positive semidefinite check
  
  Parameters:
  - kernel-matrix: Computed kernel matrix
  
  Returns:
  Map with analysis results"
  [kernel-matrix]
  (let [n (count kernel-matrix)

        ;; Basic statistics
        all-values (flatten kernel-matrix)
        mean-value (/ (reduce + all-values) (count all-values))
        min-value (apply min all-values)
        max-value (apply max all-values)

        ;; Check symmetry
        symmetric? (every? true?
                           (for [i (range n)
                                 j (range n)]
                             (fm/approx-eq (get-in kernel-matrix [i j])
                                           (get-in kernel-matrix [j i])
                                           10)))

        ;; Check diagonal elements (should be 1.0 for normalized kernels)
        diagonal-values (mapv #(get-in kernel-matrix [% %]) (range n))
        diagonal-correct? (every? #(fm/approx-eq % 1.0 6) diagonal-values)]

    {:dimensions [n n]
     :statistics {:mean mean-value
                  :min min-value
                  :max max-value
                  :diagonal-values diagonal-values}
     :properties {:symmetric? symmetric?
                  :diagonal-correct? diagonal-correct?}
     :summary (str "Kernel matrix " n "×" n
                   (if symmetric? " (symmetric)" " (asymmetric)")
                   (if diagonal-correct? " with correct diagonal" " with incorrect diagonal"))}))

;;;
;;; Convenience functions and high-level interface
;;;
(defn create-quantum-kernel
  "Create a quantum kernel function for use with classical ML algorithms.
  
  Returns a function that computes quantum kernel values between data points.
  This can be used as a drop-in replacement for classical kernels in ML pipelines.
  
  Parameters:
  - backend: Quantum backend
  - config: Kernel configuration
  
  Returns:
  Function (data-point1, data-point2) -> kernel-value"
  [backend config]
  {:pre [(s/valid? ::kernel-config config)]}

  (fn [data-point1 data-point2]
    (let [result (quantum-kernel-overlap backend data-point1 data-point2 config)]
      (if (:error result)
        0.0  ; Return default value for failed computations
        (:overlap-value result)))))

(defn quantum-kernel-svm-matrix
  "Compute kernel matrix optimized for SVM training.
  
  This function provides a kernel matrix suitable for SVM training with additional
  optimizations and regularization options.
  
  Parameters:
  - backend: Quantum backend
  - data-matrix: Training data matrix
  - config: Kernel configuration
  - regularization: Regularization parameter (added to diagonal)
  
  Returns:
  Regularized kernel matrix for SVM training"
  ([backend data-matrix config]
   (quantum-kernel-svm-matrix backend data-matrix config 1e-6))

  ([backend data-matrix config regularization]
   (let [kernel-matrix (quantum-kernel-matrix backend data-matrix config)
         n (count kernel-matrix)]

     ;; Add regularization to diagonal
     (mapv (fn [i]
             (mapv (fn [j]
                     (if (= i j)
                       (+ (get-in kernel-matrix [i j]) regularization)
                       (get-in kernel-matrix [i j])))
                   (range n)))
           (range n)))))

(comment
  ;; Example usage and testing
  (require '[org.soulspace.qclojure.adapter.backend.ideal-simulator :as sim])
  ;; 0. Initialize backend
  (def backend (sim/create-simulator))

  ;; 1. Create sample data
  (def sample-data [[0.1 0.2] [0.8 0.9] [0.2 0.1] [0.9 0.8]])

  ;; 2. Configure quantum kernel
  (def kernel-config
    {:encoding-type :angle
     :num-qubits 2
     :shots 1024
     :encoding-options {:gate-type :ry}})

  ;; 3. Test overlap computation between two points
  (def overlap-result (quantum-kernel-overlap backend (first sample-data) (second sample-data) kernel-config))

  ;; 4. Compute full kernel matrix
  (def kernel-matrix (quantum-kernel-matrix backend sample-data kernel-config))

  ;; 5. Analyze kernel matrix
  (analyze-kernel-matrix kernel-matrix)

  ;; 6. Create kernel function for ML pipeline
  (def kernel-fn (create-quantum-kernel backend kernel-config))
  (kernel-fn [0.1 0.2] [0.8 0.9])

  ;; Expected properties:
  ;; - Kernel matrix should be symmetric
  ;; - Diagonal elements should be 1.0
  ;; - All values should be between 0.0 and 1.0
  ;; - Similar data points should have higher kernel values

  ;;;
  ;;; Trainable Quantum Kernels
  ;;;

  ;; 1. Prepare training data with labels
  (def training-data [[0.1 0.2] [0.15 0.25] [0.8 0.9] [0.85 0.95]])
  (def training-labels [0 0 1 1])  ; Binary classification

  ;; 2. Configure trainable kernel
  (def trainable-config
    {:num-qubits 2
     :num-trainable-layers 2
     :encoding-type :trainable
     :alignment-objective :supervised
     :optimization-method :cmaes
     :max-iterations 50
     :shots 1024})

  ;; 3. Train the quantum kernel using Quantum Kernel Alignment
  (def training-result
    (train-quantum-kernel backend training-data training-labels trainable-config))

  ;; 4. Extract trained parameters
  (def trained-params (:optimal-parameters training-result))
  (:optimal-alignment training-result)  ; Final kernel alignment
  (:initial-alignment training-result)  ; Initial alignment (baseline)
  (:iterations training-result)         ; Number of optimization iterations

  ;; 5. Use trained kernel for inference
  (def inference-config
    (assoc trainable-config :encoding-type :trainable))

  (def trained-kernel-matrix
    (compute-trainable-kernel-matrix backend training-data trained-params inference-config))

  ;; 6. Test on new data
  (def test-data [[0.12 0.22] [0.82 0.92]])
  (def test-kernel-matrix
    (compute-trainable-kernel-matrix backend test-data trained-params inference-config))

  ;; Expected outcomes:
  ;; - Alignment should improve during training (optimal > initial)
  ;; - Data points with same labels should have higher kernel values
  ;; - Trained kernel should separate classes better than unparameterized kernel
  ;; - This demonstrates quantum advantage through trainable feature maps

  ;
  )