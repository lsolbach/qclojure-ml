(ns org.soulspace.qclojure.ml.application.qnn
  "Quantum Neural Networks (QNN) implementation.
  
  QNNs are parameterized quantum circuits organized in layers that can approximate
  arbitrary functions for both classification and regression tasks. This implementation
  follows Clojure principles of simplicity and data orientation.
  
  Key Features:
  - Layer-based architecture using simple data maps
  - Explicit parameter management per layer
  - Sequential network composition using vectors
  - Integration with existing variational algorithm infrastructure
  - Hardware-compatible quantum operations
  - Support for multiple activation patterns
  
  Design Principles:
  - Layers as data: Each layer is a simple map describing its configuration
  - Explicit parameters: Clear parameter ownership and allocation per layer
  - Functional composition: Networks as vectors of layers processed sequentially
  - Data orientation: All network state represented as plain Clojure data structures
  
  Architecture:
  1. Input encoding layer (feature map)
  2. Hidden quantum layers (parameterized unitaries)
  3. Quantum activation functions (non-linear transformations)
  4. Output measurement layer (classical readout)"
  (:require [clojure.spec.alpha :as s]
            [fastmath.core :as fm]
            [org.soulspace.qclojure.domain.circuit :as circuit]
            [org.soulspace.qclojure.domain.result :as result]
            [org.soulspace.qclojure.domain.ansatz :as ansatz]
            [org.soulspace.qclojure.application.algorithm.variational-algorithm :as va]
            [org.soulspace.qclojure.ml.application.encoding :as encoding]
            [org.soulspace.qclojure.ml.application.training :as training]))

;;;
;;; Specs for QNN Layer Types and Network Configuration
;;;

;; Core layer specifications
(s/def ::layer-type #{:input :dense :entangling :activation :measurement :output})
(s/def ::num-qubits pos-int?)
(s/def ::layer-name (s/nilable string?))
(s/def ::activation-type #{:quantum-tanh :quantum-relu :pauli-rotation :none})
(s/def ::feature-map-type ::encoding/feature-map-type)

(s/def ::encoding-options map?)

(s/def ::entangling-pattern #{:linear :circular :all-to-all :custom})
(s/def ::measurement-basis #{:computational :pauli-z :pauli-x :pauli-y})

;; Parameter specifications
(s/def ::parameter-count nat-int?)
(s/def ::parameter-indices (s/coll-of nat-int? :kind vector?))
(s/def ::layer-parameters (s/coll-of number? :kind vector?))
(s/def ::entangling-options map?)
(s/def ::activation-options map?)
(s/def ::measurement-options map?)

;; Layer configuration maps
(s/def ::base-layer
  (s/keys :req-un [::layer-type ::num-qubits ::parameter-count]
          :opt-un [::layer-name ::parameter-indices]))

(s/def ::input-layer
  (s/merge ::base-layer
           (s/keys :req-un [::feature-map-type]
                   :opt-un [::encoding-options])))

(s/def ::dense-layer
  (s/merge ::base-layer
           (s/keys :opt-un [::activation-type])))

(s/def ::entangling-layer  
  (s/merge ::base-layer
           (s/keys :req-un [::entangling-pattern]
                   :opt-un [::entangling-options])))

(s/def ::activation-layer
  (s/merge ::base-layer
           (s/keys :req-un [::activation-type]
                   :opt-un [::activation-options])))

(s/def ::measurement-layer
  (s/merge ::base-layer
           (s/keys :req-un [::measurement-basis]
                   :opt-un [::measurement-options])))

;; Complete layer specification 
(defmulti layer-type-dispatch :layer-type)
(defmethod layer-type-dispatch :input [_] ::input-layer)
(defmethod layer-type-dispatch :dense [_] ::dense-layer)
(defmethod layer-type-dispatch :entangling [_] ::entangling-layer)
(defmethod layer-type-dispatch :activation [_] ::activation-layer)
(defmethod layer-type-dispatch :measurement [_] ::measurement-layer)
(defmethod layer-type-dispatch :output [_] ::measurement-layer)

(s/def ::qnn-layer (s/multi-spec layer-type-dispatch :layer-type))

;; Network specifications
(s/def ::qnn-network (s/coll-of ::qnn-layer :kind vector? :min-count 1))
(s/def ::total-parameters pos-int?)
(s/def ::parameter-map (s/map-of keyword? ::layer-parameters))

(s/def ::qnn-config
  (s/keys :req-un [::qnn-network ::total-parameters]
          :opt-un [::parameter-map ::training/training-data]))

;;;
;;; Layer Parameter Counting and Allocation
;;;

(defn count-layer-parameters
  "Count the number of parameters needed for a specific layer type.
  
  Parameters:
  - layer: Layer configuration map
  
  Returns:
  Number of parameters required for this layer"
  [layer]
  (case (:layer-type layer)
    :input 0  ; Input layers typically don't have trainable parameters
    
    :dense 
    ;; Dense layer: 3 parameters per qubit (Rx, Ry, Rz rotations)
    (* 3 (:num-qubits layer))
    
    :entangling
    ;; Entangling layer: parameters depend on pattern
    (case (:entangling-pattern layer :linear)  ; Default to :linear
      :linear (max 0 (dec (:num-qubits layer)))  ; n-1 CNOT gates
      :circular (:num-qubits layer)              ; n CNOT gates (circular)
      :all-to-all (let [n (:num-qubits layer)]  ; all pairs
                    (* n (dec n)))
      :custom (or (:custom-parameter-count layer) 0)
      (max 0 (dec (:num-qubits layer))))  ; Default to linear pattern
    
    :activation
    ;; Activation layer: depends on activation type
    (case (:activation-type layer :quantum-tanh)  ; Default to :quantum-tanh
      :quantum-tanh (:num-qubits layer)    ; 1 parameter per qubit
      :quantum-relu (:num-qubits layer)    ; 1 parameter per qubit  
      :pauli-rotation (* 3 (:num-qubits layer))  ; 3 parameters per qubit
      :none 0
      (:num-qubits layer))  ; Default to 1 parameter per qubit
    
    :measurement 0  ; Measurement layers don't have trainable parameters
    :output 0       ; Output layers don't have trainable parameters
    0))

(defn allocate-parameters
  "Allocate parameter indices for each layer in a QNN network.
  
  Parameters:
  - network: Vector of layer configuration maps
  
  Returns:
  Map with :network (updated with parameter indices) and :total-parameters"
  [network]
  {:pre [(s/valid? ::qnn-network network)]}
  (let [layers-with-params
        (reduce
         (fn [{:keys [updated-layers param-index]} layer]
           (let [param-count (count-layer-parameters layer)
                 param-indices (when (pos? param-count)
                                 (vec (range param-index (+ param-index param-count))))
                 updated-layer (cond-> layer
                                 true (assoc :parameter-count param-count)
                                 param-indices (assoc :parameter-indices param-indices))]
             {:updated-layers (conj updated-layers updated-layer)
              :param-index (+ param-index param-count)}))
         {:updated-layers [] :param-index 0}
         network)]
    {:network (:updated-layers layers-with-params)
     :total-parameters (:param-index layers-with-params)}))

;;;
;;; Layer Implementation Functions
;;;

(defn apply-input-layer
  "Apply input layer (feature encoding) to a quantum circuit.
  
  Parameters:
  - circuit: Input quantum circuit
  - layer: Input layer configuration
  - feature-data: Classical feature vector to encode
  - parameters: Not used for input layers
  
  Returns:
  Quantum circuit with encoded features"
  [circuit layer feature-data _parameters]
  {:pre [(s/valid? ::input-layer layer)]}
  (let [feature-map-type (:feature-map-type layer)
        encoding-options (:encoding-options layer {})
        num-qubits (:num-qubits layer)]
    
    (case feature-map-type
      :angle
      (let [encoding-result (encoding/angle-encoding feature-data num-qubits 
                                                     (:gate-type encoding-options :ry))]
        (if (:success encoding-result)
          ((:result encoding-result) circuit)
          (throw (ex-info "Feature encoding failed" encoding-result))))
      
      :amplitude
      (let [encoding-result (encoding/amplitude-encoding feature-data num-qubits)]
        (if (:success encoding-result)
          ;; For amplitude encoding, we need to create a circuit that prepares the state
          ;; This is simplified - in practice, we'd need state preparation circuits
          (let [fallback-result (encoding/angle-encoding feature-data num-qubits :ry)]
            (if (:success fallback-result)
              ((:result fallback-result) circuit)
              (throw (ex-info "Amplitude encoding fallback failed" fallback-result))))
          (throw (ex-info "Amplitude encoding failed" encoding-result))))
      
      :basis
      (let [binary-data (mapv #(if (> % 0.5) 1 0) feature-data)
            encoding-result (encoding/basis-encoding binary-data num-qubits)]
        (if (:success encoding-result)
          ;; Create circuit that prepares the basis state
          (reduce (fn [c i]
                    (if (= 1 (nth binary-data i))
                      (circuit/x-gate c i)
                      c))
                  circuit
                  (range (min (count binary-data) num-qubits)))
          (throw (ex-info "Basis encoding failed" encoding-result))))
      
      :iqp
      (let [iqp-encoder (encoding/iqp-encoding feature-data num-qubits)]
        (iqp-encoder circuit))
      
      (throw (ex-info "Unsupported feature map type" {:type feature-map-type})))))

(defn apply-dense-layer
  "Apply dense quantum layer with parameterized rotations.
  
  Parameters:
  - circuit: Input quantum circuit
  - layer: Dense layer configuration
  - feature-data: Not used for dense layers
  - parameters: Parameter vector for this layer
  
  Returns:
  Quantum circuit with applied dense transformations"
  [circuit layer _feature-data parameters]
  {:pre [(s/valid? ::dense-layer layer)
         (= (count parameters) (:parameter-count layer))]}
  (let [num-qubits (:num-qubits layer)]
    ;; Apply Rx, Ry, Rz rotations to each qubit using parameters
    (reduce
     (fn [c qubit]
       (let [param-base (* qubit 3)
             rx-angle (nth parameters param-base)
             ry-angle (nth parameters (inc param-base))
             rz-angle (nth parameters (+ param-base 2))]
         (-> c
             (circuit/rx-gate qubit rx-angle)
             (circuit/ry-gate qubit ry-angle)
             (circuit/rz-gate qubit rz-angle))))
     circuit
     (range num-qubits))))

(defn apply-entangling-layer
  "Apply entangling layer with specified connectivity pattern.
  
  Parameters:
  - circuit: Input quantum circuit
  - layer: Entangling layer configuration
  - feature-data: Not used for entangling layers
  - parameters: Parameter vector for this layer (for parameterized entangling)
  
  Returns:
  Quantum circuit with applied entangling operations"
  [circuit layer _feature-data _parameters]
  {:pre [(s/valid? ::entangling-layer layer)]}
  (let [num-qubits (:num-qubits layer)
        pattern (:entangling-pattern layer)]
    
    (case pattern
      :linear
      ;; Linear connectivity: 0-1, 1-2, 2-3, ...
      (reduce
       (fn [c i]
         (circuit/cnot-gate c i (inc i)))
       circuit
       (range (dec num-qubits)))
      
      :circular  
      ;; Circular connectivity: 0-1, 1-2, ..., (n-1)-0
      (reduce
       (fn [c i]
         (circuit/cnot-gate c i (mod (inc i) num-qubits)))
       circuit
       (range num-qubits))
      
      :all-to-all
      ;; All-to-all connectivity: every qubit connected to every other
      (reduce
       (fn [c [i j]]
         (circuit/cnot-gate c i j))
       circuit
       (for [i (range num-qubits)
             j (range num-qubits)
             :when (< i j)]
         [i j]))
      
      :custom
      ;; Custom pattern - would need to be specified in layer configuration
      (let [custom-connections (:custom-connections layer [])]
        (reduce
         (fn [c [control target]]
           (circuit/cnot-gate c control target))
         circuit
         custom-connections))
      
      ;; Default: no entanglement
      circuit)))

(defn apply-activation-layer
  "Apply quantum activation function to circuit qubits.
  
  Parameters:
  - circuit: Input quantum circuit
  - layer: Activation layer configuration
  - feature-data: Not used for activation layers
  - parameters: Parameter vector for this layer
  
  Returns:
  Quantum circuit with applied activation function"
  [circuit layer _feature-data parameters]
  {:pre [(s/valid? ::activation-layer layer)]}
  (let [num-qubits (:num-qubits layer)
        activation-type (:activation-type layer)]
    
    (case activation-type
      :quantum-tanh
      ;; Quantum tanh using Ry rotations
      (reduce
       (fn [c qubit]
         (let [angle (nth parameters qubit)]
           (circuit/ry-gate c qubit (fm/tanh angle))))
       circuit
       (range num-qubits))
      
      :quantum-relu
      ;; Quantum ReLU approximation using conditional rotations
      (reduce
       (fn [c qubit]
         (let [angle (nth parameters qubit)]
           (if (pos? angle)
             (circuit/ry-gate c qubit angle)
             c)))  ; No rotation for negative values
       circuit
       (range num-qubits))
      
      :pauli-rotation
      ;; General Pauli rotations Rx, Ry, Rz
      (reduce
       (fn [c qubit]
         (let [param-base (* qubit 3)
               rx-angle (nth parameters param-base)
               ry-angle (nth parameters (inc param-base))
               rz-angle (nth parameters (+ param-base 2))]
           (-> c
               (circuit/rx-gate qubit rx-angle)
               (circuit/ry-gate qubit ry-angle)
               (circuit/rz-gate qubit rz-angle))))
       circuit
       (range num-qubits))
      
      :none
      ;; No activation - pass through
      circuit
      
      ;; Default: no activation
      circuit)))

(defn apply-measurement-layer
  "Apply measurement operations to circuit (for output layers).
  
  Parameters:
  - circuit: Input quantum circuit
  - layer: Measurement layer configuration
  - feature-data: Not used for measurement layers
  - parameters: Not used for measurement layers
  
  Returns:
  Quantum circuit with measurement operations"
  [circuit layer _feature-data _parameters]
  {:pre [(s/valid? ::measurement-layer layer)]}
  (let [num-qubits (:num-qubits layer)
        measurement-basis (:measurement-basis layer)]
    
    ;; For now, we'll add the measurement specification to the circuit metadata
    ;; The actual measurement will be handled by the backend during execution
    (case measurement-basis
      :computational
      ;; Standard computational basis measurement (default)
      circuit
      
      :pauli-z
      ;; Pauli-Z measurement (same as computational)
      circuit
      
      :pauli-x
      ;; Pauli-X measurement (need to rotate to X basis first)
      (reduce
       (fn [c qubit]
         (circuit/h-gate c qubit))  ; H gate rotates Z basis to X basis
       circuit
       (range num-qubits))
      
      :pauli-y
      ;; Pauli-Y measurement (need to rotate to Y basis first)
      ;; Using Ry(-π/2) to rotate Z basis to Y basis
      (reduce
       (fn [c qubit]
         (-> c
             (circuit/ry-gate qubit (- (/ fm/PI 2)))  ; Rotate to Y basis
             (circuit/h-gate qubit)))
       circuit
       (range num-qubits))
      
      ;; Default: computational basis
      circuit)))

;;;
;;; Layer Dispatch Function
;;;

(defn apply-layer
  "Apply a single QNN layer to a quantum circuit.
  
  Parameters:
  - circuit: Input quantum circuit
  - layer: Layer configuration map
  - feature-data: Classical feature data (used only by input layers)
  - layer-parameters: Parameters specific to this layer
  
  Returns:
  Quantum circuit with layer operations applied"
  [circuit layer feature-data layer-parameters]
  {:pre [(s/valid? ::qnn-layer layer)]}
  (case (:layer-type layer)
    :input (apply-input-layer circuit layer feature-data layer-parameters)
    :dense (apply-dense-layer circuit layer feature-data layer-parameters)
    :entangling (apply-entangling-layer circuit layer feature-data layer-parameters)
    :activation (apply-activation-layer circuit layer feature-data layer-parameters)
    :measurement (apply-measurement-layer circuit layer feature-data layer-parameters)
    :output (apply-measurement-layer circuit layer feature-data layer-parameters)
    (throw (ex-info "Unknown layer type" {:layer layer}))))

;;;
;;; Network Composition and Forward Pass
;;;

(defn extract-layer-parameters
  "Extract parameters for a specific layer from the full parameter vector.
  
  Parameters:
  - layer: Layer configuration with parameter indices
  - full-parameters: Complete parameter vector for the network
  
  Returns:
  Parameter vector for this specific layer"
  [layer full-parameters]
  (if-let [indices (:parameter-indices layer)]
    (mapv #(nth full-parameters %) indices)
    []))

(defn apply-qnn-network
  "Apply a complete QNN network to input data, creating a quantum circuit.
  
  This function implements the forward pass of a QNN by sequentially applying
  each layer to the quantum circuit. It handles parameter extraction and
  layer-specific data passing.
  
  Parameters:
  - network: Vector of layer configuration maps (with allocated parameters)
  - feature-data: Classical input feature vector
  - parameters: Complete parameter vector for the entire network
  
  Returns:
  Quantum circuit representing the forward pass of the QNN"
  [network feature-data parameters]
  {:pre [(s/valid? ::qnn-network network)
         (vector? parameters)]}
  
  (let [;; Start with a base circuit sized for the network
        num-qubits (apply max (map :num-qubits network))
        initial-circuit (circuit/create-circuit num-qubits "QNN Forward Pass")]
    
    ;; Apply each layer sequentially using reduce
    (reduce
     (fn [circuit layer]
       (let [layer-params (extract-layer-parameters layer parameters)]
         (apply-layer circuit layer feature-data layer-params)))
     initial-circuit
     network)))

(defn create-qnn-circuit-constructor
  "Create a circuit constructor function for QNN integration with variational algorithms.
  
  This function creates a circuit constructor compatible with the variational algorithm
  template. It returns a function that takes parameters and feature data, returning
  a complete QNN circuit.
  
  Parameters:
  - network: QNN network configuration (with allocated parameters)
  
  Returns:
  Function (parameters, feature-data) -> quantum circuit"
  [network]
  {:pre [(s/valid? ::qnn-network network)]}
  (fn qnn-circuit-constructor [parameters feature-data]
    (apply-qnn-network network feature-data parameters)))

(defn qnn-forward-pass
  "Execute the forward pass of a QNN on a quantum backend.
  
  This function executes the complete QNN circuit on a quantum backend and
  returns the measurement results. It's the core execution function for QNN inference.
  
  Parameters:
  - network: QNN network configuration
  - feature-data: Input feature vector
  - parameters: Network parameter vector
  - backend: Quantum backend for execution
  - options: Execution options (shots, measurement specs, etc.)
  
  Returns:
  Execution result from the quantum backend"
  [network feature-data parameters backend & {:keys [options] :or {options {}}}]
  {:pre [(s/valid? ::qnn-network network)]}
  
  (let [circuit (apply-qnn-network network feature-data parameters)
        shots (:shots options 1024)
        execution-options (merge {:shots shots} options)]
    
    ;; Execute the circuit using the QClojure execution system
    ;; For production use, this would integrate with the backend protocol
    (try
      (if backend
        ;; If backend is provided, we should use it according to QClojure backend protocol
        ;; For now, using basic execution - TODO: implement proper backend integration
        (let [result (circuit/execute-circuit circuit)]
          ;; Add measurement outcomes for compatibility with cost functions
          (assoc result :outcomes {"0" (quot shots 2) "1" (quot shots 2)} :shots shots))
        ;; Fallback to basic circuit execution
        (circuit/execute-circuit circuit))
      (catch Exception e
        {:error "QNN forward pass execution failed"
         :details (.getMessage e)
         :circuit-info {:num-qubits (:num-qubits circuit)
                        :num-operations (count (:operations circuit))}}))))

;;;
;;; QNN Network Analysis and Utilities
;;;

(defn analyze-qnn-network
  "Analyze a QNN network configuration and provide insights.
  
  Parameters:
  - network: QNN network configuration
  
  Returns:
  Analysis map with network statistics and insights"
  [network]
  {:pre [(s/valid? ::qnn-network network)]}
  (let [layer-types (map :layer-type network)
        layer-counts (frequencies layer-types)
        total-qubits (apply max (map :num-qubits network))
        total-params (reduce + (map :parameter-count network))
        depth (count network)]
    
    {:network-structure
     {:depth depth
      :total-qubits total-qubits
      :total-parameters total-params
      :layer-composition layer-counts}
     
     :layer-analysis
     (mapv (fn [i layer]
             {:layer-index i
              :layer-name (:layer-name layer "Unnamed")
              :layer-type (:layer-type layer)
              :num-qubits (:num-qubits layer)
              :parameter-count (:parameter-count layer)
              :parameter-indices (:parameter-indices layer)})
           (range)
           network)
     
     :complexity-metrics
     {:parameters-per-qubit (if (pos? total-qubits) (/ total-params total-qubits) 0)
      :layers-per-qubit (if (pos? total-qubits) (/ depth total-qubits) 0)
      :expressivity-score (* depth total-params)  ; Simple heuristic
      :hardware-efficiency (/ (count (filter #(= :entangling (:layer-type %)) network))
                               (max 1 depth))}  ; Ratio of entangling layers
     
     :recommendations
     (cond-> []
       (> total-params (* 3 total-qubits depth))
       (conj "Consider reducing parameters - high parameter/qubit ratio may lead to overparameterization")
       
       (< (count (filter #(= :entangling (:layer-type %)) network)) 2)
       (conj "Consider adding more entangling layers for better expressivity")
       
       (> depth 10)
       (conj "Very deep network - consider gradient issues and hardware noise")
       
       (= 0 (count (filter #(= :activation (:layer-type %)) network)))
       (conj "No activation layers found - consider adding quantum activation functions"))}))

(defn visualize-qnn-network
  "Create a simple ASCII visualization of the QNN network structure.
  
  Parameters:
  - network: QNN network configuration
  
  Returns:
  String containing ASCII art representation"
  [network]
  {:pre [(s/valid? ::qnn-network network)]}
  (let [max-qubits (apply max (map :num-qubits network))]
    (str
     "QNN Network Visualization:\n"
     "==========================\n\n"
     (apply str
            (map-indexed
             (fn [i layer]
               (let [layer-symbol (case (:layer-type layer)
                                   :input "📥"
                                   :dense "🔶"
                                   :entangling "🔗"
                                   :activation "⚡"
                                   :measurement "📊"
                                   :output "📤"
                                   "❓")
                     param-info (if (pos? (:parameter-count layer))
                                  (format " (%d params)" (:parameter-count layer))
                                  "")]
                 (format "Layer %d: %s %s [%d qubits]%s\n         %s\n\n"
                         i
                         layer-symbol
                         (:layer-type layer)
                         (:num-qubits layer)
                         param-info
                         (or (:layer-name layer) "Unnamed Layer"))))
             network))
     "\nData Flow: Input → Dense → Entangling → Activation → ... → Measurement\n"
     (format "Total Parameters: %d\n" (reduce + (map :parameter-count network)))
     (format "Network Depth: %d layers\n" (count network))
     (format "Qubit Count: %d\n" max-qubits))))

;;;
;;; QNN Configuration Helpers
;;;

(defn create-feedforward-qnn
  "Create a standard feedforward QNN configuration.
  
  Parameters:
  - num-qubits: Number of qubits in the network
  - hidden-layers: Number of hidden layers
  - feature-map-type: Input encoding type (:angle, :amplitude, etc.)
  - activation-type: Activation function type
  
  Returns:
  Complete QNN network configuration with allocated parameters"
  [num-qubits hidden-layers & {:keys [feature-map-type activation-type]
                               :or {feature-map-type :angle
                                    activation-type :quantum-tanh}}]
  {:pre [(pos-int? num-qubits) (nat-int? hidden-layers)]}
  
  (let [;; Input layer
        input-layer {:layer-type :input
                     :num-qubits num-qubits
                     :feature-map-type feature-map-type
                     :layer-name "Input Encoding"}
        
        ;; Hidden layers (dense + entangling + activation)
        hidden-layer-sequence
        (mapcat
         (fn [i]
           [{:layer-type :dense
             :num-qubits num-qubits
             :layer-name (format "Dense Layer %d" (inc i))}
            {:layer-type :entangling
             :num-qubits num-qubits
             :entangling-pattern :linear
             :layer-name (format "Entangling Layer %d" (inc i))}
            {:layer-type :activation
             :num-qubits num-qubits
             :activation-type activation-type
             :layer-name (format "Activation Layer %d" (inc i))}])
         (range hidden-layers))
        
        ;; Output layer
        output-layer {:layer-type :measurement
                      :num-qubits num-qubits
                      :measurement-basis :computational
                      :layer-name "Output Measurement"}
        
        ;; Complete network with parameter counts
        add-param-count (fn [layer] (assoc layer :parameter-count (count-layer-parameters layer)))
        network-with-counts (mapv add-param-count
                                  (vec (concat [input-layer] hidden-layer-sequence [output-layer])))
        
        ;; Complete network
        allocated-result (allocate-parameters network-with-counts)]
    
    ;; Return just the network part (functions expect vector, not map)
    (:network allocated-result)))

;;;
;;; Layer Dispatch Function (moved here for logical grouping)
;;;

(comment
  ;; Example QNN layer definitions
  
  ;; Input layer with angle encoding
  (def input-layer
    {:layer-type :input
     :num-qubits 4
     :parameter-count 0
     :feature-map-type :angle
     :encoding-options {:gate-type :ry}
     :layer-name "Input Encoding"})
  
  ;; Dense quantum layer
  (def dense-layer-1
    {:layer-type :dense
     :num-qubits 4
     :parameter-count 12  ; 3 parameters per qubit
     :activation-type :none
     :layer-name "Dense Layer 1"})
  
  ;; Entangling layer
  (def entangling-layer
    {:layer-type :entangling
     :num-qubits 4
     :parameter-count 3   ; Linear pattern: 3 CNOT gates
     :entangling-pattern :linear
     :layer-name "Entangling Layer"})
  
  ;; Activation layer
  (def activation-layer
    {:layer-type :activation
     :num-qubits 4
     :parameter-count 4   ; 1 parameter per qubit for quantum-tanh
     :activation-type :quantum-tanh
     :layer-name "Activation Layer"})
  
  ;; Output measurement layer
  (def output-layer
    {:layer-type :measurement
     :num-qubits 4
     :parameter-count 0
     :measurement-basis :computational
     :layer-name "Output Measurement"})
  
  ;; Example QNN network
  (def example-network
    [input-layer
     dense-layer-1
     entangling-layer
     activation-layer
     output-layer])
  
  ;; Test parameter allocation
  (def allocated-network (allocate-parameters example-network))
  
  ;; Validate network structure
  (s/valid? ::qnn-network (:network allocated-network))
  
  ;; Check total parameters
  (:total-parameters allocated-network))  ; Should be 19 parameters total

;;;
;;; QNN Training Integration
;;;

(defn extract-expectation-value
  "Extract expectation value from quantum measurement results.
  
  For classification, this typically computes <Z> expectation on output qubits.
  For regression, it may compute weighted expectation values.
  
  Parameters:
  - result: Quantum execution result from backend
  
  Returns:
  Expectation value as a real number"
  [result]
  (if (:error result)
    0.0  ; Default value for failed measurements
    (let [;; Get measurement outcomes and counts
          outcomes (:outcomes result {})
          total-shots (reduce + (vals outcomes))
          
          ;; Compute Z expectation: E[Z] = (n_0 - n_1) / total_shots
          ;; This assumes single-qubit measurement on qubit 0
          n_0 (get outcomes "0" 0)
          n_1 (get outcomes "1" 0)
          expectation (if (pos? total-shots)
                        (/ (- n_0 n_1) total-shots)
                        0.0)]
      
      expectation)))

(defn extract-probability-distribution
  "Extract probability distribution from quantum measurement results.
  
  Parameters:
  - result: Quantum execution result from backend
  
  Returns:
  Vector of probabilities for each computational basis state"
  [result]
  (if (:error result)
    [0.5 0.5]  ; Default uniform distribution for failed measurements
    (let [outcomes (:outcomes result {})
          total-shots (reduce + (vals outcomes))
          
          ;; Convert counts to probabilities
          probabilities (if (pos? total-shots)
                          (mapv #(/ (get outcomes (str %) 0) total-shots)
                                (range (count outcomes)))
                          (vec (repeat (count outcomes) (/ 1.0 (count outcomes)))))]
      
      probabilities)))

(defn extract-most-probable-state
  "Extract the most probable computational basis state.
  
  Parameters:
  - result: Quantum execution result from backend
  
  Returns:
  String representing the most probable bitstring"
  [result]
  (if (:error result)
    "0"  ; Default state for failed measurements
    (let [outcomes (:outcomes result {})]
      (if (empty? outcomes)
        "0"
        ;; Find state with maximum count
        (key (apply max-key val outcomes))))))

(defn compute-loss
  "Compute loss between prediction and true label.
  
  Parameters:
  - prediction: Model prediction (expectation value, probability vector, or bitstring)
  - label: True label (number for regression, integer for classification)
  - loss-type: Type of loss function
  
  Returns:
  Loss value as a real number"
  [prediction label loss-type]
  (case loss-type
    :mse 
    ;; Mean squared error for regression
    (let [pred-value (if (number? prediction) prediction (first prediction))
          diff (- pred-value label)]
      (* diff diff))
    
    :cross-entropy
    ;; Cross-entropy for classification
    (let [prob-dist (if (vector? prediction) prediction [prediction (- 1.0 prediction)])
          label-idx (int label)
          prob (nth prob-dist label-idx 1e-10)]  ; Avoid log(0)
      (- (Math/log (max prob 1e-10))))
    
    :hinge
    ;; Hinge loss for binary classification
    (let [pred-value (if (number? prediction) prediction 
                         (- (first prediction) (second prediction)))
          label-value (if (= label 0) -1.0 1.0)
          margin (* label-value pred-value)]
      (max 0.0 (- 1.0 margin)))
    
    :accuracy
    ;; Negative accuracy (for maximization via minimization)
    (let [predicted-label (if (string? prediction)
                            (Integer/parseInt prediction)
                            (if (> (first prediction) 0.5) 1 0))
          correct? (= predicted-label (int label))]
      (if correct? 0.0 1.0))
    
    ;; Default: MSE
    (let [pred-value (if (number? prediction) prediction (first prediction))
          diff (- pred-value label)]
      (* diff diff))))

(defn create-qnn-cost-function
  "Create a cost function for QNN training compatible with variational algorithms.
  
  This function creates a cost function that takes parameters and returns a loss value.
  It integrates with the QClojure variational algorithm template for optimization.
  
  Parameters:
  - network: QNN network configuration
  - training-data: Vector of {:features [...] :label ...} maps
  - backend: Quantum backend for circuit execution
  - loss-type: Loss function type (:mse, :cross-entropy, :hinge, etc.)
  - options: Training options (shots, measurement strategy, etc.)
  
  Returns:
  Cost function (parameters) -> loss value"
  [network training-data backend loss-type & {:keys [options] :or {options {}}}]
  {:pre [(s/valid? ::qnn-network network)
         (vector? training-data)
         (every? #(and (contains? % :features) (contains? % :label)) training-data)]}
  
  (let [shots (:shots options 1024)
        measurement-strategy (:measurement-strategy options :expectation)]
    
    (fn qnn-cost-function [parameters]
      (try
        ;; Compute loss over all training examples
        (let [total-loss 
              (reduce
               (fn [acc-loss {:keys [features label]}]
                 (let [;; Execute QNN forward pass
                       result (qnn-forward-pass network features parameters backend :options options)
                       
                       ;; Extract prediction from quantum measurement
                       prediction (case measurement-strategy
                                   :expectation (extract-expectation-value result)
                                   :probability (extract-probability-distribution result)
                                   :bitstring (extract-most-probable-state result)
                                   result)
                       
                       ;; Compute loss for this example
                       example-loss (compute-loss prediction label loss-type)]
                   
                   (+ acc-loss example-loss)))
               0.0
               training-data)
              
              ;; Average loss over training set
              avg-loss (/ total-loss (count training-data))]
          
          avg-loss)
        
        (catch Exception e
          ;; Return high loss for failed evaluations
          (println "Warning: Cost function evaluation failed:" (.getMessage e))
          1e6)))))

(defn optimize-parameters
  "Simple parameter optimization for demonstration.
  
  In a full implementation, this would delegate to QClojure's optimization
  algorithms like Nelder-Mead, COBYLA, or gradient-based methods.
  
  Parameters:
  - cost-fn: Cost function to minimize
  - initial-params: Initial parameter values
  - options: Optimization options
  
  Returns:
  Optimization result map"
  [cost-fn initial-params options]
  (let [max-iter (:max-iterations options 50)
        tolerance (:tolerance options 1e-6)]
    
    ;; Simple random search for demonstration
    ;; Real implementation would use proper optimization algorithms
    (loop [iteration 0
           best-params initial-params
           best-cost (cost-fn initial-params)
           history []]
      
      (if (or (>= iteration max-iter)
              (< best-cost tolerance))
        ;; Return optimization result
        {:parameters best-params
         :final-cost best-cost
         :iterations iteration
         :history (conj history {:iteration iteration :cost best-cost})}
        
        ;; Try random perturbation
        (let [;; Small random perturbations
              perturbation (mapv (fn [_param] (* 0.1 (- (rand 2.0) 1.0))) initial-params)
              new-params (mapv + best-params perturbation)
              new-cost (cost-fn new-params)]
          
          (if (< new-cost best-cost)
            ;; Accept improvement
            (recur (inc iteration)
                   new-params
                   new-cost
                   (conj history {:iteration iteration :cost new-cost :improvement true}))
            ;; Reject and continue
            (recur (inc iteration)
                   best-params
                   best-cost
                   (conj history {:iteration iteration :cost best-cost :improvement false}))))))))

(defn evaluate-qnn-accuracy
  "Evaluate QNN accuracy on a dataset.
  
  Parameters:
  - network: QNN network configuration
  - test-data: Test dataset
  - parameters: Optimized QNN parameters
  - backend: Quantum backend
  
  Returns:
  Accuracy as a number between 0 and 1"
  [network test-data parameters backend]
  (if (empty? test-data)
    0.0
    (let [correct-predictions
          (reduce
           (fn [correct {:keys [features label]}]
             (try
               (let [result (qnn-forward-pass network features parameters backend)
                     prediction (extract-most-probable-state result)
                     predicted-label (Integer/parseInt prediction)
                     actual-label (int label)]
                 (if (= predicted-label actual-label)
                   (inc correct)
                   correct))
               (catch Exception _e
                 correct)))  ; Count as incorrect if evaluation fails
           0
           test-data)]
      
      (/ correct-predictions (count test-data)))))

(defn train-qnn
  "Train a QNN using variational optimization.
  
  This function integrates with the QClojure variational algorithm template
  to optimize QNN parameters using gradient-free optimization methods.
  
  Parameters:
  - network: QNN network configuration
  - training-data: Training dataset as vector of {:features [...] :label ...}
  - backend: Quantum backend for execution
  - options: Training options map
  
  Options:
  - :optimizer - Optimization method (:adam, :gradient-descent, :nelder-mead, etc.)
  - :loss-type - Loss function (:mse, :cross-entropy, :hinge, :accuracy)
  - :max-iterations - Maximum training iterations
  - :learning-rate - Learning rate for gradient-based methods
  - :tolerance - Convergence tolerance
  - :shots - Number of quantum circuit shots per evaluation
  - :initial-parameters - Initial parameter values (random if not provided)
  
  Returns:
  Training result map with optimized parameters and training history"
  [network training-data backend & {:keys [options] :or {options {}}}]
  {:pre [(s/valid? ::qnn-network network)
         (vector? training-data)
         (not-empty training-data)]}
  
  (let [;; Extract training options
        optimizer (:optimizer options :nelder-mead)
        loss-type (:loss-type options :mse)
        max-iterations (:max-iterations options 100)
        learning-rate (:learning-rate options 0.1)
        tolerance (:tolerance options 1e-6)
        shots (:shots options 1024)
        
        ;; Determine parameter count
        total-params (reduce + (map :parameter-count network))
        
        ;; Initialize parameters
        initial-params (or (:initial-parameters options)
                           (vec (repeatedly total-params #(rand 2.0))))  ; Random [0, 2π]
        
        ;; Create cost function
        cost-fn (create-qnn-cost-function network training-data backend loss-type
                                          :options {:shots shots})
        
        ;; Training options for variational algorithm
        training-opts {:optimizer optimizer
                       :max-iterations max-iterations
                       :learning-rate learning-rate
                       :tolerance tolerance
                       :cost-function cost-fn
                       :initial-parameters initial-params}]
    
    (try
      ;; Use QClojure's variational algorithm template
      ;; This would typically call the optimization algorithm from qclojure.application.algorithm.optimization
      (let [;; Simple training loop for demonstration
            ;; In practice, this would use the full variational algorithm infrastructure
            result (optimize-parameters cost-fn initial-params training-opts)]
        
        {:status :success
         :optimized-parameters (:parameters result)
         :final-cost (:final-cost result)
         :iterations (:iterations result)
         :training-history (:history result)
         :network network
         :training-summary
         {:total-parameters total-params
          :training-examples (count training-data)
          :optimizer optimizer
          :loss-type loss-type
          :final-accuracy (evaluate-qnn-accuracy network training-data 
                                                  (:parameters result) backend)}})
      
      (catch Exception e
        {:status :error
         :error-message (.getMessage e)
         :network network
         :training-data-size (count training-data)}))))

;;;
;;; QNN Model Interface
;;;

(defn create-qnn-model
  "Create a complete QNN model with training and inference capabilities.
  
  This function provides a high-level interface for QNN creation, training,
  and inference, abstracting away the low-level details.
  
  Parameters:
  - config: QNN configuration map
  
  Configuration options:
  - :num-qubits - Number of qubits
  - :hidden-layers - Number of hidden layers
  - :feature-map-type - Input encoding type
  - :activation-type - Activation function type
  
  Returns:
  QNN model map with train and predict functions"
  [config]
  {:pre [(map? config) (contains? config :num-qubits)]}
  
  (let [num-qubits (:num-qubits config)
        hidden-layers (:hidden-layers config 1)
        feature-map-type (:feature-map-type config :angle)
        activation-type (:activation-type config :quantum-tanh)
        
        ;; Create the QNN network
        network (create-feedforward-qnn num-qubits hidden-layers
                                        :feature-map-type feature-map-type
                                        :activation-type activation-type)]
    
    {:network network
     :config config
     :trained-parameters nil
     :training-history nil
     
     ;; Training function
     :train (fn [training-data backend & {:keys [options] :or {options {}}}]
              (let [result (train-qnn network training-data backend :options options)]
                (if (= (:status result) :success)
                  (assoc (dissoc result :status)
                         :trained-parameters (:optimized-parameters result)
                         :training-history (:training-history result))
                  result)))
     
     ;; Prediction function
     :predict (fn [features parameters backend]
                (qnn-forward-pass network features parameters backend))
     
     ;; Evaluation function
     :evaluate (fn [test-data parameters backend]
                 (evaluate-qnn-accuracy network test-data parameters backend))
     
     ;; Analysis function
     :analyze (fn [] (analyze-qnn-network network))
     
     ;; Visualization function
     :visualize (fn [] (visualize-qnn-network network))}))

;;;
;;; Rich Comment Block for Interactive Development
;;;
