(ns org.soulspace.qclojure.application.ml.vqe-extensions
  "Extensions to VQE for Quantum Machine Learning
  
  While VQE focuses on minimizing Hermitian Hamiltonians for ground state
  energy calculations, QML requires additional cost function types and
  measurement strategies. This namespace extends the VQE framework to
  support general variational quantum algorithms for machine learning.
  
  Key Extensions:
  - Non-Hermitian cost functions (classification losses)
  - Measurement-based objectives (probability distributions)
  - Multi-objective optimization (adversarial training)
  - Parameter shift rule for gradient-based optimization
  - Fidelity and overlap measurements for kernel methods"
  (:require [clojure.spec.alpha :as s]
            [fastmath.core :as m]
            [org.soulspace.qclojure.application.algorithm.vqe :as vqe]
            [org.soulspace.qclojure.application.backend :as qb]
            [org.soulspace.qclojure.domain.circuit :as qc]
            [org.soulspace.qclojure.domain.state :as qs]))

;; Specs for QML extensions
(s/def ::loss-function #{:cross-entropy :mse :hinge :wasserstein :kl-divergence})
(s/def ::measurement-strategy #{:computational-basis :pauli-expectation :fidelity})
(s/def ::gradient-method #{:parameter-shift :finite-difference :simultaneous-perturbation})

(s/def ::qml-config
  (s/keys :req-un [::loss-function ::measurement-strategy]
          :opt-un [::gradient-method ::regularization ::batch-size]))

;;
;; Measurement-Based Cost Functions
;;
(defn measurement-probabilities
  "Extract measurement probabilities from quantum circuit execution.
  
  Parameters:
  - circuit: Quantum circuit to execute
  - backend: Quantum backend
  - shots: Number of measurements (default: 1024)
  
  Returns:
  Vector of probabilities for computational basis states"
  ([circuit backend]
   (measurement-probabilities circuit backend 1024))
  ([circuit backend shots]
   (let [num-qubits (:num-qubits circuit)
         num-states (int (m/pow 2 num-qubits))
         ;; Execute circuit and get measurement results
         execution-result (qb/execute-circuit backend circuit {:shots shots})
         measurement-counts (or (:measurement-counts execution-result)
                                ;; Fallback to simulation if no measurements
                                (let [final-state (qc/execute-circuit circuit (qs/zero-state num-qubits))
                                      probs (qs/measurement-probabilities final-state)]
                                  ;; Convert probabilities to synthetic counts
                                  (into {} (map-indexed (fn [i prob]
                                                          [(format (str "%0" num-qubits "d")
                                                                   (Integer/toString i 2))
                                                           (int (* prob shots))])
                                                        probs))))]

     ;; Convert counts to probabilities
     (mapv (fn [i]
             (let [bit-string (format (str "%0" num-qubits "d")
                                      (Integer/toString i 2))
                   count (get measurement-counts bit-string 0)]
               (/ count shots)))
           (range num-states)))))

(defn classification-loss
  "Calculate classification loss from measurement probabilities.
  
  Parameters:
  - predicted-probs: Vector of predicted class probabilities
  - true-labels: Vector of true class labels (one-hot or indices)
  - loss-type: Type of loss function
  
  Returns:
  Loss value (real number)"
  [predicted-probs true-labels loss-type]
  (case loss-type
    :cross-entropy
    (let [true-probs (if (vector? (first true-labels))
                       true-labels  ; already one-hot
                       (mapv (fn [label]
                               (assoc (vec (repeat (count predicted-probs) 0))
                                      label 1))
                             true-labels))]
      (->> (map vector predicted-probs true-probs)
           (map (fn [[pred-vec true-vec]]
                  (->> (map vector pred-vec true-vec)
                       (map (fn [[p t]] (if (> t 0)
                                          (* t (m/log (max p 1e-10)))
                                          0)))
                       (reduce +))))
           (reduce +)
           (-)))  ; negative log-likelihood

    :mse
    (let [true-probs (if (vector? (first true-labels))
                       true-labels
                       (mapv (fn [label]
                               (assoc (vec (repeat (count predicted-probs) 0))
                                      label 1))
                             true-labels))]
      (->> (map vector predicted-probs true-probs)
           (map (fn [[pred-vec true-vec]]
                  (->> (map vector pred-vec true-vec)
                       (map (fn [[p t]] (* (- p t) (- p t))))
                       (reduce +))))
           (reduce +)))

    :hinge
    (let [margins (map (fn [pred-vec label]
                         (let [correct-class-prob (nth pred-vec label)
                               other-class-probs (map-indexed
                                                  (fn [i p] (if (= i label) 0 p))
                                                  pred-vec)
                               max-other-prob (apply max other-class-probs)]
                           (- correct-class-prob max-other-prob)))
                       predicted-probs true-labels)]
      (->> margins
           (map (fn [margin] (max 0 (- 1 margin))))
           (reduce +)))))

(defn fidelity-based-cost
  "Create fidelity-based cost function for quantum kernel methods.
  
  Parameters:
  - target-state: Target quantum state to match
  - ansatz-fn: Parameterized circuit function
  - backend: Quantum backend
  
  Returns:
  Cost function that returns 1 - fidelity (to minimize)"
  [target-state ansatz-fn backend]
  (fn cost-fn [parameters]
    (let [circuit (ansatz-fn parameters)
          prepared-state (if backend
                           (let [result (qb/execute-circuit backend circuit {})]
                             (:final-state result))
                           (qc/execute-circuit circuit (qs/zero-state (:num-qubits circuit))))
          fidelity (qs/fidelity prepared-state target-state)]
      (- 1.0 fidelity))))

;;
;; Parameter Shift Rule for Gradients
;;
(defn parameter-shift-gradient
  "Calculate gradient using parameter shift rule.
  
  For gates with the form exp(-iθG/2) where G² = I, the gradient is:
  ∇θ⟨H⟩ = (⟨H⟩(θ+π/2) - ⟨H⟩(θ-π/2))/2
  
  Parameters:
  - cost-fn: Cost function to differentiate
  - parameters: Current parameter values
  - shift-amount: Shift amount (default: π/2)
  
  Returns:
  Vector of partial derivatives"
  ([cost-fn parameters]
   (parameter-shift-gradient cost-fn parameters (/ m/PI 2)))
  ([cost-fn parameters shift-amount]
   (mapv (fn [i]
           (let [params-plus (assoc parameters i (+ (nth parameters i) shift-amount))
                 params-minus (assoc parameters i (- (nth parameters i) shift-amount))
                 cost-plus (cost-fn params-plus)
                 cost-minus (cost-fn params-minus)]
             (/ (- cost-plus cost-minus) 2)))
         (range (count parameters)))))

(defn gradient-descent-step
  "Perform one gradient descent step with parameter shift rule.
  
  Parameters:
  - cost-fn: Cost function to minimize
  - parameters: Current parameters
  - learning-rate: Step size for gradient descent
  - gradient-method: Method for computing gradients
  
  Returns:
  Updated parameters"
  [cost-fn parameters learning-rate gradient-method]
  (let [gradients (case gradient-method
                    :parameter-shift (parameter-shift-gradient cost-fn parameters)
                    :finite-difference (vqe/finite-difference-gradient cost-fn parameters 1e-5)
                    ;:simultaneous-perturbation (spsa-gradient cost-fn parameters)
                    )]
    (mapv (fn [param grad]
            (- param (* learning-rate grad)))
          parameters gradients)))

;;
;; Extended VQE for General Variational Algorithms
;;
(defn create-qml-objective
  "Create objective function for quantum machine learning.
  
  This generalizes the VQE objective to support:
  - Classification and regression losses
  - Measurement-based cost functions  
  - Regularization terms
  - Batch processing
  
  Parameters:
  - ansatz-fn: Parameterized quantum circuit function
  - training-data: Training dataset
  - backend: Quantum backend
  - config: QML configuration
  
  Returns:
  Objective function for optimization"
  [ansatz-fn training-data backend config]
  (let [loss-fn (:loss-function config)
        measurement-strategy (:measurement-strategy config :computational-basis)
        batch-size (:batch-size config (count (:features training-data)))
        regularization (:regularization config 0.0)]

    (fn objective [parameters]
      (try
        (let [features (:features training-data)
              labels (:labels training-data)
              ;; Process in batches if specified
              batch-indices (take batch-size (shuffle (range (count features))))
              batch-features (mapv #(nth features %) batch-indices)
              batch-labels (mapv #(nth labels %) batch-indices)

              ;; Calculate loss for batch
              batch-loss
              (case measurement-strategy
                :computational-basis
                (->> (map vector batch-features batch-labels)
                     (map (fn [[feature-vec label]]
                            ;; Create circuit with feature encoding and ansatz
                            (let [circuit (ansatz-fn parameters feature-vec)
                                  probs (measurement-probabilities circuit backend)
                                  ;; For classification, use relevant output qubits
                                  class-probs (take 2 probs)] ; assume binary classification
                              (classification-loss [class-probs] [label] loss-fn))))
                     (reduce +)
                     (/ (count batch-features)))

                :pauli-expectation
                ;; Use Pauli measurements for cost (like original VQE)
                (let [hamiltonian (:hamiltonian config)]
                  (->> (map vector batch-features batch-labels)
                       (map (fn [[feature-vec _label]]
                              (let [circuit (ansatz-fn parameters feature-vec)
                                    final-state (qc/execute-circuit circuit (qs/zero-state (:num-qubits circuit)))]
                                (vqe/hamiltonian-expectation hamiltonian final-state))))
                       (reduce +)
                       (/ (count batch-features)))))

              ;; Add regularization
              regularization-term (* regularization
                                     (reduce + (map #(* % %) parameters)))]

          (+ batch-loss regularization-term))

        (catch Exception e
          (println "QML objective evaluation failed:" (.getMessage e))
          1000.0)))))

(defn variational-quantum-algorithm
  "General variational quantum algorithm for machine learning.
  
  This extends VQE to support arbitrary cost functions for QML applications.
  
  Parameters:
  - ansatz-fn: Parameterized quantum circuit function
  - training-data: Training dataset  
  - backend: Quantum backend
  - config: Algorithm configuration
  
  Returns:
  Training results and learned parameters"
  [ansatz-fn training-data backend config]
  (let [initial-params (:initial-parameters config
                                            (vec (repeatedly (:num-parameters config 6)
                                                             #(* 0.1 (- (rand) 0.5)))))
        max-iterations (:max-iterations config 100)
        learning-rate (:learning-rate config 0.01)
        gradient-method (:gradient-method config :parameter-shift)
        tolerance (:tolerance config 1e-6)

        ;; Create objective function
        objective-fn (create-qml-objective ansatz-fn training-data backend config)

        ;; Training loop with gradient-based optimization
        start-time (System/currentTimeMillis)]

    (loop [params initial-params
           iteration 0
           history []
           prev-cost (objective-fn initial-params)]

      (if (or (>= iteration max-iterations)
              (< (abs (- (objective-fn params) prev-cost)) tolerance))

        ;; Return results
        {:success true
         :optimal-parameters params
         :optimal-cost (objective-fn params)
         :iterations iteration
         :training-history history
         :execution-time-ms (- (System/currentTimeMillis) start-time)}

        ;; Continue training
        (let [current-cost (objective-fn params)
              new-params (gradient-descent-step objective-fn params learning-rate gradient-method)
              new-history (conj history {:iteration iteration
                                         :cost current-cost
                                         :parameters params})]

          (when (zero? (mod iteration 10))
            (println (format "Iteration %d: Cost = %.6f" iteration current-cost)))

          (recur new-params
                 (inc iteration)
                 new-history
                 current-cost))))))

(comment
  ;; Rich comment block for REPL experimentation

  ;; Test classification cost function
  (def test-probs [[0.8 0.2] [0.3 0.7] [0.9 0.1]])
  (def test-labels [0 1 0])

  (classification-loss test-probs test-labels :cross-entropy)
  (classification-loss test-probs test-labels :mse)

  ;; Test parameter shift gradient
  (defn simple-cost-fn [params]
    (+ (* (first params) (first params))
       (* (second params) (second params))))

  (def test-params [0.5 -0.3])
  (parameter-shift-gradient simple-cost-fn test-params)

  ;; Expected gradient: [2*0.5, 2*(-0.3)] = [1.0, -0.6]
  )