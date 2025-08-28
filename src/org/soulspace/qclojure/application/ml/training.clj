(ns org.soulspace.qclojure.application.ml.training
  "Training algorithms and cost functions for quantum machine learning"
  (:require [org.soulspace.qclojure.application.algorithm.vqe :as vqe]
            [org.soulspace.qclojure.application.backend :as qb]
            [org.soulspace.qclojure.application.ml.encoding :as encoding]
            [org.soulspace.qclojure.domain.circuit :as qc]
            [org.soulspace.qclojure.domain.circuit-composition :as cc]
            [org.soulspace.qclojure.domain.state :as qs]
            [org.soulspace.qclojure.adapter.backend.simulator :as sim]
            [fastmath.optimization :as opt]
            [fastmath.core :as m]
            [tablecloth.api :as tc]
            [scicloj.metamorph.ml :as ml]
            [clojure.spec.alpha :as s]))

;; QML-specific cost functions
(defn classification-cost
  "Calculate classification cost for quantum machine learning.
  
  Parameters:
  - parameters: Circuit parameters
  - ansatz-fn: Parameterized quantum circuit function
  - features: Training features (X data)
  - labels: Training labels (y data)
  - backend: Quantum backend
  
  Returns:
  Cost value (real number)"
  [parameters ansatz-fn features labels backend]
  (try
    (let [num-samples (count features)
          total-cost (reduce
                      (fn [acc-cost idx]
                        (let [feature-vec (nth features idx)
                              true-label (nth labels idx)

                              ;; Create circuit with feature encoding + ansatz
                              base-circuit (qc/create-circuit
                                            (max 2 (int (m/ceil (m/log2 (max 2 (count feature-vec)))))))

                              ;; Apply angle encoding for features
                              encoder-result (encoding/angle-encoding feature-vec (:num-qubits base-circuit) :ry)
                              encoded-circuit (if (:success encoder-result)
                                                ((:result encoder-result) base-circuit)
                                                (throw (ex-info "Feature encoding failed" encoder-result)))

                              ;; Apply variational ansatz using the provided ansatz-fn
                              ansatz-circuit (ansatz-fn parameters)
                              
                              ;; Combine encoding and ansatz using circuit composition
                              final-circuit (cc/compose-circuits encoded-circuit ansatz-circuit)

                              ;; Execute circuit using backend
                              probs (let [execution-result (qb/execute-circuit backend final-circuit {:shots 1024})
                                          measurement-counts (:measurement-counts execution-result)
                                          num-qubits (:num-qubits final-circuit)
                                          num-states (int (m/pow 2 num-qubits))]
                                      ;; Convert counts to probabilities using proper bit string formatting
                                      (mapv (fn [i]
                                              (let [bit-string (qs/basis-string i num-qubits)
                                                    count (get measurement-counts bit-string 0)]
                                                (/ count 1024.0)))
                                            (range num-states)))

                              ;; For binary classification, use first probability as class 0 probability
                              class-0-prob (first probs)
                              class-1-prob (- 1.0 class-0-prob)

                              ;; Calculate cross-entropy loss
                              predicted-prob (if (= true-label 0) class-0-prob class-1-prob)
                              sample-cost (- (m/log (max predicted-prob 1e-10)))]

                          (+ acc-cost sample-cost)))
                      0.0
                      (range num-samples))]
      (/ total-cost num-samples))
    
    (catch Exception e
      (println "Error in classification-cost:" (.getMessage e))
      1000.0))) ; Return high cost on error

(defn parameter-shift-gradient
  "Calculate gradient using parameter shift rule for quantum circuits.
  
  Parameters:
  - cost-fn: Cost function to differentiate
  - parameters: Current parameter values
  - shift-amount: Shift amount (default: π/2)
  
  Returns:
  Vector of partial derivatives"
  ([cost-fn parameters]
   (parameter-shift-gradient cost-fn parameters (/ m/PI 2)))
  ([cost-fn parameters shift-amount]
   (try
     {:success true
      :result (mapv (fn [i]
                      (let [params-plus (assoc parameters i (+ (nth parameters i) shift-amount))
                            params-minus (assoc parameters i (- (nth parameters i) shift-amount))
                            cost-plus (cost-fn params-plus)
                            cost-minus (cost-fn params-minus)]
                        (/ (- cost-plus cost-minus) 2)))
                    (range (count parameters)))}
     (catch Exception e
       {:success false
        :error (str "Gradient computation failed: " (.getMessage e))
        :exception e}))))

(defn train-qml-model
  "Train a quantum machine learning model using gradient-based optimization.
  
  Parameters:
  - ansatz-fn: Parameterized quantum circuit function  
  - training-data: Map with :features and :labels
  - config: Training configuration
  
  Returns:
  Trained model parameters and metrics"
  [ansatz-fn training-data config]
  (let [features (:features training-data)
        labels (:labels training-data)
        backend (:backend config :simulator)
        max-iterations (:max-iterations config 50)
        learning-rate (:learning-rate config 0.01)
        tolerance (:tolerance config 1e-6)

        ;; Initialize parameters
        num-params (:num-parameters config
                                    (* (count (first features)) 3)) ; 3 rotations per feature
        initial-params (vec (repeatedly num-params #(* 0.1 (- (rand) 0.5))))

        ;; Create cost function
        cost-fn (fn [params]
                  (classification-cost params ansatz-fn features labels backend))

        ;; Training loop
        start-time (System/currentTimeMillis)]

    (loop [params initial-params
           iteration 0
           best-cost (cost-fn initial-params)
           training-history []]

      (if (or (>= iteration max-iterations)
              (< best-cost tolerance))

        ;; Return results
        {:success true
         :optimal-parameters params
         :optimal-cost best-cost
         :iterations iteration
         :training-history training-history
         :execution-time-ms (- (System/currentTimeMillis) start-time)}

        ;; Continue training
        (let [current-cost (cost-fn params)
              gradients (parameter-shift-gradient cost-fn params)
              new-params (mapv (fn [param grad]
                                 (- param (* learning-rate grad)))
                               params gradients)
              new-cost (cost-fn new-params)
              history-entry {:iteration iteration
                             :cost current-cost
                             :parameters params}]

          (when (zero? (mod iteration 5))
            (println (format "Iteration %d: Cost = %.6f" iteration current-cost)))

          (recur new-params
                 (inc iteration)
                 (min best-cost new-cost)
                 (conj training-history history-entry)))))))

;; Rich comment block for testing
(comment
  ;; Test data preparation
  (def test-features [[0.1 0.2] [0.3 0.4] [0.5 0.6] [0.7 0.8]])
  (def test-labels [0 0 1 1])
  (def test-training-data {:features test-features :labels test-labels})

  ;; Test cost function
  (def test-ansatz (vqe/hardware-efficient-ansatz 2 1))
  (def test-params (vec (repeatedly 6 #(* 0.1 (rand)))))

  (classification-cost test-params test-ansatz test-features test-labels :simulator)

  ;; Test training
  (def training-config {:max-iterations 10
                        :learning-rate 0.1
                        :num-parameters 6})

  (train-qml-model test-ansatz test-training-data training-config))
  
