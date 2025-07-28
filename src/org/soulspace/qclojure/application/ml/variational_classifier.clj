(ns org.soulspace.qclojure.application.ml.variational-classifier
  "Variational Quantum Classifier using qclojure core + noj for ML pipeline"
 (:require [org.soulspace.qclojure.application.algorithm.vqe :as vqe]
           [org.soulspace.qclojure.application.ml.encoding :as encoding]
           [org.soulspace.qclojure.application.ml.training :as training]
           [tablecloth.api :as tc]
           [scicloj.metamorph.ml :as ml]
           [fastmath.optimization :as opt]))

(defn create-classifier-ansatz
  "Create variational circuit for classification"
  [num-features num-classes layers]
  (let [encoding-qubits (int (m/ceil (m/log2 num-features)))
        total-qubits (+ encoding-qubits (int (m/ceil (m/log2 num-classes))))]
    (vqe/hardware-efficient-ansatz total-qubits layers :cnot)))

(defn variational-quantum-classifier
  "Train a VQC using qclojure + noj integration"
  [dataset label-column backend options]
  (let [;; Use tablecloth for data preparation
        feature-columns (tc/column-names (tc/drop-columns dataset [label-column]))
        X (tc/select-columns dataset feature-columns)
        y (tc/get-column dataset label-column)

        ;; Split using metamorph.ml
        split-data (ml/train-test-split {:train-ratio 0.8} {:X X :y y})

        ;; Create quantum classifier
        num-features (tc/column-count X)
        num-classes (count (distinct y))
        ansatz-fn (create-classifier-ansatz num-features num-classes 2)

        ;; Training loop (simplified)
        cost-fn (fn [params]
                  ;; Encode training data → quantum states → measure → calculate cost
                  (training/classification-cost params ansatz-fn
                                                (:X-train split-data)
                                                (:y-train split-data)
                                                backend))

        ;; Optimize using fastmath
        initial-params (vec (repeatedly (* num-features 6) #(* 0.1 (- (rand) 0.5))))
        result (opt/minimize :nelder-mead cost-fn {:initial initial-params})]

    {:model {:ansatz ansatz-fn
             :optimal-params (:arg result)
             :num-features num-features
             :num-classes num-classes}
     :training {:cost (:value result)
                :iterations (:iterations result)}
     :data-split split-data}))