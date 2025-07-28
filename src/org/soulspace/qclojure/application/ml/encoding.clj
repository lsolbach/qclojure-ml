(ns org.soulspace.qclojure.application.ml.encoding
 "Quantum data encoding strategies using tablecloth for data manipulation"
 (:require [tablecloth.api :as tc]
           [tech.v3.datatype :as dtype]
           [tech.v3.tensor :as tensor]
           [qclojure.domain.state :as qs]
           [qclojure.domain.circuit :as qc]
           [fastmath.core :as m]))

(defn normalize-features
  "Normalize dataset features for quantum encoding"
  [dataset feature-columns]
  (tc/map-columns dataset
                  (zipmap feature-columns
                          (repeat (fn [col]
                                    (let [col-min (dtype/reduce-min col)
                                          col-max (dtype/reduce-max col)]
                                      (dtype/emap #(/ (- % col-min) (- col-max col-min)) col)))))))

(defn amplitude-encoding
  "Encode classical data vector into quantum amplitudes using tech.ml.dataset"
  [data-row num-qubits]
  {:pre [(= (count data-row) (m/pow 2 num-qubits))]}
  (let [normalized-amplitudes (-> data-row
                                  dtype/->double-array
                                  tensor/->tensor
                                  (tensor/normalize :l2))
        amplitudes-map (zipmap (range (count normalized-amplitudes))
                               (dtype/->reader normalized-amplitudes))]
    (qs/from-amplitudes amplitudes-map num-qubits)))
