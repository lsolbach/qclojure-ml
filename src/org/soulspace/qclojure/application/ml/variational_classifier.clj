(ns org.soulspace.qclojure.application.ml.variational-classifier
  "Variational Quantum Classifier using QClojure core + noj for ML pipeline integration"
  (:require [clojure.spec.alpha :as s]
            [tablecloth.api :as tc]
            [fastmath.core :as m]
            [org.soulspace.qclojure.domain.state :as qs]
            [org.soulspace.qclojure.domain.circuit :as qc]
            [org.soulspace.qclojure.domain.ansatz :as ansatz]
            [org.soulspace.qclojure.application.ml.encoding :as encoding]
            [org.soulspace.qclojure.application.ml.training :as training]))

;; Specs for VQC
(s/def ::num-features pos-int?)
(s/def ::num-classes pos-int?)  
(s/def ::layers pos-int?)
(s/def ::vqc-config (s/keys :req-un [::num-features ::num-classes ::layers]))

(defrecord VariationalQuantumClassifier [num-features num-classes layers parameters ansatz-fn]
  ;; Implementation of QML model protocol would go here
  )

(defn create-classifier-ansatz
  "Create variational circuit ansatz for classification.
   
   This creates a quantum circuit suitable for classification tasks by
   combining feature encoding with a parameterized ansatz.
   
   Parameters:
   - num-features: Number of features in the dataset
   - num-classes: Number of classes in the target variable  
   - layers: Number of layers in the ansatz
   
   Returns:
   - Function that creates quantum circuit for classification
   
   Example:
   (create-classifier-ansatz 4 2 2)"
  [num-features num-classes layers]
  (let [;; Calculate qubits needed for feature encoding
        encoding-qubits (max 2 (int (m/ceil (m/log2 (max 2 num-features)))))
        ;; Add qubits for classification output  
        class-qubits (max 1 (int (m/ceil (m/log2 num-classes))))
        total-qubits (+ encoding-qubits class-qubits)
        
        ;; Create base ansatz using VQE hardware-efficient ansatz
        base-ansatz (ansatz/hardware-efficient-ansatz total-qubits layers)]
    
    (fn classifier-circuit [parameters features]
      (let [;; Create base circuit
            base-circuit (qc/create-circuit total-qubits "VQC Circuit")
            
            ;; Apply feature encoding (angle encoding)
            feature-encoder (encoding/angle-encoding features encoding-qubits :ry)
            encoded-circuit (feature-encoder base-circuit)
            
            ;; Apply variational ansatz
            ansatz-circuit (base-ansatz parameters)]
        
        ;; Combine encoding and ansatz by merging operations
        (update ansatz-circuit :operations 
                #(vec (concat (:operations encoded-circuit) %)))))))

(defn train-vqc
  "Train a Variational Quantum Classifier.
   
   This function integrates with the Clojure data science ecosystem (noj, tablecloth)
   to provide a complete QML workflow from data preparation to model training.
   
   Parameters:
   - dataset: Tablecloth dataset with features and labels
   - label-column: Name of the target variable column
   - config: Configuration map for the VQC
   
   Returns:
   - Map containing trained model, training metrics, and evaluation results
   
   Example:
   (train-vqc dataset :label {:num-features 4 :num-classes 2 :layers 2})"
  [dataset label-column config]
  {:pre [(s/valid? ::vqc-config config)]}
  
  (let [;; Data preparation using tablecloth
        feature-columns (remove #{label-column} (tc/column-names dataset))
        X (tc/select-columns dataset feature-columns)  
        y (get dataset label-column)  ; Use get instead of tc/get-column
        
        ;; Convert to vectors for training
        features (mapv (fn [row-map] (mapv #(get row-map %) feature-columns))
                       (tc/rows X :as-maps true))
        labels (vec y)
        
        ;; Simple train-test split (80-20)
        n-samples (count features)
        n-train (int (* 0.8 n-samples))
        indices (shuffle (range n-samples))
        train-indices (take n-train indices)
        test-indices (drop n-train indices)
        
        split-data {:X-train (mapv #(nth features %) train-indices)
                    :y-train (mapv #(nth labels %) train-indices)
                    :X-test (mapv #(nth features %) test-indices)
                    :y-test (mapv #(nth labels %) test-indices)}
        
        ;; Extract configuration
        num-features (:num-features config (tc/column-count X))  
        num-classes (:num-classes config (count (distinct y)))
        layers (:layers config 2)
        
        ;; Create classifier ansatz
        ansatz-fn (create-classifier-ansatz num-features num-classes layers)
        
        ;; Prepare training data
        training-data {:features (:X-train split-data)
                       :labels (:y-train split-data)}
        
        ;; Training configuration
        training-config {:max-iterations (:max-iterations config 50)
                         :learning-rate (:learning-rate config 0.01)
                         :num-parameters (* num-features layers 3)
                         :backend (:backend config :simulator)}
        
        ;; Train the model
        training-result (training/train-qml-model ansatz-fn training-data training-config)
        
        ;; Create final model
        trained-model (->VariationalQuantumClassifier 
                       num-features 
                       num-classes 
                       layers
                       (:optimal-parameters training-result)
                       ansatz-fn)]
    
    {:model trained-model
     :training-result training-result  
     :data-split split-data
     :config config
     :feature-columns feature-columns}))

(defn predict-vqc
  "Make predictions using a trained VQC.
   
   Parameters:
   - model: Trained VariationalQuantumClassifier
   - features: Feature vectors for prediction
   
   Returns:
   - Vector of predicted class probabilities"
  [model features]
  (let [ansatz-fn (:ansatz-fn model)
        parameters (:parameters model)]
    
    (mapv (fn [feature-vec]
            (let [;; Create circuit for this sample
                  circuit (ansatz-fn parameters feature-vec)
                  
                  ;; Execute circuit
                  final-state (qc/execute-circuit circuit (qs/zero-state (:num-qubits circuit)))
                  
                  ;; Get measurement probabilities  
                  probs (qs/measurement-probabilities final-state)
                  
                  ;; For binary classification, return [P(class=0), P(class=1)]
                  class-0-prob (first probs)
                  class-1-prob (- 1.0 class-0-prob)]
              
              [class-0-prob class-1-prob]))
          features)))

(defn evaluate-vqc
  "Evaluate VQC performance on test data.
   
   Parameters:
   - model: Trained VariationalQuantumClassifier
   - test-features: Test feature vectors
   - test-labels: True test labels
   
   Returns:
   - Map with evaluation metrics"
  [model test-features test-labels]
  (let [predictions (predict-vqc model test-features)
        predicted-classes (mapv (fn [probs] (if (> (first probs) (second probs)) 0 1)) predictions)
        correct-predictions (map = predicted-classes test-labels)
        accuracy (/ (count (filter identity correct-predictions)) (count test-labels))
        
        ;; Calculate confusion matrix for binary classification
        true-positives (count (filter (fn [[pred true-label]] (and (= pred 1) (= true-label 1)))
                                      (map vector predicted-classes test-labels)))
        false-positives (count (filter (fn [[pred true-label]] (and (= pred 1) (= true-label 0)))
                                       (map vector predicted-classes test-labels)))
        false-negatives (count (filter (fn [[pred true-label]] (and (= pred 0) (= true-label 1)))
                                       (map vector predicted-classes test-labels)))
        true-negatives (count (filter (fn [[pred true-label]] (and (= pred 0) (= true-label 0)))
                                      (map vector predicted-classes test-labels)))]
    
    {:accuracy accuracy
     :predictions predictions
     :predicted-classes predicted-classes
     :confusion-matrix {:true-positives true-positives
                        :false-positives false-positives  
                        :false-negatives false-negatives
                        :true-negatives true-negatives}
     :precision (if (> (+ true-positives false-positives) 0)
                  (/ true-positives (+ true-positives false-positives))
                  0.0)
     :recall (if (> (+ true-positives false-negatives) 0)
               (/ true-positives (+ true-positives false-negatives))
               0.0)}))

;; Rich comment block for testing
(comment
  ;; Create synthetic dataset for testing
  
  ;; Create test dataset
  (def test-dataset
    (tc/dataset {:feature1 [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8]
                 :feature2 [0.2 0.1 0.4 0.3 0.6 0.5 0.8 0.7]  
                 :label [0 0 0 0 1 1 1 1]}))
  
  ;; Train VQC
  (def vqc-config {:num-features 2
                   :num-classes 2
                   :layers 1
                   :max-iterations 10})
  
  (def trained-vqc (train-vqc test-dataset :label vqc-config))
  
  ;; Evaluate model
  (def test-features [[0.15 0.25] [0.75 0.85]])
  (def test-labels [0 1])
  
  (def evaluation (evaluate-vqc (:model trained-vqc) test-features test-labels))
  (println "Evaluation results:" evaluation)
  )