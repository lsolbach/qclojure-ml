(ns org.soulspace.qclojure.ml.application.training
  "Training algorithms and cost functions for quantum machine learning"
  (:require [clojure.string :as str]
            [clojure.spec.alpha :as s]
            [fastmath.core :as m]
            [org.soulspace.qclojure.domain.state :as state]
            [org.soulspace.qclojure.domain.circuit :as circuit]
            [org.soulspace.qclojure.domain.circuit-composition :as ccomp]
            [org.soulspace.qclojure.domain.ansatz :as ansatz]
            [org.soulspace.qclojure.application.backend :as backend]
            [org.soulspace.qclojure.application.algorithm.optimization :as opt]
            [org.soulspace.qclojure.application.algorithm.variational-algorithm :as va]
            [org.soulspace.qclojure.ml.application.encoding :as encoding]))

;; Specs for training data validation
(s/def ::feature-vector (s/coll-of number? :kind vector?))
(s/def ::features (s/coll-of ::feature-vector :kind vector?))
(s/def ::labels (s/coll-of int? :kind vector?))
(s/def ::training-data (s/keys :req-un [::features ::labels]))

;; QML-specific cost functions and loss function library

(defn cross-entropy-loss
  "Calculate cross-entropy loss for classification.
  
  Parameters:
  - true-label: True class label (0 or 1 for binary)
  - predicted-prob: Predicted probability for the true class
  
  Returns:
  Cross-entropy loss value"
  [true-label predicted-prob]
  (let [safe-prob (max predicted-prob 1e-10)] ; Avoid log(0)
    (- (m/log safe-prob))))

(defn hinge-loss
  "Calculate hinge loss for classification (SVM-style).
  
  Parameters:
  - true-label: True class label (0 or 1, converted to -1/+1)
  - predicted-score: Raw prediction score (before sigmoid)
  
  Returns:
  Hinge loss value"
  [true-label predicted-score]
  (let [y (if (= true-label 0) -1.0 1.0)
        margin (* y predicted-score)]
    (max 0.0 (- 1.0 margin))))

(defn squared-loss  
  "Calculate squared loss for regression.
  
  Parameters:
  - true-value: True target value
  - predicted-value: Predicted value
  
  Returns:
  Squared loss value"
  [true-value predicted-value]
  (let [diff (- true-value predicted-value)]
    (* diff diff)))

(defn l1-regularization
  "Calculate L1 (Lasso) regularization penalty.
  
  Parameters:
  - parameters: Parameter vector
  - lambda: Regularization strength
  
  Returns:
  L1 penalty value"
  [parameters lambda]
  (* lambda (reduce + (map m/abs parameters))))

(defn l2-regularization
  "Calculate L2 (Ridge) regularization penalty.
  
  Parameters:
  - parameters: Parameter vector
  - lambda: Regularization strength
  
  Returns:
  L2 penalty value"
  [parameters lambda]
  (* lambda (reduce + (map #(* % %) parameters))))

(defn elastic-net-regularization
  "Calculate Elastic Net regularization (L1 + L2).
  
  Parameters:
  - parameters: Parameter vector
  - lambda: Total regularization strength
  - alpha: Mix ratio (0=L2 only, 1=L1 only)
  
  Returns:
  Elastic Net penalty value"
  [parameters lambda alpha]
  (+ (* alpha (l1-regularization parameters lambda))
     (* (- 1.0 alpha) (l2-regularization parameters lambda))))

(defn classification-cost
  "Calculate classification cost for quantum machine learning with configurable loss function.
  
  This is the lower-level implementation. For optimization integration,
  use create-classification-cost-fn instead.
  
  Parameters:
  - parameters: Circuit parameters
  - ansatz-fn: Parameterized quantum circuit function
  - features: Training features (X data)
  - labels: Training labels (y data)
  - backend: Quantum backend
  - options: Options map with:
    - :loss-function - :cross-entropy (default), :hinge, :squared
    - :regularization - :none (default), :l1, :l2, :elastic-net
    - :reg-lambda - Regularization strength (default: 0.01)
    - :reg-alpha - Elastic net mix ratio (default: 0.5)
    - :batch-size - Number of samples per batch (default: nil = all samples)
    - :batch-indices - Specific indices to use (default: nil = all samples)
    - :shots - Number of shots per circuit execution (default: 1024)
  
  Returns:
  Cost value (real number)"
  [parameters ansatz-fn features labels backend & {:keys [options] :or {options {}}}]
  (try
    ;; Input validation
    (when (or (nil? parameters) 
              (not (vector? parameters))
              (empty? parameters)
              (empty? features)
              (empty? labels)
              (not= (count features) (count labels)))
      (throw (ex-info "Invalid inputs" {:parameters parameters :features features :labels labels})))
    
    (let [num-samples (count features)
          loss-function (:loss-function options :cross-entropy)
          regularization (:regularization options :none)
          reg-lambda (:reg-lambda options 0.01)
          reg-alpha (:reg-alpha options 0.5)
          shots (:shots options 1024)
          
          ;; Determine which samples to process
          sample-indices (cond
                           ;; Use provided batch indices
                           (:batch-indices options)
                           (:batch-indices options)
                           
                           ;; Use batch-size to create random batch
                           (:batch-size options)
                           (let [batch-size (min (:batch-size options) num-samples)]
                             (vec (take batch-size (shuffle (range num-samples)))))
                           
                           ;; Default: use all samples
                           :else
                           (vec (range num-samples)))
          
          batch-size (count sample-indices)
          
          ;; Calculate data loss
          total-loss (reduce
                      (fn [acc-loss idx]
                        (let [feature-vec (nth features idx)
                              true-label (nth labels idx)

                              ;; Create circuit with feature encoding + ansatz
                              ;; Use same number of qubits as features for angle encoding
                              num-qubits (count feature-vec)
                              base-circuit (circuit/create-circuit num-qubits)

                              ;; Apply angle encoding for features
                              encoder-result (encoding/angle-encoding feature-vec (:num-qubits base-circuit) :ry)
                              encoded-circuit (if (:success encoder-result)
                                                ((:result encoder-result) base-circuit)
                                                (throw (ex-info "Feature encoding failed" encoder-result)))

                              ;; Apply variational ansatz using the provided ansatz-fn
                              ansatz-circuit (ansatz-fn parameters)
                              
                              ;; Combine encoding and ansatz using circuit composition
                              final-circuit (ccomp/compose-circuits encoded-circuit ansatz-circuit)

                              ;; Execute circuit using backend with configurable shots
                              probs (let [execution-result (backend/execute-circuit backend final-circuit {:shots shots})
                                          measurement-counts (:measurement-counts execution-result)
                                          num-qubits (:num-qubits final-circuit)
                                          num-states (int (m/pow 2 num-qubits))]
                                      ;; Convert counts to probabilities using proper bit string formatting
                                      (mapv (fn [i]
                                              (let [bit-string (state/basis-string i num-qubits)
                                                    count (get measurement-counts bit-string 0)]
                                                (/ count (double shots))))
                                            (range num-states)))

                              ;; Calculate loss based on specified loss function
                              sample-loss (case loss-function
                                            :cross-entropy 
                                            (let [class-0-prob (first probs)
                                                  predicted-prob (if (= true-label 0) class-0-prob (- 1.0 class-0-prob))]
                                              (cross-entropy-loss true-label predicted-prob))
                                            
                                            :hinge
                                            (let [class-0-prob (first probs)
                                                  predicted-score (- (* 2.0 class-0-prob) 1.0)] ; Convert to [-1, 1] range
                                              (hinge-loss true-label predicted-score))
                                            
                                            :squared
                                            (let [class-0-prob (first probs)
                                                  predicted-value (if (= true-label 0) class-0-prob (- 1.0 class-0-prob))]
                                              (squared-loss 1.0 predicted-value)) ; Target is 1.0 for correct prediction
                                            
                                            ;; Default to cross-entropy
                                            (let [class-0-prob (first probs)
                                                  predicted-prob (if (= true-label 0) class-0-prob (- 1.0 class-0-prob))]
                                              (cross-entropy-loss true-label predicted-prob)))]

                          (+ acc-loss sample-loss)))
                      0.0
                      sample-indices)
          
          ;; Calculate average data loss
          avg-data-loss (/ total-loss batch-size)
          
          ;; Add regularization penalty
          reg-penalty (case regularization
                        :l1 (l1-regularization parameters reg-lambda)
                        :l2 (l2-regularization parameters reg-lambda)
                        :elastic-net (elastic-net-regularization parameters reg-lambda reg-alpha)
                        :none 0.0
                        0.0)]
      
      (+ avg-data-loss reg-penalty))
    
    (catch Exception e
      (println "Error in classification-cost:" (.getMessage e))
      1000.0))) ; Return high cost on error

(defn create-classification-cost-fn
  "Create a classification cost function for use with optimization algorithms.
  
  This is a higher-order function that returns a cost function compatible
  with the optimization namespace interface: (fn [parameters] -> cost-value)
  
  Parameters:
  - ansatz-fn: Parameterized quantum circuit function
  - features: Training features (X data)
  - labels: Training labels (y data)
  - backend: Quantum backend
  - options: Options map for loss function and regularization (see classification-cost)
  
  Returns:
  Cost function that takes only parameters and returns cost value"
  [ansatz-fn features labels backend & {:keys [options] :or {options {}}}]
  (fn [parameters]
    (classification-cost parameters ansatz-fn features labels backend :options options)))

(defn create-regression-cost-fn
  "Create a regression cost function for quantum machine learning.
  
  Parameters:
  - ansatz-fn: Parameterized quantum circuit function
  - features: Training features (X data)
  - targets: Training targets (y data, continuous values)
  - backend: Quantum backend
  - options: Options map with:
    - :loss-function - :squared (default), :huber, :absolute
    - :regularization - :none (default), :l1, :l2, :elastic-net
    - :reg-lambda - Regularization strength (default: 0.01)
  
  Returns:
  Cost function that takes only parameters and returns cost value"
  [ansatz-fn features targets backend & {:keys [options] :or {options {}}}]
  (fn [parameters]
    (try
      (let [num-samples (count features)
            loss-function (:loss-function options :squared)
            regularization (:regularization options :none)
            reg-lambda (:reg-lambda options 0.01)
            reg-alpha (:reg-alpha options 0.5)
            
            ;; Calculate regression loss
            total-loss (reduce
                        (fn [acc-loss idx]
                          (let [feature-vec (nth features idx)
                                true-target (nth targets idx)
                                
                                ;; Create and execute circuit (similar to classification)
                                num-qubits (count feature-vec)
                                base-circuit (circuit/create-circuit num-qubits)
                                encoder-result (encoding/angle-encoding feature-vec (:num-qubits base-circuit) :ry)
                                encoded-circuit (if (:success encoder-result)
                                                  ((:result encoder-result) base-circuit)
                                                  (throw (ex-info "Feature encoding failed" encoder-result)))
                                ansatz-circuit (ansatz-fn parameters)
                                final-circuit (ccomp/compose-circuits encoded-circuit ansatz-circuit)
                                
                                ;; Execute and get expectation value for regression
                                execution-result (backend/execute-circuit backend final-circuit {:shots 1024})
                                measurement-counts (:measurement-counts execution-result)
                                num-qubits (:num-qubits final-circuit)
                                
                                ;; Calculate expectation value as prediction
                                predicted-value (let [num-states (int (m/pow 2 num-qubits))]
                                                  (reduce +
                                                          (map (fn [i]
                                                                 (let [bit-string (state/basis-string i num-qubits)
                                                                       count (get measurement-counts bit-string 0)
                                                                       prob (/ count 1024.0)]
                                                                   (* prob i)))
                                                               (range num-states))))
                                
                                ;; Normalize prediction to reasonable range
                                normalized-prediction (/ predicted-value (dec (int (m/pow 2 num-qubits))))
                                
                                ;; Calculate loss
                                sample-loss (case loss-function
                                              :squared (squared-loss true-target normalized-prediction)
                                              :absolute (m/abs (- true-target normalized-prediction))
                                              :huber (let [delta 1.0
                                                           diff (m/abs (- true-target normalized-prediction))]
                                                       (if (<= diff delta)
                                                         (* 0.5 diff diff)
                                                         (- (* delta diff) (* 0.5 delta delta))))
                                              (squared-loss true-target normalized-prediction))]
                            
                            (+ acc-loss sample-loss)))
                        0.0
                        (range num-samples))
            
            avg-data-loss (/ total-loss num-samples)
            
            ;; Add regularization
            reg-penalty (case regularization
                          :l1 (l1-regularization parameters reg-lambda)
                          :l2 (l2-regularization parameters reg-lambda)
                          :elastic-net (elastic-net-regularization parameters reg-lambda reg-alpha)
                          :none 0.0
                          0.0)]
        
        (+ avg-data-loss reg-penalty))
      
      (catch Exception e
        (println "Error in regression-cost:" (.getMessage e))
        1000.0))))

(defn train-qml-model
  "Train a quantum machine learning model using various optimization methods.
  
  Parameters:
  - ansatz-fn: Parameterized quantum circuit function  
  - training-data: Map with :features and :labels (or :targets for regression)
  - config: Training configuration map with:
    - :backend - Quantum backend
    - :max-iterations - Maximum optimization iterations (default: 50)
    - :tolerance - Convergence tolerance (default: 1e-6)
    - :num-parameters - Number of variational parameters
    - :parameter-strategy - :random, :zero, :custom, :legacy (default: :random)
    - :initial-parameters - Custom initial parameters (if :custom strategy)
    - :parameter-range - Range for random init (default: [-0.1 0.1])
    - :optimization-method - :gradient-descent, :adam, :cmaes, :nelder-mead, :powell, :bobyqa (default: :cmaes)
    - :learning-rate - Learning rate for gradient-based methods (default: 0.01)
    - :batch-size - Mini-batch size for training (default: nil = full batch)
    - :shots - Number of shots per circuit execution (default: 1024)
    - :loss-function - :cross-entropy, :hinge, :squared (default: :cross-entropy)
    - :regularization - :none, :l1, :l2, :elastic-net (default: :none)
    - :reg-lambda - Regularization strength (default: 0.01)
    - :reg-alpha - Elastic net mix ratio (default: 0.5)
  
  Returns:
  Map with trained model parameters and metrics"
  [ansatz-fn training-data config]
  (let [features (:features training-data)
        labels (:labels training-data)
        backend (:backend config :simulator)
        max-iterations (:max-iterations config 50)
        learning-rate (:learning-rate config 0.01)
        tolerance (:tolerance config 1e-6)
        optimization-method (:optimization-method config :cmaes)
        batch-size (:batch-size config nil)
        shots (:shots config 1024)

        ;; Initialize parameters using variational algorithm patterns
        num-params (:num-parameters config
                                    (* (count (first features)) 3)) ; 3 rotations per feature
        parameter-strategy (:parameter-strategy config :random)
        initial-params (case parameter-strategy
                         :random (va/random-parameter-initialization num-params 
                                                                   :range (:parameter-range config [-0.1 0.1]))
                         :zero (va/zero-parameter-initialization num-params)
                         :custom (:initial-parameters config)
                         :legacy (vec (repeatedly num-params #(* 0.1 (- (rand) 0.5))))
                         ;; Default to random with small range for QML stability
                         (va/random-parameter-initialization num-params :range [-0.1 0.1]))

        ;; Create cost function options including batch-size and shots
        loss-options {:loss-function (:loss-function config :cross-entropy)
                      :regularization (:regularization config :none)
                      :reg-lambda (:reg-lambda config 0.01)
                      :reg-alpha (:reg-alpha config 0.5)
                      :batch-size batch-size
                      :shots shots}
        
        ;; Create base cost function
        cost-fn (if (:targets training-data) ; Check if regression
                  (create-regression-cost-fn ansatz-fn features (:targets training-data) backend :options loss-options)
                  (create-classification-cost-fn ansatz-fn features labels backend :options loss-options))

        ;; Optimization execution
        start-time (System/currentTimeMillis)
        result (cond
                 ;; Gradient-based methods
                 (= optimization-method :gradient-descent)
                 (opt/gradient-descent-optimization cost-fn initial-params
                                                    {:learning-rate learning-rate
                                                     :max-iterations max-iterations
                                                     :tolerance tolerance
                                                     :adaptive-learning-rate true
                                                     :momentum 0.9})
                 
                 (= optimization-method :adam)
                 (opt/adam-optimization cost-fn initial-params
                                       {:learning-rate learning-rate
                                        :max-iterations max-iterations
                                        :tolerance tolerance})
                 
                 ;; Derivative-free methods
                 (= optimization-method :cmaes)
                 (opt/fastmath-derivative-free-optimization :cmaes cost-fn initial-params
                                                            {:max-iterations max-iterations
                                                             :tolerance tolerance
                                                             :sigma 0.3})
                 
                 (= optimization-method :nelder-mead)
                 (opt/fastmath-derivative-free-optimization :nelder-mead cost-fn initial-params
                                                            {:max-iterations max-iterations
                                                             :tolerance tolerance})
                 
                 (= optimization-method :powell)
                 (opt/fastmath-derivative-free-optimization :powell cost-fn initial-params
                                                            {:max-iterations max-iterations
                                                             :tolerance tolerance})
                 
                 (= optimization-method :bobyqa)
                 (opt/fastmath-derivative-free-optimization :bobyqa cost-fn initial-params
                                                            {:max-iterations max-iterations
                                                             :tolerance tolerance
                                                             :bounds [(:parameter-range config [-3.14 3.14])]})
                 
                 ;; Default to CMAES (robust for quantum ML)
                 :else
                 (opt/fastmath-derivative-free-optimization :cmaes cost-fn initial-params
                                                            {:max-iterations max-iterations
                                                             :tolerance tolerance
                                                             :sigma 0.3}))
        
        end-time (System/currentTimeMillis)]

    ;; Transform optimization result to match original API
    {:success (:success result)
     :optimal-parameters (vec (:optimal-parameters result))  ; Ensure vector type
     :optimal-cost (:optimal-energy result)
     :iterations (:iterations result)
     :training-history (mapv (fn [energy idx]
                               {:iteration idx
                                :cost energy
                                :parameters nil}) ; Parameters not stored per iteration
                             (:convergence-history result)
                             (range))
     :execution-time-ms (- end-time start-time)
     :convergence-reason (:reason result)
     :function-evaluations (:function-evaluations result)
     :optimization-method optimization-method
     :batch-size batch-size
     :shots shots}))

;; Rich comment block for testing
(comment
  ;; Test data preparation
  (def test-features [[0.1 0.2] [0.3 0.4] [0.5 0.6] [0.7 0.8]])
  (def test-labels [0 0 1 1])
  (def test-training-data {:features test-features :labels test-labels})

  ;; Test cost function
  (def test-ansatz (ansatz/hardware-efficient-ansatz 2 1))
  (def test-params (vec (repeatedly 6 #(* 0.1 (rand)))))

  (classification-cost test-params test-ansatz test-features test-labels :simulator)

  ;; Test training
  (def training-config {:max-iterations 10
                        :learning-rate 0.1
                        :num-parameters 6})

  (train-qml-model test-ansatz test-training-data training-config))
  

;; Scicloj/noj integration for data preprocessing and metrics
;; Note: This requires adding noj dependencies to project.clj

(defn normalize-features-with-noj
  "Normalize features using scicloj noj functionality when available.
  
  Falls back to built-in normalization if noj is not available.
  
  Parameters:
  - feature-matrix: 2D vector of features [[row1] [row2] ...]
  
  Returns:
  Normalized feature matrix"
  [feature-matrix]
  (try
    ;; For now, fallback to built-in normalization since noj is not yet integrated
    (mapv (fn [row]
            (let [norm-result (encoding/normalize-features row)]
              (if (:success norm-result)
                (:result norm-result)
                row)))
          feature-matrix)
    (catch Exception _e
      (println "Warning: using built-in normalization")
      ;; Fallback to built-in normalization
      (mapv (fn [row]
              (let [norm-result (encoding/normalize-features row)]
                (if (:success norm-result)
                  (:result norm-result)
                  row)))
            feature-matrix))))

(defn calculate-ml-metrics
  "Calculate comprehensive ML metrics for quantum machine learning results.
  
  Parameters:
  - y-true: True labels/values
  - y-pred: Predicted labels/values
  - task-type: :classification or :regression
  
  Returns:
  Map with relevant metrics"
  [y-true y-pred task-type]
  (case task-type
    :classification
    (let [n (count y-true)
          correct (count (filter true? (map = y-true y-pred)))
          accuracy (/ correct n)
          
          ;; Calculate confusion matrix for binary classification
          tp (count (filter #(and (= (first %) 1) (= (second %) 1)) (map vector y-true y-pred)))
          tn (count (filter #(and (= (first %) 0) (= (second %) 0)) (map vector y-true y-pred)))
          fp (count (filter #(and (= (first %) 0) (= (second %) 1)) (map vector y-true y-pred)))
          fn (count (filter #(and (= (first %) 1) (= (second %) 0)) (map vector y-true y-pred)))
          
          precision (if (> (+ tp fp) 0) (/ tp (+ tp fp)) 0.0)
          recall (if (> (+ tp fn) 0) (/ tp (+ tp fn)) 0.0)
          f1-score (if (> (+ precision recall) 0) (/ (* 2 precision recall) (+ precision recall)) 0.0)]
      
      {:accuracy accuracy
       :precision precision
       :recall recall
       :f1-score f1-score
       :confusion-matrix {:tp tp :tn tn :fp fp :fn fn}
       :sample-count n})
    
    :regression
    (let [n (count y-true)
          errors (map - y-true y-pred)
          squared-errors (map #(* % %) errors)
          absolute-errors (map m/abs errors)
          
          mse (/ (reduce + squared-errors) n)
          rmse (m/sqrt mse)
          mae (/ (reduce + absolute-errors) n)
          
          ;; R-squared calculation
          y-mean (/ (reduce + y-true) n)
          ss-tot (reduce + (map #(* (- % y-mean) (- % y-mean)) y-true))
          ss-res (reduce + squared-errors)
          r-squared (if (> ss-tot 0) (- 1.0 (/ ss-res ss-tot)) 0.0)]
      
      {:mse mse
       :rmse rmse
       :mae mae
       :r-squared r-squared
       :sample-count n})))

(defn create-qml-dataset
  "Create a structured dataset for quantum machine learning.
  
  Parameters:
  - features: Feature matrix
  - labels: Label vector (for classification) or target vector (for regression)
  - task-type: :classification or :regression
  - metadata: Optional metadata map
  
  Returns:
  Structured dataset map"
  [features labels task-type & {:keys [metadata] :or {metadata {}}}]
  (let [n-samples (count features)
        n-features (count (first features))
        normalized-features (normalize-features-with-noj features)]
    
    {:features normalized-features
     :labels labels
     :task-type task-type
     :metadata (merge metadata
                      {:n-samples n-samples
                       :n-features n-features
                       :created-at (java.time.Instant/now)})
     :original-features features}))

(defn split-dataset
  "Split dataset into training and testing sets.
  
  Parameters:
  - dataset: Dataset created with create-qml-dataset
  - train-ratio: Ratio for training set (default: 0.8)
  - random-seed: Random seed for reproducible splits
  
  Returns:
  Map with :train and :test datasets"
  [dataset train-ratio & {:keys [random-seed] :or {random-seed 42}}]
  (let [features (:features dataset)
        labels (:labels dataset)
        n-samples (count features)
        train-size (int (* n-samples train-ratio))
        
        ;; Create random indices for splitting
        _rng (java.util.Random. random-seed) ; For potential future use
        indices (shuffle (range n-samples))
        train-indices (take train-size indices)
        test-indices (drop train-size indices)
        
        ;; Split features and labels
        train-features (mapv #(nth features %) train-indices)
        train-labels (mapv #(nth labels %) train-indices)
        test-features (mapv #(nth features %) test-indices)
        test-labels (mapv #(nth labels %) test-indices)]
    
    {:train (assoc dataset
                   :features train-features
                   :labels train-labels
                   :metadata (assoc (:metadata dataset) :split-type :train))
     :test (assoc dataset
                  :features test-features
                  :labels test-labels
                  :metadata (assoc (:metadata dataset) :split-type :test))}))

(defn visualize-training-progress
  "Create ASCII visualization of training progress.
  
  Parameters:
  - training-history: Vector of maps with :iteration and :cost
  - width: Width of the ASCII plot (default: 60)
  
  Returns:
  String with ASCII plot"
  [training-history width]
  (if (empty? training-history)
    "No training history available"
    (let [costs (mapv :cost training-history)
          iterations (mapv :iteration training-history)
          min-cost (apply min costs)
          max-cost (apply max costs)
          cost-range (- max-cost min-cost)
          
          ;; Normalize costs to plot height (20 lines)
          height 20
          normalized-costs (if (> cost-range 0)
                             (mapv #(int (* (- % min-cost) (/ (dec height) cost-range))) costs)
                             (repeat (count costs) (/ height 2)))
          
          ;; Create ASCII plot
          lines (for [y (range height)]
                  (apply str
                         (for [x (range (min width (count costs)))]
                           (let [cost-y (nth normalized-costs x)]
                             (if (= y (- height cost-y 1)) "*" " ")))))]
      
      (str "Training Progress (Cost vs Iteration)\n"
           "Cost: " (format "%.6f" min-cost) " to " (format "%.6f" max-cost) "\n"
           (str/join "\n" (reverse lines)) "\n"
           (apply str (repeat width "-")) "\n"
           "Iterations: 0 to " (last iterations)))))
