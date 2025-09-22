(ns build-notebooks
  (:require [scicloj.clay.v2.api :as clay]))

(def example-base-config
  "Base configuration for building the examples."
  {:base-source-path "notebooks"
   :source-path ["tutorial.clj"]
   :remote-repo {:git-url "https://github.com/lsolbach/qclojure-ml"
                 :branch "main"}
   :title "QClojure ML Examples"})

