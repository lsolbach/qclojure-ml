(defproject org.soulspace/qclojure-ml "0.1.0-SNAPSHOT"
  :description "Quantum Machine Learning algorithms for QClojure"
  :license {:name "Eclipse Public License 1.0"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [org.soulspace/qclojure "0.17.0"]
                 [org.scicloj/noj "2-beta18"]
                 [org.scicloj/clay "2-beta52"]]

  :scm {:name "git" :url "https://github.com/lsolbach/qclojure-ml"}
  :deploy-repositories [["clojars" {:sign-releases false :url "https://clojars.org/repo"}]])

