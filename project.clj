(defproject org.soulspace/qclojure-ml "0.2.0"
  :description "Quantum Machine Learning algorithms for QClojure"
  :license {:name "Eclipse Public License 1.0"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.4"]
                 [org.scicloj/noj "2-beta19.1"]
                 [org.slf4j/slf4j-nop "2.0.17"] ; supress logging warnings
                 [org.soulspace/qclojure "0.24.0"]]

  :profiles {:dev [:user {}]
             :sim-heavy {:jvm-opts ["-Xms8g" "-Xmx32g"
                                    "-XX:MaxGCPauseMillis=200"
                                    "-XX:+AlwaysPreTouch"]}
             :container {:jvm-opts ["-XX:InitialRAMPercentage=2.0"
                                    "-XX:MaxRAMPercentage=60.0"]}
             :clay {:dependencies [[org.scicloj/clay "2.0.5"]]
                    :source-paths ["src" "notebooks"]}}

  :scm {:name "git" :url "https://github.com/lsolbach/qclojure-ml"}
  :deploy-repositories [["clojars" {:sign-releases false :url "https://clojars.org/repo"}]])
