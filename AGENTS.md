# Project Scope
In this project we build a full featured, production ready Quantum Machine Learning library based on the QClojure quantum computing library, not some toy implementation.

Don't implement simplified solutions, implement solutions that are correct and will run on real quantum hardware. Don't be lazy!
For the quantum algorithms we use the QClojure library.
For the math we use core and complex from the fastmath library and the QClojure complex linear algebra API.
For machine learning and data science we use scicloj/noj and its dependencies.
Double precision is used for the math, but we can switch to arbitrary precision later.
The backend protocol is the basis for the integration of real quantum computers, e.g. cloud based quantum services, and is used to run circuits.
For visualization we focus on SVG and ASCII first.

The QClojure codebase is available in the directory 'reference/qclojure' or as github repo 'lsolbach/qclojure'.
Some namespaces relevant to quantum machine learning in the QClojure codebase are
* `org.soulspace.qclojure.domain.gate` - quantum gates
* `org.soulspace.qclojure.domain.circuit` - quantum circuits
* `org.soulspace.qclojure.domain.observable` - observables
* `org.soulspace.qclojure.domain.hamiltonian` - hamiltonians
* `org.soulspace.qclojure.domain.ansatz` - ansatzes
* `org.soulspace.qclojure.application.backend` - backend protocols
* `org.soulspace.qclojure.adapter.backend.simulator` - simulator backend
* `org.soulspace.qclojure.adapter.backend.hardware-simulator` - noisy simulator backend
* `org.soulspace.qclojure.application.algorithm.optimization` - optimization algorithms
* `org.soulspace.qclojure.application.algorithm.variational-algorithm` - variational algorithm template
* `org.soulspace.qclojure.application.algorithm.vqe` - variational quantum eigensolver algorithm
* `org.soulspace.qclojure.application.algorithm.qaoa` - quantum approximate

