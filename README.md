# Probabilistic Graphical Models (PGM)

A **Probabilistic Graphical Model (PGM)** is a probabilistic framework in which a graph expresses the **conditional dependence structure** among random variables.

<img src="https://user-images.githubusercontent.com/31917400/92264526-4702cf80-eed6-11ea-97e7-72618640aa8d.jpg" alt="Overview of Probabilistic Graphical Models" />

---

## Table of Contents

- [Overview & Motivation](#overview--motivation)
- [Types of Graphical Models](#types-of-graphical-models)
  - [[a] Bayesian Networks (Directed)](#a-bayesian-networks-directed)
  - [[b] Markov Networks (Undirected)](#b-markov-networks-undirected)
- [Quick Examples](#quick-examples)
  - [Directed PGM & OLS Regression](#directed-pgm--ols-regression)
  - [Directed PGM & Naive Bayes](#directed-pgm--naive-bayes)
- [A. Representation](#a-representation)
  - [1. Bayesian Network](#1-bayesian-network)
    - [Evidence & Conditioning Patterns](#evidence--conditioning-patterns)
    - [D-Separation (Explaining Away)](#d-separation-explaining-away)
    - [Template Models & Global Structure](#template-models--global-structure)
    - [Local Structure in CPDs](#local-structure-in-cpds)
  - [2. Markov Network](#2-markov-network)
- [B. Inference](#b-inference)
  - [Inference 01: Exact Inference (Enumeration & Variable Elimination)](#inference-01-exact-inference-enumeration--variable-elimination)
  - [Inference 02: Approximate Inference via Sampling](#inference-02-approximate-inference-via-sampling)
    - [1. Importance Sampling](#1-importance-sampling)
    - [2. Inverse Transform Sampling](#2-inverse-transform-sampling)
    - [3. Rejection Sampling](#3-rejection-sampling)
    - [4. MCMC & Gibbs Sampling](#4-mcmc--gibbs-sampling)
- [Bayesian Neural Networks (BNNs)](#bayesian-neural-networks-bnns)
  - [Scaling & Stochastic Gradient Langevin Dynamics (SGLD)](#scaling--stochastic-gradient-langevin-dynamics-sgld)
  - [Dynamic Bayesian Neural Networks](#dynamic-bayesian-neural-networks)
- [Python Code Examples](#python-code-examples)
  - [1. Bayesian Network & Exact Inference (`pgmpy`)](#1-bayesian-network--exact-inference-pgmpy)
  - [2. Importance & Rejection Sampling (`numpy`)](#2-importance--rejection-sampling-numpy)
  - [3. Gibbs Sampling (`numpy`)](#3-gibbs-sampling-numpy)
  - [4. Stochastic Gradient Langevin Dynamics (`torch`)](#4-stochastic-gradient-langevin-dynamics-torch)

---

## Overview & Motivation

<img src="https://user-images.githubusercontent.com/31917400/52655869-1a025c80-2eed-11e9-82ba-1fdea931596f.jpg" alt="Graphical model motivation and dependency" />

- **Independent vs. Dependent Data:** Standard classifiers often treat data points as independent (e.g., predicting whether an image contains a cat or a dog). In sequential or structured domains—such as `"I"`, `"like"`, `"machine"`, `"learning"`—words depend on context (`"learning"` can act as a noun or verb depending on surrounding tokens). PGMs provide a structured framework to model these **dependencies**.
- **Graph Components:** A PGM combines a graph with probability theory:
  - **Nodes (vertices):** Represent random variables.
  - **Edges:** Encode probabilistic relationships and dependencies between variables.
- **Core Advantages:**
  1. **Compact & Intuitive Representation:** Captures high-dimensional joint distributions clearly.
  2. **General-Purpose Inference:** Supports systematic probabilistic reasoning and querying via graph algorithms.
  3. **Parameter Efficiency:** Factorizing along conditional independencies drastically reduces the number of parameters needed compared to a full joint table—by dropping unnecessary edges whenever conditional independence holds.

<img src="https://user-images.githubusercontent.com/31917400/69966964-58feae80-150f-11ea-852b-1b4efa485e71.jpg" alt="Factor reduction via conditional independence" />

---

## Types of Graphical Models

Depending on whether edges are **directed** or **undirected**, graphical models fall into two main families:

### [a] Bayesian Networks (Directed)
Defined over an ordered **Directed Acyclic Graph (DAG)** representing a joint distribution $P(X_1, X_2, \dots, X_n)$. While variables may be marginally dependent, encoding **conditional independence** simplifies how the joint distribution factors.

<img src="https://user-images.githubusercontent.com/31917400/69970023-6880f600-1515-11ea-82e4-ed9e176cc946.jpg" alt="Bayesian network DAG structure" />

#### Key Rules: Factorization of Joint Distributions
Using the chain rule of probability:

$$P(a, b, c) = P(a \mid b, c)\,P(b, c) = P(a \mid b, c)\,P(b \mid c)\,P(c)$$

$$P(a, b, c) = P(a, b \mid c)\,P(c)$$

- **Conditional vs. Joint under conditioning:**
  - Conditional decomposition: $P(a \mid b, C) = \dfrac{P(a, b \mid C)}{P(b \mid C)}$
  - Joint decomposition: $P(a, b \mid C) = P(a \mid b, C)\,P(b \mid C)$
- **Notation Rule:** If a conditioning bar `|` is already present, append additional conditioning variables after a comma; otherwise introduce `|` before the conditioning variable(s).

### [b] Markov Networks (Undirected)
Use an **undirected graph** to model symmetric interactions and potential functions (clique factors) rather than directed causal edges.

---

## Quick Examples

### Directed PGM & OLS Regression
<img src="https://user-images.githubusercontent.com/31917400/69980377-07aee900-1528-11ea-8fad-d6758bf79b65.jpg" alt="Directed PGM and OLS Regression" />

### Directed PGM & Naive Bayes
In a **Naive Bayes** graphical model, a single class node $Y$ acts as the parent of conditionally independent feature nodes $X_1, X_2, \dots, X_n$ given $Y$:

$$P(Y, X_1, \dots, X_n) = P(Y)\prod_{i=1}^{n} P(X_i \mid Y)$$

```text
          ( Y )         <-- Class Label
        /   |   \
       v    v    v
    (X_1) (X_2) (X_n)   <-- Conditionally Independent Features given Y
```

---

# A. Representation

## 1. Bayesian Network

<img src="https://user-images.githubusercontent.com/31917400/52790734-36bea180-305f-11e9-83b4-d831b3ac13eb.png" alt="Bayesian Network representation" />

A **Bayesian Network** represents a joint probability distribution over random variables $X_1, X_2, \dots, X_n$ using a **DAG**:

$$P(X_1, X_2, \dots, X_n) = \prod_{i=1}^{n} P\big(X_i \mid \text{Pa}(X_i)\big)$$

where $\text{Pa}(X_i)$ denotes the parents of $X_i$ in the graph.

- Marries **uncertainty** with structured graph factorization.
- Makes direct and indirect influences between uncertain variables transparent.

### Evidence & Conditioning Patterns

- **Single Observation:**
  <img src="https://user-images.githubusercontent.com/31917400/67561343-8a1be000-f714-11e9-9583-74bcd84a13ad.jpg" alt="Single test result in a Bayesian network" />

- **Multiple Observations (`result01`, `result02`, ...):**
  <img src="https://user-images.githubusercontent.com/31917400/67567229-6ca14300-f721-11e9-95f0-7493cd287be1.jpg" alt="Multiple test results in a Bayesian network" />

- **Conditional Independence Across Observations:** Given a common cause/parent $C$, its children are conditionally independent ($P(\text{result02} \mid \text{result01}, C) = P(\text{result02} \mid C)$).
  <img src="https://user-images.githubusercontent.com/31917400/67569336-7aa59280-f726-11e9-8a70-2e7ed57ed008.jpg" alt="Conditional independence given parent node" />

- **Multiple Latent Variables (Hidden Causes):**
  <img src="https://user-images.githubusercontent.com/31917400/67586119-c5d09d00-f748-11e9-894d-4e9f5f37de97.jpg" alt="Multiple latent variables in a Bayesian network" />

### Why Bayesian Networks?
Rather than enumerating every joint state combination explicitly, each node defines a local Conditional Probability Distribution (CPD) given its parents, reducing parameter count and enabling compact representation and inference.

<img src="https://user-images.githubusercontent.com/31917400/67589802-3ed3f280-f751-11e9-8de9-a5eca09adb8b.jpg" alt="Parameter reduction in Bayesian networks" />

### Worked Example
<img src="https://user-images.githubusercontent.com/31917400/70064102-dd6d3200-15e0-11ea-9868-a00ccd008e22.jpg" alt="Bayesian network example" />

### D-Separation (Explaining Away)
In a **v-structure** ($A \rightarrow C \leftarrow B$), observing a shared child (or a descendant of that child) couples previously independent causes—a phenomenon known as **explaining away**.

<img src="https://user-images.githubusercontent.com/31917400/67591895-6083a880-f756-11e9-992b-6bb649a348e4.jpg" alt="D-separation and explaining away in v-structures" />

---

### Template Models & Global Structure
**Template Models** extend graphical models to handle repeated variables, time series, and shared structure across instances or time steps:
- Replicate **template variables** and share **CPDs (parameter sharing)** across repeated instances.
- Act as a compact specification of a larger unrolled ("ground") Bayesian network without increasing expressiveness beyond the unrolled DAG.
- Common examples include **Dynamic Bayesian Networks (DBNs)** for temporal sequences and **Object-Relational Models** (directed or undirected).

<img src="https://user-images.githubusercontent.com/31917400/52852977-071ca180-3112-11e9-8928-f44a07c0f347.jpg" alt="Template model overview" />

- **Temporal Models (DBNs & HMMs):**
  - **Dynamic Bayesian Networks (DBNs)** unroll a template network across discrete time slices to model evolving systems.
  - **Hidden Markov Models (HMMs)** are a specialized subclass with latent state chains and observations at each step.
- **Plate Models:**
  - Enclose repeated/indexed variables inside a **plate** (a box indicating indexing over items such as trials, users, or documents). Parameters outside all plates are shared across the replicated variables.

<img src="https://user-images.githubusercontent.com/31917400/52870574-11ee2b00-3140-11e9-9ac1-a8ce07c4bb91.jpg" alt="Plate model representation" />

  - **Index Subset Rule:** For a template variable $A\{X_1, \dots, X_k\}$ indexed by $X$, parent template variables (such as $B$ indexed by $U$) must have indices compatible with the child's scope so that grounding produces a valid dependency graph.
  - **Ground Network:** Unrolling plates over concrete indices yields the **ground network**, enabling collective inference across correlated entities.

<img src="https://user-images.githubusercontent.com/31917400/52872945-aad37500-3145-11e9-87b8-838e19aeed7b.jpg" alt="Ground network unrolled from plates" />

---

### Local Structure in CPDs

<img src="https://user-images.githubusercontent.com/31917400/52900377-03aa1880-31ed-11e9-8100-4386992ee220.jpg" alt="Local structure in CPDs" />

Instead of storing full tabular CPDs whose size grows exponentially with parent count, **local structure** parameterizes each node's CPD compactly:

#### (a) Deterministic Structure
A child variable is a deterministic function of its parents, and **Context-Specific Independence (CSI)** holds when independence between variables appears only for specific values of a conditioning variable $C$.

<img src="https://user-images.githubusercontent.com/31917400/52900959-f98c1800-31f4-11e9-8daa-67710a994fa0.jpg" alt="Deterministic CPD structure" />

#### (b) Tree Structure & Multiplexer Model
Represents a CPD as a decision tree where internal nodes test parent values and leaves hold local distributions:

<img src="https://user-images.githubusercontent.com/31917400/52902425-6c52be80-3208-11e9-8f57-130d90e55ef1.jpg" alt="Tree-structured CPD" />

- **Multiplexer Model:** Uses a selector/multiplexer variable to route dependencies to different parent subsets, drastically cutting parameter counts.

<img src="https://user-images.githubusercontent.com/31917400/52902687-1c75f680-320c-11e9-9b0d-ca8af4f6ca8f.jpg" alt="Multiplexer tree structure" />

- **Practical Application (Printer Troubleshooting):** A selector/routing variable (local vs. network print path) determines which failure factors apply, reducing parameter count (e.g., from 145 down to ~55 in a classic diagnostic example).

<img src="https://user-images.githubusercontent.com/31917400/52902788-978bdc80-320d-11e9-935b-691a91a1e2a2.jpg" alt="Multiplexer tree example for printer diagnostics" />

#### (c) Noisy-OR & Noisy-AND Structure
Models **independence of causal influence**, where multiple parents $X_1, \dots, X_k$ affect a target $Y$ through independent intermediate mechanisms $Z_i$:

<img src="https://user-images.githubusercontent.com/31917400/52904371-4e468780-3223-11e9-8c78-c72c7ab90c28.jpg" alt="Noisy causal influence structure" />

- **Noisy-OR:** $Y$ becomes true if at least one active cause (or a background leak factor) succeeds in turning it on:

$$P(Y = 1 \mid \mathbf{x}) = 1 - (1 - \lambda_0)\prod_{i : x_i = 1} (1 - \lambda_i)$$

<img src="https://user-images.githubusercontent.com/31917400/52904648-b9de2400-3226-11e9-8a9d-7dee3fabeaa2.jpg" alt="Noisy-OR formulation" />

- **Noisy-AND (and Noisy-MAX):** Aggregates independent causal contributions $Z_i$ into a combined effect to determine $Y$.

<img src="https://user-images.githubusercontent.com/31917400/52904801-f874de00-3228-11e9-99cc-56067beca889.jpg" alt="Noisy-AND and Noisy-MAX formulation" />

#### (d) Logistic Structure
Computes a linear combination of parent influences plus a bias term $w_0$, then passes the result through a sigmoid function:

$$z = w_0 + \sum_{i} w_i x_i, \qquad P(Y = 1 \mid \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

<img src="https://user-images.githubusercontent.com/31917400/52904828-6e794500-3229-11e9-9e37-aca824bc0268.jpg" alt="Logistic CPD structure" />

#### (e) Continuous Structure
Used when variables are continuous—for example, modeling a noisy sensor measurement around an underlying continuous temperature using a Gaussian (normal) distribution:

$$p(y \mid x) = \mathcal{N}\!\left(y;\, \mu_0 + w x,\, \sigma^2\right)$$

<img src="https://user-images.githubusercontent.com/31917400/52906319-31b94800-3241-11e9-80b0-3e7b2fd646bf.jpg" alt="Continuous random variables in graphical models" />
<img src="https://user-images.githubusercontent.com/31917400/52906349-c7ed6e00-3241-11e9-8859-d0ac04e8257d.jpg" alt="Gaussian sensor model example" />

---

## 2. Markov Network

A **Markov Network** (Markov Random Field) uses an **undirected graph** $G = (V, E)$ to represent conditional independencies via graph separation: two subsets of nodes $A$ and $B$ are conditionally independent given $C$ ($A \perp B \mid C$) if every path between $A$ and $B$ passes through $C$.

```text
    (X_1) --- (X_2)
      |         |      Clique Potentials: phi_1(X_1, X_2), phi_2(X_2, X_3), ...
    (X_4) --- (X_3)
```

### Global Structure
The joint distribution factorizes over maximal cliques (or factor potentials $\phi_c(\mathbf{x}_c)$):

$$P(\mathbf{x}) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \phi_c(\mathbf{x}_c), \qquad Z = \sum_{\mathbf{x}} \prod_{c \in \mathcal{C}} \phi_c(\mathbf{x}_c)$$

where $Z$ is the **partition function** (normalizing constant). Unlike Bayesian networks, edges in a Markov network represent symmetric affinity rather than causal direction.

### Local Structure
Factor potentials $\phi_c(\mathbf{x}_c) > 0$ are commonly parameterized in log-linear (energy-based) form:

$$\phi_c(\mathbf{x}_c) = \exp\big(-E_c(\mathbf{x}_c)\big) = \exp\!\left(\sum_k w_k f_k(\mathbf{x}_c)\right)$$

using feature functions $f_k$ and weights $w_k$, reducing parameter count when features are shared across cliques or spatial positions.

---

# B. Inference

<img src="https://user-images.githubusercontent.com/31917400/67621845-b1090d80-f80b-11e9-8081-dcd66dd9a27e.jpg" alt="Inference overview in graphical models" />

Common inference tasks include:
1. **Posterior Inference (Marginal / Conditional Queries):** Computing $P(\mathbf{Q} \mid \mathbf{E} = \mathbf{e})$ for query variables $\mathbf{Q}$ given observed evidence $\mathbf{E}$.
2. **MAP / MPE Inference:** Finding the most probable assignment $\arg\max_{\mathbf{q}} P(\mathbf{q} \mid \mathbf{e})$.
3. **Bidirectional / Diagnostic Reasoning:** Inferring causes from effects (e.g., inferring whether a burglary occurred given reports that an alarm sounded).

<img src="https://user-images.githubusercontent.com/31917400/67622039-e0207e80-f80d-11e9-98c9-a121ac5855b4.jpg" alt="Diagnostic reasoning example" />

---

## Inference 01: Exact Inference (Enumeration & Variable Elimination)

**Enumeration** computes $P(Q \mid \mathbf{E} = \mathbf{e})$ by summing joint probabilities over all hidden (non-query, non-evidence) variables $\mathbf{H}$:

$$P(Q \mid \mathbf{E} = \mathbf{e}) = \frac{P(Q, \mathbf{E} = \mathbf{e})}{P(\mathbf{E} = \mathbf{e})} = \frac{\sum_{\mathbf{h}} P(Q, \mathbf{E} = \mathbf{e}, \mathbf{H} = \mathbf{h})}{\sum_{q', \mathbf{h}} P(Q = q', \mathbf{E} = \mathbf{e}, \mathbf{H} = \mathbf{h})}$$

- **Example (Burglary / Alarm / John & Mary):** Given evidence that both John and Mary called, summing over the hidden variables yields $P(\text{Burglary} = \text{True} \mid \text{JohnCalls}, \text{MaryCalls}) \approx 0.284$.

<img src="https://user-images.githubusercontent.com/31917400/67623036-735eb180-f818-11e9-9ad2-cbbcc6141961.jpg" alt="Burglary alarm inference problem" />
<img src="https://user-images.githubusercontent.com/31917400/67623078-0992d780-f819-11e9-88b7-c4b0bae56255.jpg" alt="Rewriting conditional probability as joint probabilities" />
<img src="https://user-images.githubusercontent.com/31917400/67623248-dcdfbf80-f81a-11e9-87fe-b187545b1c73.jpg" alt="Summing over hidden variables" />
<img src="https://user-images.githubusercontent.com/31917400/67623317-8fb01d80-f81b-11e9-9653-07cc7f9cafdd.jpg" alt="Expanding factors using Bayesian network CPDs" />
<img src="https://user-images.githubusercontent.com/31917400/67623855-0cde9100-f822-11e9-8f82-bb14e7c04135.jpg" alt="Summation over products of factors" />

### Speeding Up Exact Inference: Variable Elimination
Plain enumeration repeats identical sub-expressions across terms. **Variable Elimination** interleaves factor multiplication and marginalization (summing out variables one at a time) so intermediate factors remain compact.

<img src="https://user-images.githubusercontent.com/31917400/67634221-f59ab480-f8b0-11e9-8dc4-1e9a03ddbc90.jpg" alt="Factor formation in variable elimination" />
<img src="https://user-images.githubusercontent.com/31917400/67634554-8e333380-f8b5-11e9-9007-af2d5aca5ce5.jpg" alt="Marginalizing factors step by step" />

---

## Inference 02: Approximate Inference via Sampling

When exact inference is intractable, **Monte Carlo sampling** approximates expectations $\mathbb{E}_{P}[f(X)] \approx \frac{1}{N}\sum_{i=1}^{N} f(x^{(i)})$ by drawing samples and averaging evaluations.

<img src="https://user-images.githubusercontent.com/31917400/68958025-2e98bb80-07c3-11ea-9ebb-9503bedfb3c7.jpg" alt="Monte Carlo sampling overview" />

### 1. Importance Sampling
When sampling directly from $P(x)$ is difficult, draw samples from a **proposal distribution** $q(x)$ and weight each sample by $w(x) = \frac{P(x)}{q(x)}$:

$$\mathbb{E}_{P}[f(X)] = \int f(x)\frac{P(x)}{q(x)} q(x)\,dx \approx \frac{1}{N}\sum_{i=1}^{N} \frac{P(x^{(i)})}{q(x^{(i)})} f(x^{(i)}), \quad x^{(i)} \sim q(x)$$

If $P(x) = \tilde{P}(x)/Z_P$ and $q(x) = \tilde{q}(x)/Z_q$ have unknown normalizing constants, **self-normalized importance sampling** uses unnormalized weights $\tilde{w}(x) = \frac{\tilde{P}(x)}{\tilde{q}(x)}$:

$$\mathbb{E}_{P}[f(X)] \approx \frac{\sum_{i=1}^{N} \tilde{w}(x^{(i)}) f(x^{(i)})}{\sum_{i=1}^{N} \tilde{w}(x^{(i)})}$$

<img src="https://user-images.githubusercontent.com/31917400/68993028-06fd2e00-086b-11ea-805d-775da9338e2a.jpg" alt="Importance sampling proposal distribution" />
<img src="https://user-images.githubusercontent.com/31917400/68996149-28bbdc80-088e-11ea-8869-07ca8b8c7b31.jpg" alt="Importance sampling Monte Carlo approximation" />
<img src="https://user-images.githubusercontent.com/31917400/68995569-315ce480-0887-11ea-99cf-472009504fca.jpg" alt="Unnormalized distributions in importance sampling" />
<img src="https://user-images.githubusercontent.com/31917400/68995679-438b5280-0888-11ea-9345-a6bdac11e7d1.jpg" alt="Normalizing constants Z_p and Z_q" />
<img src="https://user-images.githubusercontent.com/31917400/68995820-eabcb980-0889-11ea-89e8-224c317db213.jpg" alt="Estimating normalizing constant ratio" />
<img src="https://user-images.githubusercontent.com/31917400/68996074-0d040680-088d-11ea-8587-ffbf437c73f9.jpg" alt="Self-normalized importance sampling estimator" />

### 2. Inverse Transform Sampling
Given a target distribution with Cumulative Distribution Function (CDF) $F_X(x)$:
- **Invertible CDF:** Draw $u \sim \text{Uniform}(0, 1)$ and set $x = F_X^{-1}(u)$.
- **Non-invertible / Discrete CDF:** Partition $[0, 1)$ into intervals corresponding to cumulative probability masses and map $u$ to the matching interval/value.

<img src="https://user-images.githubusercontent.com/31917400/68999966-4ace5280-08c0-11ea-9600-9497a37949d4.jpg" alt="Inverse transform sampling with invertible CDF" />
<img src="https://user-images.githubusercontent.com/31917400/69006832-eeede300-092c-11ea-8fb1-a12e13d7c021.jpg" alt="Inverse transform sampling with intervals" />

### 3. Rejection Sampling
- **Discrete / Set Sampling:** Sample uniformly from an enclosing bounding region and keep points falling inside the target set.
- **Continuous / Density Sampling:** Draw candidate samples $x \sim q(x)$ from a tractable proposal scaled by $M$ so $M\,q(x) \ge \tilde{p}(x)$, and accept each draw with probability $\frac{\tilde{p}(x)}{M\,q(x)}$.

<img src="https://user-images.githubusercontent.com/31917400/69007867-86f2c900-093b-11ea-9ad7-bea3631487c9.jpg" alt="Rejection sampling over bounding region" />
<img src="https://user-images.githubusercontent.com/31917400/69014553-53d42800-0983-11ea-8912-34609a85d966.jpg" alt="Continuous rejection sampling with proposal envelope" />

### 4. MCMC & Gibbs Sampling
**Markov Chain Monte Carlo (MCMC)** constructs a Markov chain whose stationary distribution matches the target posterior $P(\mathbf{x})$:
- **Metropolis & Metropolis–Hastings:** Propose a candidate state $x'$ from $q(x' \mid x)$ and accept with probability $\alpha = \min\!\left(1, \frac{\tilde{p}(x')\,q(x \mid x')}{\tilde{p}(x)\,q(x' \mid x)}\right)$.
- **Gibbs Sampling:** Updates one variable (or block) at a time by sampling from its full conditional distribution $P(x_i \mid \mathbf{x}_{-i})$ given the current values of all other variables.

<img src="https://user-images.githubusercontent.com/31917400/69341186-fb41ab00-0c60-11ea-8b34-17e8a851b5a6.jpg" alt="MCMC and neural network sampling" />
<img src="https://user-images.githubusercontent.com/31917400/69055084-e23cbe00-0a04-11ea-96f6-b56046fa7153.jpg" alt="Markov chain stationary distribution" />
<img src="https://user-images.githubusercontent.com/31917400/69059687-ea4d2b80-0a0d-11ea-9719-d43990924537.jpg" alt="Metropolis-Hastings transition and acceptance" />
<img src="https://user-images.githubusercontent.com/31917400/69253066-e9ea9700-0bab-11ea-99ff-e4b73ed0cc48.jpg" alt="Gibbs sampling coordinate updates" />
<img src="https://user-images.githubusercontent.com/31917400/69254772-ac3b3d80-0bae-11ea-8bd3-e9e40b7f8625.jpg" alt="Gibbs sampling iteration dynamics" />

---

## Bayesian Neural Networks (BNNs)

<img src="https://user-images.githubusercontent.com/31917400/69341233-14e2f280-0c61-11ea-80ef-26159e0fd58b.jpg" alt="Bayesian Neural Networks overview" />

Standard neural networks learn **point estimates** of weights $\mathbf{w}$, which often yield overconfident predictions in regions with scarce or out-of-distribution data. **Bayesian Neural Networks** place a **prior** $p(\mathbf{w})$ over weights and infer a **posterior** $p(\mathbf{w} \mid \mathcal{D})$ given data $\mathcal{D}$:

$$p(\mathbf{w} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \mathbf{w})\,p(\mathbf{w})}{p(\mathcal{D})}$$

Predictions marginalize over weight uncertainty:

$$p(y^* \mid \mathbf{x}^*, \mathcal{D}) = \int p(y^* \mid \mathbf{x}^*, \mathbf{w})\,p(\mathbf{w} \mid \mathcal{D})\,d\mathbf{w}$$

### Scaling & Stochastic Gradient Langevin Dynamics (SGLD)
Exact posterior inference is intractable for deep networks, so practitioners rely on **Variational Inference** or **Stochastic Gradient MCMC**.

<img src="https://user-images.githubusercontent.com/31917400/69342568-bb2ff780-0c63-11ea-9287-69eccb0dbc83.jpg" alt="Scalability challenges in Bayesian Neural Networks" />

**Stochastic Gradient Langevin Dynamics (SGLD)** combines minibatch stochastic gradients with injected Gaussian noise $\boldsymbol{\eta}_t \sim \mathcal{N}(\mathbf{0}, \epsilon_t \mathbf{I})$ so parameter updates converge toward samples from the posterior:

$$\Delta \mathbf{w}_t = \frac{\epsilon_t}{2}\left(\nabla \log p(\mathbf{w}_t) + \frac{N}{n}\sum_{i \in \mathcal{B}_t} \nabla \log p(y_i \mid \mathbf{x}_i, \mathbf{w}_t)\right) + \boldsymbol{\eta}_t$$

<img src="https://user-images.githubusercontent.com/31917400/69347963-49f54200-0c6d-11ea-97eb-62eb7defcbd2.jpg" alt="Langevin Monte Carlo for BNN inference" />

#### Limitations of Gibbs Sampling in BNNs
- **High Dimensionality:** Updating one weight coordinate at a time scales poorly when a network has millions of parameters.
- **Minibatch Constraints:** Standard Gibbs updates typically require full-dataset conditionals rather than minibatch stochastic estimates.
- **Burn-In Cost:** Chains require convergence during training, though collected posterior weight samples can be reused at test time for uncertainty-aware ensembling.

### Dynamic Bayesian Neural Networks
**Dynamic Bayesian Neural Networks** combine temporal Bayesian modeling across time steps $t-1 \to t$ with neural network parameterizations to capture non-linear state transitions and uncertainty over sequential data:

```text
  z_{t-1} ----[ Bayesian NN Transition ]----> z_t
     |                                         |
  [ Emission ]                              [ Emission ]
     v                                         v
  x_{t-1}                                     x_t
```

---

## Python Code Examples

### 1. Bayesian Network & Exact Inference (`pgmpy`)

This example builds the classic **Burglary–Earthquake–Alarm–JohnCalls–MaryCalls** network and computes $P(\text{Burglary} \mid \text{JohnCalls}=1, \text{MaryCalls}=1)$ using Variable Elimination.

```bash
pip install pgmpy
```

```python
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define DAG structure
model = DiscreteBayesianNetwork([
    ("Burglary", "Alarm"),
    ("Earthquake", "Alarm"),
    ("Alarm", "JohnCalls"),
    ("Alarm", "MaryCalls"),
])

# 2. Define Conditional Probability Distributions (CPDs)
cpd_b = TabularCPD(variable="Burglary", variable_card=2, values=[[0.999], [0.001]])
cpd_e = TabularCPD(variable="Earthquake", variable_card=2, values=[[0.998], [0.002]])

cpd_a = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[
        # P(Alarm=0 | B, E)
        [0.999, 0.71, 0.06, 0.05],
        # P(Alarm=1 | B, E)
        [0.001, 0.29, 0.94, 0.95],
    ],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
)

cpd_j = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.10], [0.05, 0.90]],
    evidence=["Alarm"],
    evidence_card=[2],
)

cpd_m = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.99, 0.30], [0.01, 0.70]],
    evidence=["Alarm"],
    evidence_card=[2],
)

model.add_cpds(cpd_b, cpd_e, cpd_a, cpd_j, cpd_m)
assert model.check_model()

# 3. Exact inference via Variable Elimination
infer = VariableElimination(model)
posterior = infer.query(
    variables=["Burglary"],
    evidence={"JohnCalls": 1, "MaryCalls": 1},
)
print(posterior)
# P(Burglary=1 | JohnCalls=1, MaryCalls=1) ≈ 0.2842
```

---

### 2. Importance & Rejection Sampling (`numpy`)

Approximating $\mathbb{E}_P[X^2]$ under an unnormalized target density $\tilde{p}(x) = \exp(-0.5 x^2)(1 + \sin^2(x))$ using a Gaussian proposal $q(x) = \mathcal{N}(0, 1.5^2)$.

```python
import numpy as np

rng = np.random.default_rng(42)

def p_tilde(x):
    return np.exp(-0.5 * x**2) * (1.0 + np.sin(x) ** 2)

def q_pdf(x, sigma=1.5):
    return np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

# --- Self-Normalized Importance Sampling ---
N = 50_000
sigma = 1.5
samples_q = rng.normal(0.0, sigma, size=N)
weights = p_tilde(samples_q) / q_pdf(samples_q, sigma)
weights /= np.sum(weights)

est_importance = np.sum(weights * (samples_q**2))
print(f"Importance Sampling E[X^2]: {est_importance:.4f}")

# --- Rejection Sampling ---
# Bound M such that M * q_pdf(x) >= p_tilde(x) for all x
M = 2.0 * np.sqrt(2 * np.pi) * sigma
candidates = rng.normal(0.0, sigma, size=N)
u = rng.uniform(0.0, 1.0, size=N)
accepted = candidates[u < p_tilde(candidates) / (M * q_pdf(candidates, sigma))]

print(f"Rejection Sampling E[X^2]:  {np.mean(accepted**2):.4f} (acceptance rate: {len(accepted)/N:.2%})")
```

---

### 3. Gibbs Sampling (`numpy`)

Sampling from a correlated 2D Gaussian $\mathcal{N}\!\left(\begin{bmatrix}0 \\ 0\end{bmatrix}, \begin{bmatrix}1 & \rho \\ \rho & 1\end{bmatrix}\right)$ by alternating between exact 1D conditional distributions $x_1 \mid x_2 \sim \mathcal{N}(\rho x_2, 1 - \rho^2)$ and $x_2 \mid x_1 \sim \mathcal{N}(\rho x_1, 1 - \rho^2)$.

```python
import numpy as np

def gibbs_bivariate_normal(n_samples=10_000, burn_in=1_000, rho=0.8, seed=42):
    rng = np.random.default_rng(seed)
    cond_std = np.sqrt(1.0 - rho**2)
    x1, x2 = 0.0, 0.0
    samples = np.empty((n_samples + burn_in, 2))

    for t in range(n_samples + burn_in):
        x1 = rng.normal(rho * x2, cond_std)
        x2 = rng.normal(rho * x1, cond_std)
        samples[t] = (x1, x2)

    return samples[burn_in:]

chain = gibbs_bivariate_normal()
print("Empirical Mean:", np.round(chain.mean(axis=0), 3))
print("Empirical Corr:", np.round(np.corrcoef(chain.T)[0, 1], 3))
```

---

### 4. Stochastic Gradient Langevin Dynamics (`torch`)

A minimal SGLD step for Bayesian regression/classification that injects calibrated Gaussian noise into minibatch gradient updates to collect posterior weight samples.

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

# Synthetic regression data
X = torch.linspace(-3, 3, 200).unsqueeze(1)
y = torch.sin(X) + 0.15 * torch.randn_like(X)

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 1),
)

N = len(X)
batch_size = 50
prior_std = 1.0
noise_var = 0.15**2
posterior_samples = []

for step in range(1, 1501):
    # Decaying step size satisfying Robbins-Monro conditions
    lr = 1e-3 * (step ** -0.55)
    idx = torch.randperm(N)[:batch_size]
    xb, yb = X[idx], y[idx]

    model.zero_grad()
    pred = model(xb)

    # Negative log-likelihood + Gaussian prior penalty
    nll = 0.5 * torch.mean((yb - pred) ** 2) / noise_var
    log_prior = sum(0.5 * torch.sum(p**2) / (prior_std**2) for p in model.parameters()) / N
    loss = nll + log_prior
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            grad = N * p.grad
            noise = torch.randn_like(p) * (lr**0.5)
            p.sub_(0.5 * lr * grad - noise)

    # Collect thinned samples after burn-in
    if step > 800 and step % 50 == 0:
        posterior_samples.append([p.detach().clone() for p in model.parameters()])

print(f"Collected {len(posterior_samples)} posterior weight samples via SGLD.")
```