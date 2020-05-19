# Notes on Kernelized Stein Discrepancy

## Relation between Stein and KL-divergence

Let $k$ be a positive definite kernel on $\mathbb R^d$ with RKHS $H$ and write $H^d = H \times \dots \times H$.

Let $\phi \in H^d$, and set $T_\varepsilon = \text{id} + \varepsilon \phi$. Further let $q$ and $p$ be distributions on $\mathbb R^d$. Then (as we know from the SVGD paper),
$$- \frac{d}{d\varepsilon} \text{KL}(q_{T_\varepsilon} \ \Vert \ p) \Big \vert_{\varepsilon = 0} = E\big[ \mathcal A_p^T [\phi](x) \big]$$,

where $\mathcal A$ is the Stein operator. The direction $\phi^* \in H^d$ in which the gradient is maximal is given by $\phi^*_{q,p}$ as defined in the SVGD paper. We have the following:

$$
\begin{align*}
- \sup_{\Vert \phi \Vert_{H^d} \leq 1} \frac{d}{d\varepsilon} \text{KL}(q_{T_\varepsilon} \ \Vert \ p) \Big \vert_{\varepsilon = 0} &= E\big[ \mathcal A_p^T [\phi^*_{q,p}](x) \big] \cdot \frac{1}{\Vert \phi^*_{p,q} \Vert_{H^d}} \\
&= \big \Vert \phi^*_{p,q} \big \Vert_{H^d} \\
&= \text{KSD}(q \ \Vert \ p)
\end{align*}
$$

In other words, one step of SVGD reduces the KL divergence by approximately $\varepsilon \cdot \text{KSD}(q \ \Vert \ p)$, where $\varepsilon$ is the step size.

Writing $k$-KSD for the Stein discrepancy computed using kernel $k$, this means that in the context of SVGD the $k$-KSD is best understood as the magnitude of the reduction in KL divergence after taking an SVGD step computed using $k$.

In particular, this means that $k$-KSDs computed using different $k$s are in fact comparable: they all measure the reduction in (a linear approximation of) the KL divergence after one step of SVGD.

## Proposed scheme for learning the kernel parameters

This is a proposal for a 'greedy' algorithm that at each step wants to maximize the reduction in KL-divergence. We want to choose the kernel $k$ such that the $k$-KSD is maximal before each SVGD step.

Concretely it could look something like this:

Initialize particles $X_1, \dots, X_n \sim q$ and bandwidth $h_0 = 1$. Then repeat:

1. Write $\hat q$ for the empirical distribution of the current particles $X_1, \dots, X_n$. Update the bandwidth $$ h_\text{new} = h_\text{old} + \eta \nabla_h \text{KSD}_{h_\text{old}}(\hat q \ \Vert \ p)$$. 
2. Compute SVGD update step using new bandwidth $h_\text{new}$: $$X_i = X_i + \phi_{\hat q, p}^*(X_i)$$



I'll set up some experiments and see how that goes. Let me know if you have any thoughts on the idea.