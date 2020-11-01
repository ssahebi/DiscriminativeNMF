# DiscriminativeNMF
We use the iterative Gradient Descent optimization algorithm, illustrated in Algorithm 1, tominimize the objective function (6.4) according to the following gradients 
\begin{equation}
    \begin{split}
        \frac{\partial L}{\partial W_{1,c} } = 
        -2\gamma (X_1-W_1H_1^T)H^T_{1,c}+\\ 2\alpha (W_{1,c}-W_{2,c}) + 
        2W_{1,c}-2\times 2\delta(S-\varepsilon WW^T)W_{1,c}
    \end{split}
    \label{eq:W1c}
\end{equation}
