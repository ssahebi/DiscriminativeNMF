# DiscriminativeNMF

Code accomanying the papers: 
* Mirzaei, M., Sahebi, S., & Brusilovsky, P. (2020, May). Detecting trait versus performance student behavioral patterns using discriminative non-negative matrix factorization. In The Thirty-Third International Flairs Conference.
* Mirzaei, M., Sahebi, S., & Brusilovsky, P. (2020, December). SB-DNMF: A structure based discriminative non-negativematrix factorization model for detecting inefficient learning behaviors. In The 2020IEEE/WIC/ACMInternational Joint Conference On Web Intelligence And Intelligent Agent Technology, WI-IAT.

-----------------------------------------------------------------------------------------------------------------------------

We use the iterative Gradient Descent optimization algorithm, to minimize the objective function in the paper.

GD.py

The main methods are:

runs the Gradient Descent algorithm to minimize the cost function considering similarities of the patterns
It takes the total number of latent factors k, number of common latent factors kc and number of eopcs
alpha, beta, delta are the parameters of the objective function
use 'grid' for running the grid search and calculates the error for a range of parameters
use 'test' for running the algorithm with specific parameters and build decompose matrices.
the inputs are pattern-student matrices for low and high performance students in 'l.txt' and 'h.txt and similarity matrix in 'nmp.csv'

gd_structure(k, kc, pathin, pathout, alpha, beta, delta, eps, 'test', epoc)

runs the Gradient Descent algorithm to minimize the cost function without considering similarities
It takes the total number of latent factors k, number of common latent factors kc and number of eopcs
alpha and beta are the parameters of the objective function
use 'grid' for running the grid search and calculates the error for a range of parameters
use 'test' for running the algorithm with specific parameters and build decompose matrices.
the inputs are pattern-student matrices for low and high performance students in 'l.txt' and 'h.txt. 

gd_eps_no_structure(k, kc, pathin, pathout, alpha, beta, epoc)

-----------------------------------------------------------------------------------------------------------------------------
The similarity matrix based on the proposed modified levenstein distance measure.

distance.py

calculates the distance between each two patterns based on the modified levenshtein distance and build the distance matrix
create_matrix(path)

normalizes the distance matrix and build similarity matrix
normal_s(path)
