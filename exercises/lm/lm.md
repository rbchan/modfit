Exercise I: Fitting a linear model using maximum likelihood and Gibbs sampling
================
Richard Chandler
Warnell School of Forestry and Natural Resources
University of Georgia
<rchandler@warnell.uga.edu>

\date{\today}


# Introduction

These assignments are meant for new graduate students in my lab. They
assume that students have some basic familiarity with statistical
inference, but most students will need help getting through the first
few exercises before they are comfortable with the process of writing
likelihood equations and joint posterior distributions.

There are likely to be mistakes in these exercises. Feel free to let
me know if you find any, and I will try to correct them.


# The model

Simple linear regression is one of the most basic statistical
models. There are several ways to describe the model. Here is one
option:
$$y_i \sim \mathrm{Norm}(\mu_i,\sigma^2)$$
where $\mu_i = \beta_0 + \beta_1 x_i$ and $x_i$ is a continuous covariate.

Here's another:
$$y_i = \beta_0 + \beta_1 x_i + \epsilon_i$$
where $\epsilon_i \sim \mathrm{Norm}(0, \sigma^2)$.

# Assignment


1. Simulate a dataset in **R** using $\beta_0=-1$, $\beta_1=1$,
    $\sigma^2=4$. Let $n=100$ be the sample size, and generate a
    single continuous covariate from a standard normal distribution.
2. Write the equation for the likelihood in \LaTeX.
3. Obtain the MLEs in **R** by minimizing the negative log-likelihood
4. Write the joint posterior distribution in \LaTeX
5. Describe a Gibbs sampler for obtaining posterior samples
6. Implement the Gibbs sampler in **R** using the dataset that
    you simulated earlier.
7. Use **JAGS** to fit the model.




