# Credit Risk Model

## Project Overview

This project develops an end-to-end credit risk modeling pipeline for buy-now-pay-later services using transaction data.

The objective is to engineer behavioral features, create a proxy credit risk variable, train predictive models, and deploy the resulting model as an API service.

---

# Credit Scoring Business Understanding

## Basel II and Interpretability

Basel II emphasizes risk measurement, transparency, and documentation. Therefore, credit scoring models must be interpretable and explainable because financial institutions must justify lending decisions to regulators and customers.

Models that cannot explain predictions create regulatory risks and operational uncertainty.

## Why Proxy Variables Are Necessary

The dataset does not contain a direct default label.

Therefore, behavioral proxies must be created using transaction patterns.

Proxy variables introduce risks because proxy behavior may not perfectly represent true default behavior, potentially causing misclassification and bias.

## Model Trade-Offs

### Interpretable Models

Advantages:

* Easier regulatory compliance
* Explainable decisions
* Easier monitoring

Disadvantages:

* Lower predictive power

### High Performance Models

Advantages:

* Better predictive accuracy
* Captures nonlinear relationships

Disadvantages:

* Reduced interpretability
* Higher regulatory scrutiny

A balanced approach combining interpretability and predictive performance is required.
