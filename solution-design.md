# Solution Design: Delivery Time & Transit Delay Estimator

## What Is Decided

| Area                         | Decision                                                                               | Why                                                                                          |
| ---------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Inputs**                   | Package weight, origin-destination distance, carrier tier, warehouse workload          | The four drivers named in the project brief                                                  |
| **Output**                   | Transit time estimate plus a confidence interval                                       | A range is more honest than a single number, and it lets us set better customer expectations |
| **Approach**                 | Regression model: it learns from past orders to predict a continuous duration          | The target is a number of days, not a category                                               |
| **Model development**        | Google Colab notebook                                                                  | Free compute, easy to share, suited to experimentation                                       |
| **User interface**           | Streamlit web app                                                                      | Quick to build, interactive, fine for an internal dispatch tool                              |
| **Hand-off between the two** | One saved model file                                                                   | Keeps the app simple: it only loads the file and predicts                                    |
| **Success measure**          | Average error in days, and how often the actual delivery falls inside the stated range | Both are easy to explain to non-technical stakeholders                                       |

## There are two stages:

1. **Build stage (Colab):** prepare data, train several models, pick the best, and add the confidence range.
2. **Use stage (Streamlit):** a dispatcher types in order details and gets an answer immediately.

## How We Will Judge the Model

- **Better than a naive guess.** We compare against simply predicting the average transit time. The model has to beat it clearly.
- **Accurate in plain terms.** Reported as "typically off by X days".
- **Honest ranges.** If we say 90% confidence, about 90% of real deliveries should land inside the range.

## Open Decision: The Dataset

The dataset has not been chosen yet. This was deliberately deferred.

**The constraint:** we need data containing all four inputs plus the actual delivery duration. No single public dataset is confirmed to have all of them.

**Options at the time of decision:**

| Option                                         | Pros                                                    | Cons                                                                            |
| ---------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Real e-commerce data, with some fields derived | Realistic patterns                                      | Some inputs (e.g. carrier tier, warehouse workload) may need to be approximated |
| Fully synthetic data                           | Has all four inputs by construction; no download issues | Model learns our assumptions, not real-world behaviour                          |
| Hybrid (real base plus synthetic gaps)         | Balanced realism and completeness                       | Needs clear documentation of what is real                                       |

**Impact on the design:** none. The rest of the solution is built around a fixed data layout, so whichever option is chosen plugs in without redesign.
