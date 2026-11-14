
Domain: Support

# Swiggy Support Decision Intelligence

IIT Mandi Data Science and Machine Learning Capstone Project.

This project analyses customer-support ticket escalation using statistical methods, machine learning, clustering, ensemble learning and neural-network architecture calculations.

## How to run

Clone the repository and enter the project directory:

```bash
git clone https://github.com/parnitha-sv/Swiggy-Support-Decision-Intelligence.git
cd Swiggy-Support-Decision-Intelligence
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the scripts in the following order:

```bash
python generate_data.py
python part1.py
python part2.py
python part3.py
python part4.py
```

The generated datasets are stored in `data/` and the analysis results are stored in `outputs/`.


## Part 1 findings

Part 1 analyses 800 customer-support records using statistical estimation, hypothesis testing and regression techniques.

### R1.1 Estimation and hypothesis testing

The sample mean resolution time was 8.9382 minutes, compared with the planning assumption of 8.0 minutes. The 95% confidence interval was [8.3026, 9.5738] minutes.

The calculated z-statistic was 2.8930, which exceeds the two-tailed critical value of 1.96 at the 5% significance level. Therefore, the null hypothesis was rejected. There is sufficient statistical evidence that the average resolution time differs from the assumed 8.0 minutes.

A Type I error in this context would mean concluding that the actual average resolution time differs from 8.0 minutes when, in reality, the true population mean is 8.0 minutes.

The difference between the observed mean and the planning assumption is 0.9382 minutes. This is a relatively modest difference for an individual ticket, but it may become operationally meaningful across a large number of support tickets.

### R1.2 Sample size estimation

The required sample size was 1293 records to estimate the mean within a margin of error of 0.5 minutes at 95% confidence. Only 800 records were available, which is 493 fewer than required.

This indicates that the available dataset does not meet the desired precision for the planning team's estimation target. Additional records would be needed to achieve the specified margin of error.

### R1.3 Gradient descent

Gradient descent reduced the cost from 163.9178 at iteration 0 to 40.3698 at iteration 100 and remained at 40.3698 at iteration 500.

The difference between the gradient-descent slope and the closed-form regression slope was 0.0000.

Gradient descent reached the closed-form solution because the linear regression cost function is convex, and the standardized predictor with the selected learning rate allowed the full-batch updates to converge towards the minimum.

### R1.4 Residual analysis and log transformation

The raw regression model produced a residual standard deviation ratio of 3.7525 between the upper and lower fitted-value groups. This indicates a funnel pattern, where residual variability increases with fitted values.

After applying a log-log transformation, the residual standard deviation ratio decreased to 1.0396. This indicates that the log transformation substantially reduced the funnel pattern and made the residual spread more consistent across the two groups.

The log-log model achieved an R-squared value of 0.6565 and an adjusted R-squared of 0.6561.

### R1.5 Multiple regression


The multiple regression model achieved an R-squared value of 0.7425 and an adjusted R-squared of 0.7403.

The estimated coefficients were:

- X1: 1.1576
- X2: -0.0074
- X3: 0.0940
- X4: 0.1067
- X5: 0.0028
- X6: 0.1518
- X7: 0.0172

Using an absolute coefficient magnitude greater than 0.05 as a descriptive cutoff, X1, X3, X4 and X6 have coefficients exceeding this threshold.

The multiple regression adjusted R-squared of 0.7403 was higher than the log-log model adjusted R-squared of 0.6561, an improvement of 0.0842. This indicates better explanatory performance after accounting for model complexity. Coefficient magnitudes should be interpreted with consideration of the different predictor scales.

## Part 2 findings

Part 2 implements logistic regression and KNN to predict support-ticket escalation.

The dataset was divided into 560 training records and 240 test records. There were 119 positive training records and 51 positive test records.

Logistic regression achieved:

- Test binary cross-entropy: 0.3818.
- At threshold 0.5, accuracy was 0.8042 and F1 was 0.5053.
- At threshold 0.3, precision was 0.4634, recall was 0.7451 and F1 was 0.5714.

The selected KNN value was K = 5. Its test accuracy was 0.7583 and F1 was 0.3696.

Cost-based threshold analysis selected a threshold of 0.15. Using the specified cost rule, the training cost was INR 12,150 and test cost was INR 6,050. The test cost at threshold 0.5 was INR 11,800.

## Part 3 findings

Part 3 performs customer segmentation using K-means clustering and hierarchical clustering, followed by Principal Component Analysis (PCA).

The elbow method selected three clusters.

| Segment | Size | Outcome rate | Mean value |
|---|---:|---:|---:|
| A | 322 | 0.0186 | 3.5976 |
| B | 255 | 0.2353 | 17.1248 |
| C | 223 | 0.4664 | 7.2883 |

The three segments show different customer characteristics and escalation rates.

PCA produced five eigenvalues with a total of 5.0000. The first three principal components explained 86.16% of the total variance. Therefore, three components were sufficient to retain more than 80% of the variance.

The hierarchical clustering analysis also compared single, complete and average linkage methods.

## Part 4 findings

Part 4 compares decision-tree splits, classification models, voting methods, AdaBoost and neural-network architectures.

### R4.1 Decision-tree split analysis

Four candidate splits were evaluated using entropy, Gini impurity, information gain and Gini gain.

The root entropy was 0.7462. S4 (X1 <= 24) was the best split because it produced the greatest reduction in impurity among the four candidates, resulting in the highest information gain and Gini gain.

Growing a decision tree until every leaf is pure can cause overfitting because the tree may learn noise and specific patterns in the training data rather than generalisable relationships. A maximum-depth constraint can prevent the tree from growing excessively deep.

### R4.2 Model comparison and voting

The five models were evaluated using precision, recall and F1-score on the 240 test records.

| Model | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Logistic regression | 0.4634 | 0.7451 | 0.5714 |
| KNN (K=5) | 0.4146 | 0.3333 | 0.3696 |
| Decision-tree stump | 0.4333 | 0.7647 | 0.5532 |
| Hard voting | 0.4375 | 0.6863 | 0.5344 |
| Soft voting | 0.4368 | 0.7451 | 0.5507 |

Logistic regression is the recommended model for the initial production system because it achieved the highest F1-score of 0.5714 with a recall of 0.7451. Although the decision-tree stump achieved a slightly higher recall of 0.7647, its F1-score was lower at 0.5532. Neither hard voting nor soft voting improved the F1-score over logistic regression, indicating that combining these models did not provide an overall improvement.

### R4.3 AdaBoost

Two rounds of AdaBoost were implemented manually.

- Round 1 weighted error: 0.2500.
- Round 1 alpha: 0.5493.
- Round 2 weighted error: 0.1667.
- Round 2 alpha: 0.8047.
- Ensemble accuracy: 0.7500.

The two misclassified points after round 1 normalization carried a combined weight of 0.5000.

Round 2 assigns greater importance to h2 because its weighted error is lower than that of h1.

### R4.4 Architecture analysis

The out-of-bag fraction for 560 bootstrap samples was 0.3676.

Model A: Image classification

| CNN layer | Output shape | Parameters |
|---|---|---:|
| Conv1 | 28 x 28 x 6 | 156 |
| Max-pool 1 | 14 x 14 x 6 | 0 |
| Conv2 | 10 x 10 x 16 | 2416 |
| Max-pool 2 | 5 x 5 x 16 | 0 |
| Flatten | 400 | 0 |
| Dense 1 | 32 | 12832 |
| Dense 2 | 10 | 330 |

The CNN total is 15,734 parameters. The fully connected alternative contains 13,002 parameters.

For a 128 x 128 x 3 input and 1,024 neurons, the fully connected layer has 50,332,672 parameters, compared with 1,792 parameters for a convolutional layer with 64 filters of size 3 x 3.

Model B: Recurrent neural network

| Weight or bias | Shape / size |
|---|---:|
| W_xh | 128 x 64 |
| W_hh | 128 x 128 |
| W_hy | 16 x 128 |
| b_h | 128 |
| b_y | 16 |

The total RNN parameter count is 26,768.

The RNN parameter count does not increase with sequence length because the same weights are reused at every time step.

A tanh hidden activation and softmax output with categorical cross-entropy loss are suitable for the 16-class task.

Very deep plain networks can suffer from vanishing gradients, preventing early layers from learning effectively. Skip connections provide shorter gradient paths and help information flow through deeper networks.

## Model-choice memo

To: Head of Customer-Support Operations

Subject: Recommended escalation prediction model

I recommend logistic regression as the initial production model for structured support-ticket escalation prediction. At the Task 4.2 threshold of 0.30, it achieved an F1 score of 0.5714 and recall of 0.7451, outperforming KNN (F1 0.3696), hard voting (0.5344) and soft voting (0.5507). Combining models did not improve the F1 score.

For production, I recommend the cost-selected threshold of 0.15 from Task 2.8. Its test cost was INR 6,050 under the specified cost rule of INR 50 per false positive and INR 400 per false negative. This is the logistic-regression comparator for operational decision-making.

The evidence is limited by the synthetic dataset and the relatively small number of positive test records. The model should be validated on real historical tickets before deployment.

Model B, the ticket-text RNN, is worth exploring as a future complementary system, but it should not delay validation and deployment of the structured-data model.

## Limitations

1. The dataset is synthetic and may not represent the actual distribution of customer-support tickets.
2. The evaluation uses a single training/test split, so performance may vary with a different split.
3. The test set contains only 51 positive escalation records, limiting the certainty of recall and F1 estimates.
4. The analysis does not include external validation, real-world deployment or monitoring for data drift.
5. Model A and Model B are evaluated through architecture and parameter calculations only; their predictive performance has not been tested through model training.
6. The results do not establish causal relationships between customer attributes and ticket escalation.
