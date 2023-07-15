import numpy as np
import util
import sys
from random import random
import matplotlib.pyplot as plt


sys.path.append("../linearclass")

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = "X"
# Ratio of class 0 to class 1
kappa = 0.1


def eval(model, eval_x, eval_y):
    raw_preds = np.array([model.predict(x) for x in eval_x])
    preds = np.array([round(x, 0) for x in raw_preds])
    numCorrect = 0.0
    tp = 0.0
    tn = 0.0
    fn = 0.0
    fp = 0.0
    for idx, val in enumerate(preds):
        if val == 1 and eval_y[idx] == 1:
            numCorrect += 1.0
            tp += 1.0
        elif val == 0 and eval_y[idx] == 0:
            tn += 1.0
            numCorrect += 1.0
        elif val == 0 and eval_y[idx] == 1:
            fn += 1.0
        else:
            fp += 1.0
    accuracy = round(numCorrect / len(preds), 3)
    a_0 = round(tn / (tn + fp), 3)
    a_1 = round(tp / (tp + fn), 3)
    balanced = 0.5 * (a_0 + a_1)
    return {
        "acc": accuracy,
        "balanced_acc": balanced,
        "a_0": a_0,
        "a_1": a_1,
        "preds": preds,
        "raw_preds": raw_preds,
    }


def main(train_path, validation_path, save_path, plot_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, "naive")
    output_path_upsampling = save_path.replace(WILDCARD, "upsampling")
    plot_path_upsampling = plot_path.replace(WILDCARD, "upsampling")
    plot_path_naive = plot_path.replace(WILDCARD, "naive")

    train_x, train_y = util.load_dataset(train_path, add_intercept=True)
    eval_x, eval_y = util.load_dataset(validation_path, add_intercept=True)

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    model = LogisticRegression(verbose=False)
    theta = model.fit(train_x, train_y)
    eval_results = eval(model, eval_x, eval_y)
    print("EVAL NAIVE REGRESSION\n", eval_results)
    np.savetxt(output_path_naive, np.array([eval_y, eval_results["raw_preds"]]))

    util.plot(eval_x, eval_y, theta, plot_path_naive)

    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    # Part (d): Upsampling minority class
    upsampled_x = train_x
    upsampled_y = train_y
    for idx, val in enumerate(train_y):
        if val == 1:
            for i in range(int(1 / kappa)):
                upsampled_x = np.append(upsampled_x, [train_x[idx]], axis=0)
                upsampled_y = np.append(upsampled_y, val)

    upsampled_model = LogisticRegression(verbose=True)
    theta_u = upsampled_model.fit(upsampled_x, upsampled_y)
    eval_results_u = eval(upsampled_model, eval_x, eval_y)
    print("EVAL UPSAMPLED REGRESSION\n", eval_results_u)
    np.savetxt(output_path_upsampling, np.array([eval_y, eval_results_u["raw_preds"]]))

    util.plot(eval_x, eval_y, theta_u, plot_path_upsampling)

    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    # *** END CODE HERE


if __name__ == "__main__":
    main(
        train_path="/Users/theboss/Documents/GitHub/CS229/pset-1/imbalanced/train.csv",
        validation_path="/Users/theboss/Documents/GitHub/CS229/pset-1/imbalanced/validation.csv",
        save_path="/Users/theboss/Documents/GitHub/CS229/pset-1/imbalanced/imbalanced_X_pred.txt",
        plot_path="/Users/theboss/Documents/GitHub/CS229/pset-1/imbalanced/plot_X.png",
    )
