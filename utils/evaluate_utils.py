import numpy as np
def evaluate(matches, ground_truth):

    # Step 2: Compute Precision and Recall
    total_correct = 0
    total_predicted = 0
    total_true = len(ground_truth)

    for google_id, predicted_amazon_ids in matches.items():
        true_amazon_id = ground_truth.get(google_id)
        if true_amazon_id is None:
            # No ground truth for this Google ID, skip
            continue

        total_predicted += len(predicted_amazon_ids)

        if true_amazon_id in predicted_amazon_ids:
            total_correct += 1  # We found the true match!

    # Precision = How many of our predictions were correct
    precision = total_correct / total_predicted if total_predicted > 0 else 0.0

    # Recall = How many of the true matches we found
    recall = total_correct / total_true if total_true > 0 else 0.0

    # f-1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


def evaluate_multiple_matches(matches, ground_truth):
    """
    Evaluate retrieval results.

    Args:
      matches: dict[qid] → list of predicted IDs (length = top_k)
      ground_truth: dict[qid] → list of true IDs

    Returns:
      dict with keys 'precision', 'recall', 'f1' giving the
      macro-averaged scores over all qids in ground_truth.
    """
    print("Evaluating matches...")
    precisions = []
    recalls = []
    f1s = []
    ids = np.arange(1, 1025)
    for qid, true_list in ground_truth.items():
        pred_list = matches.get(qid, [])
        true_set = set(true_list)
        pred_set = set(pred_list)
        print(f"qid: {qid}, true_list: {true_list}, pred_list: {pred_list}")

        # true positives
        tp = len(true_set & pred_set)

        # avoid division by zero
        precision = tp / len(pred_set) if pred_set else 0.0
        recall    = tp / len(true_set) if true_set else 0.0
        if precision + recall:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    n = len(precisions)
    prec_avg = sum(precisions) / n
    rec_avg  = sum(recalls)    / n
    f1_avg   = sum(f1s)        / n

    print(f"Precision: {prec_avg:.4f}")
    print(f"Recall:    {rec_avg:.4f}")
    print(f"F1 Score:  {f1_avg:.4f}")
