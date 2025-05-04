def evaluate(matches, ground_truth):

    # Step 2: Compute Precision and Recall
    total_correct = 0
    total_predicted = 0
    total_true = len(ground_truth)

    for google_id, predicted_amazon_ids in matches.items():
        true_amazon_ids = ground_truth.get(google_id)
        if true_amazon_ids is None:
            # No ground truth for this Google ID, skip
            continue

        total_predicted += len(predicted_amazon_ids)

        for predicted_amazon_id in predicted_amazon_ids:
            if predicted_amazon_id == true_amazon_ids:
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