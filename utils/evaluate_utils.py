import csv


def evaluate(matches, ground_truth, type='amazon-google'):

    # Step 2: Compute Precision and Recall
    total_correct = 0
    total_predicted = 0
    total_true = len(ground_truth)

    groundTruthMatches = []
    if type == 'amazon-google':
        fieldnames = ['googleId', 'predictedId', 'check']

        for google_id, predicted_amazon_ids in matches.items():
            true_amazon_id = ground_truth.get(google_id)
            if true_amazon_id is None:
                # No ground truth for this Google ID, skip
                continue

            total_predicted += len(predicted_amazon_ids)

            for predicted_amazon_id in predicted_amazon_ids:
                groundTruthMatches.append(
                    {
                        'googleId': google_id,
                        'predictedId': predicted_amazon_id,
                        'check': true_amazon_id == predicted_amazon_id
                    }
                )
                if true_amazon_id == predicted_amazon_id:
                    total_correct += 1  # We found the true match!

        with open('ground_truth_matches.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(groundTruthMatches)
    elif type == 'songs':
        fieldnames = ['song2Id', 'predictedSong1Id', 'check']

        for song2_id, predicted_song1_ids in matches.items():
            true_song1_id = ground_truth.get(song2_id)
            if true_song1_id is None:
                # No ground truth for this Google ID, skip
                continue

            total_predicted += len(predicted_song1_ids)

            for predicted_song1_id in predicted_song1_ids:
                groundTruthMatches.append(
                    {
                        'song2Id': song2_id,
                        'predictedSong1Id': predicted_song1_id,
                        'check': true_song1_id == predicted_song1_id
                    }
                )
                if true_song1_id == predicted_song1_id:
                    total_correct += 1  # We found the true match!

        with open(f"./out-data/ground_truth_matches-{type}.csv", 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(groundTruthMatches)

    # Precision = How many of our predictions were correct
    precision = total_correct / total_predicted if total_predicted > 0 else 0.0

    # Recall = How many of the true matches we found
    recall = total_correct / total_true if total_true > 0 else 0.0

    # f-1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"Type: {type}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")