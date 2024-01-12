import re

log_line = "Epoch: [6] [ 50/2208] eta: 0:06:45 lr: 0.010000 loss: 0.2625 (0.2609) bbox_regression: 0.1634 (0.1642) classification: 0.0842 (0.0887) keyp_regression: 0.0083 (0.0080) time: 0.1812 data: 0.0046 max mem: 3205"

pattern = re.compile(
    r"loss: (\d+\.\d+) \((\d+\.\d+)\) bbox_regression: (\d+\.\d+) \((\d+\.\d+)\) classification: (\d+\.\d+) \((\d+\.\d+)\) keyp_regression: (\d+\.\d+) \((\d+\.\d+)\)"
)

match = pattern.search(log_line)

if match:
    (
        total_loss,
        avg_loss,
        bbox_regression_loss,
        avg_bbox_regression_loss,
        classification_loss,
        avg_classification_loss,
        keyp_regression_loss,
        avg_keyp_regression_loss,
    ) = map(float, match.groups())

    print("Total Loss:", total_loss)
    print("Average Loss:", avg_loss)

    print("Bbox Regression Loss:", bbox_regression_loss)
    print("Average Bbox Regression Loss:", avg_bbox_regression_loss)

    print("Classification Loss:", classification_loss)
    print("Average Classification Loss:", avg_classification_loss)

    print("Keyp Regression Loss:", keyp_regression_loss)
    print("Average Keyp Regression Loss:", avg_keyp_regression_loss)
