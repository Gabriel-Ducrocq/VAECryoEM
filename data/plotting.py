import numpy as np
import argparse 
import matplotlib.pyplot as plt

parser_arg = argparse.ArgumentParser()
parser_arg.add_argument('--true_distances', type=str, required=True)
parser_arg.add_argument('--predicted_distances', type=str, required=True)
parser_arg.add_argument('--output_path', type=str, required=True)
args = parser_arg.parse_args()
true_distances = args.true_distances
predicted_distances = args.predicted_distances
output_path = args.output_path

true_distances = np.load(true_distances)
predicted_distances = np.load(predicted_distances)[:, 0]


plt.hist(true_distances, bins=30, density=True, label="True distances", alpha=0.5)
plt.hist(predicted_distances, bins=30, density=True, label="Predicted distances", alpha=0.5)
plt.xlabel("Distances in Å")
plt.legend(loc="upper right")
plt.savefig(output_path + "histogram.png")
plt.close()

plt.scatter(true_distances, predicted_distances, s=0.01)
plt.xlabel("True distance in Å")
plt.ylabel("Predicted distance in Å")
plt.title("Predicted vs True distances")
plt.savefig(output_path + "scatter_plot.png")

