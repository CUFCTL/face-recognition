#!/usr/bin/python
# Create plots for log files from experiments.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os

def load_logfile(fname):
	fp = open(fname, "rt")
	X = [[], [], [], []]

	for line in fp:
		data = line.split()

		for i in xrange(4):
			X[i].append(float(data[i]))

	fp.close()

	return X

def save_plot(fname, plots, ylim, xlabel, ylabel):
	for p in plots:
		plt.plot(p[0], p[1], p[2])

	plt.ylim(ylim)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)

	if len(plots) > 1:
		plt.legend([p[3] for p in plots])

	plt.savefig(fname)
	plt.clf()

if not os.path.exists("figures"):
	os.mkdir("figures", 0755)

X_pca_n1 = load_logfile("logs/feret-pca-n1.log")
X_lda_n1 = load_logfile("logs/feret-lda-n1.log")
X_lda_n2 = load_logfile("logs/feret-lda-n2.log")
X_ica_n1 = load_logfile("logs/feret-ica-n1.log")
X_ica_n2 = load_logfile("logs/feret-ica-n2.log")
X_knn_k  = load_logfile("logs/feret-knn-k.log")

save_plot("figures/pca_acc.eps",   [[X_pca_n1[0], X_pca_n1[1], "k"]], (0, 100), "Hyperparameter value", "Accuracy (%)")
save_plot("figures/pca_train.eps", [[X_pca_n1[0], X_pca_n1[2], "k"]], (0, 120), "Hyperparameter value", "Training time (s)")
save_plot("figures/pca_pred.eps",  [[X_pca_n1[0], X_pca_n1[3], "k"]], (0,  20), "Hyperparameter value", "Prediction time (s)")

save_plot("figures/lda_acc.eps",   [[X_lda_n1[0], X_lda_n1[1], "k--", "lda_n1"], [X_lda_n2[0], X_lda_n2[1], "k", "lda_n2"]], (0, 100), "Hyperparameter value", "Accuracy (%)")
save_plot("figures/lda_train.eps", [[X_lda_n1[0], X_lda_n1[2], "k--", "lda_n1"], [X_lda_n2[0], X_lda_n2[2], "k", "lda_n2"]], (0, 120), "Hyperparameter value", "Training time (s)")
save_plot("figures/lda_pred.eps",  [[X_lda_n1[0], X_lda_n1[3], "k--", "lda_n1"], [X_lda_n2[0], X_lda_n2[3], "k", "lda_n2"]], (0,  20), "Hyperparameter value", "Prediction time (s)")

save_plot("figures/ica_acc.eps",   [[X_ica_n1[0], X_ica_n1[1], "k--", "ica_n1"], [X_ica_n2[0], X_ica_n2[1], "k", "ica_n2"]], (0, 100), "Hyperparameter value", "Accuracy (%)")
save_plot("figures/ica_train.eps", [[X_ica_n1[0], X_ica_n1[2], "k--", "ica_n1"], [X_ica_n2[0], X_ica_n2[2], "k", "ica_n2"]], (0, 120), "Hyperparameter value", "Training time (s)")
save_plot("figures/ica_pred.eps",  [[X_ica_n1[0], X_ica_n1[3], "k--", "ica_n1"], [X_ica_n2[0], X_ica_n2[3], "k", "ica_n2"]], (0,  20), "Hyperparameter value", "Prediction time (s)")

save_plot("figures/knn_acc.eps",   [[X_knn_k[0], X_knn_k[1], "k"]], (0, 100), "Hyperparameter value", "Accuracy (%)")
save_plot("figures/knn_train.eps", [[X_knn_k[0], X_knn_k[2], "k"]], (0, 120), "Hyperparameter value", "Training time (s)")
save_plot("figures/knn_pred.eps",  [[X_knn_k[0], X_knn_k[3], "k"]], (0,  20), "Hyperparameter value", "Prediction time (s)")
