//============================================================================
// Name        : vffnet2.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <ctime>
#include <armadillo>

#include "MNISTLoader.h"

using namespace std;

arma::Col<double> sigmoid(arma::Col<double> z) {
	arma::Col<double> ones(z.size(), arma::fill::ones);
	return ones / (ones+arma::exp(-z));
}

arma::Col<double> sigmoid_prime(arma::Col<double> z) {
	arma::Col<double> ones(z.size(), arma::fill::ones);
	return sigmoid(z) % (ones - sigmoid(z));
}

arma::Col<double> softmax(arma::Col<double> z) {
	arma::Col<double> exp_z = arma::exp(z);
	return exp_z / arma::accu(exp_z);
}

arma::Col<double> label2onehot(unsigned char label, int domain_size) {
	arma::Col<double> result(domain_size, arma::fill::zeros);
	result[static_cast<int>(label)] = 1.0;
	return result;
}

int main() {

	/** Load Data */
	MNIST_Loader loader;
	std::vector<arma::Col<double>> train_images = loader.load_images(
			"./data/train-images-idx3-ubyte");
	std::vector<arma::Col<double>> val_images = loader.load_images(
			"./data/t10k-images-idx3-ubyte");
	std::vector<unsigned char> train_labels = loader.load_labels(
			"./data/train-labels-idx1-ubyte");
	std::vector<unsigned char> val_labels = loader.load_labels(
			"./data/t10k-labels-idx1-ubyte");

	int NUM_PIXELS = 784;
	int HIDDEN_LAYER_SIZE = 100;
	int OUTPUT_SIZE = 10;
	double ETA = 0.3;
	int NUM_ITERS = 5000;

	arma::Mat<double> w1 = arma::Mat<double>(HIDDEN_LAYER_SIZE, NUM_PIXELS, arma::fill::randn) / 28.0;
	arma::Col<double> b1 = arma::Col<double>(HIDDEN_LAYER_SIZE, arma::fill::zeros);
	arma::Mat<double> w2 = arma::Mat<double>(OUTPUT_SIZE, HIDDEN_LAYER_SIZE, arma::fill::randn) / 10.0;
	arma::Col<double> b2 = arma::Col<double>(OUTPUT_SIZE, arma::fill::zeros);

	cout << "TRAINING OVER " << NUM_ITERS << " SAMPLES" << endl;

	clock_t start = clock();

	int randIndex;
	arma::Col<double> x;
	unsigned char y;
	arma::Col<double> z1, y1, z2, y2;
	arma::Col<double> output;
	arma::Col<double> err2, dL_dz2, err1, dL_dz1;

	for (int i = 0; i < NUM_ITERS; i++) {
		randIndex = std::rand() % train_images.size();
		x = train_images[randIndex];
		y = train_labels[randIndex];

		z1 = w1 * x + b1;
		y1 = sigmoid(z1);
		z2 = w2 * y1 + b2;
		y2 = z2;

		output = softmax(y2);

		err2 = output - label2onehot(y, OUTPUT_SIZE);
		dL_dz2 = err2;
		err1 = w2.t() * dL_dz2;
		dL_dz1 = err1 % sigmoid_prime(z1);

		w2 -= ETA*dL_dz2*y1.t();
		b2 -= ETA*dL_dz2;
		w1 -= ETA*dL_dz1*x.t();
		b1 -= ETA*dL_dz1;
	}

	double training_time = (clock() - start) / (double) CLOCKS_PER_SEC;

	cout << "TRAINING TIME: " << training_time << endl;


	cout << "VALIDATING" << endl;
	double total_correct = 0;
	unsigned char prediction;
	for (int i=0; i < val_images.size(); i++) {
		x = val_images[i];
		y = val_labels[i];

		z1 = w1 * x + b1;
		y1 = sigmoid(z1);
		z2 = w2 * y1 + b2;
		y2 = z2;

		output = softmax(y2);

		prediction = output.index_max();

		if (static_cast<int>(prediction) == static_cast<int>(y)) {
			++total_correct;
		}
	}
	cout << "fraction correct: " << static_cast<double>(total_correct / val_images.size()) << endl;

	return 0;
}

