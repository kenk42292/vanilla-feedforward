/*
 * MNISTLoader.h
 *
 *  Created on: Aug 17, 2016
 *      Author: ken
 */

#ifndef MNISTLOADER_H_
#define MNISTLOADER_H_

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <armadillo>
using std::ifstream;
using std::string;
using std::cout;
using std::endl;

class MNIST_Loader {
private:
	int reverseInt(int i);
public:
	MNIST_Loader();
	virtual ~MNIST_Loader();
	std::vector< arma::Col<double> > load_images(string images_path);	// Return 784 long vector
	std::vector<unsigned char> load_labels(string labels_path);	// Return label as a single unsigned char
};

#endif /* MNISTLOADER_H_ */
