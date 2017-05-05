/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "particle_filter.h"

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  num_particles = 10;

  // Create generator for pseudo random numbers.
  std::default_random_engine gen;

  // Create normal distribution for x, y, theta.
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Resize already intialized vector.
  particles.resize(num_particles);
  for (int i = 0; i < num_particles; i++) {
    particles[i].id = i;
    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen);
    particles[i].weight = 1;
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  // Create normal distribution to account for car not exactly following the
  // velocity and yaw_rate measured.
  std::default_random_engine gen;
  // Center on zero since this noise will be added to particles updated
  // position calculation.
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  // Update new particles position based on velocity and raw rate and also and
  // Gaussian noise.
  for (int i = 0; i < particles.size(); i++) {
    particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t)
                      - sin(particles[i].theta)) + dist_x(gen);
    particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta
                      + yaw_rate*delta_t)) + dist_y(gen);
    // Okay to use theta in above calculations since it doesn't get updated till here.
    particles[i].theta += yaw_rate*delta_t + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html



  for (int i = 0; i < particles.size(); i++) {
    // Transform vehcile observations to current particles location with map
    // coordinates. Set id to zero since no match has been given yet to map
    // landmark.
    std::vector<LandmarkObs> transformed_observations = observations;

    for (int j = 0; j < observations.size(); j++) {
      transformed_observations[j].id = 0;
      transformed_observations[j].x = observations[j].x*cos(particles[i].theta) +
                                      observations[j].y*sin(particles[i].theta) +
                                      particles[i].x;
      transformed_observations[j].y = observations[j].x*sin(particles[i].theta) +
                                      observations[j].y*cos(particles[i].theta) +
                                      particles[i].y;
    }

    // For each observation find the closest map landmark and set as its id.
    // To reduce number of map landmarks that need to be searched, only search those
    // map landmarks within sensor_range + some error.
    std::vector<LandmarkObs> landmarks_in_sensor_range;

    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      float xf = map_landmarks.landmark_list[j].x_f;
      float yf = map_landmarks.landmark_list[j].y_f;
      int id_i = map_landmarks.landmark_list[j].id_i;
      double sensor = sensor_range*1.1; // Add 10% for sensor error.


      if ( (std::abs(particles[i].x - xf) < sensor) && (std::abs(particles[i].y - yf) < sensor) ) {
        LandmarkObs templandmark;
        templandmark.x = xf;
        templandmark.y = yf;
        templandmark.id = id_i;

        landmarks_in_sensor_range.push_back(templandmark);
      }
    }
    // Match vehicles observations transformed to map coordinates, with all the
    // landmarks within sensor range of the particle.
    dataAssociation(landmarks_in_sensor_range, transformed_observations);

    // Update weights for each particle. Weights are the product of all differences
    // between vehicle measurement and map landmark for each particle with some
    // added math.

    if (landmarks_in_sensor_range.size() == 0) {
      if (observations.size() != 0) {   // If there are no landmarks in sensor range
        particles[i].weight = 0;        // and there are observations, set weight
      }                                 // to zero.
      // If observations == 0 then weight will remain unchanged.
    } else {
      double total_weight;
      double sigma_xx = pow(std_landmark[0], 2);
      double sigma_yy = pow(std_landmark[1], 2);
      for (int j = 0; j < transformed_observations.size(); j++) {
        // Calculate difference between matched vehicle measurement and map landmark.
        double diff_x = transformed_observations[j].x -
                        landmarks_in_sensor_range[transformed_observations[j].id].x;
        double diff_y = transformed_observations[j].y -
                        landmarks_in_sensor_range[transformed_observations[j].id].y;

        // Calculate numerator and denominator of weight fuction.
        double numerator = exp(-0.5 * (pow(diff_x, 2) * pow(sigma_xx, -1) +
                           pow(diff_y, 2) * pow(sigma_yy, -1)) );
        double denominator = sqrt(2 * M_PI * sigma_xx * sigma_yy);

        double current_weight = numerator / denominator;

        if (j == 0) {  // First weight, iniialize total weight.
          total_weight = current_weight;
        } else {  // Total weight of particle is sum of weight for each measurement.
          total_weight *= current_weight;
        }
      }
      // Update particles weight.
      particles[i].weight = total_weight;
    } // NOTE: This solution does not update weights when there are no observed
      // measurements but there are landmarks within sensor range of a particle.

  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}
