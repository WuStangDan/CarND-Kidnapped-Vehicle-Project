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
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

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
    particles[i].id = 0; // not used.
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

  // If yaw rate is too small, assume it's zero due to being used in division
  // which will change the equations.
  double min_yaw_rate = 1e-3;

  for (int i = 0; i < particles.size(); i++) {

    if (std::fabs(yaw_rate) < min_yaw_rate) {
      particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
      particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
      particles[i].theta += dist_theta(gen);
    } else {
      particles[i].x += (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t)
                        - sin(particles[i].theta)) + dist_x(gen);
      particles[i].y += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta
                        + yaw_rate*delta_t)) + dist_y(gen);
      // Okay to use theta in above calculations since it doesn't get updated till here.
      particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {

	  for (int i = 0; i < observations.size(); i++) {
	    double min_squared_dist;
	    for (int j = 0; j < predicted.size(); j++) {
	      double distance_x = observations[i].x - predicted[j].x;
	      double distance_y = observations[i].y - predicted[j].y;

	      double squared_dist = pow(distance_x,2) + pow(distance_y, 2);

	      if (j == 0) { // On first run intialize min distance.
	        min_squared_dist = squared_dist;
	        observations[i].id = j;
	      } else {
	        if (min_squared_dist > squared_dist) {
	          min_squared_dist = squared_dist;
	          observations[i].id = j; // j or indices for predicted (landmarks_in_sensor_range)
	        }                         // will be used to call landmark closest to observation.
	      }
	    }

	  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {

	  for (int i = 0; i < particles.size(); i++) {
	    // Transform vehcile observations to current particles location with map
	    // coordinates. Set id to zero since no match has been given yet to map
	    // landmark.
	    std::vector<LandmarkObs> transformed_observations = observations;

	    for (int j = 0; j < observations.size(); j++) {
	      transformed_observations[j].id = 0;
	      transformed_observations[j].x = observations[j].x * cos(particles[i].theta) -
	                                      observations[j].y * sin(particles[i].theta) +
	                                      particles[i].x;
	      transformed_observations[j].y = observations[j].x * sin(particles[i].theta) +
	                                      observations[j].y * cos(particles[i].theta) +
	                                      particles[i].y;
	    }

	    // For each observation find the closest map landmark and set as its id.
	    // To reduce number of map landmarks that need to be searched, only search those
	    // map landmarks within sensor_range + some error.
	    std::vector<LandmarkObs> landmarks_in_sensor_range;

	    for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
	      float xf = map_landmarks.landmark_list[j].x_f;
	      float yf = map_landmarks.landmark_list[j].y_f;
	      int id_i = 0; // landmarks_in_sensor_range won't need map id since it will be
	                    // referenced directly when difference between observations
	                    // needs to be calculated.
	      double sensor = sensor_range * 1.2; // Add 20% for sensor error.


	      if ( (std::fabs(particles[i].x - xf) < sensor) && (std::fabs(particles[i].y - yf) < sensor) ) {
	        LandmarkObs templandmark;
	        templandmark.x = xf;
	        templandmark.y = yf;
	        templandmark.id = id_i;

	        landmarks_in_sensor_range.push_back(templandmark);
	      }
	    }
	    // Match vehicles observations transformed to map coordinates, with all the
	    // landmarks within sensor range of the particle. Indicate match by
	    // setting transformed_observations.id to matching index of landmarks_in_sensor_range.
	    dataAssociation(landmarks_in_sensor_range, transformed_observations);
	    // Update weights for each particle. Weights are the product of all differences
	    // between vehicle measurement and map landmark for each particle with some
	    // added math.

	    if (landmarks_in_sensor_range.size() == 0) {
	      if (observations.size() != 0) {   // If there are no landmarks in sensor range
	        particles[i].weight = 0;        // and there are observations, set weight
	      }                                 // to zero.
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
	double sum_weight = 0;
  std::vector<double> particle_weights; // Vector required fro discrete distribution.
  std::vector<Particle> particles_resampled; // particles that will be resampled will go here.

  for (int i = 0; i < particles.size(); i++) {
    sum_weight += particles[i].weight;
    particle_weights.push_back(particles[i].weight);
  }

  // Normalize all weights between 1 and 0.
  for (int i = 0; i < particles.size(); i++) {
    particle_weights[i] /= sum_weight;
  }

  std::default_random_engine gen;
  // Discrete distribution will allow a distribution to be created based on
  // particle weights and will output corresponding indices.
  std::discrete_distribution<int> discrete_dist (particle_weights.begin(),
                                                  particle_weights.end());

  // Resample particles based on previous weights.
  // Higher the weights, the more indentical copies of that particle
  // will be used in the next step.
  for (int i = 0; i < particles.size(); i++) {
    int particle_index = discrete_dist(gen);
    particles_resampled.push_back(particles[particle_index]);
  }


  sum_weight = 0;
  for (int i = 0; i < particles_resampled.size(); i++) {
    sum_weight += particles_resampled[i].weight;
  }
  particles = particles_resampled;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
