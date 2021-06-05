#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono> 

using namespace cv;
using namespace std;
using namespace std::chrono;

const int channels = 3;
const double infinity = 10000000000;
const int colors_draw[channels + 1][channels] = { {0, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 0, 0} };

const double weight = 1.25;
const int first_iter = 500;
const int iter = 100;
const double accuracy = 0.1;
const double epsilon = 10;
const double perc_h = 0.3;
const double perc_w = 1.;

double* calculate_mu(const int height_start, const int height_finish, const int width_left, const int width_right, double*** colors, const int amount)
{
	// Initialize result
	double* result = new double[channels]();

	// To be safe
	for (int c = 0; c < channels; ++c)
		result[c] = 0.;

	// Sum all the pixel colors
	for (int x = width_left; x < width_right; ++x)
		for (int y = height_start; y < height_finish; ++y)
			for (int c = 0; c < channels; ++c)
				result[c] += double(colors[x][y][c]);

	// Devide by amount
	for (int c = 0; c < channels; ++c)
		result[c] /= double(amount);

	return result;
}

double** calculate_ep(const int height_start, const int height_finish, const int width_left, const int width_right, double*** colors, const int amount, double* mu)
{
	// Initialize result
	double** result = new double* [channels]();
	for (int c = 0; c < channels; ++c)
		result[c] = new double[channels]();

	// To be safe
	for (int c1 = 0; c1 < channels; ++c1)
		for (int c2 = 0; c2 < channels; ++c2)
			result[c1][c2] = 0.;

	// Sum all the elements
	for (int x = width_left; x < width_right; ++x)
		for (int y = height_start; y < height_finish; ++y)
			for (int c1 = 0; c1 < channels; ++c1)
				for (int c2 = 0; c2 < channels; ++c2)
					// TODO: check if this is correct
					result[c1][c2] += double(colors[x][y][c1] - mu[c1]) * double(colors[x][y][c2] - mu[c2]);

	// Devide by amount
	for (int c1 = 0; c1 < channels; ++c1)
		for (int c2 = 0; c2 < channels; ++c2)
			result[c1][c2] /= amount;

	return result;
}

void get_distributions(double* &mu_down, double** &ep_down,
	                   double* &mu_mid, double** &ep_mid,
	                   double* &mu_up, double** &ep_up,
	                   const int height, const int width, double*** colors,
	                   const double perc_h, const double perc_w)
{
	// Pictures (0, 0) is in the top left corner
	const int height_down = int(height * (1 - perc_h));
	const int height_up = int(height * perc_h);

	const int width_left = width * (1. - perc_w) * 0.5;
	const int width_right = width * (1 - (1. - perc_w) * 0.5);

	const int amount_down = (width_right - width_left) * (height - height_down);
	const int amount_up = (width_right - width_left) * height_up;

	// Calculate all mus
	mu_down = calculate_mu(height_down, height, width_left, width_right, colors, amount_down);
	mu_up = calculate_mu(0, height_up, width_left, width_right, colors, amount_up);
	mu_mid = new double[channels]();
	for (int i = 0; i < channels; ++i)
		mu_mid[i] = mu_up[i];

	// Calculate all eps
	ep_down = calculate_ep(height_down, height, width_left, width_right, colors, amount_down, mu_down);
	ep_up = calculate_ep(0, height_up, width_left, width_right, colors, amount_up, mu_up);
	ep_mid = new double* [channels]();
	for (int c = 0; c < channels; ++c)
		ep_mid[c] = new double[channels]();
	for (int i = 0; i < channels; ++i)
		for (int j = 0; j < channels; ++j)
			ep_mid[i][j] = ep_up[i][j];
}

// Scalar mult
double vec_mult(double* x, double* y)
{
	double sum = 0.;
	for (int c = 0; c < channels; ++c)
		sum += x[c] * y[c];

	return sum;
}

// vec(x) * mat(a)
double* mat_mult(double* x, double** a)
{
	double* res = new double[channels];
	for (int c = 0; c < channels; ++c)
		res[c] = vec_mult(x, a[c]);

	return res;
}

// q function
double q(double* x, double* mu, double** ep)
{
	// TODO: check if this is correct
	double* x_ = new double[channels];
	for (int c = 0; c < channels; ++c)
		x_[c] = x[c] - mu[c];
	double* z = mat_mult(x_, ep);
	double result = vec_mult(x_, z);

	delete[] z;
	delete[] x_;

	return -result;
}

double g_plus(const int x1, const int y1, const int x2, const int y2, const int k1, const int k2)
{
	if ((k1 == 0) && (k2 == 0))
	{
		return 0;
	}

	if ((k1 == 1) && (k2 == 1))
	{
		if (x1 != x2)
			return 0;
		else
			return -infinity;
	}

	if ((k1 == 2) && (k2 == 2))
	{
		return 0;
	}

	if ((k1 == 1) && (k2 == 0))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 0) && (k2 == 1))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 2) && (k2 == 0))
	{
		return -infinity;
	}

	if ((k1 == 0) && (k2 == 2))
	{
		return -infinity;
	}

	if ((k1 == 2) && (k2 == 1))
	{
		if (y1 > y2)
			return -infinity;
		else
			return -weight;
	}

	if ((k1 == 1) && (k2 == 2))
	{
		if (y2 > y1)
			return -infinity;
		else
			return -weight;
	}

	cout << "Something is wrong... I can feel it" << endl;
	return -infinity;
}

int amount_of_neightbors_plus(bool left, bool right, bool top, bool bottom)
{
	int result = 0;

	// Checking left side
	if (left)
		result++;

	// Checking right side
	if (right)
		result++;

	// Checking top pixel
	if (top)
		result++;

	// Checking bottom pixel
	if (bottom)
		result++;

	return result;
}

int* get_neighbors_plus(int& nt, int x, int y, int w, int h)
{
	bool left = (x != 0);
	bool right = (x != w - 1);
	bool top = (y != 0);
	bool bottom = (y != h - 1);

	// Calculate amount of neightbors
	nt = amount_of_neightbors_plus(left, right, top, bottom);

	// Initialize neighbors array
	int* Neighbors = new int[nt];

	// To be safe
	for (int n = 0; n < nt; ++n)
		Neighbors[n] = -1;

	int Index = 0;

	// Checking left side
	if (left)
	{
		Neighbors[Index] = (x - 1) * h + y;
		Index++;
	}

	// Checking right side
	if (right)
	{
		Neighbors[Index] = (x + 1) * h + y;
		Index++;
	}

	// Checking top pixel
	if (top)
	{
		Neighbors[Index] = x * h + (y - 1);
		Index++;
	}

	// Checking bottom pixel
	if (bottom)
	{
		Neighbors[Index] = x * h + (y + 1);
	}

	return Neighbors;
}

int find(int* arr, const int length, const int t)
{
	int index = -1;
	for (int i = 0; i < length; ++i)
	{
		if (arr[i] == t)
		{
			return i;
		}
	}
}

void diffusion(const int iter, double** mus, double*** eps, int* nt, int** tau, double*** phi, const int width, const int height, double*** colors)
{ 
	const int modT = width * height;
	for (int it = 0; it < iter; ++it)
	{
		cout << "Diffusion --- " << it << " / " << iter << endl;
		for (int t = 0; t < modT; ++t)
		{
			for (int c = 0; c < channels; ++c)
			{
				// Finding k_best for each neighbor
				int* k_star = new int[nt[t]];
				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t][t_];
					int k_best = -1;
					double sum_best = -infinity * 100.;
					for (int c_ = 0; c_ < channels; ++c_)
					{
						double sum = g_plus(t / height, t % height, t__ / height, t__ % height, c, c_) - phi[t][t_][c] - phi[t__][find(tau[t__], nt[t__], t)][c_];
						if (sum > sum_best)
						{
							k_best = c_;
							sum_best = sum;
						}
					}
					
					k_star[t_] = k_best;
				}
				// Calculating Constant C for further update of phi
				double Con = 0.;

				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t][t_];
					Con += g_plus(t / height, t % height, t__ / height, t__ % height, c, k_star[t_]) - phi[t__][find(tau[t__], nt[t__], t)][k_star[t_]];
				}

				//cout << q(colors[t / height][t % height], mus[c], eps[c]) << endl;

				Con += q(colors[t / height][t % height], mus[c], eps[c]);

				Con /= double(nt[t]);

				// Updating phi
				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t][t_];
					phi[t][t_][c] = g_plus(t / height, t % height, t__ / height, t__ % height, c, k_star[t_]) - phi[t__][find(tau[t__], nt[t__], t)][k_star[t_]] - Con;
				}

				delete[] k_star;
			}
		}
	}
}

int* get_result(int* nt, int** tau, double*** phi, const int width, const int height)
{
	const int modT = width * height;
	int* result = new int[modT];

	for (int t = 0; t < modT; ++t)
	{
		// Finding k_best for first neightbor
		int t__ = tau[t][0];
		int k_best = -1;
		double sum_best = -infinity * 100;
		for (int c = 0; c < channels; ++c)
		{
			int k__best = -1;
			double sum__best = -infinity * 100;
			for (int c_ = 0; c_ < channels; ++c_)
			{
				double sum = g_plus(t / height, t % height, t__ / height, t__ % height, c, c_) - phi[t][0][c] - phi[t__][find(tau[t__], nt[t__], t)][c_];
				if (sum > sum__best)
				{
					k__best = c_;
					sum__best = sum;
				}
			}
			if (sum__best > sum_best)
			{
				k_best = c;
				sum_best = sum__best;
			}
		}

		result[t] = k_best;
	}

	return result;
}

void get_or_and_problem(bool*** &gs, bool** &qs, int* nt, int** tau, double*** phi, const int width, const int height, double** mus, double*** eps, double*** colors, double epsilon)
{
	// Inititalize needed variable
	const int modT = width * height;

	// Initialize qs and gs
	qs = new bool* [modT];
	for (int t = 0; t < modT; ++t)
		qs[t] = new bool[channels];

	gs = new bool** [modT];
	for (int t = 0; t < modT; ++t)
	{
		gs[t] = new bool* [nt[t]];
		for (int t_ = 0; t_ < nt[t]; ++t_)
			gs[t][t_] = new bool[channels * channels];
	}

	// TODO: check if we should compare with gs, not other qs
	// Calculating qs
	for (int t = 0; t < modT; ++t)
	{
		// Finding max qs[t][k*] and leaving only those qs[t][k], for which holds (qs[t][k*] - qs[t][k] < epsilon)
		double max = -infinity;
		double* current_q = new double[channels];
		for (int c = 0; c < channels; ++c)
		{
			// Calculating current q
			current_q[c] = q(colors[t / height][t % height], mus[c], eps[c]);
			for (int t_ = 0; t_ < nt[t]; ++t_)
				current_q[c] += phi[t][t_][c];

			// Comparing to max
			if (current_q[c] > max)
				max = current_q[c];
		}

		// Calculating qs[t][k]
		for (int c = 0; c < channels; ++c)
			qs[t][c] = (abs(max - current_q[c]) < epsilon);

		// Delete current qs
		delete[] current_q;
	}

	// Calculating gs
	for (int t = 0; t < modT; ++t)
	{
		// Find max gs[t][t_*][k*][k_*] and leaving only those gs[t][t_][k][k_], for which holds (gs[t][t_*][k*][k_*] - gs[t][t_][k][k_] < epsilon)
		double max = -infinity;
		double** current_g = new double* [nt[t]];
		
		for (int t_ = 0; t_ < nt[t]; ++t_)
		{
			int t__ = tau[t][t_];
			current_g[t_] = new double[channels * channels];
			for (int c = 0; c < channels; ++c)
			{
				for (int c_ = 0; c_ < channels; ++c_)
				{
					// Calculating current g
					current_g[t_][c * channels + c_] = g_plus(t / height, t % height, t__ / height, t__ % height, c, c_) - phi[t][t_][c] - phi[t__][find(tau[t__], nt[t__], t)][c_];

					// Comparing to max
					if (current_g[t_][c * channels + c_] > max)
						max = current_g[t_][c * channels + c_];
				}
			}
		}

		// Calculating gs[t][t_][k][k_]
		for (int t_ = 0; t_ < nt[t]; ++t_)
			for (int c = 0; c < channels; ++c)
				for (int c_ = 0; c_ < channels; ++c_)
					gs[t][t_][c * channels + c_] = (abs(max - current_g[t_][c * channels + c_]) < epsilon);

		// Delete current gs
		for (int t_ = 0; t_ < nt[t]; ++t_)
			delete[] current_g[t_];
		delete[] current_g;
	}
	
}

void cross(bool*** gs, bool** qs, int* nt, int** tau, const int width, const int height)
{
	// Inititalize needed variables
	const int modT = width * height;
	bool changed = true;

	// Repeat untill something changes
	while (changed)
	{
		changed = false;
		// Update qs
		for (int t = 0; t < modT; ++t)
		{
			for (int c = 0; c < channels; ++c)
			{
				if (qs[t][c])
					for (int t_ = 0; t_ < nt[t]; ++t_)
					{
						// Calculate or over all possible ks
						int t__ = tau[t][t_];
						bool result = false;
						for (int c_ = 0; c_ < channels; ++c_)
							result = result || (gs[t][t_][c * channels + c_] && qs[t__][c_]);

						// If the result is false, then whole AND will be false
						if (!result)
							{
								qs[t][c] = false;
								changed = true;
								break;
							}
					}
			}
		}

		// Update gs
		for (int t = 0; t < modT; ++t)
		{
			for (int t_ = 0; t_ < nt[t]; ++t_)
			{
				int t__ = tau[t][t_];
				for (int c = 0; c < channels; ++c)
				{
					for (int c_ = 0; c_ < channels; ++c_)
						if (gs[t][t_][c * channels + c_])
						{
							gs[t][t_][c * channels + c_] = qs[t][c] && qs[t__][c_];
							if (!gs[t][t_][c * channels + c_])
								changed = true;
						}
				}
			}
		}
	}
}

bool f(bool** qs)
{
	// Checks if there is a markup after crossing
	bool check = false;
	for (int c = 0; c < channels; ++c)
		check = check || qs[0][c];
	return check;
}

void copy(bool*** gs, bool** qs, bool*** gs_, bool** qs_, const int modT, int* nt)
{
	for (int t = 0; t < modT; ++t)
	{
		for (int c = 0; c < channels; ++c)
		{
			qs_[t][c] = qs[t][c];

			for (int t_ = 0; t_ < nt[t]; ++t_)
				for (int c_ = 0; c_ < channels; ++c_)
					gs_[t][t_][c * channels + c_] = gs[t][t_][c * channels + c_];
		}
	}
}

bool self_control(int* answer, bool*** gs, bool** qs, int* nt, int** tau, const int width, const int height)
{
	if (f(qs))
	{
		bool result = false;

		// Inititalize needed variables
		const int modT = width * height;

		for (int t = 0; t < modT; ++t)
			answer[t] = -1;

		// Initialize qs_ and gs_
		bool** qs_ = new bool* [modT];
		for (int t = 0; t < modT; ++t)
			qs_[t] = new bool[channels];

		bool*** gs_ = new bool** [modT];
		for (int t = 0; t < modT; ++t)
		{
			gs_[t] = new bool* [nt[t]];
			for (int t_ = 0; t_ < nt[t]; ++t_)
				gs_[t][t_] = new bool[channels * channels];
		}

		// Start main loop
		for (int t = 0; t < modT; ++t)
		{
			cout << "Self control --- " << t << " / " << modT << endl; 
			result = false;
			for (int c = 0; c < channels; ++c)
			{
				// If qs[t][k] is true then check if there is a markup after cross
				if (qs[t][c])
				{
					copy(gs, qs, gs_, qs_, modT, nt);

					// Making other qs equal to false
					for (int c_ = 0; c_ < channels; ++c_)
						if (c != c_)
							qs_[t][c_] = false;

					// Using cross
					cross(gs_, qs_, nt, tau, width, height);

					// Checking if there is markup
					if (f(qs_))
					{
						result = true;
						answer[t] = c;
						break;
					}
				}
			}

			// If we didnt find markup then break
			if (!result)
				break;
		}

		// Delete qs_ and gs_
		for (int t = 0; t < modT; ++t)
			delete[] qs_[t];
		delete[] qs_;

		for (int t = 0; t < modT; ++t)
		{
			for (int t_ = 0; t_ < nt[t]; ++t_)
				delete[] gs_[t][t_];
			delete[] gs_[t];
		}
		delete[] gs_;

		return result;
	}
	else
	{
		return false;
	}
}

void save_and_show(int* res, const int width, const int height, string name, bool save=false)
{
	Mat* result = new Mat[3];
	for (int c = 0; c < channels; ++c)
	{
		result[c] = Mat::zeros(Size(width, height), CV_8UC1);
		for (int x = 0; x < width; ++x)
			for (int y = 0; y < height; ++y)
			{
				result[c].at<uchar>(y, x) = uchar(colors_draw[1 + res[x * height + y]][c]);
			}
	}

	Mat rez;
	vector<Mat> channels;

	channels.push_back(result[0]);
	channels.push_back(result[1]);
	channels.push_back(result[2]);

	merge(channels, rez);

	namedWindow(name, WINDOW_AUTOSIZE);
	imshow(name, rez);
	if (save)
		imwrite(name + ".png", rez);

	delete[] result;
}

int* iterations(const int first_iter, const int iter,
	const double accuracy, double epsilon,
	double** mus, double*** eps,
	int* nt, int** tau,
	const int width, const int height,
	double*** colors)
{
	double prev_epsilon = 2 * epsilon;
	double current_epsilon = epsilon;
	const int modT = width * height;
	bool stop = false;
	int counter = 0;
	int* last_res = new int[modT]();
	for (int t = 0; t < modT; ++t)
		last_res[t] = -1;
	int* current_res = new int[modT]();
	for (int t = 0; t < modT; ++t)
		current_res[t] = -1;

	// Initialize phi
	double*** phi = new double** [modT];
	for (int t = 0; t < modT; ++t)
	{
		phi[t] = new double* [nt[t]];
		for (int t_ = 0; t_ < nt[t]; ++t_)
		{
			phi[t][t_] = new double[channels]();
			for (int c = 0; c < channels; ++c)
				phi[t][t_][c] = 0.;
		}
	}

	// First Diffusion
	diffusion(first_iter, mus, eps, nt, tau, phi, width, height, colors);
	
	while (!stop)
	{
		cout << prev_epsilon << " - " << current_epsilon << " <> " <<  accuracy << endl;
		// Get or and problem
		bool*** gs;
		bool** qs;
		get_or_and_problem(gs, qs, nt, tau, phi, width, height, mus, eps, colors, current_epsilon);

		// Use crossing
		cross(gs, qs, nt, tau, width, height);

		bool check;
		bool have_result = false;

		if ((prev_epsilon - current_epsilon) < 2 * accuracy)
		{
			check = f(qs);
			if (check)
			{
				check = self_control(current_res, gs, qs, nt, tau, width, height);
				have_result = true;
			}
		}
		else
		{
			check = f(qs);
		}

		if (check)
		{
			if (have_result)
				for (int t = 0; t < modT; ++t)
					last_res[t] = current_res[t];

			prev_epsilon = current_epsilon;
			current_epsilon *= 0.5;
		}
		else
		{
			current_epsilon = (prev_epsilon + current_epsilon) * 0.5;
		}

		if ((prev_epsilon - current_epsilon) < accuracy)
		{
			stop = true;
		}
		else
		{
			diffusion(iter, mus, eps, nt, tau, phi, width, height, colors);
		}

		for (int t = 0; t < modT; ++t)
			delete[] qs[t];
		delete[] qs;

		for (int t = 0; t < modT; ++t)
		{
			for (int t_ = 0; t_ < nt[t]; ++t_)
				delete[] gs[t][t_];
			delete[] gs[t];
		}
		delete[] gs;
	}

	delete[] current_res;
	
	return last_res;
}

int main()
{
	Mat image_, image[channels];
	image_ = imread("6.jpg", IMREAD_UNCHANGED);
	split(image_, image);

	auto start = high_resolution_clock::now();

	const int height = image[0].size().height;
	const int width = image[0].size().width;

	// Get array from Mat
	double*** colors = new double** [width];
	for (int x = 0; x < width; ++x)
	{
		colors[x] = new double* [height];
		for (int y = 0; y < height; ++y)
		{
			colors[x][y] = new double[channels];
			for (int c = 0; c < channels; ++c)
				colors[x][y][c] = double(image[c].at<uchar>(y, x)) / 100.;
		}
	}

	// Form single array for mus and eps
	// 0 - down
	// 1- mid
	// 2 - up
	double** mus = new double* [channels];
	double*** eps = new double** [channels];

	get_distributions(mus[0], eps[0],
		              mus[1], eps[1],
		              mus[2], eps[2],
		              height, width, colors,
		              perc_h, perc_w);
	for (int i = 0; i < channels; ++i)
		cout << mus[0][i] << " " << endl;

	cout << endl;

	for (int i = 0; i < channels; ++i)
		cout << mus[2][i] << " " << endl;

	cout << endl;

	// Create neighbour structure
	const int modT = width * height;
	int** tau = new int* [modT];
	int* nt = new int[modT];
	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
			tau[x * height + y] = get_neighbors_plus(nt[x * height + y], x, y, width, height);
	}

	int* res = iterations(first_iter, iter, accuracy, epsilon, mus, eps, nt, tau, width, height, colors);

	// Measuring time taken
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << "Time used: " << double(duration.count()) / 1000000. << endl;

	if (res[0] != -1)
	{
		save_and_show(res, width, height, "res", true);
	}
	else
	{
		cout << "Bad params" << endl;
		cout << "Try to change epsilon and/or accuracy" << endl;
	}

	waitKey(0);
	return 0;
}