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

// Dont touch these
const int modK = 3;
const int modNt = 4;
const double infinity = 10000000000;
const int colors_draw[modK + 1][3] = { {0, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 0, 0} };

// Better dont touch these
const double color_scale = 1. / 100.;
const double weight = 1.25;

// Actual params, which can be changed
const int first_iter = 300;
const int iter = 50;
const double epsilon = 0.1;
// Iterations will stop when prev epsilon and cur epsilon wont differ more then on this value
const double accuracy = 0.0001;

// Set hack_value to zero if u are not hacker
// Empirical rule: if current epsilon is close to zero
// That means that diffusion made all (or almost all) max arcs equal
// Which means that we can get solution from the updated weights of diffusion
// So instead of finding markup which epsilon close to best we can find best itself
const double hack_value = 0.005;

// Percentages of image taken for calculating distribution
const double perc_h = 0.3;
const double perc_w = 1.;

double* calculate_mu(const int height_start, const int height_finish,
	const int width_left, const int width_right,
	const int height, const int width,
	int* colors, const int amount)
{
	// Initialize result
	double* result = new double[3]();

	// To be safe
	for (int c = 0; c < 3; ++c)
		result[c] = 0.;

	// Sum all the pixel colors
	for (int x = width_left; x < width_right; ++x)
	{
		const int x_ = x * height * 3;
		for (int y = height_start; y < height_finish; ++y)
		{
			const int y_ = y * 3;
			for (int c = 0; c < 3; ++c)
				result[c] += double(colors[x_ + y_ + c]) * color_scale;
		}
	}

	// Devide by amount
	for (int c = 0; c < 3; ++c)
		result[c] /= double(amount);

	return result;
}

double* calculate_ep(const int height_start, const int height_finish,
	const int width_left, const int width_right,
	const int height, const int width,
	int* colors, const int amount, double* mu)
{
	// Initialize result
	double* result = new double[3 * 3]();

	// To be safe
	for (int c1 = 0; c1 < 3; ++c1)
		for (int c2 = 0; c2 < 3; ++c2)
			result[c1 * 3 + c2] = 0.;

	// Sum all the elements
	for (int x = width_left; x < width_right; ++x)
	{
		const int x_ = x * height * 3;
		for (int y = height_start; y < height_finish; ++y)
		{
			const int y_ = y * 3;
			for (int c1 = 0; c1 < 3; ++c1)
				for (int c2 = 0; c2 < 3; ++c2)
					// TODO: check if this is correct
					result[c1 * 3 + c2] += double(colors[x_ + y_ + c1] * color_scale - mu[c1]) * double(colors[x_ + y_ + c2] * color_scale - mu[c2]);
		}
	}

	// Devide by amount
	for (int c1 = 0; c1 < 3; ++c1)
		for (int c2 = 0; c2 < 3; ++c2)
			result[c1 * 3 + c2] /= amount;

	return result;
}

void get_distributions(double* mus, double* eps,
	const int height, const int width, int* colors)
{
	// Pictures (0, 0) is in the top left corner
	const int height_down = int(height * (1 - perc_h));
	const int height_up = int(height * perc_h);

	const int width_left = width * (1. - perc_w) * 0.5;
	const int width_right = width * (1 - (1. - perc_w) * 0.5);

	const int amount_down = (width_right - width_left) * (height - height_down);
	const int amount_up = (width_right - width_left) * height_up;

	// Calculate all mus
	double* mu_down = calculate_mu(height_down, height, width_left, width_right, height, width, colors, amount_down);
	double* mu_up = calculate_mu(0, height_up, width_left, width_right, height, width, colors, amount_up);

	for (int i = 0; i < 3; ++i)
	{
		mus[i] = mu_down[i];
		mus[3 + i] = mu_up[i];
		mus[6 + i] = mu_up[i];
	}

	// Calculate all eps
	double* ep_down = calculate_ep(height_down, height, width_left, width_right, height, width, colors, amount_down, mu_down);
	double* ep_up = calculate_ep(0, height_up, width_left, width_right, height, width, colors, amount_up, mu_up);

	for (int i = 0; i < 9; ++i)
	{
		eps[i] = ep_down[i];
		eps[9 + i] = ep_up[i];
		eps[18 + i] = ep_up[i];
	}

	delete[] mu_down;
	delete[] mu_up;
	delete[] ep_down;
	delete[] ep_up;
}

// Scalar mult
double vec_mult(double* x, double* y)
{
	double sum = 0.;
	for (int c = 0; c < modK; ++c)
		sum += x[c] * y[c];

	return sum;
}

// vec(x) * mat(a)
double* mat_mult(double* x, double** a)
{
	double* res = new double[modK];
	for (int c = 0; c < modK; ++c)
		res[c] = vec_mult(x, a[c]);

	return res;
}

// q function
double q_func(int x[3], double* mus, double* eps, const int k)
{
	// TODO: check if this is correct
	double* x_ = new double[3];
	for (int c = 0; c < 3; ++c)
		x_[c] = x[c] * color_scale - mus[k * 3 + c];

	double** ep = new double* [3];
	for (int c = 0; c < 3; ++c)
		ep[c] = new double[3];

	for (int c = 0; c < 3; ++c)
		for (int c_ = 0; c_ < 3; ++c_)
			ep[c][c_] = eps[k * 9 + c * 3 + c_];

	double* z = mat_mult(x_, ep);
	double result = vec_mult(x_, z);

	delete[] z;
	delete[] x_;

	for (int c = 0; c < 3; ++c)
		delete[] ep[c];
	delete[] ep;

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

void get_neighbors_plus(int& nt, int* tau, int x, int y, int w, int h)
{
	bool left = (x != 0);
	bool right = (x != w - 1);
	bool top = (y != 0);
	bool bottom = (y != h - 1);

	// Calculate amount of neightbors
	nt = amount_of_neightbors_plus(left, right, top, bottom);

	int Index = 0;

	// Checking left side
	if (left)
	{
		tau[x * h * modNt + y * modNt + Index] = (x - 1) * h + y;
		Index++;
	}

	// Checking right side
	if (right)
	{
		tau[x * h * modNt + y * modNt + Index] = (x + 1) * h + y;
		Index++;
	}

	// Checking top pixel
	if (top)
	{
		tau[x * h * modNt + y * modNt + Index] = x * h + (y - 1);
		Index++;
	}

	// Checking bottom pixel
	if (bottom)
	{
		tau[x * h * modNt + y * modNt + Index] = x * h + (y + 1);
	}
}

int find(int* arr, const int start, const int length, const int t)
{
	for (int i = 0; i < length; ++i)
		if (arr[start + i] == t)
			return i;

	cout << "Something is wrong... I can feel it" << endl;
	return -1;
}

inline int get_g_ind(const int t, const int t__, const int height)
{
	return 2 * (abs(t - t__) == height) + (t__ > t);
}

inline int get_q_ind(const int t, int* colors)
{
	return ((colors[t * 3] * 256 + colors[t * 3 + 1]) * 256 + colors[t * 3 + 2]) * modK;
}

void diffusion(const int iter, double* mus, double* eps,
	int* nt, int* tau,
	double* phi, const int width, const int height,
	int* colors, double* g, double* q)
{
	const int modT = width * height;
	for (int it = 0; it < iter; ++it)
	{
		cout << "Diffusion --- " << it << " / " << iter << endl;
		for (int t = 0; t < modT; ++t)
		{
			const int _t = t * modNt * modK;
			for (int c = 0; c < modK; ++c)
			{
				// Finding k_best for each neighbor
				int* k_star = new int[nt[t]];
				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t * modNt + t_];
					int k_best = -1;
					double sum_best = -infinity * 100.;
					for (int c_ = 0; c_ < modK; ++c_)
					{
						double sum = g[get_g_ind(t, t__, height) * modK * modK + c * modK + c_] -
							phi[_t + t_ * modK + c] -
							phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + c_];
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
					int t__ = tau[t * modNt + t_];
					Con += g[get_g_ind(t, t__, height) * modK * modK + c * modK + k_star[t_]] -
						phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + k_star[t_]];
				}

				Con += q[get_q_ind(t, colors) + c];
				Con /= double(nt[t]);

				// Updating phi
				for (int t_ = 0; t_ < nt[t]; ++t_)
				{
					int t__ = tau[t * modNt + t_];
					phi[_t + t_ * modK + c] = g[get_g_ind(t, t__, height) * modK * modK + c * modK + k_star[t_]] -
						phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + k_star[t_]] - Con;
				}

				delete[] k_star;
			}
		}
	}
}

int* get_result(int* nt, int* tau, double* phi, const int width, const int height, double* g)
{
	const int modT = width * height;
	int* result = new int[modT];

	for (int t = 0; t < modT; ++t)
	{
		const int _t = t * modNt * modK;
		// Finding k_best for first neightbor
		int t__ = tau[t * modNt];
		int k_best = -1;
		double sum_best = -infinity * 100;
		for (int c = 0; c < modK; ++c)
		{
			int k__best = -1;
			double sum__best = -infinity * 100;
			for (int c_ = 0; c_ < modK; ++c_)
			{
				double sum = g[get_g_ind(t, t__, height) * modK * modK + c * modK + c_] -
					phi[_t + c] -
					phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + c_];
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

void get_or_and_problem(bool*& gs, bool*& qs, int* nt, int* tau, double* phi, const int width, const int height, double* mus, double* eps, int* colors, double epsilon, double* g, double* q)
{
	// Inititalize needed variable
	const int modT = width * height;

	// Initialize qs and gs
	qs = new bool[modT * modK];
	gs = new bool[modT * modNt * modK * modK];

	// TODO: check if we should compare with gs, not other qs
	// Calculating qs
	for (int t = 0; t < modT; ++t)
	{
		const int _t = t * modNt * modK;
		// Finding max qs[t][k*] and leaving only those qs[t][k], for which holds (qs[t][k*] - qs[t][k] < epsilon)
		double max = -infinity;
		double* current_q = new double[modK];
		for (int c = 0; c < modK; ++c)
		{
			// Calculating current q
			current_q[c] = q[get_q_ind(t, colors) + c];
			for (int t_ = 0; t_ < nt[t]; ++t_)
				current_q[c] += phi[_t + t_ * modK + c];

			// Comparing to max
			if (current_q[c] > max)
				max = current_q[c];
		}

		// Calculating qs[t][k]
		for (int c = 0; c < modK; ++c)
			qs[t * modK + c] = (abs(max - current_q[c]) < epsilon);

		// Delete current qs
		delete[] current_q;
	}

	// Calculating gs
	for (int t = 0; t < modT; ++t)
	{
		const int _t = t * modNt * modK;
		// Find max gs[t][t_*][k*][k_*] and leaving only those gs[t][t_][k][k_], for which holds (gs[t][t_*][k*][k_*] - gs[t][t_][k][k_] < epsilon)
		double max = -infinity;
		double** current_g = new double* [nt[t]];

		for (int t_ = 0; t_ < nt[t]; ++t_)
		{
			int t__ = tau[t * modNt + t_];
			current_g[t_] = new double[modK * modK];
			for (int c = 0; c < modK; ++c)
			{
				for (int c_ = 0; c_ < modK; ++c_)
				{
					// Calculating current g
					current_g[t_][c * modK + c_] = g[get_g_ind(t, t__, height) * modK * modK + c * modK + c_] -
						phi[_t + t_ * modK + c] -
						phi[t__ * modNt * modK + find(tau, t__ * modNt, nt[t__], t) * modK + c_];

					// Comparing to max
					if (current_g[t_][c * modK + c_] > max)
						max = current_g[t_][c * modK + c_];
				}
			}
		}

		// Calculating gs[t][t_][k][k_]
		for (int t_ = 0; t_ < nt[t]; ++t_)
			for (int c = 0; c < modK; ++c)
				for (int c_ = 0; c_ < modK; ++c_)
					gs[_t * modK + t_ * modK * modK + c * modK + c_] = (abs(max - current_g[t_][c * modK + c_]) < epsilon);

		// Delete current gs
		for (int t_ = 0; t_ < nt[t]; ++t_)
			delete[] current_g[t_];
		delete[] current_g;
	}

}

void cross(bool* gs, bool* qs, int* nt, int* tau, const int width, const int height)
{
	// Inititalize needed variables
	const int modT = width * height;
	bool changed = true;

	const int modK2 = modK * modK;
	// Repeat untill something changes
	while (changed)
	{
		changed = false;
		// Update qs
		for (int t = 0; t < modT; ++t)
		{
			const int _t = t * modNt * modK2;
			for (int c = 0; c < modK; ++c)
			{
				if (qs[t * modK + c])
					for (int t_ = 0; t_ < nt[t]; ++t_)
					{
						// Calculate or over all possible ks
						int t__ = tau[t * modNt + t_];
						bool result = false;
						for (int c_ = 0; c_ < modK; ++c_)
							result = result || (gs[_t + t_ * modK2 + c * modK + c_] && qs[t__ * modK + c_]);

						// If the result is false, then whole AND will be false
						if (!result)
						{
							qs[t * modK + c] = false;
							changed = true;
							break;
						}
					}
			}
		}

		// Update gs
		for (int t = 0; t < modT; ++t)
		{
			const int _t = t * modNt * modK2;
			for (int t_ = 0; t_ < nt[t]; ++t_)
			{
				int t__ = tau[t * modNt + t_];
				for (int c = 0; c < modK; ++c)
				{
					for (int c_ = 0; c_ < modK; ++c_)
						if (gs[_t + t_ * modK2 + c * modK + c_])
						{
							gs[_t + t_ * modK2 + c * modK + c_] = qs[t * modK + c] && qs[t__ * modK + c_];
							if (!gs[_t + t_ * modK2 + c * modK + c_])
								changed = true;
						}
				}
			}
		}
	}
}

bool f(bool* qs)
{
	// Checks if there is a markup after crossing
	bool check = false;
	for (int c = 0; c < modK; ++c)
		check = check || qs[c];
	return check;
}

bool self_control(int* answer, bool* gs, bool* qs, int* nt, int* tau, const int width, const int height)
{
	if (f(qs))
	{
		bool result = false;

		// Inititalize needed variables
		const int modT = width * height;

		for (int t = 0; t < modT; ++t)
			answer[t] = -1;

		const int size_qs = modT * modK;
		const int size_gs = modT * modNt * modK * modK;

		// Initialize qs_ and gs_
		bool* qs_ = new bool[size_qs];
		bool* gs_ = new bool[size_gs];

		double counter_copy = 0;
		double counter_cross = 0;
		double counter_trash = 0;

		// Start main loop
		for (int t = 0; t < modT; ++t)
		{
			if (t % 100 == 0)
				cout << "Self control --- " << t << " / " << modT << endl;
			result = false;
			for (int c = 0; c < modK; ++c)
			{
				// If qs[t][k] is true then check if there is a markup after cross
				if (qs[t * modK + c])
				{
					auto mark = high_resolution_clock::now();
					std::copy(qs, qs + size_qs, qs_);
					std::copy(gs, gs + size_gs, gs_);
					counter_copy += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);

					mark = high_resolution_clock::now();
					// Making other qs equal to false
					for (int c_ = 0; c_ < modK; ++c_)
						if (c != c_)
							qs_[t * modK + c_] = false;
					counter_trash += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);

					mark = high_resolution_clock::now();
					// Using cross
					cross(gs_, qs_, nt, tau, width, height);
					counter_cross += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);

					// Checking if there is markup
					if (f(qs_))
					{
						mark = high_resolution_clock::now();
						result = true;
						answer[t] = c;
						for (int c_ = 0; c_ < modK; ++c_)
							qs[t * modK + c_] = (c == c_);
						counter_trash += (double(duration_cast<microseconds>(high_resolution_clock::now() - mark).count()) / 1000000.);
						break;
					}

				}
			}

			// If we didnt find markup then break
			if (!result)
				break;
		}

		// Delete qs_ and gs_
		delete[] qs_;
		delete[] gs_;

		double sum_time = counter_trash + counter_cross + counter_copy;
		cout << "Time used for trash: " << counter_trash / sum_time << endl;
		cout << "Time used for cross: " << counter_cross / sum_time << endl;
		cout << "Time used for copy: " << counter_copy / sum_time << endl;
		cout << "Time used for self control: " << sum_time << endl;

		return result;
	}
	else
	{
		return false;
	}
}

void save_and_show(int* res, const int width, const int height, string name, bool save = false)
{
	Mat* result = new Mat[3];
	for (int c = 0; c < modK; ++c)
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
	cv::imshow(name, rez);
	if (save)
		imwrite(name + ".png", rez);

	delete[] result;
}

int* iterations(const int first_iter, const int iter,
	const double accuracy, double epsilon,
	double* mus, double* eps,
	int* nt, int* tau,
	const int width, const int height,
	int* colors,
	double* g, double* q)
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
	double* phi = new double[modT * modNt * modK]();

	// First Diffusion
	diffusion(first_iter, mus, eps, nt, tau, phi, width, height, colors, g, q);

	while (!stop)
	{
		cout << prev_epsilon << " - " << current_epsilon << " <> " << accuracy << endl;
		// Get or and problem
		bool* gs;
		bool* qs;
		get_or_and_problem(gs, qs, nt, tau, phi, width, height, mus, eps, colors, current_epsilon, g, q);

		// Use crossing
		cross(gs, qs, nt, tau, width, height);

		bool check;
		bool have_result = false;

		if (current_epsilon < 0.01 && hack_value == 0.)
			cout << "Consider using hacks" << endl;

		if ((prev_epsilon - current_epsilon) < 2 * accuracy)
		{
			check = f(qs);
			if (check)
			{
				if (current_epsilon > hack_value)
					check = self_control(current_res, gs, qs, nt, tau, width, height);
				else
				{
					check = f(qs);
					current_res = get_result(nt, tau, phi, width, height, g);
				}
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
			diffusion(iter, mus, eps, nt, tau, phi, width, height, colors, g, q);
		}

		delete[] qs;
		delete[] gs;
	}

	delete[] current_res;

	return last_res;
}

int main()
{
	Mat image_, image[3];
	image_ = imread("2.jpg", IMREAD_UNCHANGED);
	split(image_, image);

	auto start = high_resolution_clock::now();

	const int height = image[0].size().height;
	const int width = image[0].size().width;

	// Get array from Mat
	int* colors = new int[width * height * 3];
	for (int x = 0; x < width; ++x)
	{
		const int x_ = x * height * 3;
		for (int y = 0; y < height; ++y)
		{
			const int y_ = y * 3;
			for (int c = 0; c < 3; ++c)
				colors[x_ + y_ + c] = int(image[c].at<uchar>(y, x));
		}
	}

	// Form single array for mus and eps
	// 0 - down
	// 1- mid
	// 2 - up
	double* mus = new double[modK * 3];
	double* eps = new double[modK * 3 * 3];

	get_distributions(mus, eps, height, width, colors);

	for (int i = 0; i < 3; ++i)
		cout << mus[i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
		cout << mus[6 + i] << " " << endl;
	cout << endl;

	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps[i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;


	for (int i = 0; i < 3; ++i)
	{
		for (int j = 0; j < 3; ++j)
			cout << eps[18 + i * 3 + j] << " ";
		cout << endl;
	}
	cout << endl;

	cout << "Precalculating g" << endl;
	// Initialize q and g
	double* g = new double[modNt * modK * modK];

	for (int c = 0; c < modK; ++c)
		for (int c_ = 0; c_ < modK; ++c_)
			g[0 * modK * modK + c * modK + c_] = g_plus(1, 1, 1, 0, c, c_);

	for (int c = 0; c < modK; ++c)
		for (int c_ = 0; c_ < modK; ++c_)
			g[1 * modK * modK + c * modK + c_] = g_plus(1, 1, 1, 2, c, c_);

	for (int c = 0; c < modK; ++c)
		for (int c_ = 0; c_ < modK; ++c_)
			g[2 * modK * modK + c * modK + c_] = g_plus(1, 1, 0, 1, c, c_);

	for (int c = 0; c < modK; ++c)
		for (int c_ = 0; c_ < modK; ++c_)
			g[3 * modK * modK + c * modK + c_] = g_plus(1, 1, 2, 1, c, c_);
	cout << "Done" << endl;

	cout << "Precalculating q" << endl;

	double* q = new double[256 * 256 * 256 * modK];
	for (int c1 = 0; c1 < 256; ++c1)
	{
		cout << c1 << " / " << 256 << endl;
		for (int c2 = 0; c2 < 256; ++c2)
			for (int c3 = 0; c3 < 256; ++c3)
				for (int c = 0; c < modK; ++c)
				{
					int x[3] = { c1, c2, c3 };
					q[c1 * 256 * 256 * modK + c2 * 256 * modK + c3 * modK + c] = q_func(x, mus, eps, c);
				}
	}
	cout << "Done" << endl;

	// Create neighbour structure
	const int modT = width * height;
	int* tau = new int[modT * modNt];
	int* nt = new int[modT];
	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < height; ++y)
			get_neighbors_plus(nt[x * height + y], tau, x, y, width, height);
	}

	int* res = iterations(first_iter, iter, accuracy, epsilon, mus, eps, nt, tau, width, height, colors, g, q);

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