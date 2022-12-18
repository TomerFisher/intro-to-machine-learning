#################################
# Your name: Tomer Fisher
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        xs = np.random.uniform(0, 1, m)
        ys = np.empty(0)
        for index in range(0, m):
            if 0 <= xs[index] <= 0.2 or 0.4 <= xs[index] <= 0.6 or 0.8 <= xs[index] <= 1:
                y = np.random.choice([0, 1], p=[0.2, 0.8])
                ys = np.append(ys, y)
            else:
                y = np.random.choice([0, 1], p=[0.9, 0.1])
                ys = np.append(ys, y)
        samples = np.vstack((xs, ys)).T
        return samples


    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        n_steps = int((m_last - m_first)/step) + 1
        avg_errors = np.empty((n_steps, 2))
        for m in range(m_first, m_last + 1, step):
            empirical_errors = np.empty(T)
            true_errors = np.empty(T)
            for index in range(T):
                data = self.sample_from_D(m)
                sorted_xs, sorted_ys = self.get_sorted_xs_and_ys(data)
                interval_list, besterror = intervals.find_best_interval(sorted_xs, sorted_ys, k)
                empirical_errors[index] = besterror / m
                true_errors[index] = self.calc_true_error(interval_list)
            avg_errors[int((m - m_first)/step)] = np.array([np.average(empirical_errors), np.average(true_errors)])
        plt.figure(1)
        plt.xlabel("number of samples")
        plt.ylabel("error")
        plt.plot(range(m_first, m_last + 1, step), avg_errors[:, 0], color = 'blue', label = 'empirical error')
        plt.plot(range(m_first, m_last + 1, step), avg_errors[:, 1], color = 'red', label = 'true error')
        plt.legend()
        plt.show()
        return avg_errors


    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        data = self.sample_from_D(m)
        sorted_xs, sorted_ys = self.get_sorted_xs_and_ys(data)
        empirical_errors = np.empty(0)
        true_errors = np.empty(0)
        for k in range(k_first, k_last + 1, step):
            interval_list, besterror = intervals.find_best_interval(sorted_xs, sorted_ys, k)
            empirical_errors = np.append(empirical_errors, besterror/m)
            true_errors = np.append(true_errors, self.calc_true_error(interval_list))
        plt.figure(2)
        plt.xlabel("number of intervals")
        plt.ylabel("error")
        plt.plot(range(k_first, k_last + 1, step), empirical_errors, color = 'blue', label = 'empirical error')
        plt.plot(range(k_first, k_last + 1, step), true_errors, color = 'red', label = 'true error')
        plt.legend()
        plt.show()
        return np.argmin(empirical_errors)
        

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        data = self.sample_from_D(m)
        sorted_xs, sorted_ys = self.get_sorted_xs_and_ys(data)
        empirical_errors = np.empty(0)
        true_errors = np.empty(0)
        for k in range(k_first, k_last + 1, step):
            interval_list, besterror = intervals.find_best_interval(sorted_xs, sorted_ys, k)
            empirical_errors = np.append(empirical_errors, besterror/m)
            true_errors = np.append(true_errors, self.calc_true_error(interval_list))
        penalties = np.array([2*np.sqrt((2*k + np.log(2/0.1))/m) for k in range(k_first, k_last + 1, step)])
        empirical_errors_with_penalty = np.add(empirical_errors, penalties)
        plt.figure(3)
        plt.xlabel("number of intervals")
        plt.ylabel("error")
        plt.plot(range(k_first, k_last + 1, step), empirical_errors, color = 'blue', label = 'empirical error')
        plt.plot(range(k_first, k_last + 1, step), true_errors, color = 'red', label = 'true error')
        plt.plot(range(k_first, k_last + 1, step), penalties, color = 'orange', label = 'penalty')
        plt.plot(range(k_first, k_last + 1, step), empirical_errors_with_penalty, color = 'black', label = 'empirical error + penalty')
        plt.legend()
        plt.show()
        return np.argmin(empirical_errors_with_penalty)


    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        data = self.sample_from_D(m)
        holdout_cut = int(0.8*m)
        sorted_xs, sorted_ys = self.get_sorted_xs_and_ys(data[:holdout_cut, :])
        holdout_xs = data[holdout_cut:, 0]
        holdout_ys = data[holdout_cut:, 1]
        test_errors = np.empty(0)
        for k in range(1, 11, 1):
            interval_list, besterror = intervals.find_best_interval(sorted_xs, sorted_ys, k)
            test_errors = np.append(test_errors, self.calc_empirical_error(interval_list, holdout_xs, holdout_ys))
        return np.argmin(test_errors) + 1
        
    
    #################################
    # Place for additional methods
    
    def get_sorted_xs_and_ys(self, data):
        """Get xs and ys from data and sort them by xs.
        Input: data - np.ndarray of shape (_,2)
        
        Returns: Return sorted xs and sorted ys.
        """
        xs = data[:, 0]
        sorted_xs = xs[xs.argsort()]
        ys = data[:, 1]
        sorted_ys = ys[xs.argsort()]
        return sorted_xs, sorted_ys
        
    
    def calc_true_error(self, interval_list):
        """ calculata the true error of the hypothesis defines by the interval list.
        Input: interval list.
        
        Returns: The true error.
        """
        #hypothesis label 1, and x in [0,0.2], [0.4,0.6] or [0.8,1] 
        sum1 = self.calc_sum_intervals_length_in_section(interval_list, [(0, 0.2)]) \
               + self.calc_sum_intervals_length_in_section(interval_list, [(0.4, 0.6)]) \
               + self.calc_sum_intervals_length_in_section(interval_list, [(0.8, 1)])
        #hypothesis label 0, and x in [0,0.2], [0.4,0.6] or [0.8,1]
        sum2 = 0.6 - sum1
        #hypothesis label 1, and x not in [0,0.2], [0.4,0.6] or [0.8,1]
        sum3 = self.calc_sum_intervals_length_in_section(interval_list, [(0.2, 0.4)]) \
               + self.calc_sum_intervals_length_in_section(interval_list, [(0.6, 0.8)])
        #hypothesis label 0, and x not in [0,0.2], [0.4,0.6] or [0.8,1]
        sum4 = 0.4 - sum3
        #calc the true error
        true_error = 0.2*sum1 + 0.8*sum2 + 0.9*sum3 + 0.1*sum4
        return true_error
        

    def calc_sum_intervals_length_in_section(self, interval_list, section):
        """ calculata the sum length of the intervals intersaction with section.
        Input: interval_list, section.
        
        Returns: The sum.
        """
        sum_length = 0
        for interval in interval_list:
            start_interval = interval[0]
            end_interval = interval[1]
            start = max(start_interval, section[0][0])
            end = min(end_interval, section[0][1])
            if start < end:
                sum_length += (end - start)
        return sum_length
    
    
    def calc_empirical_error(self, interval_list, xs, ys):
        """ calculata the empirical error of the hypothesis defines by the interval list with the data (xs, ys).
        Input: interval list, xs, ys.
        
        Returns: The empirical error.
        """
        bad_sample_counter = 0
        for index in range(xs.size):
            if self.check_if_point_in_intervals(xs[index], interval_list) != ys[index]:
                bad_sample_counter += 1
        empirical_error = bad_sample_counter / xs.size
        return empirical_error
            
            
    def check_if_point_in_intervals(self, x, interval_list):
        """ check if x is in one of the interval from interval list.
        Input: x, interval list.
        
        Returns: 1 if x is in one of the interval from interval list, otherwise 0.
        """
        for interval in interval_list:
            if interval[0] <= x <= interval[1]:
                return 1
        return 0
    
    #################################
        
        
if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)
