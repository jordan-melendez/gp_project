import numpy as np
import pdb

class HamiltonianSampler:

    def __init__(self, U, grad_U, num_leaps, step_size, tempering):
        """Initializes the main sampler class

        MUST CALL initialize to set starting point before running algorithm

        Parameters:
        --------------------
        U: A function that evaluates the exponent of an exponential family distribution
        grad_U: A function that evaluates the gradient of U
        num_leaps: The number of leapfrog iterations to do per sample round
        step_size: Epsilon for size of leapfrog iterations

        """
        self.U = U
        self.grad_U = grad_U

        self.num_leaps = num_leaps
        self.step_size = step_size

        self.tempering = tempering

        self.current_position = None
        self.current_velocity = None
        self.num_dimensions = None

        self.total_number_of_draws = 0
        self.number_accepted = 0

    def set_seed(self, seed):
        np.random.seed(seed)

    def initialize(self, start_q):
        """Sets the starting point for the MCMC sampler, and initializes some important variables"""
        self.num_dimensions = len(start_q)

        self.upper_bounds = np.full([self.num_dimensions], np.Infinity)
        self.lower_bounds = np.full([self.num_dimensions], -np.Infinity)

        self.current_position = start_q

    def set_bounds(self, bound_dict):
        """Setter for the upper and lower bounds of the parameter spaces.
        
        Parameters
        ----------
        bound_dict: A dictionary where the keys are the parameter vector index;
                    its values are a 2-tuple with (lower, upper) bounds.
                    
                    Example: { 3: (0, np.Infinity) } to constrain to X >= 0"""

        for idx in bound_dict:
            lower, upper = bound_dict[idx]
            self.lower_bounds[idx] = lower
            self.upper_bounds[idx] = upper

    def kinetic_energy(self, velocity):
        return velocity @ velocity / 2

    def evaluate_energy(self, position, velocity):
        return (self.U(position), self.kinetic_energy(velocity))

    def _update_position(self, position, velocity, step_size):
        new_position = position + step_size * velocity

        above_bound = new_position > self.upper_bounds
        below_bound = new_position < self.lower_bounds

        new_position[above_bound] = (2 * self.upper_bounds - new_position)[above_bound]
        new_position[below_bound] = (2 * self.lower_bounds - new_position)[below_bound]

        return new_position

    def _update_velocity(self, position, velocity, step_size):
        # print("Position: ", position, "Velocity: ", velocity)
        # pdb.set_trace()
        return velocity - step_size * self.grad_U(position)

    def _leapfrog_step(self, position, velocity, temper_const):
        # print("LEAPFROG: Position ", position, ", Velocity ", velocity)
        position = self._update_position(position, velocity, self.step_size)
        velocity = temper_const * self._update_velocity(position, velocity, self.step_size)
        # print("Position ", position, ", Velocity ", velocity)
        return (position, velocity)

    def _leapfrog(self, position, velocity):
        velocity = self._update_velocity(position, velocity, self.step_size / 2)
        for i in range(self.num_leaps - 1):
            temper_const = self.tempering if i < (self.num_leaps / 2) else 1/self.tempering
            position, velocity = self._leapfrog_step(position, velocity, temper_const)

        position = self._update_position(position, velocity, self.step_size)
        velocity = self._update_velocity(position, velocity, self.step_size / 2)

        velocity = - velocity

        return (position, velocity)

    def accept_proposed_sample(self, current_PE, proposed_PE, current_KE, proposed_KE):
        print("Current PE ", current_PE, ", Proposed PE ", proposed_PE,
            ", Current KE", current_KE, ", Proposed KE ", proposed_KE)
        print(np.exp(current_PE - proposed_PE + current_KE - proposed_KE))
        print(self.current_position, self.current_velocity)
        print(self.grad_U(self.current_position))
        if np.log(np.random.rand()) < current_PE - proposed_PE + current_KE - proposed_KE:
            return True
        else:
            return False


    def _step(self):
        velocity = np.random.normal(size=[self.num_dimensions])
        self.current_velocity = velocity
        # pdb.set_trace()
        position, velocity = self._leapfrog(self.current_position, self.current_velocity)

        current_PE, current_KE = self.evaluate_energy(self.current_position, self.current_velocity)
        proposed_PE, proposed_KE = self.evaluate_energy(position, velocity)

        accept = self.accept_proposed_sample(current_PE, proposed_PE, current_KE, proposed_KE)

        if accept == True:
            self.current_position = position
            self.number_accepted += 1
        # Otherwise stay at the old value of position (self.current_position)

    def burn_in(self, burn_steps):
        pre_number_accepted = self.number_accepted
        for i in range(burn_steps):
            self._step()
        self.number_accepted = pre_number_accepted

    def sample(self, n_samples):
        self.total_number_of_draws += n_samples

        samples = np.zeros(shape=[n_samples, self.num_dimensions + 2])

        #samples[0, 0:-2] = self.current_position

        five_percent = round(0.05 * n_samples)

        for i in range(0, n_samples):
            self._step()
            samples[i, 0:-2] = self.current_position
            samples[i, -2:] = self.evaluate_energy(self.current_position, self.current_velocity)

            if i % 10 == 0:
                print("Run ", i, "/", n_samples, ": ", self.current_position)

            if i % five_percent == 0:
                print(i/n_samples* 100, "% sampled. ", self.current_position)

        return samples
