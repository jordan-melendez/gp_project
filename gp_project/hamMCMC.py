import numpy as np

class HamiltonianSampler:

    def __init__(self, U, grad_U, num_leaps, step_size):
        self.U = U
        self.grad_U = grad_U

        self.num_leaps = num_leaps
        self.step_size = step_size

        self.current_position = None
        self.current_velocity = None
        self.num_dimensions = None

    def set_seed(self, seed):
        np.random.seed(seed)

    def initialize(self, start_q):
        self.num_dimensions = len(start_q)

        self.current_position = start_q

    def kinetic_energy(self, velocity):
        return np.sum(np.square(velocity)) / 2

    def evaluate_energy(self, position, velocity):
        return (self.U(position), self.kinetic_energy(velocity))

    def _update_position(self, position, velocity, step_size):
        return position + step_size * velocity

    def _update_velocity(self, position, velocity, step_size):
        return velocity - step_size * self.grad_U(position)

    def _leapfrog_step(self, position, velocity):
        position = self._update_position(position, velocity, self.step_size)
        velocity = self._update_velocity(position, velocity, self.step_size)
        return (position, velocity)

    def _leapfrog(self, position, velocity):
        velocity = self._update_velocity(position, velocity, self.step_size / 2)
        for _ in range(self.num_leaps - 1):
            position = self._update_position(position, velocity, self.step_size)
            velocity = self._update_velocity(position, velocity, self.step_size)

        position = self._update_position(position, velocity, self.step_size)
        velocity = self._update_velocity(position, velocity, self.step_size / 2)

        velocity = - velocity

        return (position, velocity)

    def accept_proposed_sample(self, current_PE, proposed_PE, current_KE, proposed_KE):
        if np.random.rand() < np.exp(current_PE - proposed_PE + current_KE - proposed_KE):
            return True
        else:
            return False


    def _step(self):
        velocity = np.random.normal(size=[self.num_dimensions])
        self.current_velocity = velocity

        position, velocity = self._leapfrog(position, velocity)

        current_PE, current_KE = self.evaluate_energy(self.current_position, self.current_velocity)
        proposed_PE, proposed_KE = self.evaluate_energy(position, velocity)

        accept = self.accept_proposed_sample(current_PE, proposed_PE, current_KE, proposed_KE)

        if accept == True:
            self.current_position = position
        # Otherwise stay at the old value of position (self.current_position)

    def burn_in(self, burn_steps):
        for _ in range(burn_steps);
            self._step()

    def sample(self, n_samples):
        samples = np.zeros(shape=[n_samples, self.num_dimensions])

        samples[0,:] = self.current_position

        for i in range(1, n_samples):
            self._step()
            samples[i, :] = self.current_position

        return samples