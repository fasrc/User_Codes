'''
2D Lattice Kuramoto solvers using explicit fixed time-stepping
'''

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import line_profiler
import atexit

@dataclass
class KuramotoSolver:
	N: int 												# Size of the domain (N x N)
	K: float = 3.0 								# Coupling constant
	t: float = 0.0 								# Start time
	T: float = 10.0 							# Stop time
	dt: float = 0.1 							# Time step
	dtype: np.dtype = np.float64 	# Precision for computations
	record: bool = False 		# Whether to store solutions
	profile: bool = False 				# Whether to run the line-profiler


	def __post_init__(self):
		# Initialize the state
		np.random.seed(0) # For reproducibility
		self.initial_state = np.random.uniform(0, 2*np.pi, size=(self.N, self.N)).astype(self.dtype)
		self.state = self.initial_state.copy()
		self.omega = np.random.uniform(0, 1, size=(self.N, self.N)).astype(self.dtype) # Intrinsic frequencies
		self.samples = [self.state.copy()] if self.record else []
		if self.profile or self.record:
			print(f'Using dtype: {self.dtype}')
		if self.profile:
			print('Running profiler.')
			self.profiler = line_profiler.LineProfiler()
			self.dudt = self.profiler(self.dudt)


	@abstractmethod
	def dudt(self, u: npt.NDArray) -> npt.NDArray:
		'''
		Compute the vector field from the current state
		'''
		pass

	def step(self):
		self.state += self.dudt(self.state) * self.dt

	def integrate(self):
		if self.profile or self.record:
			print('Solving...')
		while self.t < self.T:
			self.step()
			if self.record:
				self.samples.append(self.state.copy())
			self.t += self.dt
		if self.profile or self.record:
			print('Finished.')
		if self.profile:
			self.profiler.print_stats()

	def play_video(self):
		fig = plt.figure(figsize=(5, 4))
		ax = fig.add_subplot(autoscale_on=False, xlim=(0, self.N-1), ylim=(0, self.N-1))
		im = ax.imshow(np.sin(self.initial_state), interpolation='None')
		im.set_clim(-1, 1)
		div = make_axes_locatable(ax)
		cax = div.append_axes('right', '5%', '5%')
		cb = fig.colorbar(im, cax=cax)

		def animate(i):
			im.set_data(np.sin(self.samples[i]))
			return [im]

		anim = animation.FuncAnimation(fig, animate, frames=len(self.samples), interval=self.dt*1000, blit=True)
		plt.show()
		return anim
