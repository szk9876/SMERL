import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_palette("colorblind")

env_names = ['HalfCheetahGoal', 'WalkerVelocity', 'HopperVelocity']
results_types = ['ObstaclesResults', 'JointsResults', 'MotorResults']

for env_name in env_names:
	for results_type in results_types:

		if results_type == 'ObstaclesResults':
			results = 'obstacles'
		elif results_type =='JointsResults':
			results = 'joints'
		elif results_type == 'MotorResults':
			results = 'motor'

		if results_type == 'ObstaclesResults':
			if env_name == 'HalfCheetahGoal':
				x = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55])
			else:
				x = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 
					0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575])
		elif results_type == 'JointsResults':
			if env_name == 'HalfCheetahGoal':
				x = 100.0 * np.arange(1, 20)
			else:
				x = 10.0 * np.arange(1, 20)
		elif results_type == 'MotorResults':
			x = np.arange(10, 110, 10)

		num_heights = len(x)

		sac_1_skill_results = np.zeros((num_heights, 3))
		sac_5_skills_results = np.zeros((num_heights, 3))
		sac_plus_diayn_results = np.zeros((num_heights, 3))
		smerl_results = np.zeros((num_heights, 3))
		diayn_results = np.zeros((num_heights, 3))

		num_seeds = 3
		for i in range(num_seeds):			
			sac_1_skill_results[:, i] = np.load('{}/{}/{}_sac_1_skill_seed{}_{}.npy'.format(env_name, results_type, env_name, i, results)).mean(axis=-1)
			sac_5_skills_results[:, i] = np.load('{}/{}/{}_sac_5_skills_seed{}_{}.npy'.format(env_name, results_type, env_name, i, results)).mean(axis=-1)
			sac_plus_diayn_results[:, i] = np.load('{}/{}/{}_sac+diayn_seed{}_{}.npy'.format(env_name, results_type, env_name, i, results)).mean(axis=-1)
			smerl_results[:, i] = np.load('{}/{}/{}_smerl_seed{}_{}.npy'.format(env_name, results_type, env_name, i, results)).mean(axis=-1)
			diayn_results[:, i] = np.load('{}/{}/{}_diayn_seed{}_{}.npy'.format(env_name, results_type, env_name, i, results)).mean(axis=-1)
		
		end = len(x)
		x = x[0:end]

		sac_1_skill_mean = sac_1_skill_results.mean(axis=-1)[0:end]
		sac_1_skill_error = sac_1_skill_results.std(axis=-1)[0:end] * 0.5

		sac_5_skills_mean = sac_5_skills_results.mean(axis=-1)[0:end]
		sac_5_skills_error = sac_5_skills_results.std(axis=-1)[0:end] * 0.5

		smerl_mean = smerl_results.mean(axis=-1)[0:end]
		smerl_error = smerl_results.std(axis=-1)[0:end] * 0.5

		diayn_mean = diayn_results.mean(axis=-1)[0:end]
		diayn_error = diayn_results.std(axis=-1)[0:end] * 0.5

		sac_plus_diayn_mean = sac_plus_diayn_results.mean(axis=-1)[0:end]
		sac_plus_diayn_error = sac_plus_diayn_results.std(axis=-1)[0:end] * 0.5

		alpha = 0.3

		plt.plot(x, sac_1_skill_mean, label='SAC (1 Policy)', marker='o', markersize=8, linewidth=2)
		plt.fill_between(x, sac_1_skill_mean-sac_1_skill_error, sac_1_skill_mean+sac_1_skill_error, alpha=alpha)

		plt.plot(x, smerl_mean, label='SMERL', marker='x', markersize=8, linewidth=2.5, color='k')
		plt.fill_between(x, smerl_mean-smerl_error, smerl_mean+smerl_error, alpha=alpha, color='k')

		plt.plot(x, sac_5_skills_mean, label='SAC (5 Policies)', marker='^', markersize=8, linewidth=2.5)
		plt.fill_between(x, sac_5_skills_mean-sac_5_skills_error, sac_5_skills_mean+sac_5_skills_error, alpha=alpha)
		
		plt.plot(x, sac_plus_diayn_mean, label='SAC+DIAYN', marker='D', markersize=8, linewidth=2.5)
		plt.fill_between(x, sac_plus_diayn_mean-sac_plus_diayn_error, sac_plus_diayn_mean+sac_plus_diayn_error, alpha=alpha)

		plt.plot(x, diayn_mean, label='DIAYN', marker='+', linewidth=2)
		plt.fill_between(x, diayn_mean-diayn_error, diayn_mean+diayn_error, alpha=alpha)

		plt.legend(fontsize=10)
		plt.xticks(fontsize=14)
		plt.yticks(fontsize=14)

		plt.savefig('{}/{}_{}.png'.format(env_name, env_name, results))
		plt.close()