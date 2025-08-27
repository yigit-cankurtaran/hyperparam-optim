"""
Complete RL Hyperparameter Tuning Demo with Optuna
This shows how to tune PPO hyperparameters on CartPole-v1
"""

import optuna
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================
N_TRIALS = 20           # How many hyperparameter combinations to try
N_STARTUP_TRIALS = 5    # Random trials before Bayesian optimization kicks in
BUDGET_TIMESTEPS = 50_000  # Budget per trial (keep low for demo)
N_EVAL_EPISODES = 5     # Episodes for final evaluation
ENV_ID = "CartPole-v1"  # Environment to tune on

# ============================================================================
# HYPERPARAMETER SAMPLING FUNCTION
# ============================================================================
def sample_ppo_params(trial):
    """
    Sample PPO hyperparameters for this trial.
    Optuna will intelligently choose these values.
    """
    return {
        # Learning rate: most important hyperparameter
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        
        # Network architecture
        'net_arch': trial.suggest_categorical('net_arch', [
            [64, 64],           # Small network
            [128, 128],         # Medium network  
            [256, 256],         # Large network
            [64, 64, 64],       # Deeper small
        ]),
        
        # PPO-specific parameters
        'n_steps': trial.suggest_categorical('n_steps', [128, 256, 512, 1024, 2048]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 10),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 5.0),
    }

# ============================================================================
# OBJECTIVE FUNCTION (THE HEART OF TUNING)
# ============================================================================
def objective(trial):
    """
    This function gets called for each trial.
    It trains a model with the sampled hyperparameters and returns a score.
    Optuna will try to MAXIMIZE this score.
    """
    
    # 1. Sample hyperparameters for this trial
    params = sample_ppo_params(trial)
    
    print(f"\n=== TRIAL {trial.number} ===")
    print(f"Testing params: {params}")
    
    try:
        # 2. Create environment
        env = make_vec_env(ENV_ID, n_envs=4)  # 4 parallel envs for speed
        
        # 3. Create model with sampled hyperparameters
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=0,  # Suppress training logs
            policy_kwargs=dict(net_arch=params['net_arch']),
            # Pass all the sampled hyperparameters
            learning_rate=params['learning_rate'],
            n_steps=params['n_steps'],
            batch_size=params['batch_size'],
            n_epochs=params['n_epochs'],
            gamma=params['gamma'],
            gae_lambda=params['gae_lambda'],
            clip_range=params['clip_range'],
            ent_coef=params['ent_coef'],
            vf_coef=params['vf_coef'],
            max_grad_norm=params['max_grad_norm'],
        )
        
        # 4. Train the model for our budget
        model.learn(total_timesteps=BUDGET_TIMESTEPS)
        
        # 5. Evaluate final performance
        eval_env = gym.make(ENV_ID)
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=N_EVAL_EPISODES, deterministic=True
        )
        
        # 6. Clean up
        env.close()
        eval_env.close()
        del model  # Free memory
        
        print(f"Trial {trial.number} result: {mean_reward:.2f} ± {std_reward:.2f}")
        
        # 7. Return the score (Optuna tries to maximize this)
        return mean_reward
        
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        # Return a very bad score for failed trials
        return -1000.0

# ============================================================================
# ADVANCED OBJECTIVE WITH INTERMEDIATE VALUES (OPTIONAL)
# ============================================================================
def objective_with_pruning(trial):
    """
    More advanced version that reports intermediate results.
    This allows Optuna to prune (stop) bad trials early.
    """
    params = sample_ppo_params(trial)
    
    try:
        env = make_vec_env(ENV_ID, n_envs=4)
        eval_env = gym.make(ENV_ID)
        
        # Custom callback to report intermediate results
        class OptunaPruningCallback(EvalCallback):
            def __init__(self, trial, eval_env, **kwargs):
                super().__init__(eval_env, **kwargs)
                self.trial = trial
                self.eval_num = 0
            
            def _on_step(self) -> bool:
                # Call parent method
                continue_training = super()._on_step()
                
                # Report intermediate value to Optuna every evaluation
                if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
                    self.eval_num += 1
                    # Use the last mean reward as intermediate value
                    if hasattr(self, 'last_mean_reward'):
                        self.trial.report(self.last_mean_reward, self.eval_num)
                        
                        # Check if trial should be pruned
                        if self.trial.should_prune():
                            print(f"Trial {self.trial.number} pruned at evaluation {self.eval_num}")
                            return False  # Stop training
                
                return continue_training
        
        # Create model
        model = PPO(
            policy='MlpPolicy',
            env=env,
            verbose=0,
            policy_kwargs=dict(net_arch=params['net_arch']),
            **{k: v for k, v in params.items() if k != 'net_arch'}
        )
        
        # Create callback that reports to Optuna
        callback = OptunaPruningCallback(
            trial=trial,
            eval_env=eval_env,
            eval_freq=BUDGET_TIMESTEPS // 4,  # Evaluate 4 times during training
            n_eval_episodes=3,
            verbose=0
        )
        
        # Train with pruning
        model.learn(total_timesteps=BUDGET_TIMESTEPS, callback=callback)
        
        # Final evaluation
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=N_EVAL_EPISODES)
        
        # Cleanup
        env.close()
        eval_env.close()
        del model
        
        return mean_reward
        
    except optuna.TrialPruned:
        # This trial was pruned
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        return -1000.0

# ============================================================================
# MAIN TUNING FUNCTION
# ============================================================================
def run_hyperparameter_tuning():
    """
    Main function that sets up and runs the hyperparameter optimization.
    """
    
    print("🚀 Starting RL Hyperparameter Tuning with Optuna")
    print(f"Environment: {ENV_ID}")
    print(f"Budget per trial: {BUDGET_TIMESTEPS:,} timesteps")
    print(f"Number of trials: {N_TRIALS}")
    print("=" * 60)
    
    # 1. Create Optuna study
    study = optuna.create_study(
        direction='maximize',  # We want to maximize reward
        
        # Bayesian optimization algorithm
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=N_STARTUP_TRIALS,  # Random trials before TPE
            multivariate=True,  # Consider parameter interactions
        ),
        
        # Optional: pruning for early stopping of bad trials
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=N_STARTUP_TRIALS,
            n_warmup_steps=2,  # Wait for 2 intermediate reports
        ),
        
        study_name=f"ppo_{ENV_ID}_tuning"
    )
    
    # 2. Run the optimization
    try:
        study.optimize(
            objective,  # Use objective_with_pruning for advanced version
            n_trials=N_TRIALS,
            timeout=None,  # No time limit
            n_jobs=1,  # Parallel jobs (be careful with GPU memory)
        )
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    
    # 3. Print results
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best reward: {study.best_value:.2f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 4. Show optimization history
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    top_trials = trials_df.nlargest(5, 'value')
    for idx, row in top_trials.iterrows():
        print(f"  Trial {row['number']}: {row['value']:.2f}")
    
    # 5. Optional: Save results
    study.trials_dataframe().to_csv(f"optuna_results_{ENV_ID}.csv")
    print(f"\nResults saved to optuna_results_{ENV_ID}.csv")
    
    return study

# ============================================================================
# VALIDATION FUNCTION
# ============================================================================
def validate_best_params(study, validation_timesteps=200_000):
    """
    Train the best configuration for longer to get a more reliable estimate.
    """
    print("\n🔬 VALIDATING BEST HYPERPARAMETERS")
    print("=" * 60)
    
    best_params = study.best_params
    print(f"Validating with {validation_timesteps:,} timesteps...")
    
    # Recreate best model
    env = make_vec_env(ENV_ID, n_envs=4)
    
    # Extract net_arch separately
    net_arch = best_params.pop('net_arch')
    
    model = PPO(
        policy='MlpPolicy',
        env=env,
        verbose=1,  # Show progress this time
        policy_kwargs=dict(net_arch=net_arch),
        **best_params
    )
    
    # Train for longer
    model.learn(total_timesteps=validation_timesteps)
    
    # Thorough evaluation
    eval_env = gym.make(ENV_ID)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    
    print(f"\n✅ Validation Results:")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Original trial reward: {study.best_value:.2f}")
    
    # Save the validated model
    model.save(f"best_model_{ENV_ID}")
    print(f"Best model saved as best_model_{ENV_ID}.zip")
    
    env.close()
    eval_env.close()
    
    return model

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Run hyperparameter tuning
    study = run_hyperparameter_tuning()
    
    # Validate best hyperparameters with longer training
    best_model = validate_best_params(study, validation_timesteps=200_000)
    
    print("\n🎉 Hyperparameter tuning complete!")
    print("Check the saved CSV file and model for results.")
