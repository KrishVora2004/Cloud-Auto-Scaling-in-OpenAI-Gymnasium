from tests.metrics import compute_metrics, aggregate


def evaluate_model(env_fn, model, scenario, seeds=[1, 42, 100]):
    all_results = []

    for seed in seeds:
        env = env_fn(scenario, seed)

        obs, _ = env.reset(seed=seed)
        done = False

        data = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            data.append(info)

        all_results.append(compute_metrics(data))

    return aggregate(all_results)