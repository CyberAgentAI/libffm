import optuna
import json
import subprocess
import multiprocessing


def get_meta_path(trial_number: int):
    return f"./data/optuna/ffm-meta-{trial_number}.json"


def objective(trial: optuna.Trial):
    lmd = trial.suggest_loguniform("lambda", 1e-6, 1)
    eta = trial.suggest_loguniform("eta", 1e-6, 1)
    json_meta_path = get_meta_path(trial.number)

    commands = [
        "./ffm-train",
        "-p", "./data/valid2.txt",
        "--auto-stop", "--auto-stop-threshold", "3",
        "-l", str(lmd),
        "-r", str(eta),
        "-k", "4",
        "-t", str(500),
        "--json-meta", json_meta_path,
        "./data/train2.txt",
    ]

    result = subprocess.run(
        commands,
        capture_output=True,
        universal_newlines=True,
        encoding='utf-8')

    trial.set_user_attr("args", result.args)
    best_iteration = None
    best_va_loss = None
    with open(json_meta_path) as f:
        json_dict = json.load(f)
        best_iteration = json_dict.get('best_iteration')
        best_va_loss = json_dict.get('best_va_loss')

    if best_iteration is None or best_va_loss is None:
        raise ValueError("failed to open json meta")

    trial.set_user_attr("best_iteration", best_iteration)
    return best_va_loss


def main():
    storage = optuna.storages.RDBStorage(
        "sqlite:///db.sqlite3",
        engine_kwargs={"pool_size": 1})
    sampler = optuna.integration.SkoptSampler()
    study = optuna.load_study(
        study_name="dynalyst-ffm-gp",
        storage=storage,
        sampler=sampler)
    study.optimize(
        objective,
        n_trials=256,
        n_jobs=multiprocessing.cpu_count() - 1,
        catch=())
    print("best_trial", study.best_trial.number)
    print("best_params", study.best_params)
    print("best_value", study.best_value)


if __name__ == '__main__':
    main()
