import json
import argparse
from functools import partial

import optuna
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
import tensorflow as tf
from tensorflow.keras import datasets, activations, Sequential, layers, callbacks
import tensorflow.keras.utils as kutils

def build_model(trial, input_shape, output_shape, n_layers, activation, n_units, units_step):
    model = Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    for i in range(n_layers-1):
        layer_units = trial.suggest_int(f'layer_units_{i}', *n_units, step=units_step)
        layer = layers.Dense(layer_units, activation=activation, name=f'dense_{i}')
        model.add(layer)
    layer = layers.Dense(output_shape, activation='softmax', name='dense_output')
    model.add(layer)
    return model

def compile_and_fit(model, train_ds, val_ds):
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc', 'mse', 'mae'])
    cb = [
        callbacks.EarlyStopping(monitor='val_loss',
                        min_delta=1e-3, patience=10, verbose=0,
                        mode='min', restore_best_weights=True),
    ]
    history_log = model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=cb)
    return history_log, model

def hpo_func(trial, n_layers, activations, n_units, units_step):
    mlflow.tensorflow.autolog()

    with mlflow.start_run(experiment_id=EXPERIMENT, nested=True, run_name='MLP Classifier'):
        n_layers = trial.suggest_int('n_layers', *n_layers)
        activation = trial.suggest_categorical('activation', choices=activations)

        mlflow.log_param('n_layers', n_layers)
        mlflow.log_param('activaion', activation)
        mnist_dataset = datasets.mnist
        (x_fit, y_fit),(x_test, y_test) = mnist_dataset.load_data()
        x_fit, x_test = x_fit / 255.0, x_test / 255.0   #Normalizing the data
        y_fit = kutils.to_categorical(y_fit)
        y_test = kutils.to_categorical(y_test)
        input_shape = x_fit.shape[1:]
        output_shape = y_fit.shape[1]

        test_size = len(x_test)
        fit_idx = np.arange(len(x_fit))
        np.random.shuffle(fit_idx)
        x_train = x_fit[fit_idx[test_size:]]
        y_train = y_fit[fit_idx[test_size:]]
        x_val = x_fit[fit_idx[:test_size:]]
        y_val = y_fit[fit_idx[:test_size]]

        train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)
        test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

        # --- Building Model ----------------------------------------------------
        model = Sequential()
        model.add(layers.Flatten(input_shape=input_shape))
        for i in range(n_layers-1):
            layer_units = trial.suggest_int(f'layer_units_{i}', *n_units, step=units_step)
            layer = layers.Dense(layer_units, activation=activation, name=f'dense_{i}')
            model.add(layer)
        layer = layers.Dense(output_shape, activation='softmax', name='dense_output')
        model.add(layer)

        # --- Fitting Model -----------------------------------------------------
        model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['acc', 'mse', 'mae'])
        cb = [
            callbacks.EarlyStopping(monitor='val_loss',
                            min_delta=1e-3, patience=3, verbose=0,
                            mode='min', restore_best_weights=True),
        ]
        model.fit(train_ds, epochs=100, validation_data=val_ds, callbacks=cb)

        eval_values = model.evaluate(test_ds)
        eval_res = {'test_' + metric: value for metric, value in zip(model.metrics_names, eval_values)}
        mlflow.log_params(eval_res)

    return eval_res['test_acc']

if __name__ == '__main__':

    KERAS_ACTIVATIONS = [act for act in dir(activations) if not act.startswith('_')]
    global EXPERIMENT_NAME, RUN_NAME, EXPERIMENT
    EXPERIMENT_NAME = 'MNIST Classif'
    RUN_NAME = 'MLP Clasissifier HPO'
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_layers', nargs=2, type=int, help='Minimum and maximum number of layers respectivelly')
    parser.add_argument('--activations', nargs='+', type=str, help='Activation functions to test', choices=KERAS_ACTIVATIONS)
    parser.add_argument('--n_units', nargs=2, type=int, help='Minimum and maximum number of units per layer respectivelly')
    parser.add_argument('--units_step', type=int, help='Step for n_units interval')
    args = parser.parse_args().__dict__
    print(args)

    with open('databases_info.json', 'r') as json_file:
        db_info = json.load(json_file)

    client = MlflowClient()
    try:
        EXPERIMENT = client.create_experiment(EXPERIMENT_NAME)
    except:
        EXPERIMENT = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

    with mlflow.start_run(experiment_id=EXPERIMENT, run_name=RUN_NAME) as run:

        mlflow.log_params(args)
        study_name = run.info.run_id
        study = optuna.create_study(study_name=study_name, direction='maximize', storage=db_info['optuna'])
        objective = partial(hpo_func, **args)
        study.optimize(objective, n_trials=3)

        history_plot = optuna.visualization.plot_optimization_history(study)
        coord_plot = optuna.visualization.plot_parallel_coordinate(study, params=["n_layers", "activation"])
        importance_plot = optuna.visualization.plot_param_importances(study)
        slice_plot = optuna.visualization.plot_slice(study, params=["n_layers", "activation"])

        mlflow.log_figure(history_plot, 'history_plot.html')
        mlflow.log_figure(coord_plot, 'coord_plot.html')
        mlflow.log_figure(importance_plot, 'importance_plot.html')
        mlflow.log_figure(slice_plot, 'slice_plot.html')