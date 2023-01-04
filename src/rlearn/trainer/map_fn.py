from tensorflow import keras

OPTIMIZER_MAP = {
    "rmsprop": lambda learning_rate, args: keras.optimizers.RMSprop(
        learning_rate=learning_rate, rho=args.get("rho", 0.9), momentum=args.get("momentum", 0.0),
        epsilon=args.get("epsilon", 1e-07)
    ),
    "adam": lambda learning_rate, args: keras.optimizers.Adam(
        learning_rate=learning_rate, beta_1=args.get("beta_1", 0.9), beta_2=args.get("beta_2", 0.999),
        epsilon=args.get("epsilon", 1e-07)
    ),
    "sgd": lambda learning_rate, args: keras.optimizers.SGD(
        learning_rate=learning_rate, momentum=args.get("momentum", 0.0)
    ),
    "adadelta": lambda learning_rate, args: keras.optimizers.Adadelta(
        learning_rate=learning_rate, rho=args.get("rho", 0.9), epsilon=args.get("epsilon", 1e-07)
    ),
}

LOSS_MAP = {
    "binary_crossentropy": lambda args: keras.losses.BinaryCrossentropy(
        from_logits=args.get("from_logits", False)),
    "categorical_crossentropy": lambda args: keras.losses.CategoricalCrossentropy(
        from_logits=args.get("from_logits", False)
    ),
    "sparse_categorical_crossentropy": lambda args: keras.losses.SparseCategoricalCrossentropy(
        from_logits=args.get("from_logits", False)
    ),
    "mean_squared_error": lambda args: keras.losses.MeanSquaredError(),
    "mean_absolute_error": lambda args: keras.losses.MeanAbsoluteError(),
}
