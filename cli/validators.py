from argparse import ArgumentTypeError

def validate_args(args) -> None:
    """
    Валидатор для параметров передаваемых через командную строку
    """
    if args.command == "train":
        if args.learning_rate <= 0:
            raise ArgumentTypeError("--learning_rate должен быть > 0")
        if args.num_train_epochs <= 0:
            raise ArgumentTypeError("--num_train_epochs должен быть > 0")