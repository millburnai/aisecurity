"""

"aisecurity.extras.demo"

Demonstration of facial recognition system.

"""


def demo(model="ms_celeb_1m", path=None, use_log=True, use_dynamic=True, verbose=False, using_picamera=False):

    # default arg values (for Pycharm, where args.* default to None)
    if model is None:
        model = "ms_celeb_1m"
    if use_log is None:
        use_log = True
    if use_dynamic is None:
        use_dynamic = True
    if verbose is None:
        verbose = False

    if not verbose:
        # SETUP
        import os

        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

        # ERROR HANDLING
        import warnings

        warnings.simplefilter(action="ignore", category=FutureWarning)
        warnings.simplefilter(action="ignore", category=UserWarning)
        warnings.simplefilter(action="ignore", category=RuntimeWarning)

        import tensorflow as tf

        try:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        except AttributeError:
            tf.logging.set_verbosity(tf.logging.ERROR)

        # PATH SETUP
        try:
            # for Pycharm
            from aisecurity.extras.paths import HOME, DATABASE

        except ModuleNotFoundError:
            # for terminal
            import sys
            from paths import HOME, DATABASE

            try:
                sys.path.insert(1, os.getenv("HOME") + "/PycharmProjects/aisecurity/")
            except FileNotFoundError:
                sys.path.insert(1, os.getenv("HOME") + "/Desktop/aisecurity/")

    from aisecurity.facenet import FaceNet, retrieve_embeds, cprint
    from aisecurity.extras.paths import DATABASE, CONFIG_HOME

    cprint("\nLoading facial recognition system", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    # facenet = FaceNet(path if path else CONFIG_HOME + "/models/{}.pb".format(model))
    facenet = FaceNet("/home/ryan/scratchpad/aisecurity/models/frozen.uff")

    cprint("\nLoading encrypted database", attrs=["bold"], end="")
    cprint("...", attrs=["bold", "blink"])
    facenet.set_data(retrieve_embeds(DATABASE, encrypted="names"))

    input("\nPress ENTER to continue:")

    facenet.real_time_recognize(use_log=use_log, use_dynamic=use_dynamic, using_picamera=using_picamera)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="(boolean) suppress warnings and TensorFlow output")
    parser.add_argument("--model", help="name of facenet model")
    parser.add_argument("--path_to_model", help="path to facenet model")
    args = parser.parse_args()

    demo(model=args.model, path=args.path_to_model, verbose=args.verbose)
