from copy import deepcopy


def save_best(module, curr_loss, curr_epoch, snapshot, wait_epochs, wait_tol):
    exit_training = False
    last_update, best_loss, _ = snapshot
    improvement = (best_loss - curr_loss) / (abs(best_loss) + 1e-6)
    if improvement > wait_tol:
        snapshot = (curr_epoch, curr_loss, deepcopy(module.state_dict()))
    if curr_epoch == snapshot[0] + wait_epochs:
        exit_training = True
    return exit_training, snapshot


def linear_decay_handle(initial_val, final_val, start, stop):
    """
    Returns function handle to use in torch.optim.lr_scheduler.LambdaLR.
    The returned function handle supplies the multiplier to decay a value linearly.
    """
    assert stop > start

    def decay_fn(counter):
        if counter <= start:
            return 1
        if counter >= stop:
            return final_val / initial_val
        time_range = stop - start
        return 1 - (counter - start) * (1 - final_val / initial_val) / time_range

    assert decay_fn(start) * initial_val == initial_val
    assert decay_fn(stop) * initial_val == final_val
    return decay_fn
