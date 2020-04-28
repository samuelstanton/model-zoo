from copy import deepcopy


def save_best(module, curr_loss, curr_epoch, snapshot, wait_epochs, wait_tol):
    exit_training = False
    last_update, best_loss, _ = snapshot
    improvement = (best_loss - curr_loss) / abs(best_loss)
    if improvement > wait_tol:
        snapshot = (curr_epoch, curr_loss, deepcopy(module.state_dict()))
    if curr_epoch == snapshot[0] + wait_epochs:
        exit_training = True
    return exit_training, snapshot
