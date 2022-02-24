import collections
import gc

import torch


def free_up_memory(reset_counters=False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if reset_counters:
            torch.cuda.reset_peak_memory_stats()
    gc.collect()


def merge_dicts(*dicts):
    res = collections.defaultdict(list)
    for d in dicts:
        for k, v in d.items():
            if len(res[k]) == 0:
                res[k] = v
            else:
                res[k] = torch.cat((res[k], v))
    return res


# this function is designed to handle and recover from OOM errors while training
# it does so by freeing up as much memory as possible and splits the batch before retrying
# all gradients get accumulated while splitting the batches
# this is mainly a workaround for gpu vram fragmentation and helps to recover from OOM while starting the training
# until the memory consumption stabilizes or helps to handle random OOMs while training.
def train_step(model, input, target, criterion_handler, forward_handler=None, stack_loss=True, cat_out=True,
               handle_out_dict=True, handle_oom=True,
               steps=1):
    oom = False

    try:
        if steps == 1:
            if forward_handler is not None:
                out = forward_handler(model, input)
            else:
                out = model(input)

            if criterion_handler is not None:
                loss = criterion_handler(output=out, target=target, steps=steps, batch_size=len(input))
            else:
                loss = None

            return out, loss, False

        batches = torch.split(input, len(input) // steps)
        targets = torch.split(target, len(target) // steps)

        print('If this error persists, you should lower the batch_size. The training will try to recover from OOM '
              'errors, however it is highly inefficient!')
        print(f'Retrying with batch size {len(target) // steps}')

        free_up_memory()

        results = []
        losses = []
        out = None
        for i, mini_batch in enumerate(batches):
            try:
                if forward_handler is not None:
                    out = forward_handler(model, mini_batch)
                else:
                    out = model(mini_batch)

                if criterion_handler is not None:
                    loss = criterion_handler(output=out, target=targets[i], steps=steps, batch_size=len(mini_batch))
                    losses.append(loss)

            except Exception as e:
                if out is not None:
                    del out
                del results
                raise e
            results.append(out)

        if stack_loss:
            losses = torch.stack(losses).mean()

        if cat_out:
            if handle_out_dict and isinstance(results[0], dict):
                results = merge_dicts(*results)
            else:
                results = torch.cat(results)

        return results, losses, steps > 1
    except RuntimeError as e:
        if "out of memory" in str(e):
            if not handle_oom:
                raise e

            # this construct releases the exception which contains a stack frame
            # otherwise we would always get OOM
            print(f'Exception: {str(e)}')
            print(
                f'OOM occurred while trying to pass {len(input) // steps} into the model '
                f'on device {next(model.parameters()).device}')

            oom = True
            del e

            for p in model.parameters():
                p.grad = None
        else:
            raise e

    if oom:
        return train_step(model, input, target, criterion_handler, forward_handler, stack_loss, cat_out,
                          handle_out_dict, handle_oom, steps * 2)