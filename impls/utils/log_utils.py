import os
import tempfile
from datetime import datetime

import absl.flags as flags
import ml_collections
import numpy as np
import wandb
import jax
from collections import defaultdict
from PIL import Image, ImageEnhance


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(seed):
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'sd{seed:03d}_'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f's_{os.environ["SLURM_JOB_ID"]}.'
    if 'SLURM_PROCID' in os.environ:
        exp_name += f'{os.environ["SLURM_PROCID"]}.'
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    return exp_name


def get_flag_dict():
    """Return the dictionary of flags."""
    flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS if '.' not in k}
    for k in flag_dict:
        if isinstance(flag_dict[k], ml_collections.ConfigDict):
            flag_dict[k] = flag_dict[k].to_dict()
    return flag_dict


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        config=get_flag_dict(),
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, newshape=(n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, newshape=(t, c, n_rows * h, n_cols * w))

    return v


def get_wandb_video(renders=None, n_cols=None, fps=15):
    """Return a Weights & Biases video.

    It takes a list of videos and reshapes them into a single video with the specified number of columns.

    Args:
        renders: List of videos. Each video should be a numpy array of shape (t, h, w, c).
        n_cols: Number of columns for the reshaped video. If None, it is set to the square root of the number of videos.
    """
    # Pad videos to the same length.
    max_length = max([len(render) for render in renders])
    for i, render in enumerate(renders):
        assert render.dtype == np.uint8

        # Decrease brightness of the padded frames.
        final_frame = render[-1]
        final_image = Image.fromarray(final_frame)
        enhancer = ImageEnhance.Brightness(final_image)
        final_image = enhancer.enhance(0.5)
        final_frame = np.array(final_image)

        pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
        renders[i] = np.concatenate([render, pad], axis=0)

        # Add borders.
        renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    renders = np.array(renders)  # (n, t, h, w, c)

    renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

    return wandb.Video(renders, fps=fps, format='mp4')


def _nparams(x):
    return int(np.prod(x.shape))


def _path_to_tuple(path):
    out = []
    for k in path:
        if hasattr(k, "key"):
            out.append(str(k.key))
        elif hasattr(k, "idx"):
            out.append(str(k.idx))
        else:
            out.append(str(k))
    return tuple(out)


def _count_tree(pytree):
    leaves = jax.tree_util.tree_leaves(pytree)
    return int(sum(np.prod(x.shape) for x in leaves))


def _collect_counts_and_kernel_shapes(module_params, inner_depth=2):
    parent_totals = defaultdict(int)
    child_totals = defaultdict(lambda: defaultdict(int))
    child_kernel_shapes = defaultdict(lambda: defaultdict(list)) 

    flat = jax.tree_util.tree_flatten_with_path(module_params)[0]
    for path, leaf in flat:
        p = _path_to_tuple(path)

        key = p[:inner_depth] if len(p) >= inner_depth else p
        parent = key[0] if len(key) >= 1 else "<root>"
        child = "/".join(key[1:]) if len(key) >= 2 else ""

        n = _nparams(leaf)
        parent_totals[parent] += n
        if child != "":
            child_totals[parent][child] += n

        if len(p) > 0 and p[-1] == "kernel" and child != "":
            child_kernel_shapes[parent][child].append(tuple(leaf.shape))

    parent_totals = dict(sorted(parent_totals.items(), key=lambda kv: kv[1], reverse=True))
    child_totals = {pa: dict(sorted(ch.items(), key=lambda kv: kv[1], reverse=True)) for pa, ch in child_totals.items()}

    for pa in child_kernel_shapes:
        for ch in child_kernel_shapes[pa]:
            shapes = child_kernel_shapes[pa][ch]
            uniq = sorted({str(s): s for s in shapes}.items(), key=lambda kv: kv[0])
            child_kernel_shapes[pa][ch] = [s for _, s in uniq]

    return parent_totals, child_totals, child_kernel_shapes


def print_network_breakdown(agent_params, names=None, inner_depth=2, max_inner=20, verbose=False):
    # --- build module dict (same as before) ---
    all_mods = {}
    for k, v in agent_params.items():
        if k.startswith("modules_"):
            name = k[len("modules_"):]
            all_mods[name] = v

    # --- make a filtered copy of agent_params for TOTAL ---
    filtered_agent_params = dict(agent_params)
    for mod_name in list(all_mods.keys()):
        if "target" in mod_name:
            filtered_agent_params.pop(f"modules_{mod_name}", None)

    print("▬"*35)
    print(" NETWORK PARAMETER BREAKDOWN")
    print(f" TOTAL: {_count_tree(filtered_agent_params):,}")
    print("▬"*35)

    if verbose:
        # --- names default ---
        if names is None:
            names = sorted(all_mods.keys())

        for name in names:
            if "target" in name:
                continue
            if name not in all_mods:
                continue

            mp = all_mods[name]
            mt = _count_tree(mp)
            print(f"\n{name}: {mt:,}")

            parent_totals, child_totals, child_kernel_shapes = _collect_counts_and_kernel_shapes(
                mp, inner_depth=inner_depth
            )

            shown_parent = 0
            for parent, pcount in parent_totals.items():
                if shown_parent >= max_inner:
                    print(f"  ... ({len(parent_totals)-max_inner} more)")
                    break
                print(f"  {parent}: {pcount:,}")

                if parent in child_totals:
                    for child, ccount in child_totals[parent].items():
                        shapes = child_kernel_shapes.get(parent, {}).get(child, [])
                        shape_str = ""
                        if len(shapes) == 1:
                            shape_str = f" {shapes[0]}"
                        elif len(shapes) > 1:
                            shape_str = " " + " ".join(str(s) for s in shapes)
                        print(f"    {child}: {ccount:,}{shape_str}")

                shown_parent += 1
