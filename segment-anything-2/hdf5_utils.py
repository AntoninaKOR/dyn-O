import h5py
import os


def is_hdf5_file_readable(filepath):
    try:
        with h5py.File(filepath, 'r') as f:
            def visit_func(name, obj):
                if isinstance(obj, h5py.Group):
                    _ = list(obj.attrs.items())     # Read attributes
                    _ = list(obj.keys())            # List datasets in the group

            # Traverse all groups
            f.visititems(visit_func)
            return True
    except Exception:
        return False


def find_latest_backup(backup_files):
    valid_backups = [f for f in backup_files if os.path.exists(f) and is_hdf5_file_readable(f)]

    if not valid_backups:
        return None

    latest_backup = max(valid_backups, key=os.path.getmtime)
    return latest_backup
