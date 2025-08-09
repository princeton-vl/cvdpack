import logging
import multiprocessing
import os.path
import queue
import subprocess
from multiprocessing import Process

try:
    import paramiko
    import paramiko_jump
except ImportError:
    paramiko = None
    paramiko_jump = None

logger = logging.getLogger("cvdpack")


def get_scp_process_queue(ssh_config):
    user = ssh_config["User"]
    hostname = ssh_config["HostName"]
    if hostname is None or hostname == "":
        return None, None
    ssh_configs = list(f'{k}=\"{os.path.expanduser(v)}\" ' for k, v in ssh_config.items() if v is not None)
    ssh_config_cmd = []
    for c in ssh_configs:
        ssh_config_cmd.extend(['-o', c])

    remote = f'{user}@{hostname}' if user else hostname
    ssh_cmd = ['ssh', *ssh_config_cmd, remote,"exit"]
    logger.debug('ssh command: %s', ' '.join(ssh_cmd))
    subprocess.call(ssh_cmd)
    logger.info('ssh connected')
    jobs = multiprocessing.Queue()

    def sftp_watch(jobs):
        while True:
            try:
                logger.debug('Waiting for job')
                j = jobs.get(timeout=30)
                logger.debug('Received job: %s', j)
                if j is None:
                    break
            except queue.Empty:
                continue
            if j is None:
                logger.debug('No jobs received')
                continue
            try:
                local_path = str(j.output_path)
                remote_path = str(j.remote_path).partition(':')[-1]
                logger.info(f"Moving {local_path} to {remote_path}")
                cmd = ['ssh', *ssh_config_cmd, remote,
                    f"mkdir -p {os.path.dirname(remote_path)}"]
                logger.debug('ssh command: %s', ' '.join(cmd))
                subprocess.call(cmd)
                cmd = ['scp', *ssh_config_cmd, local_path, f'{user}@{hostname}:{remote_path}']
                logger.debug('scp command: %s', ' '.join(cmd))
                subprocess.call(cmd)
                logger.info(f"Moving {local_path} to {remote_path} finished")
                subprocess.check_call(['rm', local_path])
            except Exception as e:
                logger.warning("Scp has failed: %s", str(e))

    return Process(target=sftp_watch, args=(jobs,)), jobs
