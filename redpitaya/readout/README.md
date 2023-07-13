# Redpitaya readout module

Copy the `readout.service` file to `/etc/systemd/system/`, then:
- enable it: `systemctl enable readout.service`
- start/stop it: `systemctl start/stop readout.service`

## Benchmarks
- Data serialisation: `benchmarks/serialisation.py`
- Data transfer: `benchmarks/transfer.py`
- Readout: `benchmarks/readout.py`
