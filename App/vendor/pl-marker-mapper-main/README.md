# Pupil Labs Marker Mapper

[![ci](https://github.com/pupil-labs/pl-marker-mapper/actions/workflows/main.yml/badge.svg)](https://github.com/pupil-labs/pl-marker-mapper/actions/workflows/main.yml)
[![documentation](https://img.shields.io/badge/docs-mkdocs-708FCC.svg?style=flat)](https://pupil-labs.github.io/pl-marker-mapper/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre_commit-black?logo=pre-commit&logoColor=FAB041)](https://github.com/pre-commit/pre-commit)
[![pypi version](https://img.shields.io/pypi/v/pupil-labs-marker-mapper.svg)](https://pypi.org/project/pupil-labs-marker-mapper/)
[![python version](https://img.shields.io/pypi/pyversions/pupil-labs-marker-mapper)](https://pypi.org/project/pupil-labs-marker-mapper/)

[![pl-marker-mapper banner](https://raw.githubusercontent.com/pupil-labs/pl-marker-mapper/refs/heads/main/docs/assets/banner.png)](https://pupil-labs.com/)

This is an implementation of the Marker Mapper functionality as it can be found in e.g. [Pupil Cloud](https://cloud.pupil-labs.com/) and [Neon Player](https://docs.pupil-labs.com/neon/neon-player/). This implementation allows both post hoc and real-time gaze mapping onto surfaces defined by AprilTags.

The examples demonstrate how to interact with the Marker Mapper's interface. There is also an example app that demonstrates the Marker Mapper functionality in real-time in combination with a Neon device.

Documentation of this library is currently still lacking, but you can find reference the [Surface Tracking plugin in Neon Player](https://github.com/pupil-labs/neon-player-beta/blob/main/src/pupil_labs/neon_player/plugins/surface_tracking/surface_tracking.py#L44) as a reference.

## Installation

```bash
pip install -e git+https://github.com/pupil-labs/pl-marker-mapper.git
```
